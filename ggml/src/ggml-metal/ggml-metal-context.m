#import "ggml-metal-context.h"

#import "ggml-impl.h"
#import "ggml-backend-impl.h"

#import "ggml-metal-impl.h"
#import "ggml-metal-common.h"
#import "ggml-metal-ops.h"

#import <Foundation/Foundation.h>

#import <Metal/Metal.h>

#include <mach/mach_time.h>
#include <stdatomic.h>

#undef MIN
#undef MAX
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

// max number of MTLCommandBuffer used to submit a graph for processing
#define GGML_METAL_MAX_COMMAND_BUFFERS 8

// max deferred host-side get_tensor copies (GGML_METAL_GET_MEMCPY=1)
#define GGML_METAL_MAX_DEFERRED_GETS 16

// per-op GPU time profiling (GGML_METAL_PROFILE=1)
// one encoder per op with timestamp samples at the encoder boundaries
#define GGML_METAL_PROF_MAX_ENTRIES 1024

struct ggml_metal_prof_entry {
    char     key[192];
    uint64_t ticks;
    uint64_t count;
};

static struct ggml_metal_prof_entry g_prof_entries[GGML_METAL_PROF_MAX_ENTRIES];
static int                          g_prof_n_entries  = 0;
static int                          g_prof_enabled    = -1;
static MTLTimestamp                 g_prof_cpu0       = 0;
static MTLTimestamp                 g_prof_gpu0       = 0;
static NSLock *                     g_prof_lock       = nil;

static bool ggml_metal_prof_enabled(void) {
    if (g_prof_enabled < 0) {
        const char * val = getenv("GGML_METAL_PROFILE");
        g_prof_enabled = val ? atoi(val) : 0;
        if (g_prof_enabled) {
            g_prof_lock = [[NSLock alloc] init];
        }
    }
    return g_prof_enabled > 0;
}

static void ggml_metal_prof_add(const char * key, uint64_t ticks) {
    [g_prof_lock lock];
    for (int i = 0; i < g_prof_n_entries; ++i) {
        if (strcmp(g_prof_entries[i].key, key) == 0) {
            g_prof_entries[i].ticks += ticks;
            g_prof_entries[i].count += 1;
            [g_prof_lock unlock];
            return;
        }
    }
    if (g_prof_n_entries < GGML_METAL_PROF_MAX_ENTRIES) {
        struct ggml_metal_prof_entry * e = &g_prof_entries[g_prof_n_entries++];
        snprintf(e->key, sizeof(e->key), "%s", key);
        e->ticks = ticks;
        e->count = 1;
    }
    [g_prof_lock unlock];
}

static void ggml_metal_prof_make_key(int prof_id, const struct ggml_tensor * node, char * key, size_t len) {
    const struct ggml_tensor * s0 = node->src[0];
    const struct ggml_tensor * s1 = node->src[1];
    snprintf(key, len, "m%d %-16s %-6s s0=[%lld,%lld,%lld] s1=[%lld,%lld] dst=[%lld,%lld]",
        prof_id,
        ggml_op_desc(node),
        s0 ? ggml_type_name(s0->type) : "-",
        s0 ? s0->ne[0] : 0, s0 ? s0->ne[1] : 0, s0 ? s0->ne[2] : 0,
        s1 ? s1->ne[0] : 0, s1 ? s1->ne[1] : 0,
        node->ne[0], node->ne[1]);
}

static int ggml_metal_prof_cmp(const void * a, const void * b) {
    const struct ggml_metal_prof_entry * ea = a;
    const struct ggml_metal_prof_entry * eb = b;
    return ea->ticks < eb->ticks ? 1 : ea->ticks > eb->ticks ? -1 : 0;
}

// submit-prof (GGML_METAL_SUBMIT_PROF=1): per-graph GPU timeline against the host encode window,
// to measure how much of the CPU submit is exposed on the round vs hidden under GPU execution.
// per-buffer GPU times come from MTLCommandBuffer GPUStartTime/GPUEndTime (same timebase as
// mach_absolute_time), so this adds no encoders and does not perturb the timings it measures.
struct ggml_metal_submit_rec {
    double entry;       // graph_compute entry (host)
    double encode_done; // dispatch_apply returned, all buffers committed (host)
    // slot 0 is the main-thread buffer (first nodes, executes first), slot 1+i is worker i
    double t0[GGML_METAL_MAX_COMMAND_BUFFERS + 1];
    double t1[GGML_METAL_MAX_COMMAND_BUFFERS + 1];
    int    n_bufs;
    int    n_nodes;
    struct ggml_metal * ctx;
    atomic_int n_done;
};

static NSLock * g_sprof_lock = nil;

static bool ggml_metal_submit_prof_enabled(void) {
    static int res = -1;
    if (res < 0) {
        const char * val = getenv("GGML_METAL_SUBMIT_PROF");
        res = val ? atoi(val) : 0;
        if (res) {
            g_sprof_lock = [[NSLock alloc] init];
        }
    }
    return res > 0;
}

static double ggml_metal_submit_prof_now(void) {
    static mach_timebase_info_data_t tb;
    if (tb.denom == 0) {
        mach_timebase_info(&tb);
    }
    return 1e-9 * mach_absolute_time() * tb.numer / tb.denom;
}

static void ggml_metal_submit_prof_complete(struct ggml_metal_submit_rec * rec);

// device limits: at most 32 live sample buffers, at most 4096 samples each -> one buffer per command buffer
#define GGML_METAL_PROF_MAX_SAMPLES 4096

static id<MTLCounterSampleBuffer> ggml_metal_prof_new_smpbuf(id<MTLDevice> device) {
    static id<MTLCounterSet> cset = nil;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        for (id<MTLCounterSet> cs in device.counterSets) {
            if ([cs.name isEqualToString:MTLCommonCounterSetTimestamp]) {
                cset = [cs retain];
                break;
            }
        }
        GGML_ASSERT(cset != nil);
    });

    MTLCounterSampleBufferDescriptor * desc = [[MTLCounterSampleBufferDescriptor alloc] init];
    desc.counterSet  = cset;
    desc.storageMode = MTLStorageModeShared;
    desc.sampleCount = GGML_METAL_PROF_MAX_SAMPLES;

    NSError * err = nil;
    id<MTLCounterSampleBuffer> res = [device newCounterSampleBufferWithDescriptor:desc error:&err];
    if (!res) {
        GGML_LOG_ERROR("%s: error: %s\n", __func__, [[err localizedDescription] UTF8String]);
    }
    [desc release];

    return res;
}

static void ggml_metal_prof_dump(id<MTLDevice> device) {
    MTLTimestamp cpu1 = 0;
    MTLTimestamp gpu1 = 0;
    [device sampleTimestamps:&cpu1 gpuTimestamp:&gpu1];

    // CPU timestamps are in ns; convert GPU ticks to ns via correlation
    const double factor = gpu1 > g_prof_gpu0 ? (double)(cpu1 - g_prof_cpu0)/(double)(gpu1 - g_prof_gpu0) : 1.0;

    qsort(g_prof_entries, g_prof_n_entries, sizeof(g_prof_entries[0]), ggml_metal_prof_cmp);

    double total_ms = 0.0;
    for (int i = 0; i < g_prof_n_entries; ++i) {
        total_ms += 1e-6*factor*g_prof_entries[i].ticks;
    }

    fprintf(stderr, "ggml_metal_prof: ts factor = %.3f, total GPU time = %.3f ms\n", factor, total_ms);
    fprintf(stderr, "ggml_metal_prof: %10s %10s %10s  %s\n", "total ms", "count", "us/call", "op");
    for (int i = 0; i < g_prof_n_entries; ++i) {
        const double ms = 1e-6*factor*g_prof_entries[i].ticks;
        fprintf(stderr, "ggml_metal_prof: %10.3f %10llu %10.2f  %s\n", ms, g_prof_entries[i].count, 1e3*ms/g_prof_entries[i].count, g_prof_entries[i].key);
    }
}

struct ggml_metal_command_buffer {
    id<MTLCommandBuffer> obj;
};

struct ggml_metal {
    char name[128];

    ggml_metal_device_t  dev;
    ggml_metal_library_t lib;

    ggml_metal_event_t ev_cpy; // for async copies

    dispatch_queue_t d_queue;

    // additional, inference-time compiled pipelines
    ggml_metal_pipelines_t pipelines_ext;

    bool use_fusion;
    bool use_concurrency;
    bool use_graph_optimize;

    int debug_graph;
    int debug_fusion;

    // how many times a given op was fused
    uint64_t fuse_cnt[GGML_OP_COUNT];

    // capture state
    int capture_compute;
    bool capture_started;

    id<MTLCaptureScope> capture_scope;

    // command buffer state
    int n_cb;           // number of extra threads used to submit the command buffers
    int n_nodes_0;      // number of nodes submitted by the main thread
    int n_nodes_1;      // remaining number of nodes submitted by the n_cb threads
    int n_nodes_per_cb;

    struct ggml_cgraph * gf;

    // the callback given to the thread pool
    void (^encode_async)(size_t ith);

    // n_cb command buffers + 1 used by the main thread
    struct ggml_metal_command_buffer cmd_bufs[GGML_METAL_MAX_COMMAND_BUFFERS + 1];

    // extra command buffers for things like getting, setting and copying tensors
    NSMutableArray * cmd_bufs_ext;

    // the last command buffer queued into the Metal queue with operations relevant to the current Metal backend
    id<MTLCommandBuffer> cmd_buf_last;

    // abort ggml_metal_graph_compute if callback returns true
    ggml_abort_callback abort_callback;
    void *              abort_callback_data;

    // error state - set when a command buffer fails during synchronize
    // once set, graph_compute will return GGML_STATUS_FAILED until the backend is recreated
    bool has_error;

    // ordinal for the profiler key - separates per-model rows (target/drafter dims collide)
    int prof_id;

    // submit-prof accumulators (GGML_METAL_SUBMIT_PROF=1), guarded by g_sprof_lock
    struct {
        int    n;
        double nodes;   // graph nodes
        double sub;     // host encode window: entry -> dispatch_apply done
        double pre;     // GPU idle before the first buffer starts
        double gaps;    // GPU idle between buffers
        double busy;    // sum of per-buffer GPU time
        double tail;    // GPU time left after the host encode window closes
        double span;    // entry -> last GPU end
        double exposed; // pre + gaps + host encode past the last GPU end
    } sprof;

    // deferred host-side get_tensor copies (GGML_METAL_GET_MEMCPY=1): instead of encoding a
    // blit into an extra command buffer behind the graph, remember the copy and do a plain
    // memcpy in synchronize after the wait - the source buffer is CPU-visible (unified memory)
    struct {
        void *        dst;
        id<MTLBuffer> src;
        size_t        offs;
        size_t        size;
    } get_deferred[GGML_METAL_MAX_DEFERRED_GETS];
    int n_get_deferred;
};

static bool ggml_metal_get_memcpy_enabled(void) {
    static int res = -1;
    if (res < 0) {
        const char * val = getenv("GGML_METAL_GET_MEMCPY");
        res = val ? atoi(val) : 0;
    }
    return res > 0;
}

static void ggml_metal_drain_deferred_gets(struct ggml_metal * ctx) {
    for (int i = 0; i < ctx->n_get_deferred; ++i) {
        memcpy(ctx->get_deferred[i].dst, (const char *) [ctx->get_deferred[i].src contents] + ctx->get_deferred[i].offs, ctx->get_deferred[i].size);
    }
    ctx->n_get_deferred = 0;
}

static void ggml_metal_submit_prof_finalize(struct ggml_metal_submit_rec * rec) {
    const int n = rec->n_bufs;

    double busy = 0.0;
    double gaps = 0.0;
    double t_first =  1e30;
    double t_last  = -1e30;
    for (int i = 0; i < n; ++i) {
        busy   += rec->t1[i] - rec->t0[i];
        t_first = MIN(t_first, rec->t0[i]);
        t_last  = MAX(t_last,  rec->t1[i]);
    }
    for (int i = 1; i < n; ++i) {
        gaps += MAX(0.0, rec->t0[i] - rec->t1[i - 1]);
    }

    struct ggml_metal * ctx = rec->ctx;

    [g_sprof_lock lock];

    ctx->sprof.n       += 1;
    ctx->sprof.nodes   += rec->n_nodes;
    ctx->sprof.sub     += rec->encode_done - rec->entry;
    ctx->sprof.pre     += t_first - rec->entry;
    ctx->sprof.gaps    += gaps;
    ctx->sprof.busy    += busy;
    ctx->sprof.tail    += MAX(0.0, t_last - rec->encode_done);
    ctx->sprof.span    += t_last - rec->entry;
    ctx->sprof.exposed += (t_first - rec->entry) + gaps + MAX(0.0, rec->encode_done - t_last);

    // windowed averages: the first window absorbs load/prefill warmup (pipeline JIT), later windows are steady state
    if (ctx->sprof.n % 64 == 0) {
        const double k = 1e3/ctx->sprof.n;
        fprintf(stderr, "ggml-metal submit-prof m%d n=%d nodes=%d avg ms: sub %.3f pre %.3f gaps %.3f busy %.3f tail %.3f span %.3f exposed %.3f\n",
            ctx->prof_id, ctx->sprof.n, (int) (ctx->sprof.nodes/ctx->sprof.n),
            k*ctx->sprof.sub, k*ctx->sprof.pre, k*ctx->sprof.gaps, k*ctx->sprof.busy,
            k*ctx->sprof.tail, k*ctx->sprof.span, k*ctx->sprof.exposed);
        memset(&ctx->sprof, 0, sizeof(ctx->sprof));
    }

    [g_sprof_lock unlock];
}

static void ggml_metal_submit_prof_complete(struct ggml_metal_submit_rec * rec) {
    // parties: one completed-handler per buffer, plus the encoding thread; the last one finalizes
    if (atomic_fetch_add(&rec->n_done, 1) == rec->n_bufs) {
        ggml_metal_submit_prof_finalize(rec);
        free(rec);
    }
}

static void ggml_metal_submit_prof_attach(struct ggml_metal_submit_rec * rec, id<MTLCommandBuffer> cmd_buf, int slot) {
    [cmd_buf addCompletedHandler:^(id<MTLCommandBuffer> cb) {
        rec->t0[slot] = [cb GPUStartTime];
        rec->t1[slot] = [cb GPUEndTime];
        ggml_metal_submit_prof_complete(rec);
    }];
}

ggml_metal_t ggml_metal_init(ggml_metal_device_t dev) {
    GGML_LOG_INFO("%s: allocating\n", __func__);

#if TARGET_OS_OSX && !GGML_METAL_NDEBUG
    // Show all the Metal device instances in the system
    NSArray * devices = MTLCopyAllDevices();
    for (id<MTLDevice> device in devices) {
        GGML_LOG_INFO("%s: found device: %s\n", __func__, [[device name] UTF8String]);
    }
    [devices release]; // since it was created by a *Copy* C method
#endif

    // init context
    ggml_metal_t res = calloc(1, sizeof(struct ggml_metal));

    id<MTLDevice> device = ggml_metal_device_get_obj(dev);

    GGML_LOG_INFO("%s: picking default device: %s\n", __func__, [[device name] UTF8String]);

    // TODO: would it be better to have one queue for the backend and one queue for the device?
    //       the graph encoders and async ops would use the backend queue while the sync ops would use the device queue?
    //res->queue = [device newCommandQueue]; [TAG_QUEUE_PER_BACKEND]
    id<MTLCommandQueue> queue = ggml_metal_device_get_queue(dev);
    if (queue == nil) {
        GGML_LOG_ERROR("%s: error: failed to create command queue\n", __func__);
        return NULL;
    }

    res->dev = dev;
    res->lib = ggml_metal_device_get_library(dev);
    if (res->lib == NULL) {
        GGML_LOG_WARN("%s: the device does not have a precompiled Metal library - this is unexpected\n", __func__);
        GGML_LOG_WARN("%s: will try to compile it on the fly\n", __func__);

        res->lib = ggml_metal_library_init(dev);
        if (res->lib == NULL) {
            GGML_LOG_ERROR("%s: error: failed to initialize the Metal library\n", __func__);

            free(res);

            return NULL;
        }
    }

    res->ev_cpy = ggml_metal_device_event_init(dev);

    const struct ggml_metal_device_props * props_dev = ggml_metal_device_get_props(dev);

    snprintf(res->name, sizeof(res->name), "%s", props_dev->name);

    res->d_queue = dispatch_queue_create("ggml-metal", DISPATCH_QUEUE_CONCURRENT);

    res->use_fusion      = getenv("GGML_METAL_FUSION_DISABLE") == nil;
    res->use_concurrency = getenv("GGML_METAL_CONCURRENCY_DISABLE") == nil;

    if (ggml_metal_prof_enabled() && g_prof_gpu0 == 0) {
        [device sampleTimestamps:&g_prof_cpu0 gpuTimestamp:&g_prof_gpu0];
        GGML_LOG_INFO("%s: per-op profiling enabled (GGML_METAL_PROFILE)\n", __func__);
    }

    {
        const char * val = getenv("GGML_METAL_GRAPH_DEBUG");
        res->debug_graph = val ? atoi(val) : 0;
    }

    {
        const char * val = getenv("GGML_METAL_FUSION_DEBUG");
        res->debug_fusion = val ? atoi(val) : 0;
    }

    res->use_graph_optimize = true;

    if (getenv("GGML_METAL_GRAPH_OPTIMIZE_DISABLE") != NULL) {
        res->use_graph_optimize = false;
    }

    memset(res->fuse_cnt, 0, sizeof(res->fuse_cnt));

    GGML_LOG_INFO("%s: use fusion         = %s\n", __func__, res->use_fusion         ? "true" : "false");
    GGML_LOG_INFO("%s: use concurrency    = %s\n", __func__, res->use_concurrency    ? "true" : "false");
    GGML_LOG_INFO("%s: use graph optimize = %s\n", __func__, res->use_graph_optimize ? "true" : "false");

    res->capture_compute = 0;
    res->capture_started = false;
    res->capture_scope = nil;

    {
        const char * val = getenv("GGML_METAL_CAPTURE_COMPUTE");
        if (val) {
            res->capture_compute = atoi(val);
        }
    }

    res->has_error = false;

    static int g_prof_next_id = 0;
    res->prof_id = g_prof_next_id++;

    res->gf = nil;
    res->encode_async = nil;
    for (int i = 0; i < GGML_METAL_MAX_COMMAND_BUFFERS; ++i) {
        res->cmd_bufs[i].obj = nil;
    }

    res->cmd_bufs_ext = [[NSMutableArray alloc] init];

    res->cmd_buf_last = nil;

    res->pipelines_ext = ggml_metal_pipelines_init();

    return res;
}

void ggml_metal_free(ggml_metal_t ctx) {
    GGML_LOG_INFO("%s: deallocating\n", __func__);

    if (ggml_metal_prof_enabled()) {
        ggml_metal_synchronize(ctx);
        ggml_metal_prof_dump(ggml_metal_device_get_obj(ctx->dev));
    }

    for (int i = 0; i < GGML_METAL_MAX_COMMAND_BUFFERS; ++i) {
        if (ctx->cmd_bufs[i].obj) {
            [ctx->cmd_bufs[i].obj release];
        }
    }

    for (int i = 0; i < (int) ctx->cmd_bufs_ext.count; ++i) {
        if (ctx->cmd_bufs_ext[i]) {
            [ctx->cmd_bufs_ext[i] release];
        }
    }

    [ctx->cmd_bufs_ext removeAllObjects];
    [ctx->cmd_bufs_ext release];

    if (ctx->pipelines_ext) {
        ggml_metal_pipelines_free(ctx->pipelines_ext);
        ctx->pipelines_ext = nil;
    }

    if (ctx->debug_fusion > 0) {
        GGML_LOG_DEBUG("%s: fusion stats:\n", __func__);
        for (int i = 0; i < GGML_OP_COUNT; i++) {
            if (ctx->fuse_cnt[i] == 0) {
                continue;
            }

            // note: cannot use ggml_log here
            GGML_LOG_DEBUG("%s: - %s: %" PRIu64 "\n", __func__, ggml_op_name((enum ggml_op) i), ctx->fuse_cnt[i]);
        }
    }

    Block_release(ctx->encode_async);

    //[ctx->queue release]; // [TAG_QUEUE_PER_BACKEND]

    dispatch_release(ctx->d_queue);

    ggml_metal_device_event_free(ctx->dev, ctx->ev_cpy);

    free(ctx);
}

const char * ggml_metal_get_name(ggml_metal_t ctx) {
    return ctx->name;
}

void ggml_metal_synchronize(ggml_metal_t ctx) {
    // wait for any backend operations to finish
    if (ctx->cmd_buf_last) {
        [ctx->cmd_buf_last waitUntilCompleted];
        ctx->cmd_buf_last = nil;
    }

    ggml_metal_drain_deferred_gets(ctx);

    // check status of all command buffers
    {
        const int n_cb = ctx->n_cb;

        for (int cb_idx = 0; cb_idx <= n_cb; ++cb_idx) {
            id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs[cb_idx].obj;
            if (!cmd_buf) {
                continue;
            }

            MTLCommandBufferStatus status = [cmd_buf status];
            if (status != MTLCommandBufferStatusCompleted) {
                GGML_LOG_ERROR("%s: error: command buffer %d failed with status %d\n", __func__, cb_idx, (int) status);
                if (status == MTLCommandBufferStatusError) {
                    GGML_LOG_ERROR("error: %s\n", [[cmd_buf error].localizedDescription UTF8String]);
                }
                ctx->has_error = true;
                return;
            }
        }
    }

    // release any completed extra command buffers
    if (ctx->cmd_bufs_ext.count > 0) {
        for (size_t i = 0; i < ctx->cmd_bufs_ext.count; ++i) {
            id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs_ext[i];

            MTLCommandBufferStatus status = [cmd_buf status];
            if (status != MTLCommandBufferStatusCompleted) {
                GGML_LOG_ERROR("%s: error: command buffer %d failed with status %d\n", __func__, (int) i, (int) status);
                if (status == MTLCommandBufferStatusError) {
                    GGML_LOG_ERROR("error: %s\n", [[cmd_buf error].localizedDescription UTF8String]);
                }

                // release this and all remaining command buffers before returning
                for (size_t j = i; j < ctx->cmd_bufs_ext.count; ++j) {
                    [ctx->cmd_bufs_ext[j] release];
                }
                [ctx->cmd_bufs_ext removeAllObjects];

                ctx->has_error = true;
                return;
            }

            [cmd_buf release];
        }

        [ctx->cmd_bufs_ext removeAllObjects];
    }
}

static struct ggml_metal_buffer_id ggml_metal_get_buffer_id(const struct ggml_tensor * t) {
    if (!t) {
        return (struct ggml_metal_buffer_id) { nil, 0 };
    }

    ggml_backend_buffer_t buffer = t->view_src ? t->view_src->buffer : t->buffer;

    return ggml_metal_buffer_get_id(buffer->context, t);
}

void ggml_metal_set_tensor_async(ggml_metal_t ctx, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    @autoreleasepool {
        // wrap the source data into a Metal buffer
        id<MTLDevice> device = ggml_metal_device_get_obj(ctx->dev);
        id<MTLBuffer> buf_src = [device newBufferWithBytes:data
                                                    length:size
                                                   options:MTLResourceStorageModeShared];

        GGML_ASSERT(buf_src);

        struct ggml_metal_buffer_id bid_dst = ggml_metal_get_buffer_id(tensor);
        if (bid_dst.metal == nil) {
            GGML_ABORT("%s: failed to find buffer for tensor '%s'\n", __func__, tensor->name);
        }

        bid_dst.offs += offset;

        // queue the copy operation into the queue of the Metal context
        // this will be queued at the end, after any currently ongoing GPU operations
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx->dev);
        id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];
        id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

        [encoder copyFromBuffer:buf_src
                   sourceOffset:0
                       toBuffer:bid_dst.metal
              destinationOffset:bid_dst.offs
                           size:size];

        [encoder endEncoding];
        [cmd_buf commit];
        [buf_src release];

        // do not wait here for completion
        //[cmd_buf waitUntilCompleted];

        // instead, remember a reference to the command buffer and wait for it later if needed
        [ctx->cmd_bufs_ext addObject:cmd_buf];
        ctx->cmd_buf_last = cmd_buf;

        [cmd_buf retain];
    }
}

void ggml_metal_get_tensor_async(ggml_metal_t ctx, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    if (ggml_metal_get_memcpy_enabled() && ctx->n_get_deferred < GGML_METAL_MAX_DEFERRED_GETS) {
        struct ggml_metal_buffer_id bid_src = ggml_metal_get_buffer_id(tensor);
        if (bid_src.metal == nil) {
            GGML_ABORT("%s: failed to find buffer for tensor '%s'\n", __func__, tensor->name);
        }

        ctx->get_deferred[ctx->n_get_deferred++] = (__typeof__(ctx->get_deferred[0])) {
            /*.dst  =*/ data,
            /*.src  =*/ bid_src.metal,
            /*.offs =*/ bid_src.offs + offset,
            /*.size =*/ size,
        };

        return;
    }

    @autoreleasepool {
        id<MTLDevice> device = ggml_metal_device_get_obj(ctx->dev);
        id<MTLBuffer> buf_dst = [device newBufferWithBytesNoCopy:data
                                                          length:size
                                                         options:MTLResourceStorageModeShared
                                                     deallocator:nil];

        GGML_ASSERT(buf_dst);

        struct ggml_metal_buffer_id bid_src = ggml_metal_get_buffer_id(tensor);
        if (bid_src.metal == nil) {
            GGML_ABORT("%s: failed to find buffer for tensor '%s'\n", __func__, tensor->name);
        }

        bid_src.offs += offset;

        // queue the copy operation into the queue of the Metal context
        // this will be queued at the end, after any currently ongoing GPU operations
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx->dev);
        id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];
        id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

        [encoder copyFromBuffer:bid_src.metal
                   sourceOffset:bid_src.offs
                       toBuffer:buf_dst
              destinationOffset:0
                           size:size];

        [encoder endEncoding];
        [cmd_buf commit];
        [buf_dst release];

        // do not wait here for completion
        //[cmd_buf waitUntilCompleted];

        // instead, remember a reference to the command buffer and wait for it later if needed
        [ctx->cmd_bufs_ext addObject:cmd_buf];
        ctx->cmd_buf_last = cmd_buf;

        [cmd_buf retain];
    }
}

bool ggml_metal_cpy_tensor_async(ggml_metal_t ctx_src, ggml_metal_t ctx_dst, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    @autoreleasepool {
        struct ggml_metal_buffer_id bid_src = ggml_metal_get_buffer_id(src);
        struct ggml_metal_buffer_id bid_dst = ggml_metal_get_buffer_id(dst);

        if (bid_src.metal == nil || bid_dst.metal == nil) {
            return false;
        }

        // queue the copy operation into the Metal context
        // this will be queued at the end, after any currently ongoing GPU operations
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx_src->dev);
        id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];
        id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

        [encoder copyFromBuffer:bid_src.metal
                   sourceOffset:bid_src.offs
                       toBuffer:bid_dst.metal
              destinationOffset:bid_dst.offs
                           size:ggml_nbytes(src)];

        [encoder endEncoding];

        ggml_metal_event_t ev_cpy = ggml_metal_get_ev_cpy(ctx_src);
        ggml_metal_event_encode_signal(ev_cpy, cmd_buf);

        [cmd_buf commit];

        // do not wait here for completion
        //[cmd_buf waitUntilCompleted];

        // instead, remember a reference to the command buffer and wait for it later if needed
        [ctx_src->cmd_bufs_ext addObject:cmd_buf];
        ctx_src->cmd_buf_last = cmd_buf;

        [cmd_buf retain];

        ggml_metal_event_wait(ctx_dst, ev_cpy);

        return true;
    }
}

enum ggml_status ggml_metal_graph_compute(ggml_metal_t ctx, struct ggml_cgraph * gf) {
    if (ctx->has_error) {
        GGML_LOG_ERROR("%s: backend is in error state from a previous command buffer failure - recreate the backend to recover\n", __func__);
        return GGML_STATUS_FAILED;
    }

    // a deferred get with no synchronize since means the caller relies on queue order - drain it
    // before this graph can overwrite the source
    if (ctx->n_get_deferred > 0) {
        if (ctx->cmd_buf_last) {
            [ctx->cmd_buf_last waitUntilCompleted];
        }
        ggml_metal_drain_deferred_gets(ctx);
    }

    // number of nodes encoded by the main thread (empirically determined)
    const int n_main = MAX(64, 0.1*gf->n_nodes);

    // number of threads in addition to the main thread
    const int n_cb = ctx->n_cb;

    struct ggml_metal_submit_rec * srec = NULL;
    if (ggml_metal_submit_prof_enabled() && ctx->capture_compute < 0 && !(ctx->abort_callback && n_cb > 2)) {
        srec = calloc(1, sizeof(*srec));
        srec->entry   = ggml_metal_submit_prof_now();
        srec->n_bufs  = n_cb + 1;
        srec->n_nodes = gf->n_nodes;
        srec->ctx     = ctx;
    }

    // keep the memory wired
    ggml_metal_device_rsets_keep_alive(ctx->dev);

    // submit the ggml compute graph to the GPU by creating command buffers and encoding the ops in them
    // the first n_nodes_0 are encoded and submitted for processing directly by the calling thread
    // while these nodes are processing, we start n_cb threads to enqueue the rest of the nodes
    // each thread creates it's own command buffer and enqueues the ops in parallel
    //
    // tests on M1 Pro and M2 Ultra using LLaMA models, show that optimal values for n_cb are 1 or 2

    @autoreleasepool {
        ctx->gf = gf;

        ctx->n_nodes_0 = MIN(n_main, gf->n_nodes);
        ctx->n_nodes_1 = gf->n_nodes - ctx->n_nodes_0;

        ctx->n_nodes_per_cb = (ctx->n_nodes_1 + ctx->n_cb - 1) / ctx->n_cb;

        if (ctx->capture_compute >= 0) {
            ctx->capture_compute--;
        }

        const bool use_capture = ctx->capture_compute == 0;
        if (use_capture) {
            ctx->capture_compute = -1;

            // make sure all previous computations have finished before starting the capture
            if (ctx->cmd_buf_last) {
                [ctx->cmd_buf_last waitUntilCompleted];
                ctx->cmd_buf_last = nil;
            }

            if (!ctx->capture_started) {
                NSString * path = [NSString stringWithFormat:@"/tmp/perf-metal-%d.gputrace", getpid()];

                GGML_LOG_WARN("%s: capturing graph in %s\n", __func__, [path UTF8String]);

                // create capture scope
                id<MTLDevice> device = ggml_metal_device_get_obj(ctx->dev);
                ctx->capture_scope = [[MTLCaptureManager sharedCaptureManager] newCaptureScopeWithDevice:device];

                MTLCaptureDescriptor * descriptor = [MTLCaptureDescriptor new];
                descriptor.captureObject = ctx->capture_scope;
                descriptor.destination = MTLCaptureDestinationGPUTraceDocument;
                descriptor.outputURL = [NSURL fileURLWithPath:path];

                NSError * error = nil;
                if (![[MTLCaptureManager sharedCaptureManager] startCaptureWithDescriptor:descriptor error:&error]) {
                    GGML_LOG_ERROR("%s: error: unable to start capture '%s'\n", __func__, [[error localizedDescription] UTF8String]);
                } else {
                    [ctx->capture_scope beginScope];
                    ctx->capture_started = true;
                }
            }
        }

        // short-hand
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx->dev);

        // the main thread commits the first few commands immediately
        // cmd_buf[n_cb]
        {
            id<MTLCommandBuffer> cmd_buf = [queue commandBufferWithUnretainedReferences];
            [cmd_buf retain];

            if (ctx->cmd_bufs[n_cb].obj) {
                [ctx->cmd_bufs[n_cb].obj release];
            }
            ctx->cmd_bufs[n_cb].obj = cmd_buf;

            if (srec) {
                ggml_metal_submit_prof_attach(srec, cmd_buf, 0);
            }

            [cmd_buf enqueue];

            ctx->encode_async(n_cb);
        }

        // remember the command buffer for the next iteration
        ctx->cmd_buf_last = ctx->cmd_bufs[n_cb].obj;

        // prepare the rest of the command buffers asynchronously (optional)
        // cmd_buf[0.. n_cb)
        for (int cb_idx = 0; cb_idx < n_cb; ++cb_idx) {
            id<MTLCommandBuffer> cmd_buf = [queue commandBufferWithUnretainedReferences];
            [cmd_buf retain];

            if (ctx->cmd_bufs[cb_idx].obj) {
                [ctx->cmd_bufs[cb_idx].obj release];
            }
            ctx->cmd_bufs[cb_idx].obj = cmd_buf;

            if (srec) {
                ggml_metal_submit_prof_attach(srec, cmd_buf, 1 + cb_idx);
            }

            // always enqueue the first two command buffers
            // enqueue all of the command buffers if we don't need to abort
            if (cb_idx < 2 || ctx->abort_callback == NULL) {
                [cmd_buf enqueue];

                // update the pointer to the last queued command buffer
                // this is needed to implement synchronize()
                ctx->cmd_buf_last = cmd_buf;
            }
        }

        dispatch_apply(n_cb, ctx->d_queue, ctx->encode_async);

        if (srec) {
            srec->encode_done = ggml_metal_submit_prof_now();
            ggml_metal_submit_prof_complete(srec);
        }

        // for debugging: block until graph is computed
        //[ctx->cmd_buf_last waitUntilCompleted];

        // enter here only when capturing in order to wait for all computation to finish
        // otherwise, we leave the graph to compute asynchronously
        if (use_capture && ctx->capture_started) {
            // wait for completion and check status of each command buffer
            // needed to detect if the device ran out-of-memory for example (#1881)
            {
                id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs[n_cb].obj;
                [cmd_buf waitUntilCompleted];

                MTLCommandBufferStatus status = [cmd_buf status];
                if (status != MTLCommandBufferStatusCompleted) {
                    GGML_LOG_INFO("%s: command buffer %d failed with status %lu\n", __func__, n_cb, status);
                    if (status == MTLCommandBufferStatusError) {
                        GGML_LOG_INFO("error: %s\n", [[cmd_buf error].localizedDescription UTF8String]);
                    }

                    return GGML_STATUS_FAILED;
                }
            }

            for (int i = 0; i < n_cb; ++i) {
                id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs[i].obj;
                [cmd_buf waitUntilCompleted];

                MTLCommandBufferStatus status = [cmd_buf status];
                if (status != MTLCommandBufferStatusCompleted) {
                    GGML_LOG_INFO("%s: command buffer %d failed with status %lu\n", __func__, i, status);
                    if (status == MTLCommandBufferStatusError) {
                        GGML_LOG_INFO("error: %s\n", [[cmd_buf error].localizedDescription UTF8String]);
                    }

                    return GGML_STATUS_FAILED;
                }

                id<MTLCommandBuffer> next_buffer = (i + 1 < n_cb ? ctx->cmd_bufs[i + 1].obj : nil);
                if (!next_buffer) {
                    continue;
                }

                const bool next_queued = ([next_buffer status] != MTLCommandBufferStatusNotEnqueued);
                if (next_queued) {
                    continue;
                }

                if (ctx->abort_callback && ctx->abort_callback(ctx->abort_callback_data)) {
                    GGML_LOG_INFO("%s: command buffer %d aborted", __func__, i);
                    return GGML_STATUS_ABORTED;
                }

                [next_buffer commit];
            }

            [ctx->capture_scope endScope];
            [[MTLCaptureManager sharedCaptureManager] stopCapture];

            ctx->capture_started = false;
        }
    }

    return GGML_STATUS_SUCCESS;
}

void ggml_metal_graph_optimize(ggml_metal_t ctx, struct ggml_cgraph * gf) {
    //const int64_t t_start = ggml_time_us();

    if (ctx->use_graph_optimize) {
        ggml_graph_optimize(gf);
    }

    //printf("%s: graph optimize took %.3f ms\n", __func__, (ggml_time_us() - t_start) / 1000.0);
}

void ggml_metal_event_record(ggml_metal_t ctx, ggml_metal_event_t ev) {
    @autoreleasepool {
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx->dev);
        id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];

        ggml_metal_event_encode_signal(ev, cmd_buf);

        [cmd_buf commit];

        [ctx->cmd_bufs_ext addObject:cmd_buf];
        ctx->cmd_buf_last = cmd_buf;

        [cmd_buf retain];
    }
}

void ggml_metal_event_wait(ggml_metal_t ctx, ggml_metal_event_t ev) {
    @autoreleasepool {
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx->dev);
        id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];

        ggml_metal_event_encode_wait(ev, cmd_buf);

        [cmd_buf commit];

        [ctx->cmd_bufs_ext addObject:cmd_buf];
        ctx->cmd_buf_last = cmd_buf;

        [cmd_buf retain];
    }
}

ggml_metal_event_t ggml_metal_get_ev_cpy(ggml_metal_t ctx) {
    return ctx->ev_cpy;
}

void ggml_metal_set_n_cb(ggml_metal_t ctx, int n_cb) {
    if (ctx->n_cb != n_cb) {
        ctx->n_cb = MIN(n_cb, GGML_METAL_MAX_COMMAND_BUFFERS);

        if (ctx->n_cb > 2) {
            GGML_LOG_WARN("%s: n_cb = %d, using n_cb > 2 is not recommended and can degrade the performance in some cases\n", __func__, n_cb);
        }
    }

    if (ctx->encode_async) {
        Block_release(ctx->encode_async);
    }

    ctx->encode_async = Block_copy(^(size_t iter) {
        const int cb_idx = iter;
        const int n_cb_l = ctx->n_cb;

        const int n_nodes_0 = ctx->n_nodes_0;
        const int n_nodes_1 = ctx->n_nodes_1;

        const int n_nodes_per_cb = ctx->n_nodes_per_cb;

        int idx_start = 0;
        int idx_end   = n_nodes_0;

        if (cb_idx < n_cb_l) {
            idx_start = n_nodes_0 + (                                         (cb_idx + 0) * n_nodes_per_cb);
            idx_end   = n_nodes_0 + (MIN((cb_idx == n_cb_l - 1) ? n_nodes_1 : (cb_idx + 1) * n_nodes_per_cb, n_nodes_1));
        }

        id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs[cb_idx].obj;

        if (ggml_metal_prof_enabled()) {
            // one encoder per op so the timestamp samples at the encoder boundaries give per-op GPU time
            id<MTLDevice> device = ggml_metal_device_get_obj(ctx->dev);

            id<MTLCounterSampleBuffer> sb = ggml_metal_prof_new_smpbuf(device);
            GGML_ASSERT(sb != nil);

            NSMutableArray * keys = [[NSMutableArray alloc] init];

            int raw = idx_start;
            int k   = 0;
            while (raw < idx_end && 2*(k + 1) <= GGML_METAL_PROF_MAX_SAMPLES) {
                ggml_metal_op_t ctx_op = ggml_metal_op_init(
                    ctx->dev,
                    cmd_buf,
                    ctx->gf,
                    raw,
                    idx_end,
                    ctx->use_fusion,
                    false,
                    ctx->capture_compute,
                    ctx->debug_graph,
                    ctx->debug_fusion,
                    (void *) sb,
                    2*k);

                if (ggml_metal_op_n_nodes(ctx_op) == 0) {
                    ggml_metal_op_free(ctx_op);
                    break;
                }

                const int res = ggml_metal_op_encode(ctx_op, 0);

                char key[192];
                ggml_metal_prof_make_key(ctx->prof_id, ctx->gf->nodes[ggml_metal_op_node_idx(ctx_op, 0)], key, sizeof(key));

                const int raw_last = ggml_metal_op_node_idx(ctx_op, (res > 0 ? res : 1) - 1);

                ggml_metal_op_free(ctx_op);

                [keys addObject:[NSString stringWithUTF8String:key]];

                k  += 1;
                raw = raw_last + 1;

                if (res == 0) {
                    break;
                }
            }

            const int n_ops = k;

            [cmd_buf addCompletedHandler:^(id<MTLCommandBuffer> cb) {
                GGML_UNUSED(cb);
                NSData * data = [sb resolveCounterRange:NSMakeRange(0, 2*n_ops)];
                if (data && data.length >= 2*n_ops*sizeof(uint64_t)) {
                    const uint64_t * ts = (const uint64_t *) data.bytes;
                    for (int i = 0; i < n_ops; ++i) {
                        if (ts[2*i] != 0 && ts[2*i + 1] != (uint64_t) -1 && ts[2*i + 1] > ts[2*i]) {
                            ggml_metal_prof_add([keys[i] UTF8String], ts[2*i + 1] - ts[2*i]);
                        }
                    }
                }
            }];

            [sb release];
            [keys release];
        } else {
            ggml_metal_op_t ctx_op = ggml_metal_op_init(
                ctx->dev,
                cmd_buf,
                ctx->gf,
                idx_start,
                idx_end,
                ctx->use_fusion,
                ctx->use_concurrency,
                ctx->capture_compute,
                ctx->debug_graph,
                ctx->debug_fusion,
                NULL,
                0);

            for (int idx = 0; idx < ggml_metal_op_n_nodes(ctx_op); ++idx) {
                const int res = ggml_metal_op_encode(ctx_op, idx);
                if (res == 0) {
                    break;
                }

                idx += res - 1;
            }

            ggml_metal_op_free(ctx_op);
        }

        if (cb_idx < 2 || ctx->abort_callback == NULL) {
            [cmd_buf commit];
        }
    });
}

void ggml_metal_set_abort_callback(ggml_metal_t ctx, ggml_abort_callback abort_callback, void * user_data) {
    ctx->abort_callback = abort_callback;
    ctx->abort_callback_data = user_data;
}

bool ggml_metal_supports_family(ggml_metal_t ctx, int family) {
    GGML_ASSERT(ctx->dev != nil);

    id<MTLDevice> device = ggml_metal_device_get_obj(ctx->dev);

    return [device supportsFamily:(MTLGPUFamilyApple1 + family - 1)];
}

void ggml_metal_capture_next_compute(ggml_metal_t ctx) {
    ctx->capture_compute = 1;
}
