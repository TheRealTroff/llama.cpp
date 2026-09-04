#pragma once

#include "llama-batch.h"
#include "llama-graph.h"
#include "llama-memory.h"

#include <map>
#include <set>
#include <vector>

//
// llama_memory_recurrent
//

// TODO: extract the cache state used for graph computation into llama_memory_recurrent_context_i
//       see the implementation of llama_kv_cache_context_i for an example how to do it
class llama_memory_recurrent : public llama_memory_i {
public:
    llama_memory_recurrent(
            const llama_model & model,
                    ggml_type   type_r,
                    ggml_type   type_s,
                         bool   offload,
                     uint32_t   mem_size,
                     uint32_t   n_seq_max,
                     uint32_t   n_rs_seq,
        const layer_filter_cb & filter);

    ~llama_memory_recurrent() = default;

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    bool prepare(const std::vector<llama_ubatch> & ubatches);

    // find a contiguous slot of memory cells and emplace the ubatch there
    bool find_slot(const llama_ubatch & ubatch);

    bool get_can_shift() const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

    uint32_t head = 0; // the location where the batch will be placed in the cache (see find_slot())
    uint32_t size = 0; // total number of cells, shared across all sequences
    uint32_t used = 0; // used cells (i.e. at least one seq_id)

    // number of recurrent-state snapshots per seq for rollback; tensors are widened to (1 + n_rs_seq) groups
    uint32_t n_rs_seq = 0;

    // per-seq rollback index
    std::vector<uint32_t> rs_idx;

    void set_rs_idx(llama_seq_id seq_id, uint32_t idx);

    // computed before each graph build
    uint32_t n = 0;

    // first zero-ed state
    int32_t rs_z = -1;

    // TODO: optimize for recurrent state needs
    struct mem_cell {
        llama_pos pos  = -1;
        int32_t   src  = -1; // used to know where states should be copied from
        int32_t   src0 = -1; // like src, but only used when setting the inputs (allowing to copy once)
        int32_t   tail = -1;

        // gdn_replay: kept tokens to re-run before this batch (set by find_slot for the batch cells)
        int32_t   xk_replay = 0;
        // gdn_replay: a rollback is pending (the logical state is not materialized in group 0)
        bool      xk_pending = false;

        std::set<llama_seq_id> seq_id;

        bool has_seq_id(const llama_seq_id & id) const {
            return seq_id.find(id) != seq_id.end();
        }

        bool is_empty() const {
            return seq_id.empty();
        }

        bool is_same_seq(const mem_cell & other) const {
            return seq_id == other.seq_id;
        }
    };

    std::vector<mem_cell> cells;

    // per layer
    std::vector<ggml_tensor *> r_l;
    std::vector<ggml_tensor *> s_l;

    // recompute-on-rollback (LLAMA_GDN_REPLAY=1, delta-net models with n_rs_seq > 0): instead of
    // 1 + n_rs_seq full-state snapshot groups, s_l holds 2 groups - group 0 the current state,
    // group 1 the state before the last n_keep tokens of the seq's last batch - and x_l keeps
    // those tokens' delta-net inputs (n_embd_x floats each). A rollback of r <= n_keep tokens
    // re-runs the recurrence over the first n_keep - r kept tokens from group 1 inside the next
    // batch's delta-net op. r_l keeps its 1 + n_rs_seq groups (small, written by the conv path).
    bool gdn_replay = false;

    uint32_t n_embd_x = 0;

    // per layer: [n_embd_x * n_rs_seq, size] (row = cell, token j at j*n_embd_x)
    std::vector<ggml_tensor *> x_l;

    // per seq: tokens kept from its last batch (bounds the rollback), 0 = none
    std::vector<uint32_t> xk_n_keep;

private:
    //const llama_model & model;
    const llama_hparams & hparams;

    const uint32_t n_seq_max = 1;

    // ggml contexts for the KV cache along with the allocated backend buffers:
    std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> ctxs_bufs;

    size_t total_size() const;

    size_t size_r_bytes() const;
    size_t size_s_bytes() const;

    void state_write_meta(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges, llama_seq_id seq_id = -1) const;
    void state_write_data(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges) const;

    // gdn_replay: one entry per written cell - its source row and, with a rollback pending,
    // how many kept tokens to replay on the CPU from group 1 to materialize the logical state
    struct xk_write_cell {
        uint32_t src;
        bool     pending;
        uint32_t n_rep;
    };
    mutable std::vector<xk_write_cell> xk_write_cells;

    void state_write_s_replay(llama_io_write_i & io, int32_t il, const xk_write_cell & c) const;

    bool state_read_meta(llama_io_read_i & io, uint32_t cell_count, llama_seq_id dest_seq_id = -1);
    bool state_read_data(llama_io_read_i & io, uint32_t cell_count);
};

class llama_memory_recurrent_context : public llama_memory_context_i {
public:
    // used for errors
    llama_memory_recurrent_context(llama_memory_status status);

    // used to create a full-cache or update context
    llama_memory_recurrent_context(
            llama_memory_recurrent * mem);

    // used to create a batch processing context from a batch
    llama_memory_recurrent_context(
            llama_memory_recurrent * mem,
            std::vector<llama_ubatch> ubatches);

    virtual ~llama_memory_recurrent_context();

    //
    // llama_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    llama_memory_status  get_status() const override;
    const llama_ubatch & get_ubatch() const override;

    //
    // llama_memory_recurrent_context specific API
    //

    uint32_t get_n_rs() const;
    uint32_t get_head() const;
    int32_t  get_rs_z() const;
    uint32_t get_size() const;

    ggml_tensor * get_r_l(int32_t il) const;
    ggml_tensor * get_s_l(int32_t il) const;

    // gdn_replay (recompute-on-rollback) accessors, see llama_memory_recurrent
    bool          gdn_replay() const;
    uint32_t      get_n_rs_seq() const;
    uint32_t      get_n_embd_x() const;
    ggml_tensor * get_x_l(int32_t il) const;
    // ssm-state source row of seq i: group 1 while a rollback is pending, else like s_copy_peek
    int32_t       s_copy_ss_peek(int i) const;
    int32_t       s_copy_ss_view_row0(uint32_t n_seqs) const;
    // plain source cell of seq i (no snapshot group)
    int32_t       s_copy_src0(int i) const;
    // kept tokens seq i re-runs before this batch
    int32_t       xk_replay(int i) const;

    int32_t s_copy(int i) const;

    // s_copy without the rollback-index reset side effect, for graph-build decisions
    int32_t s_copy_peek(int i) const;

    // true when every one of the first n_seqs sequences reads its state from its own cell
    // (row head + i), i.e. the state gather is the identity and a view of the cache rows
    // can replace the copy
    bool s_copy_is_identity(uint32_t n_seqs) const;

    // first source row when the n_seqs source rows are consecutive (row0 + i), else -1: such a
    // gather is a view of the cache rows (identity, or a uniform rollback slot; always for 1 seq)
    int32_t s_copy_view_row0(uint32_t n_seqs) const;

private:
    const llama_memory_status status;

    llama_memory_recurrent * mem;

    size_t i_next = 0;

    std::vector<llama_ubatch> ubatches;

    //
    // data needed for building the compute graph for the current ubatch:
    // TODO: extract all the state like `head` and `n` here
    //

    const bool is_full = false;
};
