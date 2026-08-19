// TurboQuant KV-cache quantization: Lloyd-Max 2/3/4-bit codebooks over L2-normalized,
// sign-WHT-rotated 128-element groups (one group == one block == one padded head_dim slice).
//
// Dequantization stays in the rotated domain: the graph applies the forward WHT to Q and
// the inverse WHT to the attention output (GGML_OP_TURBO_WHT), so no inverse rotation is
// needed per cached row. The sign vectors and centroid tables below must match the Metal
// shader copies bit-for-bit.

#define GGML_COMMON_IMPL_C
#include "ggml-common.h"

#include "ggml-quants.h"
#include "ggml-impl.h"

#include <math.h>
#include <string.h>
#include <assert.h>

// WHT sign diagonals (seed=42); forward is y = D(s2) * (1/sqrt(128)) * H * D(s1) * x
static const float turbo_s1[QK_TURBO] = {
    -1,1,1,-1,-1,1,-1,1,-1,-1,1,1,1,1,1,1,1,-1,1,-1,1,-1,-1,1,1,1,-1,1,1,-1,-1,-1,
    -1,1,1,-1,1,1,-1,1,-1,1,1,-1,-1,1,-1,1,1,1,1,-1,-1,-1,-1,-1,1,-1,1,1,1,1,-1,1,
    -1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1,1,1,1,-1,-1,1,1,-1,1,1,-1,1,-1,
    -1,1,1,-1,1,-1,1,-1,1,1,1,1,-1,1,-1,1,1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,1,1,-1,1
};

static const float turbo_s2[QK_TURBO] = {
    1,1,1,1,-1,1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,1,1,
    1,1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,1,-1,1,-1,1,1,1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,
    1,-1,1,-1,-1,-1,-1,1,-1,1,-1,1,-1,-1,1,1,-1,1,-1,1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,
    1,-1,1,1,1,-1,-1,1,-1,1,-1,1,1,-1,-1,1,-1,1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,-1
};

#define TURBO_INV_SQRT_128 0.08838834764831845f

// in-place on one 128-element group; direction 0 = forward, 1 = inverse
// (H/sqrt(128) and the ±1 diagonals are self-inverse, so the inverse just swaps s1/s2)
void ggml_turbo_wht_group(float * x, int direction) {
    const float * s_first  = direction == 0 ? turbo_s1 : turbo_s2;
    const float * s_second = direction == 0 ? turbo_s2 : turbo_s1;

    for (int i = 0; i < QK_TURBO; i++) {
        x[i] *= s_first[i];
    }

    for (int h = 1; h < QK_TURBO; h *= 2) {
        for (int i = 0; i < QK_TURBO; i += h*2) {
            for (int j = i; j < i + h; j++) {
                const float a = x[j];
                const float b = x[j + h];
                x[j]     = a + b;
                x[j + h] = a - b;
            }
        }
    }

    for (int i = 0; i < QK_TURBO; i++) {
        x[i] *= TURBO_INV_SQRT_128 * s_second[i];
    }
}

// Lloyd-Max centroids for N(0, 1/128)
static const float turbo_centroids_2bit[4] = { -0.133462f, -0.039994f, 0.039994f, 0.133462f };

static const float turbo_centroids_3bit[8] = {
    -0.190207f, -0.118786f, -0.066822f, -0.021663f,
     0.021663f,  0.066822f,  0.118786f,  0.190207f
};

static const float turbo_centroids_4bit[16] = {
    -0.241529f, -0.182877f, -0.143016f, -0.111036f,
    -0.083292f, -0.058050f, -0.034299f, -0.011349f,
     0.011349f,  0.034299f,  0.058050f,  0.083292f,
     0.111036f,  0.143016f,  0.182877f,  0.241529f
};

// decision thresholds == midpoints of adjacent centroids
static const float turbo_mid_2bit[3] = { -0.086728f, 0.000000f, 0.086728f };

static const float turbo_mid_3bit[7] = {
    -0.154496f, -0.092804f, -0.044243f, 0.000000f, 0.044243f, 0.092804f, 0.154496f
};

static const float turbo_mid_4bit[15] = {
    -0.212203f, -0.162947f, -0.127026f, -0.097164f, -0.070671f, -0.046174f, -0.022824f,
     0.000000f,
     0.022824f,  0.046174f,  0.070671f,  0.097164f,  0.127026f,  0.162947f,  0.212203f
};

static inline int turbo_nearest(const float * mid, int n_mid, float val) {
    int idx = 0;
    while (idx < n_mid && val >= mid[idx]) {
        idx++;
    }
    return idx;
}

// shared per-group quantize: normalize -> rotate -> nearest centroid; returns corrected norm
static float turbo_quantize_group(const float * x, float * rotated, uint8_t * idx,
                                  const float * mid, int n_mid, const float * centroids) {
    float norm_sq = 0.0f;
    for (int i = 0; i < QK_TURBO; i++) {
        norm_sq += x[i]*x[i];
    }
    const float grp_norm = sqrtf(norm_sq);
    const float inv_norm = grp_norm > 1e-10f ? 1.0f/grp_norm : 0.0f;

    for (int i = 0; i < QK_TURBO; i++) {
        rotated[i] = x[i]*inv_norm;
    }

    ggml_turbo_wht_group(rotated, 0);

    float recon_sq = 0.0f;
    for (int i = 0; i < QK_TURBO; i++) {
        const int q = turbo_nearest(mid, n_mid, rotated[i]);
        idx[i] = (uint8_t) q;
        recon_sq += centroids[q]*centroids[q];
    }

    // corrected norm: dequant is exactly centroid[idx]*norm with the original L2 restored
    const float recon_norm = sqrtf(recon_sq);
    return recon_norm > 1e-10f ? grp_norm/recon_norm : grp_norm;
}

void quantize_row_turbo2_0_ref(const float * GGML_RESTRICT x, block_turbo2_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO == 0);
    const int64_t nb = k/QK_TURBO;

    for (int64_t b = 0; b < nb; b++) {
        float   rotated[QK_TURBO];
        uint8_t idx[QK_TURBO];

        const float norm = turbo_quantize_group(x + b*QK_TURBO, rotated, idx,
                turbo_mid_2bit, 3, turbo_centroids_2bit);

        y[b].norm = GGML_FP32_TO_FP16(norm);
        memset(y[b].qs, 0, sizeof(y[b].qs));
        for (int j = 0; j < QK_TURBO; j++) {
            y[b].qs[j/4] |= (uint8_t)((idx[j] & 0x3) << ((j % 4)*2));
        }
    }
}

void quantize_row_turbo3_0_ref(const float * GGML_RESTRICT x, block_turbo3_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO == 0);
    const int64_t nb = k/QK_TURBO;

    for (int64_t b = 0; b < nb; b++) {
        float   rotated[QK_TURBO];
        uint8_t idx[QK_TURBO];

        const float norm = turbo_quantize_group(x + b*QK_TURBO, rotated, idx,
                turbo_mid_3bit, 7, turbo_centroids_3bit);

        y[b].norm = GGML_FP32_TO_FP16(norm);
        memset(y[b].qs,    0, sizeof(y[b].qs));
        memset(y[b].signs, 0, sizeof(y[b].signs));
        for (int j = 0; j < QK_TURBO; j++) {
            y[b].qs[j/4] |= (uint8_t)((idx[j] & 0x3) << ((j % 4)*2));
            if (idx[j] & 0x4) {
                y[b].signs[j/8] |= (uint8_t)(1 << (j % 8));
            }
        }
    }
}

void quantize_row_turbo4_0_ref(const float * GGML_RESTRICT x, block_turbo4_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO == 0);
    const int64_t nb = k/QK_TURBO;

    for (int64_t b = 0; b < nb; b++) {
        float   rotated[QK_TURBO];
        uint8_t idx[QK_TURBO];

        const float norm = turbo_quantize_group(x + b*QK_TURBO, rotated, idx,
                turbo_mid_4bit, 15, turbo_centroids_4bit);

        y[b].norm = GGML_FP32_TO_FP16(norm);
        memset(y[b].qs, 0, sizeof(y[b].qs));
        for (int j = 0; j < QK_TURBO; j++) {
            y[b].qs[j/2] |= (uint8_t)((idx[j] & 0xF) << ((j % 2)*4));
        }
    }
}

// dequantization stays in the WHT-rotated domain (see header comment)

void dequantize_row_turbo2_0(const block_turbo2_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO == 0);
    const int64_t nb = k/QK_TURBO;

    for (int64_t b = 0; b < nb; b++) {
        const float norm = GGML_FP16_TO_FP32(x[b].norm);
        for (int j = 0; j < QK_TURBO; j++) {
            const int q = (x[b].qs[j/4] >> ((j % 4)*2)) & 0x3;
            y[b*QK_TURBO + j] = turbo_centroids_2bit[q]*norm;
        }
    }
}

void dequantize_row_turbo3_0(const block_turbo3_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO == 0);
    const int64_t nb = k/QK_TURBO;

    for (int64_t b = 0; b < nb; b++) {
        const float norm = GGML_FP16_TO_FP32(x[b].norm);
        for (int j = 0; j < QK_TURBO; j++) {
            const int lo = (x[b].qs[j/4] >> ((j % 4)*2)) & 0x3;
            const int hi = (x[b].signs[j/8] >> (j % 8)) & 0x1;
            y[b*QK_TURBO + j] = turbo_centroids_3bit[lo | (hi << 2)]*norm;
        }
    }
}

void dequantize_row_turbo4_0(const block_turbo4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO == 0);
    const int64_t nb = k/QK_TURBO;

    for (int64_t b = 0; b < nb; b++) {
        const float norm = GGML_FP16_TO_FP32(x[b].norm);
        for (int j = 0; j < QK_TURBO; j++) {
            const int q = (x[b].qs[j/2] >> ((j % 2)*4)) & 0xF;
            y[b*QK_TURBO + j] = turbo_centroids_4bit[q]*norm;
        }
    }
}

size_t quantize_turbo2_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    (void) quant_weights; // not applicable: KV-cache-only type
    const size_t row_size = ggml_row_size(GGML_TYPE_TURBO2_0, n_per_row);
    for (int64_t r = 0; r < nrow; r++) {
        quantize_row_turbo2_0_ref(src + r*n_per_row, (block_turbo2_0 *)((char *) dst + r*row_size), n_per_row);
    }
    return nrow*row_size;
}

size_t quantize_turbo3_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    (void) quant_weights;
    const size_t row_size = ggml_row_size(GGML_TYPE_TURBO3_0, n_per_row);
    for (int64_t r = 0; r < nrow; r++) {
        quantize_row_turbo3_0_ref(src + r*n_per_row, (block_turbo3_0 *)((char *) dst + r*row_size), n_per_row);
    }
    return nrow*row_size;
}

size_t quantize_turbo4_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    (void) quant_weights;
    const size_t row_size = ggml_row_size(GGML_TYPE_TURBO4_0, n_per_row);
    for (int64_t r = 0; r < nrow; r++) {
        quantize_row_turbo4_0_ref(src + r*n_per_row, (block_turbo4_0 *)((char *) dst + r*row_size), n_per_row);
    }
    return nrow*row_size;
}
