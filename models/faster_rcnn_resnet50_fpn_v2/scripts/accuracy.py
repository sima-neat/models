"""Score Faster R-CNN ResNet-50 FPN v2 on COCO val2017 as torchvision evaluates it.

Each image is preprocessed as torchvision's GeneralizedRCNNTransform does, into the top-left corner of the
backbone's 1344x1344 input. The backbone returns its raw output, a C kernel this script builds with the board's
gcc selects as many proposals as the box head takes and pools their regions straight into its raw input, the box
head runs as its Inference and Postprocess fragments, and calibration.py decodes the detections, at most 100 per image,
mapped back to the original image. This is the same path performance.py measures.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import io
import json
import platform
import random
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import model_report
import numpy as np
import pyneat
from calibration import CHANNELS, GRID_ROWS, INPUT_SIZE, PITCH, decode_image, detections, preprocess, region_rows
from faster_coco_eval import COCO, COCOeval_faster

ANNOTATIONS = "annotations/instances_val2017.json"
IMAGES = "val2017"
# Metric name in model.yaml -> COCOeval_faster.stats_as_dict label. With maxDets [1, 10, 100], AR_all,
# AR_second and AR_third are the recall at 1, 10 and 100 detections per image.
METRICS = {
    "coco_map": "AP_all",
    "coco_ap50": "AP_50",
    "coco_ap75": "AP_75",
    "coco_ap_small": "AP_small",
    "coco_ap_medium": "AP_medium",
    "coco_ap_large": "AP_large",
    "coco_ar1": "AR_all",
    "coco_ar10": "AR_second",
    "coco_ar100": "AR_third",
    "coco_ar_small": "AR_small",
    "coco_ar_medium": "AR_medium",
    "coco_ar_large": "AR_large",
}
MAX_DETECTIONS = [1, 10, 100]
# The backbone's one raw output: each FPN level's cells as rows, P2-P6 in order, with P2-P5's 256 features, then the RPN
# head's 3 objectness logits and 12 box deltas scaled by these gains into the features' range, then one padding column.
LEVEL_SIZES = tuple(INPUT_SIZE // stride for stride in (4, 8, 16, 32, 64))
LEVEL_ROWS = tuple(int(np.cumsum([0] + [size * size for size in LEVEL_SIZES])[i]) for i in range(6))
ROW_STRIDE = 272
OBJECTNESS_GAIN, DELTA_GAIN = 1 / 16, 1 / 2
KIND_INT8, KIND_BF16 = 0, 1
KERNEL_THREADS = 8
# Images decode and preprocess on worker threads while the MPKs run; OpenCV releases the GIL.
WORKERS = 4
PREFETCH = 8


def tensor_options(raw_output: bool = False) -> pyneat.ModelOptions:
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Tensor
    # The script already resized, normalized and padded the image; Neat still quantizes and tessellates.
    options.preprocess.resize.enable = pyneat.AutoFlag.Off
    options.preprocess.color_convert.enable = pyneat.AutoFlag.Off
    options.preprocess.normalize.enable = pyneat.AutoFlag.Off
    options.inference_terminal.mla_only = raw_output
    return options


# The CPU steps between the components, as C built on the board by load_kernel. Pooling 1,000 regions from these outputs
# takes about 20 ms this way and about 160 ms with NumPy and OpenCV. The benchmark fingerprint covers this script, so
# the source stays in it; accuracy.py and performance.py hold the same copy, which a build test keeps equal.
KERNEL_SOURCE = r"""
// Faster R-CNN's CPU steps between its two MLA components, following torchvision at inference. It reads the backbone's
// raw rows in place and writes each pooled region straight into the box head's raw input grid, so neither buffer is
// converted to float32 by Neat. GCC turns the channel loops into NEON on the Cortex-A65.
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define CHANNELS 256
#define ANCHORS 3
#define RPN_CHANNELS 15
#define POOL 7
#define PITCH 8
// load_kernel sets REGIONS, the proposals the box head takes, and GRID_COLS, its grid's columns.
#if !defined(REGIONS) || !defined(GRID_COLS)
#error "build with -DREGIONS and -DGRID_COLS"
#endif
#define GRID_WIDTH (PITCH * GRID_COLS)
#define LEVELS 5
#define MAX_THREADS 16
#define ROI_CHUNK 16
#define COMPACT_CHUNK 8192
#define TOP_N 1000
#define MAX_CANDIDATES (LEVELS * TOP_N)
#define FIXED_ONE 8192

enum { KIND_INT8 = 0, KIND_BF16 = 1 };

typedef float alias_float __attribute__((may_alias));

typedef struct {
    int kind;
    const void *rows;
    long stride;  // Elements per row.
    float step;   // INT8: real = (q - zero_point) * step.
    int zero_point;
    const long *offsets;  // Each FPN level's first row, and the total.
    const int *sizes;     // Each FPN level's side.
} Backbone;

static inline float bf16_to_float(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

static inline uint16_t float_to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return (uint16_t)((bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16);  // To nearest, ties to even.
}

static void run_workers(void *(*worker)(void *), void *job, int threads) {
    pthread_t handles[MAX_THREADS];
    int started = 0;
    if (threads > MAX_THREADS) threads = MAX_THREADS;
    for (int t = 1; t < threads; t++) {
        if (pthread_create(&handles[started], NULL, worker, job) == 0) started++;
    }
    worker(job);
    for (int t = 0; t < started; t++) pthread_join(handles[t], NULL);
}

typedef struct {
    float low_weight[POOL * 2], high_weight[POOL * 2];
    int low[POOL * 2], high[POOL * 2];
} Axis;

// roi_align's samples along one axis, aligned=False, sampling ratio 2, with its edge rules.
static void axis_samples(float start, float end, int size, Axis *axis) {
    float length = end - start;
    float bin = (length > 1.0f ? length : 1.0f) / POOL;
    for (int p = 0; p < POOL; p++) {
        for (int i = 0; i < 2; i++) {
            int s = p * 2 + i;
            float c = start + p * bin + (i + 0.5f) * bin / 2;
            if (c < -1.0f || c > (float)size) {
                axis->low[s] = axis->high[s] = 0;
                axis->low_weight[s] = axis->high_weight[s] = 0.0f;
                continue;
            }
            if (c < 0.0f) c = 0.0f;
            int low = (int)c;
            if (low >= size - 1) {
                axis->low[s] = axis->high[s] = size - 1;
                axis->low_weight[s] = 1.0f;
                axis->high_weight[s] = 0.0f;
            } else {
                axis->low[s] = low;
                axis->high[s] = low + 1;
                axis->high_weight[s] = c - low;
                axis->low_weight[s] = 1.0f - axis->high_weight[s];
            }
        }
    }
}

// LevelMapper: the map, 0-3 for P2-P5, a box pools from.
static int roi_level(const float *box) {
    float scale = sqrtf((box[2] - box[0]) * (box[3] - box[1]));
    int level = (int)floorf(4.0f + log2f(scale / 224.0f + 1e-6f));
    return (level < 2 ? 2 : level > 5 ? 5 : level) - 2;
}

typedef struct {
    float a, b, c, d;
    long p0, p1, p2, p3;  // Rows of the four bilinear taps.
} Sample;

// The bilinear samples of one output bin; returns how many have any weight.
static int bin_samples(const Axis *ys, const Axis *xs, int size, long first_row, int py, int px, Sample *samples) {
    int count = 0;
    for (int sy = py * 2; sy < py * 2 + 2; sy++) {
        for (int sx = px * 2; sx < px * 2 + 2; sx++) {
            float wl = ys->low_weight[sy], wh = ys->high_weight[sy];
            Sample s = {wl * xs->low_weight[sx], wl * xs->high_weight[sx], wh * xs->low_weight[sx],
                        wh * xs->high_weight[sx], 0, 0, 0, 0};
            if (s.a == 0.0f && s.b == 0.0f && s.c == 0.0f && s.d == 0.0f) continue;
            long top = first_row + (long)ys->low[sy] * size, bottom = first_row + (long)ys->high[sy] * size;
            s.p0 = top + xs->low[sx];
            s.p1 = top + xs->high[sx];
            s.p2 = bottom + xs->low[sx];
            s.p3 = bottom + xs->high[sx];
            samples[count++] = s;
        }
    }
    return count;
}

typedef struct {
    const Backbone *backbone;
    const float *boxes;
    void *grid;
    float out_scale;
    int out_zero_point;
    int count, next;
} PoolJob;

// One bin on INT8 rows, into the box head's INT8 input: fixed-point weights make the loop NEON widening multiplies.
static void bin_int8(const PoolJob *job, const Sample *samples, int count, int8_t *restrict out) {
    const Backbone *backbone = job->backbone;
    const int8_t *rows = backbone->rows;
    int sum[CHANNELS];
    memset(sum, 0, sizeof(sum));
    int total_weight = 0;
    for (int s = 0; s < count; s++) {
        const short a = (short)lrintf(samples[s].a * FIXED_ONE), b = (short)lrintf(samples[s].b * FIXED_ONE);
        const short c = (short)lrintf(samples[s].c * FIXED_ONE), d = (short)lrintf(samples[s].d * FIXED_ONE);
        total_weight += a + b + c + d;
        const int8_t *restrict p0 = rows + samples[s].p0 * backbone->stride;
        const int8_t *restrict p1 = rows + samples[s].p1 * backbone->stride;
        const int8_t *restrict p2 = rows + samples[s].p2 * backbone->stride;
        const int8_t *restrict p3 = rows + samples[s].p3 * backbone->stride;
        for (int i = 0; i < CHANNELS; i++) {
            sum[i] += (short)p0[i] * a + (short)p1[i] * b + (short)p2[i] * c + (short)p3[i] * d;
        }
    }
    float to_q = backbone->step / (4.0f * FIXED_ONE) * job->out_scale, shift = (float)job->out_zero_point;
    int offset = backbone->zero_point * total_weight;
    for (int i = 0; i < CHANNELS; i++) {
        float q = (float)(sum[i] - offset) * to_q + shift;
        q += q < 0.0f ? -0.5f : 0.5f;
        int v = (int)q;
        out[i] = (int8_t)(v < -128 ? -128 : v > 127 ? 127 : v);
    }
}

// One bin on BF16 rows, into the box head's BF16 input.
static void bin_bf16(const PoolJob *job, const Sample *samples, int count, uint16_t *restrict out) {
    const Backbone *backbone = job->backbone;
    const uint16_t *rows = backbone->rows;
    float sum[CHANNELS];
    memset(sum, 0, sizeof(sum));
    for (int s = 0; s < count; s++) {
        const float a = samples[s].a, b = samples[s].b, c = samples[s].c, d = samples[s].d;
        const uint16_t *restrict p0 = rows + samples[s].p0 * backbone->stride;
        const uint16_t *restrict p1 = rows + samples[s].p1 * backbone->stride;
        const uint16_t *restrict p2 = rows + samples[s].p2 * backbone->stride;
        const uint16_t *restrict p3 = rows + samples[s].p3 * backbone->stride;
        for (int i = 0; i < CHANNELS; i++) {
            sum[i] += a * bf16_to_float(p0[i]) + b * bf16_to_float(p1[i]) + c * bf16_to_float(p2[i]) +
                      d * bf16_to_float(p3[i]);
        }
    }
    for (int i = 0; i < CHANNELS; i++) out[i] = float_to_bf16(sum[i] * 0.25f);
}

static void pool_region(const PoolJob *job, const float *box, long cell_offset) {
    const Backbone *backbone = job->backbone;
    int level = roi_level(box);
    int size = backbone->sizes[level];
    float scale = (float)size / (float)backbone->sizes[0] / 4.0f;
    Axis ys, xs;
    axis_samples(box[1] * scale, box[3] * scale, size, &ys);
    axis_samples(box[0] * scale, box[2] * scale, size, &xs);
    for (int py = 0; py < POOL; py++) {
        for (int px = 0; px < POOL; px++) {
            Sample samples[4];
            int count = bin_samples(&ys, &xs, size, backbone->offsets[level], py, px, samples);
            long bin_offset = cell_offset + ((long)py * GRID_WIDTH + px) * CHANNELS;
            if (backbone->kind == KIND_INT8) {
                bin_int8(job, samples, count, (int8_t *)job->grid + bin_offset);
            } else {
                bin_bf16(job, samples, count, (uint16_t *)job->grid + bin_offset);
            }
        }
    }
}

static long cell_offset_of(int region) {
    return ((long)(region / GRID_COLS) * PITCH * GRID_WIDTH + (long)(region % GRID_COLS) * PITCH) * CHANNELS;
}

static void *pool_worker(void *argument) {
    PoolJob *job = argument;
    for (;;) {
        int first = __atomic_fetch_add(&job->next, ROI_CHUNK, __ATOMIC_RELAXED);
        if (first >= job->count) return NULL;
        int last = first + ROI_CHUNK < job->count ? first + ROI_CHUNK : job->count;
        for (int r = first; r < last; r++) pool_region(job, job->boxes + r * 4, cell_offset_of(r));
    }
}

typedef struct {
    float score;
    int index;
} Ranked;

// By decreasing score, then increasing index, as a stable descending sort gives.
static int compare_ranked(const void *a, const void *b) {
    const Ranked *x = a, *y = b;
    if (x->score != y->score) return x->score > y->score ? -1 : 1;
    return x->index - y->index;
}

static void sift_down(Ranked *heap, int n, int i) {
    for (;;) {
        int least = i, l = 2 * i + 1, r = l + 1;
        if (l < n && compare_ranked(&heap[l], &heap[least]) > 0) least = l;
        if (r < n && compare_ranked(&heap[r], &heap[least]) > 0) least = r;
        if (least == i) return;
        Ranked t = heap[i];
        heap[i] = heap[least];
        heap[least] = t;
        i = least;
    }
}

// Greedy NMS over boxes sorted by decreasing score; returns how many stay, compacted in order.
static int nms_sorted(float *boxes, float *scores, int n, float threshold) {
    float *areas = malloc(sizeof(float) * (n + 1));
    char *dead = calloc(n + 1, 1);
    for (int i = 0; i < n; i++) areas[i] = (boxes[i * 4 + 2] - boxes[i * 4]) * (boxes[i * 4 + 3] - boxes[i * 4 + 1]);
    int kept = 0;
    for (int i = 0; i < n; i++) {
        if (dead[i]) continue;
        const float *a = boxes + i * 4;
        for (int j = i + 1; j < n; j++) {
            if (dead[j]) continue;
            const float *b = boxes + j * 4;
            float w = (a[2] < b[2] ? a[2] : b[2]) - (a[0] > b[0] ? a[0] : b[0]);
            if (w <= 0.0f) continue;
            float h = (a[3] < b[3] ? a[3] : b[3]) - (a[1] > b[1] ? a[1] : b[1]);
            if (h <= 0.0f) continue;
            float inter = w * h;
            if (inter > threshold * (areas[i] + areas[j] - inter)) dead[j] = 1;
        }
        memmove(boxes + kept * 4, a, sizeof(float) * 4);
        scores[kept++] = scores[i];
    }
    free(areas);
    free(dead);
    return kept;
}

// The 15 RPN columns of every row, gains undone, in one float buffer: scanning this instead of the rows avoids a cache
// miss per anchor cell in the output buffer.
typedef struct {
    const Backbone *backbone;
    float objectness_scale, delta_scale;
    float *compact;
    long next;
} CompactJob;

static void *compact_worker(void *argument) {
    CompactJob *job = argument;
    const Backbone *backbone = job->backbone;
    long total = backbone->offsets[LEVELS];
    for (;;) {
        long first = __atomic_fetch_add(&job->next, COMPACT_CHUNK, __ATOMIC_RELAXED);
        if (first >= total) return NULL;
        long last = first + COMPACT_CHUNK < total ? first + COMPACT_CHUNK : total;
        for (long row = first; row < last; row++) {
            float *out = job->compact + row * RPN_CHANNELS;
            long start = row * backbone->stride + CHANNELS;
            if (backbone->kind == KIND_INT8) {
                const int8_t *in = (const int8_t *)backbone->rows + start;
                for (int i = 0; i < RPN_CHANNELS; i++) out[i] = (float)(in[i] - backbone->zero_point) * backbone->step;
            } else {
                const uint16_t *in = (const uint16_t *)backbone->rows + start;
                for (int i = 0; i < RPN_CHANNELS; i++) out[i] = bf16_to_float(in[i]);
            }
            for (int i = 0; i < ANCHORS; i++) out[i] *= job->objectness_scale;
            for (int i = ANCHORS; i < RPN_CHANNELS; i++) out[i] *= job->delta_scale;
        }
    }
}

static const int ANCHOR_SIZES[LEVELS] = {32, 64, 128, 256, 512};
static const float H_RATIOS[ANCHORS] = {0.70710678f, 1.0f, 1.41421356f};  // Square roots of the aspect ratios.
static const float RPN_MIN_SIZE = 1e-3f, RPN_NMS = 0.7f, BOX_CLIP = 4.135166556742356f;

typedef struct {
    const float *compact;
    const long *offsets;
    const int *sizes;
    int input_size;
    float left, top, right, bottom;
    float *level_boxes, *level_scores;
    int level_counts[LEVELS];
    int next;
} ProposalJob;

static inline float clampf(float v, float low, float high) { return v < low ? low : v > high ? high : v; }

static void level_proposals(ProposalJob *job, int level) {
    int size = job->sizes[level];
    int total = size * size * ANCHORS;
    int k = total < TOP_N ? total : TOP_N;
    const float *rows = job->compact + job->offsets[level] * RPN_CHANNELS;
    Ranked heap[TOP_N];
    // A min-heap of the top k: nearly every anchor fails the first comparison with its root.
    for (int i = 0; i < total; i++) {
        Ranked item = {rows[(long)(i / ANCHORS) * RPN_CHANNELS + i % ANCHORS], i};
        if (i < k) {
            heap[i] = item;
            if (i == k - 1) {
                for (int h = k / 2 - 1; h >= 0; h--) sift_down(heap, k, h);
            }
        } else if (item.score > heap[0].score) {
            heap[0] = item;
            sift_down(heap, k, 0);
        }
    }
    qsort(heap, k, sizeof(Ranked), compare_ranked);
    float *boxes = job->level_boxes + level * TOP_N * 4;
    float *scores = job->level_scores + level * TOP_N;
    int stride = job->input_size / size, count = 0;
    for (int t = 0; t < k; t++) {
        int anchor = heap[t].index % ANCHORS, cell = heap[t].index / ANCHORS;
        float half_w = roundf(ANCHOR_SIZES[level] / H_RATIOS[anchor] / 2);
        float half_h = roundf(ANCHOR_SIZES[level] * H_RATIOS[anchor] / 2);
        float shift_x = (float)(cell % size * stride), shift_y = (float)(cell / size * stride);
        float x1 = shift_x - half_w, y1 = shift_y - half_h, w = half_w * 2, h = half_h * 2;
        const float *delta = rows + (long)cell * RPN_CHANNELS + ANCHORS + anchor * 4;
        float dw = delta[2] < BOX_CLIP ? delta[2] : BOX_CLIP, dh = delta[3] < BOX_CLIP ? delta[3] : BOX_CLIP;
        float cx = delta[0] * w + x1 + 0.5f * w, cy = delta[1] * h + y1 + 0.5f * h;
        float half_pw = 0.5f * expf(dw) * w, half_ph = 0.5f * expf(dh) * h;
        float bx1 = clampf(cx - half_pw, job->left, job->right), by1 = clampf(cy - half_ph, job->top, job->bottom);
        float bx2 = clampf(cx + half_pw, job->left, job->right), by2 = clampf(cy + half_ph, job->top, job->bottom);
        if (bx2 - bx1 < RPN_MIN_SIZE || by2 - by1 < RPN_MIN_SIZE) continue;
        boxes[count * 4] = bx1, boxes[count * 4 + 1] = by1, boxes[count * 4 + 2] = bx2, boxes[count * 4 + 3] = by2;
        scores[count++] = 1.0f / (1.0f + expf(-heap[t].score));
    }
    job->level_counts[level] = nms_sorted(boxes, scores, count, RPN_NMS);
}

static void *proposal_worker(void *argument) {
    ProposalJob *job = argument;
    for (;;) {
        int level = __atomic_fetch_add(&job->next, 1, __ATOMIC_RELAXED);
        if (level >= LEVELS) return NULL;
        level_proposals(job, level);
    }
}

// RegionProposalNetwork at inference: up to REGIONS (x1, y1, x2, y2) boxes, by decreasing score; returns the count.
// window: left, top, right, bottom of the image in the input; boxes clip to it.
int frcnn_proposals(int kind, const void *rows, long stride, float scale, int zero_point, float objectness_gain,
                    float delta_gain, const long *offsets, const int *sizes, int input_size, const float *window,
                    float *out_boxes, int threads) {
    Backbone backbone = {kind, rows, stride, 1.0f / scale, zero_point, offsets, sizes};
    float *compact = malloc(sizeof(float) * RPN_CHANNELS * offsets[LEVELS]);
    float *level_boxes = malloc(sizeof(float) * 4 * MAX_CANDIDATES);
    float *level_scores = malloc(sizeof(float) * MAX_CANDIDATES);
    CompactJob compact_job = {&backbone, 1.0f / objectness_gain, 1.0f / delta_gain, compact, 0};
    run_workers(compact_worker, &compact_job, threads);
    ProposalJob job = {compact, offsets, sizes, input_size, window[0], window[1], window[2], window[3],
                       level_boxes, level_scores, {0}, 0};
    run_workers(proposal_worker, &job, threads < LEVELS ? threads : LEVELS);
    Ranked *merged = malloc(sizeof(Ranked) * MAX_CANDIDATES);
    int total = 0;
    for (int level = 0; level < LEVELS; level++) {
        for (int i = 0; i < job.level_counts[level]; i++) {
            merged[total].score = level_scores[level * TOP_N + i];
            merged[total++].index = level * TOP_N + i;
        }
    }
    qsort(merged, total, sizeof(Ranked), compare_ranked);
    int count = total < REGIONS ? total : REGIONS;
    for (int i = 0; i < count; i++) memcpy(out_boxes + i * 4, level_boxes + merged[i].index * 4, sizeof(float) * 4);
    free(merged);
    free(level_boxes);
    free(level_scores);
    free(compact);
    return count;
}

// Reset cells first..last-1 of the grid to the box head's real zero, after a frame with fewer boxes than its last.
static void clear_cells(int kind, void *grid, int first, int last, int zero_point) {
    for (int r = first; r < last; r++) {
        for (int py = 0; py < POOL; py++) {
            long offset = cell_offset_of(r) + (long)py * GRID_WIDTH * CHANNELS;
            if (kind == KIND_INT8) {
                memset((int8_t *)grid + offset, zero_point, POOL * CHANNELS);
            } else {
                memset((uint16_t *)grid + offset, 0, sizeof(uint16_t) * POOL * CHANNELS);
            }
        }
    }
}

// MultiScaleRoIAlign of count boxes on P2-P5, into cells 0..count-1 of the box head's (200, 320, 256) raw input, in its
// own quantization for INT8. previous_count is what this grid last held. Fill a new INT8 grid with out_zero_point: that
// is the box head's real zero, in its gaps and unused cells.
void frcnn_roi_align(int kind, const void *rows, long stride, float scale, int zero_point, const long *offsets,
                     const int *sizes, const float *boxes, int count, void *grid, float out_scale, int out_zero_point,
                     int previous_count, int threads) {
    Backbone backbone = {kind, rows, stride, 1.0f / scale, zero_point, offsets, sizes};
    PoolJob job = {&backbone, boxes, grid, out_scale, out_zero_point, count, 0};
    run_workers(pool_worker, &job, threads);
    clear_cells(kind, grid, count, previous_count, out_zero_point);
}
"""


def load_kernel(regions: int) -> ctypes.CDLL:
    """Build KERNEL_SOURCE with the board's gcc for a box head of regions proposals. Without gcc the script stops: a
    slower fallback would measure a different pipeline."""
    gcc = shutil.which("gcc")
    if gcc is None:
        raise SystemExit("this script builds its C kernel with gcc, which is not on this board")
    target = ["-mcpu=cortex-a65"] if platform.machine() == "aarch64" else []
    with tempfile.TemporaryDirectory(prefix="faster_rcnn_kernel-") as directory:
        source, library = Path(directory) / "kernel.c", Path(directory) / "kernel.so"
        source.write_text(KERNEL_SOURCE, encoding="utf-8")
        grid = [f"-DREGIONS={regions}", f"-DGRID_COLS={regions // GRID_ROWS}"]
        subprocess.run(
            [gcc, "-O3", *target, *grid, "-shared", "-fPIC", "-pthread", "-o", library, source, "-lm"], check=True
        )
        kernel = ctypes.CDLL(str(library))
    buffer, floats = ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)
    longs, ints = ctypes.POINTER(ctypes.c_long), ctypes.POINTER(ctypes.c_int)
    number, whole = ctypes.c_float, ctypes.c_int
    kernel.frcnn_proposals.restype = whole
    kernel.frcnn_proposals.argtypes = [
        whole,
        buffer,
        ctypes.c_long,
        number,
        whole,
        number,
        number,
        longs,
        ints,
        whole,
        floats,
        floats,
        whole,
    ]
    kernel.frcnn_roi_align.restype = None
    kernel.frcnn_roi_align.argtypes = [
        whole,
        buffer,
        ctypes.c_long,
        number,
        whole,
        longs,
        ints,
        floats,
        whole,
        buffer,
        number,
        whole,
        whole,
        whole,
    ]
    return kernel


@dataclass(frozen=True)
class RawFormat:
    """How a raw buffer between the MLA and the CPU holds values: INT8 with its scale and zero point, or BF16."""

    kind: int
    dtype: type
    scale: float
    zero_point: int


@dataclass
class Grid:
    """A box head input buffer and how many regions it last held."""

    array: np.ndarray
    count: int = 0


def _manifest_plugins(mpk: Path) -> list[dict]:
    with tarfile.open(mpk, "r:gz") as stream:
        manifest = next(member for member in stream.getmembers() if member.name.endswith("_mpk.json"))
        return sorted(json.load(stream.extractfile(manifest))["plugins"], key=lambda plugin: plugin["sequence"])


def _raw_format(plugin: dict) -> RawFormat:
    """The format a package's quantize, dequantize or cast step converts to or from."""
    config = plugin["config_params"]
    if config["kernel"] in ("quantization_transform", "dequantization_transform"):
        scale, zero_point = config["params"]["channel_params"][0]
        return RawFormat(KIND_INT8, np.int8, float(scale), int(zero_point))
    if config["kernel"] != "cast_transform":
        raise SystemExit(f"unexpected package step: {config['kernel']}")
    return RawFormat(KIND_BF16, np.uint16, 1.0, 0)


def backbone_format(mpk: Path) -> RawFormat:
    """The backbone's raw output, as its package's last step converts it."""
    return _raw_format(_manifest_plugins(mpk)[-1])


def box_head_format(mpk: Path) -> RawFormat:
    """The box head's raw input, as its package's first step converts it."""
    return _raw_format(_manifest_plugins(mpk)[0])


def box_head_regions(mpk: Path) -> int:
    """The proposals the box head takes, from its float32 input's size: a grid of 25 rows of 8x8 cells of 256 values."""
    with tarfile.open(mpk, "r:gz") as stream:
        manifest = next(member for member in stream.getmembers() if member.name.endswith("_mpk.json"))
        (node,) = json.load(stream.extractfile(manifest))["input_nodes"]
    cols, extra = divmod(int(node["size"]), 4 * CHANNELS * GRID_ROWS * PITCH * PITCH)
    if extra or not cols:
        raise SystemExit(f"unexpected box head input size: {node['size']}")
    return GRID_ROWS * cols


def new_grid(raw: RawFormat, regions: int) -> Grid:
    """A box head input for regions proposals with every cell at the real zero."""
    return Grid(np.full((GRID_ROWS * PITCH, regions // GRID_ROWS * PITCH, CHANNELS), raw.zero_point, raw.dtype))


def pool_regions(
    kernel: ctypes.CDLL, raw: np.ndarray, backbone: RawFormat, box_head: RawFormat, window: Sequence[float], grid: Grid
) -> np.ndarray:
    """The backbone's raw output -> the frame's proposals, with their regions written into the grid."""
    if raw.nbytes != LEVEL_ROWS[-1] * ROW_STRIDE * np.dtype(backbone.dtype).itemsize:
        raise ValueError(f"unexpected raw backbone output: {raw.shape} {raw.dtype}")
    offsets, sizes = (ctypes.c_long * len(LEVEL_ROWS))(*LEVEL_ROWS), (ctypes.c_int * len(LEVEL_SIZES))(*LEVEL_SIZES)
    boxes = np.empty((grid.array.shape[0] // PITCH * (grid.array.shape[1] // PITCH), 4), np.float32)
    floats = ctypes.POINTER(ctypes.c_float)
    count = kernel.frcnn_proposals(
        backbone.kind,
        raw.ctypes.data,
        ROW_STRIDE,
        backbone.scale,
        backbone.zero_point,
        OBJECTNESS_GAIN,
        DELTA_GAIN,
        offsets,
        sizes,
        INPUT_SIZE,
        (ctypes.c_float * 4)(*window),
        boxes.ctypes.data_as(floats),
        KERNEL_THREADS,
    )
    kernel.frcnn_roi_align(
        backbone.kind,
        raw.ctypes.data,
        ROW_STRIDE,
        backbone.scale,
        backbone.zero_point,
        offsets,
        sizes,
        boxes.ctypes.data_as(floats),
        count,
        grid.array.ctypes.data,
        box_head.scale,
        box_head.zero_point,
        grid.count,
        KERNEL_THREADS,
    )
    grid.count = count
    return boxes[:count]


def box_head_graph(model: pyneat.Model) -> pyneat.Graph:
    """The box head's package without its input conversion: the kernel writes its input in the raw format."""
    graph = pyneat.Graph()
    graph.add(pyneat.nodes.input(pyneat.InputOptions()))
    graph.add(model.fragment(pyneat.ModelStage.Inference))
    graph.add(model.fragment(pyneat.ModelStage.Postprocess))
    graph.add(pyneat.nodes.output(pyneat.OutputOptions()))
    return graph


def grid_tensor(grid: Grid) -> pyneat.Tensor:
    """The grid as a device tensor. Making one copies the grid, so the grid is free once this returns. Neat keeps
    about 64 of a model's past inputs, which at this size overflow the 1.8 GB heap EV74 tensors use, so it uses MLA
    memory: the box head runs from there with the same results."""
    return pyneat.Tensor.from_numpy(
        grid.array, copy=False, layout=pyneat.TensorLayout.HWC, memory=pyneat.TensorMemory.MLA
    )


def prefetched(load: Callable, items: Sequence) -> Iterator:
    """Yield load(item) in order, keeping at most PREFETCH items in flight."""
    with ThreadPoolExecutor(WORKERS) as pool:
        pending = deque(pool.submit(load, item) for item in items[:PREFETCH])
        for item in items[PREFETCH:]:
            yield pending.popleft().result()
            pending.append(pool.submit(load, item))
        while pending:
            yield pending.popleft().result()


class Detector:
    """The two MPKs and the kernel between them, one image at a time."""

    def __init__(self, mpks: dict[str, Path]) -> None:
        regions = box_head_regions(mpks["box_head"])
        self.kernel = load_kernel(regions)
        self.backbone_format, self.box_head_format = (
            backbone_format(mpks["backbone"]),
            box_head_format(mpks["box_head"]),
        )
        self.backbone_model = pyneat.Model(str(mpks["backbone"]), tensor_options(raw_output=True))
        self.box_head_model = pyneat.Model(str(mpks["box_head"]), tensor_options())
        self.grid = new_grid(self.box_head_format, regions)
        self.backbone = self.box_head = None

    def __call__(self, image: np.ndarray, size: tuple[int, int]):
        """One preprocessed image -> boxes in the resized image, scores and COCO category IDs."""
        # A tensor holds a copy of the array as it was here.
        tensor = pyneat.Tensor.from_numpy(image, copy=False)
        if self.backbone is None:
            self.backbone = self.backbone_model.build([tensor])
        outputs = self.backbone.run([tensor], timeout_ms=120000)
        boxes = pool_regions(
            self.kernel,
            outputs[0].to_numpy(copy=False),
            self.backbone_format,
            self.box_head_format,
            (0.0, 0.0, float(size[1]), float(size[0])),
            self.grid,
        )
        del outputs
        tensor = grid_tensor(self.grid)
        if self.box_head is None:
            self.box_head = box_head_graph(self.box_head_model).build([tensor])
        class_logits, box_regression = (view(output) for output in self.box_head.run([tensor], timeout_ms=120000))
        count = len(boxes)
        return detections(region_rows(class_logits)[:count], region_rows(box_regression)[:count], boxes, size)


def view(tensor: pyneat.Tensor) -> np.ndarray:
    """Read-only view of an output; valid until its MPK runs again."""
    array = tensor.to_numpy(copy=False)
    if array.dtype != np.float32:
        raise ValueError(f"expected a float32 MPK output, got {tensor.dtype}")
    if not np.isfinite(array).all():
        raise ValueError("MPK output contains non-finite values")
    return array


def result_rows(image_id: int, width: int, height: int, size: tuple[int, int], found) -> np.ndarray:
    """COCO result rows [image_id, x, y, w, h, score, category_id], in the original image's pixels."""
    boxes, scores, labels = found
    boxes = boxes * np.array([width / size[1], height / size[0]] * 2, np.float32)
    ids = np.full(len(boxes), image_id)
    return np.stack(
        [ids, boxes[:, 0], boxes[:, 1], boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1], scores, labels], axis=1
    )


def coco_stats(scorer: COCOeval_faster) -> dict[str, float]:
    """Every declared metric by its COCOeval label; COCOeval reports -1 for an empty object-size range."""
    if list(scorer.params.maxDets) != MAX_DETECTIONS:
        raise ValueError(f"expected COCOeval maxDets {MAX_DETECTIONS}, got {scorer.params.maxDets}")
    labelled = scorer.stats_as_dict
    stats = {name: float(labelled[label]) for name, label in METRICS.items()}
    empty = [name for name, value in stats.items() if value < 0]
    if empty:
        raise SystemExit(f"no ground-truth objects to score for: {', '.join(empty)}")
    return stats


def parse_mpks(values: list[str]) -> dict[str, Path]:
    mpks = dict(value.split("=", 1) for value in values)
    if set(mpks) != {"backbone", "box_head"}:
        raise SystemExit("--mpk must name exactly backbone=<path> and box_head=<path>")
    return {name: Path(path) for name, path in mpks.items()}


def log(*values: object) -> None:
    print(*values, file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpk", action="append", required=True, help="component=path, for backbone and box_head")
    parser.add_argument("--dataset", required=True, help="coco-2017-val=<extracted directory>")
    parser.add_argument("--output", type=Path, required=True, help="directory for model-report.json")
    parser.add_argument("--limit", type=int, help="measure a seeded sample of images (local runs only)")
    args = parser.parse_args(argv)
    dataset_id, dataset_dir = args.dataset.split("=", 1)
    dataset = Path(dataset_dir)

    mpks = parse_mpks(args.mpk)
    with contextlib.redirect_stdout(io.StringIO()):
        ground_truth = COCO(str(dataset / ANNOTATIONS))
    available = sorted(ground_truth.getImgIds())
    image_ids = sorted(random.Random(0).sample(available, args.limit)) if args.limit else available
    records = ground_truth.loadImgs(image_ids)

    detector = Detector(mpks)
    rows: list[np.ndarray] = []
    started = time.monotonic()
    images = prefetched(
        lambda record: preprocess(decode_image((dataset / IMAGES / record["file_name"]).read_bytes())), records
    )
    for count, (record, (image, size)) in enumerate(zip(records, images), start=1):
        found = detector(image, size)
        rows.append(result_rows(record["id"], record["width"], record["height"], size, found))
        if count == 1 or count % 250 == 0 or count == len(records):
            log(f"{count}/{len(records)} images, {(time.monotonic() - started) / count * 1000:.0f} ms/image")
    inference_seconds = time.monotonic() - started

    found = np.concatenate(rows)
    if not len(found):
        raise SystemExit("the model produced no detections")
    started = time.monotonic()
    with contextlib.redirect_stdout(io.StringIO()):
        results = ground_truth.loadRes(found)
    scorer = COCOeval_faster(ground_truth, results, "bbox", print_function=log)
    scorer.params.imgIds = image_ids
    scorer.evaluate()
    scorer.accumulate()
    scorer.summarize()
    stats = coco_stats(scorer)
    scoring_seconds = time.monotonic() - started

    report = model_report.from_accuracy(
        {
            dataset_id: {
                "metrics": stats,
                "split": IMAGES,
                "evaluated_samples": len(image_ids),
                "available_samples": len(available),
                "details": {
                    "detections": len(found),
                    "inference_seconds": round(inference_seconds, 1),
                    "scoring_seconds": round(scoring_seconds, 1),
                },
            },
        }
    )
    report.save(args.output)
    measured = report.document["datasets"][dataset_id]["metrics"]
    print(json.dumps({name: round(value, 4) for name, value in measured.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
