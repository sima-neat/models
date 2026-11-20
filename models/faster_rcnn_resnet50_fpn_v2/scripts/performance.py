"""Benchmark Faster R-CNN ResNet-50 FPN v2 with Neat Core and save its report.

Given --mpk backbone=<path> --mpk box_head=<path> and --dataset coco-2017-val=<extracted directory>, the
script measures the whole model as it runs on one COCO image in a 720p frame: Neat's preprocessing and the
backbone, the proposals and their RoIAlign on the CPU, the copy to the box head's memory and the box head,
and the detections' decoding and NMS on the CPU, each stage working on a different frame at the same time. The box
head's package sets how many proposals it takes. The script also records each MPK's MLA layer and stage stats and its
execution plan.

Raw values cross between the MLA and the CPU: the backbone runs with inference_terminal.mla_only and the box head
as its Inference and Postprocess fragments, so Neat converts neither side to float32. The proposals and RoIAlign
run in a C kernel this script builds with the board's gcc. They and the detections follow torchvision's
fasterrcnn_resnet50_fpn_v2 at inference, as calibration.py documents them; this script keeps its own copy so it
runs with only NumPy, OpenCV, gcc and Neat. Neat letterboxes the frame into the 1344x1344 input, centered, where
torchvision places the image top-left: the script clips and maps boxes to the frame's place in the input.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import platform
import queue
import random
import shutil
import subprocess
import tarfile
import tempfile
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import model_report
import numpy as np
import pyneat

INPUT_SIZE = 1344
WARMUP_FRAMES = 100
MEASURED_FRAMES = 1000
MIN_DURATION_S = 60
TIMEOUT_MS = 10000
# Frames each runner may queue; deeper queues add latency without adding throughput.
QUEUE_DEPTH = 4
IMAGES = "val2017"
# The benchmark input is a 720p camera frame, letterboxed with grey padding.
FRAME_WIDTH, FRAME_HEIGHT, FILL = 1280, 720, 114
# torchvision's inference settings for the steps in Python.
BOX_WEIGHTS = (10.0, 10.0, 5.0, 5.0)
BOX_CLIP = math.log(1000.0 / 16)
SCORE_THRESHOLD = 0.05
NMS_THRESHOLD = 0.5
BOX_MIN_SIZE = 1e-2
DETECTIONS = 100
CLASSES = 91
# The backbone's one raw output: each FPN level's cells as rows, P2-P6 in order, with P2-P5's 256 features, then the RPN
# head's 3 objectness logits and 12 box deltas scaled by these gains into the features' range, then one padding column.
LEVEL_SIZES = tuple(INPUT_SIZE // stride for stride in (4, 8, 16, 32, 64))
LEVEL_ROWS = tuple(int(np.cumsum([0] + [size * size for size in LEVEL_SIZES])[i]) for i in range(6))
ROW_STRIDE = 272
OBJECTNESS_GAIN, DELTA_GAIN = 1 / 16, 1 / 2
# The box head's input: its proposals' 7x7 maps of 256 features in a grid of 25 rows at an 8-pixel pitch. Its package's
# input size gives the number of proposals: torchvision's 1,000 fill a 25x40 grid.
CHANNELS, GRID_ROWS, PITCH = 256, 25, 8
KIND_INT8, KIND_BF16 = 0, 1
KERNEL_THREADS = 8
# Box head inputs in flight between the kernel and the copy to the device.
GRIDS = 3
# Seconds a stage waits on another before treating it as stopped.
STAGE_TIMEOUT_S = 60


def image_options() -> pyneat.ModelOptions:
    """The backbone's: Neat letterboxes, normalizes, quantizes and tessellates the frame inside the pipeline."""
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Image
    options.preprocess.enable = pyneat.AutoFlag.On
    # Size Neat's input buffers for the benchmark frame; it otherwise allocates for 1920x1080.
    options.preprocess.input_max_width = FRAME_WIDTH
    options.preprocess.input_max_height = FRAME_HEIGHT
    options.preprocess.resize.enable = pyneat.AutoFlag.On
    options.preprocess.resize.mode = pyneat.ResizeMode.Letterbox
    options.preprocess.normalize.enable = pyneat.AutoFlag.On
    # Without a preset Neat only scales pixels to [0, 1]; the model expects ImageNet mean and std.
    options.preprocess.preset = pyneat.NormalizePreset.ImageNet
    return options


def backbone_options() -> pyneat.ModelOptions:
    """The image route, stopping at the MLA's raw output: Neat does not convert it to float32."""
    options = image_options()
    options.inference_terminal.mla_only = True
    return options


def tensor_options() -> pyneat.ModelOptions:
    """The box head's: its input is the tiled regions."""
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Tensor
    options.preprocess.resize.enable = pyneat.AutoFlag.Off
    options.preprocess.color_convert.enable = pyneat.AutoFlag.Off
    options.preprocess.normalize.enable = pyneat.AutoFlag.Off
    return options


@dataclass(frozen=True)
class Window:
    """Where the letterboxed frame sits in the 1344x1344 input, and its scale."""

    top: float
    left: float
    bottom: float
    right: float
    scale: float


def letterbox_window() -> Window:
    scale = min(INPUT_SIZE / FRAME_WIDTH, INPUT_SIZE / FRAME_HEIGHT)
    width, height = FRAME_WIDTH * scale, FRAME_HEIGHT * scale
    top, left = (INPUT_SIZE - height) / 2, (INPUT_SIZE - width) / 2
    return Window(top, left, top + height, left + width, scale)


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


def decode_boxes(deltas: np.ndarray, boxes: np.ndarray, weights: tuple[float, ...]) -> np.ndarray:
    widths, heights = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    center_x, center_y = boxes[:, 0] + 0.5 * widths, boxes[:, 1] + 0.5 * heights
    dx, dy = deltas[:, 0] / weights[0], deltas[:, 1] / weights[1]
    dw = np.minimum(deltas[:, 2] / weights[2], BOX_CLIP)
    dh = np.minimum(deltas[:, 3] / weights[3], BOX_CLIP)
    x, y = dx * widths + center_x, dy * heights + center_y
    half_w, half_h = np.float32(0.5) * np.exp(dw) * widths, np.float32(0.5) * np.exp(dh) * heights
    return np.stack([x - half_w, y - half_h, x + half_w, y + half_h], axis=1).astype(np.float32)


def clip_boxes(boxes: np.ndarray, window: Window) -> np.ndarray:
    return np.stack(
        [
            np.clip(boxes[:, 0], window.left, window.right),
            np.clip(boxes[:, 1], window.top, window.bottom),
            np.clip(boxes[:, 2], window.left, window.right),
            np.clip(boxes[:, 3], window.top, window.bottom),
        ],
        axis=1,
    )


def nms(boxes: np.ndarray, scores: np.ndarray, groups: np.ndarray, threshold: float) -> np.ndarray:
    if not len(boxes):
        return np.zeros(0, np.int64)
    rects = np.concatenate([boxes[:, :2], boxes[:, 2:] - boxes[:, :2]], axis=1).astype(np.float64)
    keep = np.asarray(
        cv2.dnn.NMSBoxesBatched(rects, scores.astype(np.float32), groups.astype(np.int32), -1.0, threshold),
        np.int64,
    ).reshape(-1)
    return keep[np.argsort(-scores[keep], kind="stable")]


def region_rows(output: np.ndarray) -> np.ndarray:
    """A box head output, NHWC (1, 25, cols, C) or NCHW (1, C, 25, cols) -> (25 * cols, C), one row per region."""
    if output.shape[1] == GRID_ROWS:
        return output.reshape(GRID_ROWS * output.shape[2], -1)
    return output.reshape(output.shape[1], GRID_ROWS * output.shape[3]).T


def detections(class_logits: np.ndarray, box_regression: np.ndarray, boxes: np.ndarray, window: Window) -> np.ndarray:
    """(N, 6) rows of x1, y1, x2, y2 in the 720p frame's pixels, score and COCO category ID."""
    logits = class_logits - class_logits.max(axis=1, keepdims=True)
    scores = np.exp(logits)
    scores = scores[:, 1:] / scores.sum(axis=1, keepdims=True)
    region, column = np.nonzero(scores > SCORE_THRESHOLD)
    labels = column + 1
    deltas = box_regression.reshape(-1, CLASSES, 4)[region, labels]
    decoded = clip_boxes(decode_boxes(deltas, boxes[region], BOX_WEIGHTS), window)
    scores = scores[region, column]
    keep = np.where((decoded[:, 2] - decoded[:, 0] >= BOX_MIN_SIZE) & (decoded[:, 3] - decoded[:, 1] >= BOX_MIN_SIZE))[
        0
    ]
    decoded, scores, labels = decoded[keep], scores[keep], labels[keep]
    keep = nms(decoded, scores, labels, NMS_THRESHOLD)[:DETECTIONS]
    frame = (decoded[keep] - np.array([window.left, window.top] * 2, np.float32)) / np.float32(window.scale)
    return np.concatenate([frame, scores[keep, None], labels[keep, None].astype(np.float32)], axis=1)


@dataclass
class Pipeline:
    """What the CPU stages share: the kernel, both raw formats, the frame's place in the input, and the free grids."""

    kernel: ctypes.CDLL
    backbone: RawFormat
    box_head: RawFormat
    window: Window
    grids: queue.Queue


def pooled_regions(pipeline: Pipeline, outputs: Sequence[pyneat.Tensor]) -> tuple[Grid, np.ndarray]:
    """The backbone's raw output -> a grid holding the regions, and the proposals they came from."""
    grid = pipeline.grids.get(timeout=STAGE_TIMEOUT_S)
    window = pipeline.window
    boxes = pool_regions(
        pipeline.kernel,
        outputs[0].to_numpy(copy=False),
        pipeline.backbone,
        pipeline.box_head,
        (window.left, window.top, window.right, window.bottom),
        grid,
    )
    return grid, boxes


def decode_outputs(outputs: Sequence[pyneat.Tensor], boxes: np.ndarray, window: Window) -> np.ndarray:
    class_logits, box_regression = (region_rows(output.to_numpy(copy=False)) for output in outputs)
    return detections(class_logits[: len(boxes)], box_regression[: len(boxes)], boxes, window)


def percentile_ms(seconds: list[float], percentile: float) -> float:
    return float(np.percentile(seconds, percentile) * 1000)


def mla_stats(mpk: Path) -> tuple[dict[str, object], dict[str, object]]:
    """Core's MLA layer and stage stats for one MPK, in the shape Core saves them. Their throughput is not the
    model's, so Core's default run serves."""
    options = pyneat.BenchmarkOptions()
    options.include_mla_layer_stats = True
    options.include_mla_stage_stats = True
    report = pyneat.Model(str(mpk)).benchmark(options)
    layer_stats, stage_stats = report.mla_layer_stats, report.mla_stage_stats
    layers = [
        {"stage_index": layer.stage_index, "name": layer.name, "latency_us": layer.latency_us}
        for layer in layer_stats.layers
    ]
    stages = [{"stage_index": stage.stage_index, "latency_us": stage.latency_us} for stage in stage_stats.stages]
    return {"status": layer_stats.status, "layers": layers}, {"status": stage_stats.status, "stages": stages}


def sample_image(dataset: Path) -> Path:
    """One val2017 image, chosen with a fixed seed so every run measures the same input."""
    images = sorted((dataset / IMAGES).glob("*.jpg"))
    if not images:
        raise SystemExit(f"{dataset} has no images under {IMAGES}/")
    return random.Random(0).choice(images)


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise SystemExit(f"could not read {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def letterbox_to_720p(image: np.ndarray) -> np.ndarray:
    """Scale an RGB image to fit a 1280x720 frame and pad it to the frame, centered."""
    scale = min(FRAME_WIDTH / image.shape[1], FRAME_HEIGHT / image.shape[0])
    width, height = round(image.shape[1] * scale), round(image.shape[0] * scale)
    left, top = (FRAME_WIDTH - width) // 2, (FRAME_HEIGHT - height) // 2
    frame = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), FILL, np.uint8)
    frame[top : top + height, left : left + width] = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    return frame


def frame_tensor(path: Path) -> pyneat.Tensor:
    """The image's 720p frame, placed once in the memory the backbone reads, reused for every frame."""
    frame = letterbox_to_720p(read_rgb(path))
    return pyneat.Tensor.from_numpy(
        frame, copy=True, image_format=pyneat.PixelFormat.RGB, memory=pyneat.TensorMemory.EV74
    )


def build_runners(
    models: dict[str, pyneat.Model], image: pyneat.Tensor, pipeline: Pipeline
) -> tuple[pyneat.ModelRunner, pyneat.ModelRunner]:
    """A runner for the backbone and the box head's graph. The latter is built from one real backbone output."""
    run_options = pyneat.RunOptions()
    run_options.queue_depth = QUEUE_DEPTH
    run_options.overflow_policy = pyneat.OverflowPolicy.Block
    backbone = models["backbone"].build([image], run_options=run_options)
    grid, _ = pooled_regions(pipeline, backbone.run([image], timeout_ms=TIMEOUT_MS))
    box_head = box_head_graph(models["box_head"]).build([grid_tensor(grid)], options=run_options)
    pipeline.grids.put(grid)
    return backbone, box_head


def infer_one(
    backbone: pyneat.ModelRunner, box_head: pyneat.ModelRunner, image: pyneat.Tensor, pipeline: Pipeline
) -> np.ndarray:
    grid, boxes = pooled_regions(pipeline, backbone.run([image], timeout_ms=TIMEOUT_MS))
    tensor = grid_tensor(grid)
    pipeline.grids.put(grid)
    return decode_outputs(box_head.run([tensor], timeout_ms=TIMEOUT_MS), boxes, pipeline.window)


def measure_latency(
    backbone: pyneat.ModelRunner, box_head: pyneat.ModelRunner, image: pyneat.Tensor, pipeline: Pipeline
) -> list[float]:
    """Seconds per frame through every stage with nothing else in flight, after the warm-up."""
    for _ in range(WARMUP_FRAMES):
        infer_one(backbone, box_head, image, pipeline)
    seconds: list[float] = []
    ends = time.perf_counter() + MIN_DURATION_S
    while len(seconds) < MEASURED_FRAMES or time.perf_counter() < ends:
        started = time.perf_counter()
        infer_one(backbone, box_head, image, pipeline)
        seconds.append(time.perf_counter() - started)
    return seconds


@dataclass(frozen=True)
class Throughput:
    """The measured span of the throughput pass: the frames decoded, how long they took, and the board
    telemetry sampled over the same span."""

    frames: int
    seconds: float
    telemetry: pyneat.Telemetry

    @property
    def fps(self) -> float:
        return self.frames / self.seconds


def push(runner: pyneat.ModelRunner, inputs: list[pyneat.Tensor], name: str) -> None:
    if not runner.push(inputs):
        raise RuntimeError(f"the {name} rejected a frame")


def drain(runner: pyneat.ModelRunner, name: str, pushed: dict[str, int]) -> Iterator[list[pyneat.Tensor]]:
    """Each output until the runner's end of stream. Fewer outputs than frames pushed means a stall.

    Core's pull returns no tensors both at end of stream and on a timeout, so the count tells them apart. A model's
    runner returns a sample holding its tensors, and a graph's runner returns the tensors or None.
    """
    pulled = 0
    while tensors := getattr(sample := runner.pull(TIMEOUT_MS), "tensors", sample):
        pulled += 1
        yield tensors
        # Return the outputs to the runner before waiting for the next frame.
        del sample, tensors
    if pulled != pushed[name]:
        raise RuntimeError(f"the {name} stalled: {pulled} of {pushed[name]} frames came out within {TIMEOUT_MS} ms")


def run_together(*stages: Callable[[], None], runners: tuple[pyneat.ModelRunner, ...]) -> None:
    """Run each stage on its own thread. The first failure ends every runner's input, which unblocks a stage
    waiting to push, and is raised once all stages stop. Each stage waits in Core with the GIL released."""
    errors: list[BaseException] = []

    def run(stage: Callable[[], None]) -> None:
        try:
            stage()
        except BaseException as error:
            errors.append(error)
            for runner in runners:
                runner.close_input()

    threads = [threading.Thread(target=run, args=(stage,), name=stage.__name__) for stage in stages]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]


def measure_throughput(
    backbone: pyneat.ModelRunner, box_head: pyneat.ModelRunner, image: pyneat.Tensor, pipeline: Pipeline
) -> Throughput:
    """Frames per second and board telemetry with every stage busy, each on a different frame. The measured span
    starts after WARMUP_FRAMES outputs and lasts at least MEASURED_FRAMES frames and MIN_DURATION_S seconds."""
    monitor = pyneat.TelemetryMonitor()
    measured = threading.Event()
    throughput: Throughput | None = None
    pushed = {"backbone": 0, "box head": 0}
    # Grids the kernel filled, waiting for their copy to the device; None ends the stage.
    filled: queue.Queue = queue.Queue(GRIDS)
    # The proposals of each frame in the box head, in order.
    in_flight: queue.Queue = queue.Queue()

    def push_inputs() -> None:
        try:
            while not measured.is_set():
                push(backbone, [image], "backbone")
                pushed["backbone"] += 1
        finally:
            # End of stream follows the last frame through the backbone, then the box head.
            backbone.close_input()

    def pool_stage() -> None:
        try:
            for outputs in drain(backbone, "backbone", pushed):
                filled.put(pooled_regions(pipeline, outputs), timeout=STAGE_TIMEOUT_S)
                # The kernel is done with them, so the backbone gets its output buffers back now.
                del outputs
        finally:
            filled.put(None, timeout=STAGE_TIMEOUT_S)

    def copy_stage() -> None:
        try:
            while (item := filled.get(timeout=STAGE_TIMEOUT_S)) is not None:
                grid, boxes = item
                tensor = grid_tensor(grid)
                pipeline.grids.put(grid)
                in_flight.put(boxes)
                push(box_head, [tensor], "box head")
                pushed["box head"] += 1
        finally:
            box_head.close_input()

    def decode_detections() -> None:
        nonlocal throughput
        outputs, started = 0, 0.0
        for tensors in drain(box_head, "box head", pushed):
            decode_outputs(tensors, in_flight.get(timeout=STAGE_TIMEOUT_S), pipeline.window)
            del tensors
            outputs += 1
            if outputs == WARMUP_FRAMES:
                monitor.start()
                started = time.perf_counter()
            elif outputs > WARMUP_FRAMES and not measured.is_set():
                frames, seconds = outputs - WARMUP_FRAMES, time.perf_counter() - started
                if frames >= MEASURED_FRAMES and seconds >= MIN_DURATION_S:
                    throughput = Throughput(frames, seconds, monitor.stop())
                    measured.set()
        if throughput is None:
            raise RuntimeError("the pipeline stopped before the measured span ended")

    try:
        run_together(push_inputs, pool_stage, copy_stage, decode_detections, runners=(backbone, box_head))
    finally:
        # A failure inside the measured span leaves the monitor running.
        if monitor.running():
            monitor.stop()
        backbone.close()
        box_head.close()
    return throughput


def write_report(
    output: Path,
    image: str,
    regions: int,
    latencies: list[float],
    throughput: Throughput,
    models: dict[str, pyneat.Model],
    mla_layer_stats: dict[str, dict[str, object]],
    mla_stage_stats: dict[str, dict[str, object]],
) -> None:
    benchmark = pyneat.BenchmarkReport()
    benchmark.latency_ms = float(np.mean(latencies) * 1000)
    benchmark.latency_p50_ms = percentile_ms(latencies, 50)
    benchmark.latency_p95_ms = percentile_ms(latencies, 95)
    benchmark.latency_p99_ms = percentile_ms(latencies, 99)
    benchmark.fps = throughput.fps
    benchmark.throughput_duration_ms = throughput.seconds * 1000
    benchmark.throughput_samples = throughput.frames
    report = model_report.from_performance(
        benchmark,
        models,
        workload=(
            f"{image} in a {FRAME_WIDTH}x{FRAME_HEIGHT} RGB frame, Neat preprocessing letterboxed to "
            f"{INPUT_SIZE}x{INPUT_SIZE}, backbone, {regions} proposals and RoIAlign in a C kernel on the CPU, "
            f"box head, and box decoding and NMS on the CPU, pipelined {QUEUE_DEPTH} frames deep"
        ),
        telemetry=throughput.telemetry,
        mla_layer_stats=mla_layer_stats,
        mla_stage_stats=mla_stage_stats,
    )
    report.save(output)
    print(
        f"{benchmark.fps:.1f} FPS over {throughput.frames} frames in {throughput.seconds:.1f} s; "
        f"latency {benchmark.latency_ms:.2f} ms over {len(latencies)} frames"
    )


def benchmark_pipeline(mpks: dict[str, Path], dataset_id: str, dataset: Path, output: Path) -> None:
    mla_layer_stats, mla_stage_stats = {}, {}
    for name, mpk in mpks.items():
        mla_layer_stats[name], mla_stage_stats[name] = mla_stats(mpk)
    models = {
        "backbone": pyneat.Model(str(mpks["backbone"]), backbone_options()),
        "box_head": pyneat.Model(str(mpks["box_head"]), tensor_options()),
    }
    path = sample_image(dataset)
    image = frame_tensor(path)
    backbone_raw, box_head_raw = backbone_format(mpks["backbone"]), box_head_format(mpks["box_head"])
    grids: queue.Queue = queue.Queue()
    regions = box_head_regions(mpks["box_head"])
    for _ in range(GRIDS):
        grids.put(new_grid(box_head_raw, regions))
    pipeline = Pipeline(load_kernel(regions), backbone_raw, box_head_raw, letterbox_window(), grids)
    backbone, box_head = build_runners(models, image, pipeline)
    latencies = measure_latency(backbone, box_head, image, pipeline)
    throughput = measure_throughput(backbone, box_head, image, pipeline)
    write_report(
        output, f"{dataset_id} {path.name}", regions, latencies, throughput, models, mla_layer_stats, mla_stage_stats
    )


def parse_mpks(values: list[str]) -> dict[str, Path]:
    mpks = dict(value.split("=", 1) for value in values if "=" in value)
    if len(mpks) != len(values) or set(mpks) != {"backbone", "box_head"}:
        raise SystemExit("--mpk must name exactly backbone=<path> and box_head=<path>")
    return {name: Path(path) for name, path in mpks.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpk", action="append", required=True, help="component=path, for each component")
    parser.add_argument("--dataset", required=True, help="coco-2017-val=<extracted directory>")
    parser.add_argument("--output", required=True, help="directory for model-report.json")
    parser.add_argument("--resources", help="directory holding the package's licenses folder (unused)")
    args = parser.parse_args()
    dataset_id, dataset_dir = args.dataset.split("=", 1)
    benchmark_pipeline(parse_mpks(args.mpk), dataset_id, Path(dataset_dir), Path(args.output))


if __name__ == "__main__":
    main()
