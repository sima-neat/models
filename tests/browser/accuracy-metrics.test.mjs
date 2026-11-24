import assert from 'node:assert/strict';
import {test} from 'node:test';

const {accuracyDisplay, accuracyMetricLabel, formatAccuracy, orderedMetricNames, referenceNotes, referenceScores} = await import(new URL('../../artifact-browser/accuracy-metrics.js', import.meta.url));

test('detection metrics follow the label order whatever order the registry returns', () => {
  // The registry returned RF-DETR Small's metrics in this order.
  const metrics = ['coco_ar1', 'coco_map', 'coco_ap50', 'coco_ap75', 'coco_ar10', 'coco_ar100', 'coco_ap_large',
    'coco_ap_small', 'coco_ar_large', 'coco_ar_small', 'coco_ap_medium', 'coco_ar_medium'];
  assert.deepEqual(orderedMetricNames(['custom_b', ...metrics, 'custom_a']), [
    'coco_map', 'coco_ap50', 'coco_ap75', 'coco_ap_small', 'coco_ap_medium', 'coco_ap_large',
    'coco_ar1', 'coco_ar10', 'coco_ar100', 'coco_ar_small', 'coco_ar_medium', 'coco_ar_large',
    'custom_b', 'custom_a',
  ]);
});

test('pose metrics lead with keypoint mAP', () => {
  const metrics = ['coco_person_box_ap50', 'coco_keypoint_ar', 'coco_keypoint_ap_large', 'coco_person_box_map',
    'coco_keypoint_ar75', 'coco_keypoint_map', 'coco_keypoint_ar_medium', 'coco_keypoint_ap75',
    'coco_keypoint_ar_large', 'coco_keypoint_ap50', 'coco_keypoint_ar50', 'coco_keypoint_ap_medium'];
  assert.deepEqual(orderedMetricNames(metrics), [
    'coco_keypoint_map', 'coco_keypoint_ap50', 'coco_keypoint_ap75', 'coco_keypoint_ap_medium',
    'coco_keypoint_ap_large', 'coco_keypoint_ar', 'coco_keypoint_ar50', 'coco_keypoint_ar75',
    'coco_keypoint_ar_medium', 'coco_keypoint_ar_large', 'coco_person_box_map', 'coco_person_box_ap50',
  ]);
});

test('segmentation metrics lead with mask mAP', () => {
  const metrics = ['coco_box_ap50', 'coco_mask_ar100', 'coco_mask_ap_large', 'coco_box_map', 'coco_mask_ar1',
    'coco_mask_map', 'coco_mask_ar_medium', 'coco_mask_ap75', 'coco_mask_ar_large', 'coco_mask_ap50',
    'coco_mask_ar10', 'coco_mask_ap_medium', 'coco_mask_ap_small', 'coco_mask_ar_small'];
  assert.deepEqual(orderedMetricNames(metrics), [
    'coco_mask_map', 'coco_mask_ap50', 'coco_mask_ap75', 'coco_mask_ap_small', 'coco_mask_ap_medium',
    'coco_mask_ap_large', 'coco_mask_ar1', 'coco_mask_ar10', 'coco_mask_ar100', 'coco_mask_ar_small',
    'coco_mask_ar_medium', 'coco_mask_ar_large', 'coco_box_map', 'coco_box_ap50',
  ]);
});

const reference = [
  {dataset: {id: 'coco-2017', split: 'val'}, metrics: {coco_map: 0.53, coco_ap50: 0.721}},
  {dataset: {id: 'coco-2014', split: 'minival'}, metrics: {coco_ap75: 0.6}},
];

test('reference scores sit beside measurements of the same dataset only', () => {
  assert.deepEqual(referenceScores(reference, 'coco-2017'), {coco_map: 0.53, coco_ap50: 0.721});
  assert.deepEqual(referenceScores(reference, 'imagenet-2012'), {});
  assert.deepEqual(referenceScores(undefined, 'coco-2017'), {});
});

test('published and FP32 baseline scores stay apart', () => {
  const baseline = {origin: 'fp32_baseline', dataset: {id: 'coco-2017', split: 'val'}, metrics: {coco_map: 0.51}};
  assert.deepEqual(referenceScores([...reference, baseline], 'coco-2017'), {coco_map: 0.53, coco_ap50: 0.721});
  assert.deepEqual(referenceScores([...reference, baseline], 'coco-2017', 'fp32_baseline'), {coco_map: 0.51});
  assert.deepEqual(referenceScores(reference, 'coco-2017', 'fp32_baseline'), {});
});

test('an accuracy tile notes each origin that scores its metric, published first', () => {
  const scores = {published: {pose_auc_5: 0.3902}, fp32_baseline: {pose_auc_5: 0.2865, matching_score: 0.2783}};
  assert.deepEqual(referenceNotes('pose_auc_5', scores), ['Published 39.02%', 'FP32 baseline 28.65%']);
  assert.deepEqual(referenceNotes('matching_score', scores), ['FP32 baseline 27.83%']);
  assert.deepEqual(referenceNotes('match_precision', scores), []);
});

test('ratios show as percentages and other units as reported, each with its direction', () => {
  assert.equal(formatAccuracy('coco_map', 0.401), '40.10%');
  assert.equal(formatAccuracy('epe', 0.7763), '0.776 px');
  assert.equal(formatAccuracy('d1_all', 2.8183), '2.82%');
  assert.equal(formatAccuracy('epe', undefined), '—');
  assert.equal(formatAccuracy('epe', undefined, {blank: true}), '');
  assert.deepEqual(accuracyDisplay('coco_map', 0.401), {number: 40.1, unit: '%', direction: 'higher'});
  assert.deepEqual(accuracyDisplay('epe', 0.73), {number: 0.73, unit: 'px', direction: 'lower'});
  assert.deepEqual(accuracyDisplay('d1_all', 2.82), {number: 2.82, unit: '%', direction: 'lower'});
});

test('disparity metrics lead with EPE', () => {
  assert.deepEqual(orderedMetricNames(['d1_all', 'epe']), ['epe', 'd1_all']);
});

test('depth error is dimensionless and unbounded, while delta1 is a percentage', () => {
  assert.equal(formatAccuracy('abs_rel', 0.0924), '0.092');
  assert.equal(formatAccuracy('abs_rel', 1.25), '1.250');
  assert.equal(formatAccuracy('delta1', 0.893), '89.30%');
  assert.deepEqual(accuracyDisplay('abs_rel', 0.0924), {number: 0.0924, unit: '', direction: 'lower'});
  assert.deepEqual(orderedMetricNames(['delta1', 'abs_rel']), ['abs_rel', 'delta1']);
});

test('speech error rates are lower-is-better percentages that may exceed 100%', () => {
  assert.equal(accuracyMetricLabel('wer'), 'WER');
  assert.equal(formatAccuracy('wer', 0.0842), '8.42%');
  assert.equal(formatAccuracy('wer', 1.25), '125.00%');
  assert.deepEqual(accuracyDisplay('cer', 0.031), {number: 3.1, unit: '%', direction: 'lower'});
  assert.equal(accuracyMetricLabel('wer_as_written'), 'WER as written');
  assert.equal(formatAccuracy('wer_as_written', 0.0651), '6.51%');
  assert.deepEqual(orderedMetricNames(['cer', 'wer_as_written', 'wer']), ['wer', 'wer_as_written', 'cer']);
});
