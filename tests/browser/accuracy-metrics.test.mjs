import assert from 'node:assert/strict';
import {test} from 'node:test';

const {orderedMetricNames, referenceScores} = await import(new URL('../../artifact-browser/accuracy-metrics.js', import.meta.url));

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
