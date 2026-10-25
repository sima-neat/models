// Accuracy metric labels in display order; tools/model_report/metrics.json gives the same labels.
export const ACCURACY_METRIC_LABELS = {
  top1_accuracy: "Top-1",
  top5_accuracy: "Top-5",
  coco_map: "mAP 50–95",
  coco_ap50: "AP50",
  coco_ap75: "AP75",
  coco_ap_small: "AP small",
  coco_ap_medium: "AP medium",
  coco_ap_large: "AP large",
  coco_ar1: "AR@1",
  coco_ar10: "AR@10",
  coco_ar100: "AR@100",
  coco_ar_small: "AR small",
  coco_ar_medium: "AR medium",
  coco_ar_large: "AR large",
  coco_keypoint_map: "Keypoint mAP 50–95",
  coco_keypoint_ap50: "Keypoint AP50",
  coco_keypoint_ap75: "Keypoint AP75",
  coco_keypoint_ap_medium: "Keypoint AP medium",
  coco_keypoint_ap_large: "Keypoint AP large",
  coco_keypoint_ar: "Keypoint AR",
  coco_keypoint_ar50: "Keypoint AR50",
  coco_keypoint_ar75: "Keypoint AR75",
  coco_keypoint_ar_medium: "Keypoint AR medium",
  coco_keypoint_ar_large: "Keypoint AR large",
  coco_person_box_map: "Person box mAP 50–95",
  coco_person_box_ap50: "Person box AP50",
  coco_mask_map: "Mask mAP 50–95",
  coco_mask_ap50: "Mask AP50",
  coco_mask_ap75: "Mask AP75",
  coco_mask_ap_small: "Mask AP small",
  coco_mask_ap_medium: "Mask AP medium",
  coco_mask_ap_large: "Mask AP large",
  coco_mask_ar1: "Mask AR@1",
  coco_mask_ar10: "Mask AR@10",
  coco_mask_ar100: "Mask AR@100",
  coco_mask_ar_small: "Mask AR small",
  coco_mask_ar_medium: "Mask AR medium",
  coco_mask_ar_large: "Mask AR large",
  coco_box_map: "Box mAP 50–95",
  coco_box_ap50: "Box AP50",
  pose_auc_5: "Pose AUC@5°",
  pose_auc_10: "Pose AUC@10°",
  pose_auc_20: "Pose AUC@20°",
  match_precision: "Match precision",
  matching_score: "Matching score",
};

export function accuracyMetricLabel(name) {
  return ACCURACY_METRIC_LABELS[name] || name;
}

// Known metrics follow the label order and any others follow them in the order given.
export function orderedMetricNames(names) {
  const known = Object.keys(ACCURACY_METRIC_LABELS);
  const rank = (name) => (known.includes(name) ? known.indexOf(name) : known.length);
  return [...names].sort((left, right) => rank(left) - rank(right));
}

// The score each source model's authors published for a metric on a dataset. A result measured on a
// different dataset has no comparable reference score, even when the metric names match.
export function referenceScores(reference, datasetId) {
  const scores = {};
  for (const entry of reference || []) {
    if (entry.dataset?.id !== datasetId) continue;
    for (const [name, value] of Object.entries(entry.metrics || {})) {
      if (Number.isFinite(value)) scores[name] = value;
    }
  }
  return scores;
}
