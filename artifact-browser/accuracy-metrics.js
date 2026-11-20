// Accuracy metric labels in display order; tools/model_report/metrics.json gives the same labels.
export const ACCURACY_METRIC_LABELS = {
  top1_accuracy: "Top-1",
  top5_accuracy: "Top-5",
  knn_top1_accuracy: "k-NN top-1",
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
  dota_obb_map50: "DOTA mAP50",
  dota_obb_map: "DOTA mAP 50–95",
  keypoint_repeatability: "Repeatability",
  homography_accuracy: "Homography accuracy",
  jf_mean: "J&F",
  j_mean: "J (region overlap)",
  f_mean: "F (boundary)",
  pose_auc_5: "Pose AUC@5°",
  pose_auc_10: "Pose AUC@10°",
  pose_auc_20: "Pose AUC@20°",
  match_precision: "Match precision",
  matching_score: "Matching score",
  recall_at_1: "Recall@1",
  recall_at_5: "Recall@5",
  recall_at_10: "Recall@10",
  epe: "EPE",
  d1_all: "D1-all",
  change_f1: "Change F1",
  change_precision: "Change precision",
  change_recall: "Change recall",
  abs_rel: "AbsRel",
  delta1: "δ1",
  miou: "mIoU",
};

// Accuracy metrics with non-default units, directions or display settings; matches metrics.json.
export const ACCURACY_METRIC_SCALES = {
  epe: { unit: "px", direction: "lower" },
  d1_all: { unit: "%", direction: "lower" },
  abs_rel: { unit: "ratio", direction: "lower", display: {unit: "", scale: 1} },
};

export function accuracyMetricLabel(name) {
  return ACCURACY_METRIC_LABELS[name] || name;
}

// Ratios default to percentages; an explicit display setting overrides that without changing the metric's unit.
export function accuracyDisplay(name, value) {
  const metric = ACCURACY_METRIC_SCALES[name] || {unit: "ratio", direction: "higher"};
  const display = metric.display || (metric.unit === "ratio" ? {unit: "%", scale: 100} : {unit: metric.unit, scale: 1});
  return {number: value * display.scale, unit: display.unit, direction: metric.direction};
}

// A score as text: "40.10%" for a ratio or a percentage, "0.776 px" for pixels, and "—" (or blank) when missing.
export function formatAccuracy(name, value, { blank = false } = {}) {
  if (!Number.isFinite(value)) return blank ? "" : "—";
  const { number, unit } = accuracyDisplay(name, value);
  if (unit === "%") return `${number.toFixed(2)}%`;
  return unit ? `${number.toFixed(3)} ${unit}` : number.toFixed(3);
}

// Known metrics follow the label order and any others follow them in the order given.
export function orderedMetricNames(names) {
  const known = Object.keys(ACCURACY_METRIC_LABELS);
  const rank = (name) => (known.includes(name) ? known.indexOf(name) : known.length);
  return [...names].sort((left, right) => rank(left) - rank(right));
}

// Who measured a reference score: the source model's authors, or the model's FP32 baseline script on the
// package's own inputs. Each is labelled apart, so a baseline never reads as the authors' claim.
export const REFERENCE_ORIGINS = {
  published: { label: "Published", title: "Published scores", id: "published-scores" },
  fp32_baseline: { label: "FP32 baseline", title: "FP32 baseline", id: "fp32-baseline-scores" },
};

export function referenceOrigin(entry) {
  return entry.origin || "published";
}

// The reference score of one origin for each metric on a dataset. A result measured on a different dataset has
// no comparable reference score, even when the metric names match.
export function referenceScores(reference, datasetId, origin = "published") {
  const scores = {};
  for (const entry of reference || []) {
    if (entry.dataset?.id !== datasetId || referenceOrigin(entry) !== origin) continue;
    for (const [name, value] of Object.entries(entry.metrics || {})) {
      if (Number.isFinite(value)) scores[name] = value;
    }
  }
  return scores;
}

// The reference notes under one accuracy tile, in REFERENCE_ORIGINS order, such as "FP32 baseline 28.65%". Both
// views show them, so a customer can compare a build with its baseline.
export function referenceNotes(name, scoresByOrigin) {
  return Object.entries(REFERENCE_ORIGINS).flatMap(([origin, { label }]) => {
    const score = formatAccuracy(name, scoresByOrigin[origin]?.[name], { blank: true });
    return score ? [`${label} ${score}`] : [];
  });
}

// A dataset as readers know it: the name datasets/<id>.yaml gives it, such as "COCO 2017". A dataset without
// one goes by its Dataset Registry archive, such as "coco-2017 val".
export function datasetName(dataset) {
  return dataset.name || registryArchive(dataset);
}

// A Dataset Registry archive: its ID and variant, such as "coco-2017 val".
export function registryArchive(dataset) {
  return [dataset.id, dataset.variant].filter(Boolean).join(" ");
}
