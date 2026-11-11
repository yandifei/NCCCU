"""对齐两次验证的检查概率，按患者配对重采样比较 Macro-AUC。"""

import json
from pathlib import Path

import numpy as np

from ncccu.data import CLASS_NAMES
from ncccu.evaluation import binary_auc, metrics
from ncccu.runtime import ROOT, output_dir
from ncccu.thresholds import validate_arrays


def read_validation(path: Path) -> dict[str, np.ndarray]:
    """只接受包含真实标签、唯一检查标识的验证概率文件。"""
    with np.load(path, allow_pickle=False) as archive:
        required = {"probabilities", "targets", "subject_ids", "study_ids"}
        if not required.issubset(archive.files):
            raise ValueError("需要包含标签的验证 NPZ，测试集预测不能用于模型比较")
        arrays = {key: archive[key] for key in required}
    labels, scores = validate_arrays(arrays["targets"], arrays["probabilities"])
    if scores.shape[1] != len(CLASS_NAMES):
        raise ValueError("验证概率须包含官方十个类别")
    subjects, studies = arrays["subject_ids"], arrays["study_ids"]
    if subjects.shape != (len(scores),) or studies.shape != subjects.shape:
        raise ValueError("检查标识与概率行数不一致")
    keys = list(zip(subjects.tolist(), studies.tolist()))
    if len(set(keys)) != len(keys):
        raise ValueError("验证概率包含重复检查")
    arrays.update(targets=labels, probabilities=scores)
    return arrays


def compare_predictions(
    *, baseline: str | Path, candidate: str | Path,
    name: str, repeats: int = 2000, seed: int = 0,
) -> Path:
    """以相同患者抽样同时计算两模型分差，输出百分位置信区间。"""
    if type(repeats) is not int or repeats < 1:
        raise ValueError("repeats 须为正整数")
    baseline_path, candidate_path = (ROOT / baseline).resolve(), (ROOT / candidate).resolve()
    first, second = read_validation(baseline_path), read_validation(candidate_path)
    first_keys = list(zip(first["subject_ids"].tolist(), first["study_ids"].tolist()))
    second_keys = list(zip(second["subject_ids"].tolist(), second["study_ids"].tolist()))
    if set(first_keys) != set(second_keys):
        raise ValueError("两份预测须来自完全相同的验证检查")
    positions = {key: index for index, key in enumerate(second_keys)}
    order = [positions[key] for key in first_keys]
    labels = first["targets"]
    if not np.array_equal(labels, second["targets"][order]):
        raise ValueError("两份验证记录的标签不一致")
    before, after = first["probabilities"], second["probabilities"][order]
    baseline_metrics, candidate_metrics = metrics(labels, before), metrics(labels, after)
    if baseline_metrics["auc_classes"] != len(CLASS_NAMES):
        raise ValueError("正式对照须在十个类别中均包含正负样本")
    subjects = np.unique(first["subject_ids"])
    groups = [np.flatnonzero(first["subject_ids"] == subject) for subject in subjects]
    random = np.random.default_rng(seed)
    differences = []
    for _ in range(repeats):
        selected = random.integers(len(groups), size=len(groups))
        indices = np.concatenate([groups[index] for index in selected])
        sampled_labels = labels[indices]
        # 少数类别可能在重采样后缺正例，整次丢弃以保持十类指标口径。
        positives = sampled_labels.sum(axis=0)
        if np.any((positives == 0) | (positives == len(indices))):
            continue
        delta = metrics(sampled_labels, after[indices])["macro_auc"] - metrics(
            sampled_labels, before[indices]
        )["macro_auc"]
        differences.append(delta)
    if not differences:
        raise ValueError("重采样均缺少类别正负样本，无法计算十类 AUC 区间")
    per_class = []
    for index, class_name in enumerate(CLASS_NAMES):
        left = binary_auc(labels[:, index], before[:, index])
        right = binary_auc(labels[:, index], after[:, index])
        per_class.append(dict(class_id=index + 1, name=class_name,
                              positives=int(labels[:, index].sum()),
                              baseline_auc=left, candidate_auc=right, delta=right - left))
    report = dict(
        baseline=str(baseline_path), candidate=str(candidate_path),
        studies=len(labels), patients=len(subjects), seed=seed,
        baseline_metrics=baseline_metrics, candidate_metrics=candidate_metrics,
        macro_auc_delta=candidate_metrics["macro_auc"] - baseline_metrics["macro_auc"],
        bootstrap_requested=repeats, bootstrap_valid=len(differences),
        bootstrap_skipped=repeats - len(differences),
        delta_interval_95=np.quantile(differences, [0.025, 0.975]).tolist(),
        per_class=per_class,
        interpretation="区间反映当前验证患者的抽样不确定性，仍受验证集选模和反复调参影响。",
    )
    run = output_dir("analysis", name)
    (run / "comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(f"Macro-AUC 差值 {report['macro_auc_delta']:+.6f}，95% 区间 {report['delta_interval_95']}")
    print(f"对照报告 {run / 'comparison.json'}")
    return run


if __name__ == "__main__":
    compare_predictions(
        baseline="./outputs/predict/v4_valid_for_v5a/study_probabilities.npz",  # 第四版逐图基线的官方验证概率。
        candidate="./outputs/train/v5_study_weighted_512/best_validation.npz",  # 第五版最佳轮保存的验证概率。
        name="v5a_vs_v4",  # 对照报告位于 outputs/analysis，每次使用独立名称。
        repeats=2000,  # 以患者为单位有放回抽样，同一患者的全部检查一起参与。
        seed=0,  # 固定重采样种子。
    )
