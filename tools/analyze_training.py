"""读取完成的训练，用官方验证集诊断并搜索 Macro-F1 最优阈值。"""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from ncccu.multilabel import (
    ROOT, CLASS_NAMES, ChestDataset, aggregate, device_for, evaluate, metrics, output_dir,
)
from ncccu.model import load_model
from ncccu.runtime import configure_randomness, seed_worker
from ncccu.thresholds import f1_scores, optimize_thresholds, patient_folds


def file_hash(path: Path) -> str:
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def size_statistics(dataset: ChestDataset) -> dict:
    sizes = []
    for row in tqdm(dataset.rows, desc=f'{dataset.split} image sizes'):
        with Image.open(dataset.image_path(row)) as source:
            sizes.append(source.size)
    sizes = np.asarray(sizes)
    short, long = sizes.min(1), sizes.max(1)
    return dict(images=len(sizes), width_percentiles=np.percentile(sizes[:, 0], [0, 50, 100]).tolist(),
                height_percentiles=np.percentile(sizes[:, 1], [0, 50, 100]).tolist(),
                median_short_side=float(np.median(short)), median_long_side=float(np.median(long)),
                median_content_short_side_224=float(np.median(224*short/long)),
                median_content_short_side_512=float(np.median(512*short/long)))


def plot_diagnostics(history: list[dict], result: dict, per_class: list[dict], run: Path) -> None:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    figure = Figure(figsize=(13, 9))
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2)
    epochs = [row['epoch'] for row in history]
    for name in ('train_loss', 'val_loss'):
        axes[0, 0].plot(epochs, [row[name] for row in history], label=name)
    axes[0, 0].set(title='Training / validation loss', xlabel='Epoch', yscale='log')
    for name in ('macro_auc', 'macro_f1'):
        axes[0, 1].plot(epochs, [row[name] for row in history], label=name)
    axes[0, 1].set(title='Validation metrics (F1 threshold = 0.5)', xlabel='Epoch')
    axes[1, 0].plot(result['curve_thresholds'], result['curve_macro_f1'], label='Validation Macro-F1')
    axes[1, 0].axvline(result['global_threshold'], color='red', linestyle='--', label='Best common threshold')
    axes[1, 0].set(title='Fixed best model: threshold search', xlabel='Threshold', xlim=(0, 1))
    x = np.arange(10)
    axes[1, 1].bar(x-.2, [r['f1_05'] for r in per_class], .4, label='F1 @ 0.5')
    axes[1, 1].bar(x+.2, [r['f1_tuned'] for r in per_class], .4, label='F1 @ per-class tuned threshold')
    axes[1, 1].set(title='Same validation data used to tune thresholds', xlabel='Official class ID', xticks=x, xticklabels=range(1, 11))
    for axis in axes.flat:
        axis.legend()
        axis.grid(alpha=.2)
    figure.tight_layout()
    figure.savefig(run / 'diagnostics.png', dpi=160)
    figure = Figure(figsize=(15, 7))
    FigureCanvasAgg(figure)
    for axis, row in zip(figure.subplots(2, 5).flat, per_class):
        matrix = np.array([[row['tn'], row['fp']], [row['fn'], row['tp']]])
        axis.imshow(matrix, cmap='Blues')
        for (y, x), value in np.ndenumerate(matrix):
            axis.text(x, y, str(value), ha='center', va='center',
                      color='white' if value > matrix.max()/2 else 'black')
        axis.set(title=f"{row['class_id']}: {row['class_name']}\nt={row['threshold']:.4f}",
                 xlabel='Predicted', ylabel='Actual', xticks=[0, 1], yticks=[0, 1])
    figure.tight_layout()
    figure.savefig(run / 'confusion_matrices.png', dpi=160)


def analyze_training(*, training_run: str | Path, data: str | Path, name: str,
                     device: str | int = 0, batch: int = 16, workers: int = 3) -> Path:
    training_run, data = ROOT / training_run, ROOT / data
    config = json.loads((training_run / 'args.json').read_text(encoding='utf-8'))
    with (training_run / 'results.csv').open(encoding='utf-8', newline='') as file:
        history = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(file)]
    if not history:
        raise ValueError('训练记录为空')
    configure_randomness(config['seed'], config['deterministic'])
    device = device_for(device)
    weights = training_run / 'weights/best.pt'
    model, checkpoint = load_model(weights, device)
    if checkpoint['smoke_test']:
        raise ValueError('请使用正式训练的最佳权重')
    if config['channels_last']:
        model.to(memory_format=torch.channels_last)
    valid = ChestDataset(data, 'valid', checkpoint['imgsz'])
    loader = DataLoader(valid, batch_size=batch, num_workers=workers,
                        pin_memory=device.type == 'cuda', worker_init_fn=seed_worker)
    print(f"分析 best.pt，第 {checkpoint['epoch']} 轮，{len(valid)} 张验证影像", flush=True)
    probabilities, val_loss = evaluate(model, loader, device, channels_last=config['channels_last'], verbose=True)
    keys, scores, labels = aggregate(valid, probabilities)
    result = optimize_thresholds(labels, scores)
    baseline = metrics(labels, scores)
    # 逐类调阈值的验证内收益可能偏乐观，额外按患者分组交叉拟合阈值。
    folds = patient_folds([subject for subject, _ in keys], count=5, seed=config['seed'])
    out_of_fold_global = np.empty((len(keys), 1))
    out_of_fold_class = np.empty_like(scores, dtype=np.float64)
    fold_results = []
    for fold in range(5):
        holdout = folds == fold
        fitted = optimize_thresholds(labels[~holdout], scores[~holdout])
        out_of_fold_global[holdout] = fitted['global_threshold']
        out_of_fold_class[holdout] = fitted['class_thresholds']
        fold_results.append(dict(fold=fold, holdout_studies=int(holdout.sum()),
                                 global_threshold=fitted['global_threshold'], class_thresholds=fitted['class_thresholds'].tolist()))
    tuned_f1 = f1_scores(labels, scores, result['class_thresholds'])
    per_class = []
    for index, class_name in enumerate(CLASS_NAMES):
        target, score = labels[:, index].astype(bool), scores[:, index]
        predicted = score >= result['class_thresholds'][index]
        positive, negative = score[target, None], score[None, ~target]
        auc = float(((positive > negative) + .5*(positive == negative)).mean())
        tp, fp = int((target & predicted).sum()), int((~target & predicted).sum())
        fn, tn = int((target & ~predicted).sum()), int((~target & ~predicted).sum())
        per_class.append(dict(class_id=index+1, class_name=class_name, positives=int(target.sum()),
                              auc=auc, threshold=float(result['class_thresholds'][index]),
                              f1_05=float(f1_scores(labels, scores, .5)[index]), f1_tuned=float(tuned_f1[index]),
                              precision=tp/max(tp+fp, 1), recall=tp/max(tp+fn, 1), tn=tn, fp=fp, fn=fn, tp=tp))
    train = ChestDataset(data, 'train', checkpoint['imgsz'])
    train_subjects = {row['Subject_id'] for row in train.rows}
    valid_subjects = {subject for subject, _ in keys}
    summary = dict(
        weights=str(weights), weights_sha256=file_hash(weights), best_epoch=checkpoint['epoch'],
        stopped_epoch=int(history[-1]['epoch']), configured_epochs=config['epochs'], patience=config['patience'],
        stale_epochs=int(history[-1]['epoch'])-checkpoint['epoch'],
        best_record=checkpoint['metrics'], final_record=history[-1],
        evaluation=dict(split='valid', images=len(valid), studies=len(keys), patients=len(valid_subjects),
                        imgsz=checkpoint['imgsz'], batch=batch, workers=workers, quantize=32,
                        channels_last=config['channels_last'], augment=False, aggregation='mean',
                        val_loss=val_loss, **baseline),
        global_threshold=result['global_threshold'], global_macro_f1=result['global_macro_f1'],
        class_thresholds=result['class_thresholds'].tolist(), class_macro_f1=result['class_macro_f1'],
        grouped_oof_global_macro_f1=float(f1_scores(labels, scores, out_of_fold_global).mean()),
        grouped_oof_class_macro_f1=float(f1_scores(labels, scores, out_of_fold_class).mean()),
        fold_thresholds=fold_results, per_class=per_class,
        train_valid_shared_patients=len(train_subjects & valid_subjects),
        image_sizes=dict(train=size_statistics(train), valid=size_statistics(valid)),
        comparison='probability >= threshold; among equal F1 candidates choose highest threshold',
        limitations='Thresholds are fitted on validation; model was also selected on validation. Grouped OOF only cross-fits thresholds, not model selection. Not an independent test estimate.',
    )
    run = output_dir('analysis', name)
    (run / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    with (run / 'per_class.csv').open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(per_class[0]))
        writer.writeheader()
        writer.writerows(per_class)
    with (run / 'validation_predictions.csv').open('w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Subject_id', 'Study_id', 'Predict_class', 'Target', 'Probability', 'Gate'])
        for (subject, study), target, values in zip(keys, labels, scores):
            for index in range(10):
                writer.writerow([subject, study, index+1, int(target[index]), float(values[index]), float(result['class_thresholds'][index])])
    np.savez_compressed(run / 'validation_scores.npz', labels=labels, scores=scores, folds=folds)
    plot_diagnostics(history, result, per_class, run)
    print(json.dumps({key: summary[key] for key in ('best_epoch', 'stopped_epoch', 'global_threshold', 'global_macro_f1',
                                                   'class_thresholds', 'class_macro_f1', 'grouped_oof_global_macro_f1',
                                                   'grouped_oof_class_macro_f1')}, ensure_ascii=False, indent=2), flush=True)
    print(f'诊断产出：{run}', flush=True)
    return run


# 直接修改配置；仅使用 train/valid，测试集不参与阈值拟合。
if __name__ == '__main__':
    analyze_training(
        training_run='./outputs/train/v4_fp32_lr3e5_512',
        data='./datasets',
        name='v4_fp32_lr3e5_512_thresholds',  # 再次分析时改为未使用的目录名
        device=0,
        batch=1,  # 与 predict.py 的 FP32 推理批量一致
        workers=3,
    )
