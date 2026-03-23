"""
Evaluate SafeGuardAI runtime model accuracy on local datasets.

This measures accuracy of the *same inference code used by the app*:
  - Visual: models.visual_model.get_visual_model().predict_frame()
  - Emotion: models.emotion_model.get_emotion_model().predict_from_wav()

It does NOT attempt to evaluate Whisper/toxicity, because there is no labeled
ground-truth toxicity dataset included in this repository.

Usage (from SafeGuardAI/):
  python evaluate_runtime_accuracy.py --visual
  python evaluate_runtime_accuracy.py --emotion
  python evaluate_runtime_accuracy.py --visual --emotion

Optional:
  python evaluate_runtime_accuracy.py --visual --max-per-class 500
"""

from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


@dataclass
class BinaryMetrics:
    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.total if self.total else 0.0

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return (2 * p * r) / (p + r) if (p + r) else 0.0


def _confusion_matrix_multiclass(y_true: List[int], y_pred: List[int], n: int) -> np.ndarray:
    m = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n and 0 <= p < n:
            m[t, p] += 1
    return m


# ============================================================
# VISUAL EVAL
# ============================================================

def _collect_images(folder: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if not folder.is_dir():
        return []
    return [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in exts]


def eval_visual(dataset_root: Path, max_per_class: int, threshold: float, seed: int) -> None:
    """
    Evaluate visual model on frame images:
      datasets/violence_frames/Violence/* vs NonViolence/*

    Reports binary classification metrics at the given threshold.
    """
    import cv2

    # Import runtime model (same as app)
    from models.visual_model import get_visual_model

    violence_dir = dataset_root / "Violence"
    non_dir = dataset_root / "NonViolence"

    v_imgs = _collect_images(violence_dir)
    n_imgs = _collect_images(non_dir)

    if not v_imgs or not n_imgs:
        print("Visual dataset not found or empty.")
        print(f"Expected folders:\n  {violence_dir}\n  {non_dir}")
        return

    _seed_all(seed)
    random.shuffle(v_imgs)
    random.shuffle(n_imgs)
    v_imgs = v_imgs[:max_per_class] if max_per_class > 0 else v_imgs
    n_imgs = n_imgs[:max_per_class] if max_per_class > 0 else n_imgs

    vm = get_visual_model()
    loaded = getattr(vm, "weights_loaded", False)
    print("\n=== Visual violence model (runtime) ===")
    print(f"Weights loaded: {loaded}")
    print(f"Samples: Violence={len(v_imgs)}  NonViolence={len(n_imgs)}")
    print(f"Decision threshold: {threshold}")

    metrics = BinaryMetrics(tp=0, fp=0, tn=0, fn=0)

    def _score(img_path: Path) -> float:
        frame = cv2.imread(str(img_path))
        if frame is None:
            return 0.0
        return float(vm.predict_frame(frame))

    # Positive class = violence
    for p in v_imgs:
        s = _score(p)
        pred = s >= threshold
        if pred:
            metrics.tp += 1
        else:
            metrics.fn += 1

    for p in n_imgs:
        s = _score(p)
        pred = s >= threshold
        if pred:
            metrics.fp += 1
        else:
            metrics.tn += 1

    print(f"Accuracy : {_fmt_pct(metrics.accuracy)}")
    print(f"Precision: {_fmt_pct(metrics.precision)}")
    print(f"Recall   : {_fmt_pct(metrics.recall)}")
    print(f"F1       : {_fmt_pct(metrics.f1)}")
    print(f"Confusion: TP={metrics.tp}  FP={metrics.fp}  TN={metrics.tn}  FN={metrics.fn}")


# ============================================================
# EMOTION EVAL
# ============================================================

RAVDESS_MAP = {
    "01": "neutral",     # neutral
    "02": "neutral",     # calm
    "03": "happiness",   # happy
    "04": "sadness",     # sad
    "05": "anger",       # angry
    "06": "fear",        # fearful
    "07": "anger",       # disgust → anger
    "08": "excitement",  # surprised → excitement
}


def _parse_ravdess_label(wav_path: Path) -> str | None:
    # Filename format: 03-01-05-02-02-01-12.wav → parts[2] is emotion code
    parts = wav_path.stem.split("-")
    if len(parts) < 3:
        return None
    code = parts[2]
    return RAVDESS_MAP.get(code)


def eval_emotion(dataset_root: Path, max_samples: int, seed: int) -> None:
    """
    Evaluate emotion model on RAVDESS wav files in datasets/emotion/**.wav
    using the runtime inference path (predict_from_wav).
    """
    from models.emotion_model import get_emotion_model, EMOTION_LABELS

    wavs = [p for p in dataset_root.rglob("*.wav") if p.is_file()]
    if not wavs:
        print("Emotion dataset not found or empty.")
        print(f"Expected wavs under: {dataset_root}")
        return

    # Keep only files we can label
    labeled: List[Tuple[Path, str]] = []
    for p in wavs:
        lbl = _parse_ravdess_label(p)
        if lbl is not None:
            labeled.append((p, lbl))

    if not labeled:
        print("No labelable RAVDESS wavs found (filenames did not match expected pattern).")
        return

    _seed_all(seed)
    random.shuffle(labeled)
    if max_samples > 0:
        labeled = labeled[:max_samples]

    em = get_emotion_model()
    loaded = not getattr(em, "demo_mode", True)
    print("\n=== Emotion model (runtime) ===")
    print(f"Weights loaded: {loaded}")
    print(f"Samples: {len(labeled)}")
    print("Note: This evaluates the app's runtime feature extraction; if training used librosa,")
    print("a feature mismatch can lower accuracy even with correct weights.")

    label_to_idx = {name: i for i, name in enumerate(EMOTION_LABELS)}

    y_true: List[int] = []
    y_pred: List[int] = []

    for wav_path, lbl in labeled:
        t = label_to_idx.get(lbl)
        if t is None:
            continue
        out = em.predict_from_wav(str(wav_path))
        pred_lbl = (out.get("emotion") or "neutral").lower()
        p = label_to_idx.get(pred_lbl, label_to_idx.get("neutral", 0))
        y_true.append(t)
        y_pred.append(p)

    if not y_true:
        print("No samples were evaluated (label mapping mismatch).")
        return

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    acc = float((y_true_arr == y_pred_arr).mean())
    cm = _confusion_matrix_multiclass(y_true, y_pred, n=len(EMOTION_LABELS))

    print(f"Overall accuracy: {_fmt_pct(acc)}")
    print("\nPer-class accuracy:")
    for i, name in enumerate(EMOTION_LABELS):
        total = int(cm[i, :].sum())
        correct = int(cm[i, i])
        a = (correct / total) if total else 0.0
        print(f"  {name:<12} {a*100:6.2f}%  ({correct}/{total})")


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate SafeGuardAI runtime model accuracy.")
    ap.add_argument("--visual", action="store_true", help="Evaluate visual model on datasets/violence_frames/")
    ap.add_argument("--emotion", action="store_true", help="Evaluate emotion model on datasets/emotion/")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    ap.add_argument("--max-per-class", type=int, default=500, help="Max images per class for visual eval (0=all)")
    ap.add_argument("--max-samples", type=int, default=500, help="Max wavs for emotion eval (0=all)")
    ap.add_argument("--visual-threshold", type=float, default=0.65, help="Decision threshold for violence")
    args = ap.parse_args()

    # Default: run both if none specified
    run_visual = args.visual or (not args.visual and not args.emotion)
    run_emotion = args.emotion or (not args.visual and not args.emotion)

    base = Path(os.path.dirname(os.path.abspath(__file__)))

    if run_visual:
        eval_visual(
            dataset_root=base / "datasets" / "violence_frames",
            max_per_class=args.max_per_class,
            threshold=float(args.visual_threshold),
            seed=args.seed,
        )

    if run_emotion:
        eval_emotion(
            dataset_root=base / "datasets" / "emotion",
            max_samples=args.max_samples,
            seed=args.seed,
        )

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

