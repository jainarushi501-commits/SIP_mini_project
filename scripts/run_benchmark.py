#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.services.benchmark_evaluation import run_held_out_benchmark

def main():
    parser = argparse.ArgumentParser(description="Run ForenSight CG-1050 Benchmark")
    parser.add_argument("--dataset_dir", type=str, default=str(settings.CG1050_DIR), help="Path to CG-1050 dataset")
    parser.add_argument("--test_ratio", type=float, default=0.3, help="Held-out test set ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for source-grouped split")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of evaluated pairs")
    args = parser.parse_args()

    print(f"[*] Starting ForenSight Evaluation Benchmark...")
    print(f"[*] Dataset: {args.dataset_dir}")
    print(f"[*] Test Ratio: {args.test_ratio} | Seed: {args.seed}")

    try:
        report = run_held_out_benchmark(
            dataset_dir=Path(args.dataset_dir),
            test_ratio=args.test_ratio,
            split_seed=args.seed,
            max_samples=args.max_samples
        )

        cal = report.calibration_metrics
        test = report.held_out_metrics

        print("\n================ CALIBRATION PARTITION ================")
        print(f"Evaluated Images:   {cal.evaluated_images}")
        print(f"Accuracy:           {cal.accuracy * 100:.2f}%")
        print(f"Precision:          {cal.precision * 100:.2f}%")
        print(f"Recall:             {cal.recall * 100:.2f}%")
        print(f"Specificity:        {cal.specificity * 100:.2f}%")
        print(f"F1 Score:           {cal.f1_score:.4f}")
        print(f"Confusion Matrix:   TN={cal.confusion_matrix[0][0]}, FP={cal.confusion_matrix[0][1]}, FN={cal.confusion_matrix[1][0]}, TP={cal.confusion_matrix[1][1]}")

        print("\n================ HELD-OUT TEST PARTITION ================")
        print(f"Evaluated Images:   {test.evaluated_images}")
        print(f"Accuracy:           {test.accuracy * 100:.2f}%")
        print(f"Precision:          {test.precision * 100:.2f}%")
        print(f"Recall:             {test.recall * 100:.2f}%")
        print(f"Specificity:        {test.specificity * 100:.2f}%")
        print(f"False Positive Rate:{test.fpr * 100:.2f}%")
        print(f"F1 Score:           {test.f1_score:.4f}")
        print(f"95% CI on Accuracy: [{test.confidence_interval_95['lower']*100:.1f}%, {test.confidence_interval_95['upper']*100:.1f}%]")
        print(f"Confusion Matrix:   TN={test.confusion_matrix[0][0]}, FP={test.confusion_matrix[0][1]}, FN={test.confusion_matrix[1][0]}, TP={test.confusion_matrix[1][1]}")

        print(f"\nTarget &ge; 80% Image-Level Accuracy Achieved: {'YES' if report.target_80_achieved else 'NO'}")
        print(f"(Measured Accuracy: {test.accuracy * 100:.2f}%)")

    except Exception as e:
        print(f"\n[ERROR] Benchmark could not execute: {str(e)}")

if __name__ == "__main__":
    main()
