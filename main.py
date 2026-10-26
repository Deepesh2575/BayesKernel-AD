"""Command-line entry point for BayesKernel-AD experiments."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas as pd

from data.generate_synthetic_stream import generate_synthetic_stream
from evaluation.experiments import evaluate_drift_adaptation, run_experiments
from core.pipeline import BayesKernelAD
from config import PipelineConfig


def run_demo(show_plot: bool = True) -> None:
    """Run a chronological synthetic-stream demo with a drift marker."""
    stream = generate_synthetic_stream()
    warmup = 200
    detector = BayesKernelAD(
        PipelineConfig(min_train_size=warmup, drift_window=30, drift_threshold=1.25)
    ).fit(stream.features[:warmup], stream.labels[:warmup])
    probabilities = []
    decisions = []
    detected = []
    for index in range(warmup, len(stream.labels)):
        result = detector.predict(stream.features[index : index + 1])
        probabilities.append(result.probabilities[0, 1])
        decisions.append(result.decisions[0])
        detected.append(result.drift_detected)

    frame = pd.DataFrame(
        {
            "sample": range(warmup, len(stream.labels)),
            "anomaly_probability": probabilities,
            "true_anomaly": stream.labels[warmup:],
            "detected_drift": detected,
        }
    )
    print(frame.head(10).to_string(index=False))
    print(f"\nDrift adaptation delay: {evaluate_drift_adaptation():.1f} samples")
    if show_plot:
        plt.figure(figsize=(11, 4))
        plt.plot(frame["sample"], frame["anomaly_probability"], label="P(anomaly)")
        plt.scatter(
            frame.loc[frame["true_anomaly"] == 1, "sample"],
            frame.loc[frame["true_anomaly"] == 1, "anomaly_probability"],
            color="crimson",
            s=12,
            label="true anomaly",
        )
        plt.axvline(int(len(stream.labels) * 0.55), color="black", linestyle="--", label="drift")
        plt.xlabel("Stream sample")
        plt.ylabel("Anomaly posterior")
        plt.title("BayesKernel-AD on synthetic drifting stream")
        plt.legend()
        plt.tight_layout()
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("demo", "evaluate"), nargs="?", default="demo"
    )
    parser.add_argument("--no-plot", action="store_true", help="Do not display demo plot")
    args = parser.parse_args()
    if args.command == "demo":
        run_demo(show_plot=not args.no_plot)
    else:
        print(run_experiments().to_string(index=False))


if __name__ == "__main__":
    main()
