"""Exploratory CPU vs Apple Silicon MPS smoke check for Yug.

Run from the repository root with ``python examples/apple_silicon_smoke.py``.
This checks that both backends produce finite forecasts and reports numerical
and timing differences. It does not set a performance or parity threshold.
"""

from __future__ import annotations

import gc
import statistics
import time

import numpy as np
import torch

from yug import YugPipeline

MODEL_ID = "birlaailabs/yug"
CONTEXT_LENGTH = 256
PREDICTION_LENGTH = 64
NUM_SAMPLES = 4
REPEATS = 3


def synchronize(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()


def run_backend(device: str, context: np.ndarray) -> tuple[np.ndarray, float]:
    print(f"Loading {MODEL_ID} on {device} (first run may download model weights)...")
    pipeline = YugPipeline.from_pretrained(MODEL_ID, device_map=device)

    def forecast() -> np.ndarray:
        result = pipeline.predict(
            context,
            prediction_length=PREDICTION_LENGTH,
            freq="D",
            num_samples=NUM_SAMPLES,
            seed=7,
        )
        return result.values[0]

    # Warm up kernels/cache, then report median over synchronized measurements.
    forecast()
    timings = []
    output = None
    for _ in range(REPEATS):
        synchronize(device)
        start = time.perf_counter()
        output = forecast()
        synchronize(device)
        timings.append(time.perf_counter() - start)

    assert output is not None
    if not np.isfinite(output).all():
        raise RuntimeError(f"{device} forecast contains NaN or infinite values")
    del pipeline
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    return output, statistics.median(timings)


def main() -> None:
    if not torch.backends.mps.is_available():
        raise SystemExit(
            "MPS is unavailable. Run this on an Apple Silicon Mac with a "
            "PyTorch build that supports MPS. No model was loaded."
        )

    rng = np.random.default_rng(7)
    context = rng.normal(size=CONTEXT_LENGTH).astype(np.float32)

    mps_forecast, mps_seconds = run_backend("mps", context)
    cpu_forecast, cpu_seconds = run_backend("cpu", context)
    max_abs_difference = float(np.max(np.abs(mps_forecast - cpu_forecast)))

    print(f"Forecast shape: {mps_forecast.shape}")
    print(f"MPS median warm forecast: {mps_seconds:.3f}s")
    print(f"CPU median warm forecast: {cpu_seconds:.3f}s")
    print(f"Maximum absolute CPU/MPS difference: {max_abs_difference:.6g}")
    print("Exploratory smoke check only; no speedup or parity threshold is implied.")


if __name__ == "__main__":
    main()
