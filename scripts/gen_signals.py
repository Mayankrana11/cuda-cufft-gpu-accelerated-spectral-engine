#!/usr/bin/env python3
"""gen_signals.py - Generates synthetic time-domain audio/sensor signals for cuFFT pipeline."""

import argparse
import math
import os
import random

def generate_signal(num_samples: int, sample_rate: int, seed: int):
    rng = random.Random(seed)
    signal = [0.0] * num_samples
    t_step = 1.0 / sample_rate

    # Generate 3-5 harmonic sinusoidal tones (e.g. musical chords or sensor vibrations)
    n_harmonics = rng.randint(3, 6)
    base_freq = rng.uniform(120.0, 800.0)
    for h in range(1, n_harmonics + 1):
        freq = base_freq * h + rng.uniform(-10.0, 10.0)
        amp = rng.uniform(0.15, 0.7) / h
        phase = rng.uniform(0.0, 2.0 * math.pi)
        for i in range(num_samples):
            t = i * t_step
            signal[i] += amp * math.sin(2.0 * math.pi * freq * t + phase)

    # Add frequency chirp sweep in second half
    sweep_start = rng.uniform(300.0, 1200.0)
    sweep_rate = rng.uniform(500.0, 2000.0)
    for i in range(num_samples // 2, num_samples):
        t = (i - num_samples // 2) * t_step
        instant_freq = sweep_start + 0.5 * sweep_rate * t
        signal[i] += 0.3 * math.sin(2.0 * math.pi * instant_freq * t)

    # Add Gaussian-like white noise
    for i in range(num_samples):
        noise = (rng.random() + rng.random() + rng.random() - 1.5) * 0.15
        signal[i] += noise

    # Normalize to [-1.0, 1.0]
    max_val = max(max(abs(v) for v in signal), 1e-6)
    return [round(v / max_val, 5) for v in signal]

def write_csv(path: str, samples: list, sample_rate: int):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# sample_rate: {sample_rate}\n")
        f.write(f"# num_samples: {len(samples)}\n")
        for s in samples:
            f.write(f"{s}\n")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=100, help="Number of signal files")
    parser.add_argument("--samples", type=int, default=16384, help="Samples per file")
    parser.add_argument("--rate", type=int, default=16000, help="Sample rate in Hz")
    parser.add_argument("--output", default="data/input", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    print(f"Generating {args.count} signal datasets ({args.samples} samples @ {args.rate} Hz)...")

    digits = len(str(args.count))
    for i in range(1, args.count + 1):
        filename = f"signal_{str(i).zfill(digits)}.csv"
        path = os.path.join(args.output, filename)
        samples = generate_signal(args.samples, args.rate, seed=i * 42)
        write_csv(path, samples, args.rate)
        if i % 25 == 0 or i == args.count:
            print(f"  Generated {i}/{args.count} signals...")

    print(f"Done! {args.count} signals written to {args.output}")

if __name__ == "__main__":
    main()
