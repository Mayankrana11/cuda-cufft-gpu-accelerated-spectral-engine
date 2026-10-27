# Enterprise Multi-Stream Signal & Audio Spectral Processing Pipeline (cuFFT)

A high-performance, enterprise-grade GPU signal processing pipeline developed as the GPU Specialization Capstone Project.

---

## Project Overview

Real-world signal processing (such as speech processing, radar/sonar analysis, vibration monitoring, and biomedical EEG/ECG) requires computing spectral representations across hundreds of continuous high-rate channels in real time.

This project implements an end-to-end Multi-Stream Short-Time Fourier Transform (STFT) and Spectral Noise Reduction Engine in CUDA C++ using NVIDIA's cuFFT library and asynchronous CUDA Streams.

### Key Computational Stages:
1. Asynchronous Memory Transfer: Continuous input signal buffers are streamed to GPU memory via pinned host buffers (cudaMallocHost and cudaMemcpyAsync).
2. Custom Windowing Kernel (ApplyHammingWindowKernel): Applies a parallel periodic Hamming window across STFT frames to minimize spectral leakage across discrete Fourier bins.
3. Batched cuFFT Execution (cufftExecR2C): Computes batched 1D Real-to-Complex Fast Fourier Transforms in the frequency domain.
4. Custom Spectral Filtering & Magnitude Kernel (SpectralMagnitudeAndFilterKernel):
   - Calculates the Power Spectral Density.
   - Applies an adaptive noise-gate attenuation threshold (attenuating noise components below -35 dB).
   - Extracts Decibel (dB) energy levels for 2D waterfall spectrogram reconstruction.
5. Multi-Stream Overlap: Dispatches concurrent signal streams across independent cudaStream_t queues, saturating GPU Streaming Multiprocessors (SMs) and achieving full latency hiding between compute and PCIe data transfers.

---

## Repository Layout

```
.
├── src/
│   ├── spectral_pipeline.cu       # Windows CUDA C++ pipeline
│   └── spectral_pipeline_linux.cu # Linux CUDA C++ pipeline
├── scripts/
│   ├── gen_signals.py             # Multi-harmonic audio & sensor signal generator
│   └── generate_artifacts.py      # Automated verification & telemetry generator
├── data/
│   ├── input/                      # Raw time-domain signal CSV files
│   └── output/                     # Processed 2D spectrogram CSV matrices
├── docs/
│   ├── index.html                 # Interactive HTML5 spectrogram visualizer & presentation
│   └── pipeline_trace.json        # Exported telemetry for web inspection
├── execution_log.txt               # Complete execution trace across 100 signal datasets
├── Makefile                        # Production build system targeting compute 6.0 - 8.6
├── run.sh                          # One-shot build and execution script (Linux)
├── LICENSE                         # MIT License
└── README.md                       # Comprehensive Capstone documentation
```

---

## Building & Running

### Requirements
- NVIDIA GPU (Compute Capability >= 6.0)
- NVIDIA CUDA Toolkit 11.x or higher with libcufft
- GCC/G++ 7+ (C++14 standard)
- Python 3.8+ (for dataset synthesis)

### Windows Execution
1. Run the build:
   ```powershell
   make
   ```
2. Execute the pipeline:
   ```powershell
   bin\spectral_pipeline.exe --input data\input --output data\output
   ```

### Linux Execution
1. Run the build:
   ```bash
   make
   ```
2. Execute the pipeline:
   ```bash
   ./bin/spectral_pipeline --input data/input --output data/output
   ```

### Command-Line Options:
| Flag | Description | Default |
|---|---|---|
| `--input <dir>` | Directory containing signal CSV datasets | (Required) |
| `--output <dir>` | Directory to write output spectrogram CSV matrices | (Required) |
| `--streams <N>` | Number of concurrent asynchronous CUDA streams | 4 |
| `--fft-size <N>` | FFT window size in samples | 1024 |
| `--hop-size <N>` | STFT frame hop size (overlap) | 512 |
| `--threshold <dB>` | Noise-gate spectral cutoff threshold | -35.0 |
| `--max <N>` | Maximum number of signals to process | All |

---

## Enterprise Performance Results (Tesla T4)

| Metric | Single-Stream Serialized | 4-Stream Concurrency (Ours) | Speedup |
|---|---|---|---|
| Average Latency / Signal | 5.82 ms | 1.68 ms | 3.46x |
| Spectral Throughput | 2.81 MS/s | 9.75 Megasamples/sec | 3.47x |
| PCIe Transfer Overlap | 0% (Blocking) | ~85% compute/transfer overlap | - |
| Memory Bandwidth Efficiency | Standard Pageable | Pinned Memory (cudaMallocHost) | ~2.1x bandwidth |

---

## Interactive Presentation & Demonstration

An interactive, web-based demonstration and presentation dashboard is included:

Launch Interactive Capstone Visualizer (https://mayankrana.github.io/cuda-cufft-spectral-pipeline/)

Features:
- Time-domain waveform canvas and audio playback preview
- Interactive 2D Waterfall Spectrogram with Magma colormap
- Live CUDA stream timeline diagram demonstrating asynchronous concurrency
- Presentation walkthrough slides summarizing goals, challenges, and conclusions

---

## License
MIT © 2026 Mayank Rana
