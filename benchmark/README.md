# Platform Benchmark Suite

Evaluation scripts and raw results for the CPU/GPU/FPGA comparison (Table 3 of the paper).

## Protocol

- Single-sample inference, batch size 1 (streaming BMS operating point)
- FP32, PyTorch default eager execution mode
- Latency: 1,000 warm-up runs, then 10,000 timed runs; mean and standard deviation reported
- GPU latency measured on-device (input resident in GPU memory, `torch.cuda.synchronize()` around the model call); host-device transfer excluded
- Power: average over 60 s of sustained batch-1 inference, sampled at 2 Hz
  - CPU: package power from the processor's power sensors (LibreHardwareMonitor)
  - GPU: board power from driver telemetry (`nvidia-smi --query-gpu=power.draw`)

## Files

| File | Purpose |
|---|---|
| `bench_latency.py` | CPU + GPU (transfer-inclusive) latency benchmark |
| `gpu_bench_5060.py` | GPU on-device latency benchmark + sustained load for power sampling |
| `cpu_load.py` | CPU latency benchmark + sustained load for power sampling (single thread, pinned, high priority) |
| `measure_cpu_power.ps1` | CPU package-power sampling around `cpu_load.py` (Windows, LibreHardwareMonitor) |
| `bench_result.json` | Raw latency results, CPU and GPU with transfer |
| `bench_gpu_pure.json` | Raw GPU on-device latency results |
| `cpu_power_7800x3d.json` | Raw CPU power sampling results |

## Reported numbers

| Platform | Latency (batch 1) | Power (sustained load) |
|---|---|---|
| CPU (PyTorch 2.6, FP32 eager) | 25.7 ± 0.5 µs | 40.1 W package |
| GPU (PyTorch 2.11 + CUDA 12.8, FP32 eager) | 55.5 ± 1.8 µs on-device | 31.6 W board |
| FPGA (this work) | 0.69 µs (69 cycles @ 100 MHz, RTL simulation) | 0.182 W (post-implementation analysis) |
