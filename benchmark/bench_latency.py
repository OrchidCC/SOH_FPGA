#!/usr/bin/env python3
"""Batch-1 latency benchmark for the PINN SOH estimator.

Protocol (as stated in the response letter):
- single-sample inference, batch size 1
- FP32, PyTorch eager
- warm-up phase before timing, then averaged over repeated runs
- GPU timing includes host-device transfer (H2D input + D2H output)
"""
import time, json, platform, statistics
import torch
from Model import Solution_u

WARMUP = 1000
RUNS = 10000

ck = torch.load("model_XJTU_0.pth", map_location="cpu", weights_only=False)
model = Solution_u()
model.load_state_dict(ck["solution_u"])
model.eval()

x = torch.randn(1, 17, dtype=torch.float32)

def bench_cpu():
    m = model
    with torch.no_grad():
        for _ in range(WARMUP):
            m(x)
        ts = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            y = m(x)
            ts.append(time.perf_counter() - t0)
    return ts

def bench_gpu():
    m = Solution_u()
    m.load_state_dict(ck["solution_u"])
    m.eval().cuda()
    with torch.no_grad():
        for _ in range(WARMUP):
            xg = x.cuda(non_blocking=False)
            y = m(xg)
            _ = y.cpu()
        torch.cuda.synchronize()
        ts = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            xg = x.cuda(non_blocking=False)
            y = m(xg)
            _ = y.cpu()
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
    return ts

def stats(ts):
    us = [t * 1e6 for t in ts]
    us.sort()
    n = len(us)
    return {
        "mean_us": sum(us) / n,
        "median_us": us[n // 2],
        "std_us": statistics.pstdev(us),
        "p5_us": us[int(0.05 * n)],
        "p95_us": us[int(0.95 * n)],
        "min_us": us[0],
    }

out = {
    "cpu_name": platform.processor(),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "warmup": WARMUP, "runs": RUNS,
}
print("benchmarking CPU...", flush=True)
out["cpu"] = stats(bench_cpu())
if torch.cuda.is_available():
    print("benchmarking GPU...", flush=True)
    out["gpu"] = stats(bench_gpu())
print(json.dumps(out, indent=1))
with open("bench_result.json", "w") as f:
    json.dump(out, f, indent=2)
