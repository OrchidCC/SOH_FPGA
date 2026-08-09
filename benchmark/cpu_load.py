import sys, time, statistics, torch
sys.path.insert(0, r"D:\CEM_REVISE\code\SOH_FPGA-master\SOH_FPGA-master")
from models.Model import Solution_u

ck = torch.load(r"D:\CEM_REVISE\code\PINN4SOH-main\PINN4SOH-main\pretrained model\model_XJTU_0.pth",
                map_location="cpu", weights_only=False)
import psutil
torch.set_num_threads(1)
psutil.Process().cpu_affinity([2])
psutil.Process().nice(psutil.HIGH_PRIORITY_CLASS)
m = Solution_u(); m.load_state_dict(ck["solution_u"]); m.eval()
x = torch.randn(1, 17, dtype=torch.float32)

with torch.no_grad():
    for _ in range(1000):
        m(x)
    ts = []
    t_end = time.time() + 15
    while time.time() < t_end:
        t0 = time.perf_counter()
        m(x)
        ts.append(time.perf_counter() - t0)
    us = sorted(t * 1e6 for t in ts)
    n = len(us)
    print(f"LAT mean={sum(us)/n:.2f} med={us[n//2]:.2f} std={statistics.pstdev(us):.2f} p95={us[int(0.95*n)]:.2f} n={n}", flush=True)
    while True:  # sustained load for power sampling; killed externally
        m(x)
