import time, json, statistics, torch
from Model import Solution_u
ck = torch.load('model_XJTU_0.pth', map_location='cpu', weights_only=False)
m = Solution_u(); m.load_state_dict(ck['solution_u']); m.eval().cuda()
xg = torch.randn(1, 17, dtype=torch.float32, device='cuda')
with torch.no_grad():
    for _ in range(1000):
        y = m(xg)
    torch.cuda.synchronize()
    ts = []
    for _ in range(10000):
        t0 = time.perf_counter()
        y = m(xg)
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    us = sorted(t*1e6 for t in ts); n = len(us)
    out = {'torch': torch.__version__, 'cuda': torch.version.cuda,
           'gpu': torch.cuda.get_device_name(0),
           'mean_us': sum(us)/n, 'median_us': us[n//2],
           'std_us': statistics.pstdev(us), 'p95_us': us[int(0.95*n)]}
    print('LATENCY ' + json.dumps(out), flush=True)
    json.dump(out, open('gpu5060_latency.json', 'w'), indent=2)
    while True:  # sustained load for power sampling
        y = m(xg)
