import os
import gc
import io
import sys
import time
import math
import json
import random
import logging
import platform
import threading
import contextlib
import tracemalloc
import multiprocessing as mp
from copy import deepcopy

import numpy as np
import psutil
import torch
import torch.nn as nn

# ==========================================
# 0. Global reproducibility configuration
# ==========================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_grad_enabled(False)
torch.backends.mkldnn.enabled = True


# ==========================================
# 1. Benchmark configuration
# ==========================================
class CpuEvalConfig:
    IMG_SIZE        = 224
    DEVICE          = 'cpu'
    PTH_DIR         = './'
    WARMUP_REPS     = 100
    TEST_REPS       = 200
    INNER_REPS      = 10
    DISCARD_RATIO   = 0.10
    NUM_TRIALS      = 3
    SLEEP_BETWEEN   = 5.0
    NUM_THREADS     = 1
    FUSE_BN         = True
    SHUFFLE_ORDER   = True
    MP_START_METHOD = "fork"   # Notebook-friendly; change to "spawn" for pure .py


# ==========================================
# 2. Model architecture
# ==========================================
class PConv(nn.Module):
    def __init__(self, dim, ratio=0.25):
        super().__init__()
        self.dim_conv = max(1, int(dim * ratio))
        self.dim_untouched = dim - self.dim_conv
        self.conv = nn.Conv2d(self.dim_conv, self.dim_conv, 3, 1, 1, bias=False)
        self.bn   = nn.BatchNorm2d(self.dim_conv)
        self.act  = nn.ReLU6(inplace=True)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.dim_conv, self.dim_untouched], dim=1)
        x1 = self.act(self.bn(self.conv(x1)))
        return torch.cat((x1, x2), dim=1)


class ECA(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        k = int(abs((math.log2(channels) / 2) + 0.5))
        k = k if k % 2 == 1 else k + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.gap(x).squeeze(-1).transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y)


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, use_pconv=False, use_eca=False):
        super().__init__()
        self.use_res_connect = (stride == 1 and inp == oup)
        hidden_dim = int(round(inp * expand_ratio))

        layers = []
        if expand_ratio != 1:
            layers += [nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                       nn.BatchNorm2d(hidden_dim),
                       nn.ReLU6(inplace=True)]

        if use_pconv and stride == 1:
            layers.append(PConv(hidden_dim, ratio=0.25))
        else:
            layers += [nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1,
                                 groups=hidden_dim, bias=False),
                       nn.BatchNorm2d(hidden_dim),
                       nn.ReLU6(inplace=True)]

        if use_eca:
            layers.append(ECA(hidden_dim))

        layers += [nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                   nn.BatchNorm2d(oup)]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.conv(x) if self.use_res_connect else self.conv(x)


class ProposedMobileNetV2(nn.Module):
    def __init__(self, num_classes=3, width_mult=0.25, expand_ratio=6,
                 use_pconv=False, use_eca=False):
        super().__init__()
        settings = [[1,  16, 1, 1],
                    [6,  24, 2, 2],
                    [6,  32, 3, 2],
                    [6,  64, 4, 2],
                    [6,  96, 3, 1],
                    [6, 160, 3, 2],
                    [6, 320, 1, 1]]

        def _make_divisible(v, divisor=8):
            new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v

        input_channel    = _make_divisible(32 * width_mult)
        self.last_channel = _make_divisible(1280 * max(1.0, width_mult))

        features = [nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
                    nn.BatchNorm2d(input_channel),
                    nn.ReLU6(inplace=True)]

        for t, c, n, s in settings:
            output_channel = _make_divisible(c * width_mult)
            t_actual = t if t == 1 else expand_ratio
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidual(
                    input_channel, output_channel, stride, t_actual,
                    use_pconv=use_pconv, use_eca=use_eca))
                input_channel = output_channel

        features += [nn.Conv2d(input_channel, self.last_channel, 1, 1, bias=False),
                     nn.BatchNorm2d(self.last_channel),
                     nn.ReLU6(inplace=True)]

        self.features   = nn.Sequential(*features)
        self.classifier = nn.Sequential(nn.Dropout(0.2),
                                        nn.Linear(self.last_channel, num_classes))

    def forward(self, x):
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.classifier(x)


# ==========================================
# 3. Conv-BN fusion
# ==========================================
def fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    fused = nn.Conv2d(conv.in_channels, conv.out_channels,
                      conv.kernel_size, conv.stride, conv.padding,
                      conv.dilation, conv.groups, bias=True)
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    bn_std = (bn.eps + bn.running_var).sqrt()
    w_bn   = torch.diag(bn.weight / bn_std)
    fused.weight.copy_(torch.mm(w_bn, w_conv).view(fused.weight.size()))
    b_conv = (torch.zeros(conv.out_channels, device=conv.weight.device)
              if conv.bias is None else conv.bias)
    b_bn = bn.bias - bn.weight * bn.running_mean / bn_std
    fused.bias.copy_(b_conv + b_bn)
    return fused


def fuse_model(model: nn.Module) -> nn.Module:
    for name, module in model.named_children():
        if isinstance(module, nn.Sequential):
            new_layers = []
            i = 0
            children = list(module.children())
            while i < len(children):
                if (i + 1 < len(children)
                        and isinstance(children[i], nn.Conv2d)
                        and isinstance(children[i+1], nn.BatchNorm2d)):
                    new_layers.append(fuse_conv_bn(children[i], children[i+1]))
                    i += 2
                else:
                    fuse_model(children[i])
                    new_layers.append(children[i])
                    i += 1
            setattr(model, name, nn.Sequential(*new_layers))
        else:
            fuse_model(module)
    return model


# ==========================================
# 4. Params / MACs / Weight-size statistics
# ==========================================
def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_weight_mb(model: nn.Module) -> float:
    """Weight memory footprint in MB — the most stable metric for papers"""
    return sum(p.numel() * p.element_size()
               for p in model.parameters()) / (1024 ** 2)


def count_flops(model: nn.Module, input_size=(1, 3, 224, 224)) -> float:
    """Count MACs; element-wise ops are excluded per academic convention. Suppresses unsupported-op warnings."""
    x = torch.randn(*input_size)

    # Prefer thop
    try:
        from thop import profile
        buf_out, buf_err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
            macs, _ = profile(deepcopy(model), inputs=(x,), verbose=False)
        return float(macs)
    except ImportError:
        pass

    # Fallback to fvcore
    try:
        from fvcore.nn import FlopCountAnalysis
        logging.getLogger("fvcore").setLevel(logging.ERROR)
        fa = FlopCountAnalysis(model, x)
        fa.unsupported_ops_warnings(False)
        fa.uncalled_modules_warnings(False)
        return float(fa.total())
    except ImportError:
        return float('nan')


# ==========================================
# 5. Memory monitoring thread
# ==========================================
class CPUMemoryMonitor(threading.Thread):
    def __init__(self, delay=0.001):
        super().__init__(daemon=True)
        self.delay = delay
        self.peak  = 0.0
        self.running = True
        self.proc = psutil.Process(os.getpid())

    def run(self):
        while self.running:
            try:
                rss = self.proc.memory_info().rss / (1024 ** 2)
                if rss > self.peak:
                    self.peak = rss
            except Exception:
                pass
            time.sleep(self.delay)

    def stop(self):
        self.running = False


# ==========================================
# 6. Single measurement within subprocess
# ==========================================
def _worker_measure(cfg: dict, exp: dict, return_dict):
    try:
        torch.manual_seed(SEED)
        torch.set_num_threads(cfg["NUM_THREADS"])
        torch.set_grad_enabled(False)

        # ---- 1) Build model ----
        model = ProposedMobileNetV2(
            num_classes=3, width_mult=0.25,
            expand_ratio=exp["t"],
            use_pconv=exp["use_pconv"],
            use_eca=exp["use_eca"]).eval()

        # ---- 2) Load weights ----
        pth_path = os.path.join(cfg["PTH_DIR"],
                                f"best_{exp['name'].replace(' ', '_')}_fold1.pth")
        if os.path.exists(pth_path):
            try:
                sd = torch.load(pth_path, map_location='cpu', weights_only=True)
                model.load_state_dict(sd)
                load_status = "loaded"
            except Exception as e:
                load_status = f"load_fail({type(e).__name__})"
        else:
            load_status = "random_init"

        # ---- 3) Params / MACs / Weight size ----
        params    = count_params(model)
        macs      = count_flops(model, (1, 3, cfg["IMG_SIZE"], cfg["IMG_SIZE"]))
        weight_mb = count_weight_mb(model)

        # ---- 4) Conv-BN fusion ----
        if cfg["FUSE_BN"]:
            model = fuse_model(model)
            model.eval()
        weight_mb_fused = count_weight_mb(model)

        # ---- 5) Input ----
        x = torch.randn(1, 3, cfg["IMG_SIZE"], cfg["IMG_SIZE"])

        # ---- 6) Warmup ----
        for _ in range(cfg["WARMUP_REPS"]):
            _ = model(x)

        # ---- 7) Latency measurement (outer iterations × inner batch) ----
        timings = []
        for _ in range(cfg["TEST_REPS"]):
            t0 = time.perf_counter()
            for _ in range(cfg["INNER_REPS"]):
                _ = model(x)
            t1 = time.perf_counter()
            timings.append((t1 - t0) * 1000.0 / cfg["INNER_REPS"])

        timings = np.array(timings)
        n_drop  = int(len(timings) * cfg["DISCARD_RATIO"])
        timings = timings[n_drop:]

        lat_median = float(np.median(timings))
        lat_mean   = float(np.mean(timings))
        lat_std    = float(np.std(timings))
        lat_p95    = float(np.percentile(timings, 95))
        fps_median = 1000.0 / lat_median

        # ---- 8) Memory measurement (tracemalloc + RSS dual-path) ----
        gc.collect()
        time.sleep(0.2)

        # (a) tracemalloc: precise Python-level allocation
        tracemalloc.start()
        snap_before = tracemalloc.take_snapshot()

        # (b) RSS: high-frequency physical memory sampling
        base_mem = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
        monitor = CPUMemoryMonitor(delay=0.001)
        monitor.start()
        time.sleep(0.05)

        # Run multiple iterations to amplify signal
        for _ in range(500):
            _ = model(x)

        monitor.stop()
        monitor.join()

        snap_after = tracemalloc.take_snapshot()
        stats = snap_after.compare_to(snap_before, 'lineno')
        py_alloc_kb = sum(max(0, s.size_diff) for s in stats) / 1024.0
        tracemalloc.stop()

        peak_rss_inc = max(0.0, monitor.peak - base_mem)

        # ---- 9) Single-forward activation memory peak (tracemalloc) ----
        gc.collect()
        tracemalloc.start()
        _ = model(x)
        _, peak_act_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        single_fwd_peak_kb = peak_act_bytes / 1024.0

        return_dict.update({
            "status":             load_status,
            "params":             params,
            "flops":              macs,
            "weight_mb":          weight_mb,
            "weight_mb_fused":    weight_mb_fused,
            "lat_median":         lat_median,
            "lat_mean":           lat_mean,
            "lat_std":            lat_std,
            "lat_p95":            lat_p95,
            "fps_median":         fps_median,
            "py_alloc_kb":        py_alloc_kb,
            "peak_rss_inc":       peak_rss_inc,
            "single_fwd_peak_kb": single_fwd_peak_kb,
        })
    except Exception as e:
        import traceback
        return_dict["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"


def measure_in_subprocess(cfg: dict, exp: dict) -> dict:
    ctx = mp.get_context(cfg["MP_START_METHOD"])
    manager = ctx.Manager()
    rd = manager.dict()
    p = ctx.Process(target=_worker_measure, args=(cfg, exp, rd))
    p.start()
    p.join()
    if "error" in rd:
        raise RuntimeError(f"[{exp['name']}] {rd['error']}")
    return dict(rd)


# ==========================================
# 7. Robust multi-trial aggregation
# ==========================================
def measure_robust(cfg, exp):
    trials = []
    for k in range(cfg["NUM_TRIALS"]):
        print(f"   trial {k+1}/{cfg['NUM_TRIALS']}", flush=True)
        try:
            r = measure_in_subprocess(cfg, exp)
            if "lat_median" not in r:
                raise RuntimeError(f"incomplete result: {dict(r)}")
            trials.append(r)
        except Exception as e:
            print(f"   ❌ trial {k+1} failed: {e}")
        time.sleep(cfg["SLEEP_BETWEEN"])

    if not trials:
        return {"status": "ALL_TRIALS_FAILED",
                "params": 0, "flops": float('nan'),
                "weight_mb": float('nan'), "weight_mb_fused": float('nan'),
                "lat_median": float('nan'), "lat_mean": float('nan'),
                "lat_std": float('nan'), "lat_p95": float('nan'),
                "fps_median": float('nan'),
                "py_alloc_kb": float('nan'), "peak_rss_inc": float('nan'),
                "single_fwd_peak_kb": float('nan'),
                "lat_trial_std": float('nan'),
                "lat_trial_min": float('nan'),
                "lat_trial_max": float('nan')}

    lats = [t["lat_median"] for t in trials]
    idx  = int(np.argsort(lats)[len(lats) // 2])
    rep  = trials[idx]
    rep["lat_trial_std"] = float(np.std(lats))
    rep["lat_trial_min"] = float(np.min(lats))
    rep["lat_trial_max"] = float(np.max(lats))
    return rep


# ==========================================
# 8. Environment information
# ==========================================
def print_env_info():
    print("=" * 100)
    print("ENVIRONMENT")
    print("-" * 100)
    print(f"Python      : {sys.version.split()[0]}")
    print(f"PyTorch     : {torch.__version__}")
    print(f"NumPy       : {np.__version__}")
    print(f"OS          : {platform.platform()}")
    print(f"CPU arch    : {platform.processor()}")
    try:
        import cpuinfo
        info = cpuinfo.get_cpu_info()
        print(f"CPU brand   : {info.get('brand_raw', 'N/A')}")
        print(f"CPU Hz adv  : {info.get('hz_advertised_friendly', 'N/A')}")
    except ImportError:
        pass
    print(f"Logical CPUs: {psutil.cpu_count(logical=True)}")
    print(f"RAM total   : {psutil.virtual_memory().total / (1024**3):.1f} GB")
    print(f"Threads used: {CpuEvalConfig.NUM_THREADS}")
    print(f"MP method   : {CpuEvalConfig.MP_START_METHOD}")
    print("=" * 100)


# ==========================================
# 9. Main entry point
# ==========================================
def main():
    print_env_info()

    cfg = {k: getattr(CpuEvalConfig, k) for k in dir(CpuEvalConfig)
           if not k.startswith("_")}

    experiments = [
        {"name": "Baseline (t=6)",  "t": 6, "use_pconv": False, "use_eca": False},
        {"name": "t=6+PConv",       "t": 6, "use_pconv": True,  "use_eca": False},
        {"name": "t=3",             "t": 3, "use_pconv": False, "use_eca": False},
        {"name": "t=3+PConv",     "t": 3, "use_pconv": True,  "use_eca": False},
        {"name": "t=3+PConv+ECA",   "t": 3, "use_pconv": True,  "use_eca": True},
    ]

    test_order = list(range(len(experiments)))
    if cfg["SHUFFLE_ORDER"]:
        random.shuffle(test_order)
    print(f"\nTest order (shuffled): {[experiments[i]['name'] for i in test_order]}\n")

    results = {}
    for i, idx in enumerate(test_order):
        exp = experiments[idx]
        print(f"\n[{i+1}/{len(experiments)}] >>> {exp['name']}")
        results[exp["name"]] = measure_robust(cfg, exp)

    # ===== Output table (in original order) =====
    print("\n\n" + "=" * 145)
    print(f"{'BENCHMARK SUMMARY (median of ' + str(cfg['NUM_TRIALS']) + ' trials, BN-fused)':^145}")
    print("=" * 145)
    header = (f"{'Experiment':<22}{'Status':<13}"
              f"{'Params(M)':>10}{'MACs(M)':>10}"
              f"{'Lat med(ms)':>12}{'Lat P95':>10}{'FPS':>8}"
              f"{'Wgt(MB)':>10}{'WgtFu(MB)':>11}"
              f"{'FwdAct(KB)':>12}{'TrialStd':>10}")
    print(header)
    print("-" * 145)
    for exp in experiments:
        r = results[exp["name"]]
        macs_m = r["flops"] / 1e6 if not math.isnan(r["flops"]) else float('nan')
        print(f"{exp['name']:<22}{r['status']:<13}"
              f"{r['params']/1e6:>10.3f}"
              f"{macs_m:>10.2f}"
              f"{r['lat_median']:>12.3f}"
              f"{r['lat_p95']:>10.3f}"
              f"{r['fps_median']:>8.2f}"
              f"{r['weight_mb']:>10.3f}"
              f"{r['weight_mb_fused']:>11.3f}"
              f"{r['single_fwd_peak_kb']:>12.2f}"
              f"{r['lat_trial_std']:>10.4f}")
    print("=" * 145)

    # ===== Save raw JSON =====
    out = {"config": cfg,
           "env": {"python": sys.version, "torch": torch.__version__,
                   "platform": platform.platform()},
           "results": results}
    with open("benchmark_results.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("\nRaw results saved to: benchmark_results.json")


if __name__ == "__main__":
    mp.set_start_method(CpuEvalConfig.MP_START_METHOD, force=True)
    main()