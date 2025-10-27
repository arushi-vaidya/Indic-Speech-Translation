import argparse
import subprocess
import time
import statistics
import concurrent.futures
import importlib
import csv
import sys
import platform
from typing import Callable, List

def run_cmd_once(cmd: List[str], input_text: str, timeout: int = 60):
    start = time.perf_counter()
    p = subprocess.run(cmd, input=input_text.encode('utf-8'), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    end = time.perf_counter()
    return end - start, p.stdout.decode('utf-8', errors='ignore'), p.stderr.decode('utf-8', errors='ignore')

def load_callable(path: str) -> Callable[[str], str]:
    module_name, func_name = path.split(':', 1)
    mod = importlib.import_module(module_name)
    return getattr(mod, func_name)

def bench_cmd(cmd_template, inputs: List[str], warmup: int, iters: int, concurrency: int, timeout: int):
    cmd_list = cmd_template.split()
    for i in range(warmup):
        run_cmd_once(cmd_list, inputs[i % len(inputs)], timeout=timeout)
    times = []
    def task(i):
        return run_cmd_once(cmd_list, inputs[i % len(inputs)], timeout=timeout)
    if concurrency <= 1:
        for i in range(iters):
            t, out, err = task(i)
            times.append(t)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
            futures = [ex.submit(task, i) for i in range(iters)]
            for f in concurrent.futures.as_completed(futures):
                t, out, err = f.result()
                times.append(t)
    return times

def bench_callable(callable_fn: Callable[[str], str], inputs: List[str], warmup: int, iters: int, concurrency: int):
    for i in range(warmup):
        callable_fn(inputs[i % len(inputs)])
    times = []
    def task(i):
        start = time.perf_counter()
        callable_fn(inputs[i % len(inputs)])
        return time.perf_counter() - start
    if concurrency <= 1:
        for i in range(iters):
            times.append(task(i))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
            futures = [ex.submit(task, i) for i in range(iters)]
            for f in concurrent.futures.as_completed(futures):
                times.append(f.result())
    return times

def percentiles(times: List[float], ps=(50,90,95,99)):
    times_sorted = sorted(times)
    n = len(times_sorted)
    out = {}
    for p in ps:
        k = max(0, min(n-1, int(round((p/100.0)*(n-1)))))
        out[p] = times_sorted[k]
    return out

def histogram(times: List[float], bins=10):
    mn, mx = min(times), max(times)
    if mn == mx:
        return [(mn, mx, len(times))]
    width = (mx - mn) / bins
    hist = []
    for i in range(bins):
        low = mn + i*width
        high = low + width
        cnt = sum(1 for t in times if (t >= low and (t < high or i == bins-1)))
        hist.append((low, high, cnt))
    return hist

def summary(times):
    total = sum(times)
    n = len(times)
    mean = statistics.mean(times) if n else 0.0
    med = statistics.median(times) if n else 0.0
    ps = percentiles(times, (50,90,95,99))
    return {
        "count": n,
        "total_s": total,
        "mean_s": mean,
        "median_s": med,
        "p90_s": ps[90],
        "p95_s": ps[95],
        "p99_s": ps[99],
        "throughput_rps": n / total if total > 0 else float('inf'),
        "min_s": min(times) if n else 0.0,
        "max_s": max(times) if n else 0.0
    }

def load_inputs(path: str, default_samples= ["This is a test sentence.", "Hello world!", "How are you?"]):
    if path is None:
        return default_samples
    with open(path, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip() and not l.strip().startswith('#')]
    return lines or default_samples

def save_csv(path: str, times: List[float], meta: dict):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["meta", "value"])
        for k,v in meta.items():
            writer.writerow([k, v])
        writer.writerow([])
        writer.writerow(["iteration","time_s"])
        for i,t in enumerate(times, start=1):
            writer.writerow([i, f"{t:.9f}"])

def print_sys_info():
    print("System:")
    print(f"  platform: {platform.platform()}")
    print(f"  python: {sys.version.splitlines()[0]}")

def main():
    parser = argparse.ArgumentParser(description="Improved generic inference benchmark tool")
    parser.add_argument('--mode', choices=['cmd','python'], default='cmd')
    parser.add_argument('--cmd', help='Command to call for each input. Example: "python inference/custom_interactive.py --some-arg"')
    parser.add_argument('--callable', help='Python callable module:function for in-process calls')
    parser.add_argument('--input-file', help='File with one input per line')
    parser.add_argument('--iters', type=int, default=100)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--concurrency', type=int, default=1)
    parser.add_argument('--timeout', type=int, default=120)
    parser.add_argument('--output-csv', help='Path to write CSV of raw times and metadata')
    parser.add_argument('--bins', type=int, default=10, help='Histogram bins')
    args = parser.parse_args()

    print_sys_info()

    inputs = load_inputs(args.input_file)
    if args.mode == 'cmd':
        if not args.cmd:
            parser.error('--cmd is required for mode=cmd')
        print("Running benchmark (cmd) ...")
        times = bench_cmd(args.cmd, inputs, args.warmup, args.iters, args.concurrency, args.timeout)
    else:
        if not args.callable:
            parser.error('--callable is required for mode=python')
        fn = load_callable(args.callable)
        print("Running benchmark (python callable) ...")
        times = bench_callable(fn, inputs, args.warmup, args.iters, args.concurrency)

    stats = summary(times)
    print("\nRESULTS")
    for k,v in stats.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")

    print("\nHistogram (seconds):")
    for low, high, cnt in histogram(times, bins=args.bins):
        print(f"  {low:.6f} - {high:.6f}: {cnt}")
    if args.output_csv:
        meta = {k: (f"{v:.6f}" if isinstance(v, float) else v) for k,v in stats.items()}
        save_csv(args.output_csv, times, meta)
        print(f"\nSaved raw times + meta to {args.output_csv}")

if __name__ == '__main__':
    main()