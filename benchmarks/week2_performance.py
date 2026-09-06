# File: benchmarks/week2_performance.py
"""
Week 2 performance benchmark: all five sorting algorithms compared.

Runs bubble, selection, insertion (Week 1) and merge, quick (Week 2)
on 6 data types at sizes 100 .. 50,000, then produces:

- benchmarks/results/comparison_table.csv   (raw timing data)
- benchmarks/results/<data_type>.png        (one plot per data type)
- benchmarks/results/complexity_summary.csv (empirical fits, random data)

The run takes a long time (the O(n²) sorts at n = 50,000 need ~1 min
per execution), so the script is RESUMABLE: every completed measurement
is appended to the CSV immediately, and on restart any (algorithm,
data type, size) combination already present is skipped.
"""
import csv
import os
import random
import sys
import time

import matplotlib
matplotlib.use("Agg")  # headless: write PNGs, never open a window
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.sorting.basic_sorts import bubble_sort, selection_sort, insertion_sort
from src.sorting.merge_sort import merge_sort
from src.sorting.quick_sort import quick_sort
from src.utils.benchmark import AlgorithmBenchmark, BenchmarkResult

RESULTS_DIR = os.path.join(PROJECT_ROOT, "benchmarks", "results")
CSV_PATH = os.path.join(RESULTS_DIR, "comparison_table.csv")
CSV_FIELDS = ["algorithm", "data_type", "size", "runs",
              "avg_time", "std_dev", "min_time", "max_time"]

SEED = 42
SIZES = [100, 500, 1000, 5000, 10000, 50000]

ALGORITHMS = {
    "bubble_sort": bubble_sort,
    "selection_sort": selection_sort,
    "insertion_sort": insertion_sort,
    "merge_sort": merge_sort,
    "quick_sort": quick_sort,
}

PLOT_COLORS = {
    "bubble_sort": "tab:red",
    "selection_sort": "tab:orange",
    "insertion_sort": "tab:green",
    "merge_sort": "tab:blue",
    "quick_sort": "tab:purple",
}


# ---------------------------------------------------------------------------
# Data generation (seeded for reproducibility)
# ---------------------------------------------------------------------------

def _nearly_sorted(size, rng):
    """Sorted list with ~5% of elements displaced (95% sorted)."""
    data = list(range(size))
    for _ in range(max(1, size // 40)):  # each swap displaces 2 elements
        a, b = rng.randrange(size), rng.randrange(size)
        data[a], data[b] = data[b], data[a]
    return data


DATA_GENERATORS = {
    "random_data": lambda size, rng: [rng.randint(0, size) for _ in range(size)],
    "sorted_data": lambda size, _rng: list(range(size)),
    "reverse_data": lambda size, _rng: list(range(size, 0, -1)),
    "nearly_sorted": _nearly_sorted,
    # Assignment spec: "many duplicates (only 10 unique values)"
    "many_duplicates": lambda size, rng: [rng.randint(0, 9) for _ in range(size)],
    # Assignment spec: "few unique (only 3 unique values)"
    "few_unique": lambda size, rng: [rng.choice([1, 2, 3]) for _ in range(size)],
}


def runs_for(size):
    """5 timed runs at small sizes; 3 at n >= 10,000 to keep the O(n²)
    algorithms' total wall time manageable (documented in the report)."""
    return 5 if size < 10000 else 3


# ---------------------------------------------------------------------------
# Resumable measurement loop
# ---------------------------------------------------------------------------

def load_completed():
    """Return {(algorithm, data_type, size)} already present in the CSV."""
    done = set()
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, newline="") as f:
            for row in csv.DictReader(f):
                done.add((row["algorithm"], row["data_type"], int(row["size"])))
    return done


def append_row(row):
    new_file = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def run_benchmarks():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    done = load_completed()
    total = len(DATA_GENERATORS) * len(SIZES) * len(ALGORITHMS)
    completed = len(done)
    print(f"Resuming: {completed}/{total} measurements already recorded"
          if completed else f"Starting fresh: {total} measurements to run")

    # One warmup at large sizes (a warmup of bubble sort at n=50,000 costs
    # a full minute); two at small sizes, matching Week 1 methodology.
    bench_small = AlgorithmBenchmark(warmup_runs=2)
    bench_large = AlgorithmBenchmark(warmup_runs=1)

    suite_start = time.perf_counter()
    for data_type, generate in DATA_GENERATORS.items():
        for size in SIZES:
            rng = random.Random(SEED)  # same data for every algorithm
            data = generate(size, rng)
            for name, func in ALGORITHMS.items():
                if (name, data_type, size) in done:
                    continue
                bench = bench_small if size < 10000 else bench_large
                result = bench.time_algorithm(
                    func, data, runs=runs_for(size))
                append_row({
                    "algorithm": name,
                    "data_type": data_type,
                    "size": size,
                    "runs": runs_for(size),
                    "avg_time": f"{result.average_time:.6f}",
                    "std_dev": f"{result.std_deviation:.6f}",
                    "min_time": f"{result.min_time:.6f}",
                    "max_time": f"{result.max_time:.6f}",
                })
                completed += 1
                elapsed = time.perf_counter() - suite_start
                print(f"[{completed:3d}/{total}] {data_type:15s} n={size:<6d} "
                      f"{name:15s} {result.average_time:.6f}s "
                      f"(elapsed {elapsed / 60:.1f} min)", flush=True)


# ---------------------------------------------------------------------------
# Plots and complexity analysis (read everything back from the CSV)
# ---------------------------------------------------------------------------

def load_rows():
    with open(CSV_PATH, newline="") as f:
        return list(csv.DictReader(f))


def plot_data_type(rows, data_type):
    """One log-log plot per data type: time vs n for all 5 algorithms."""
    plt.figure(figsize=(10, 6))
    for name in ALGORITHMS:
        series = sorted(
            ((int(r["size"]), float(r["avg_time"]), float(r["std_dev"]))
             for r in rows
             if r["algorithm"] == name and r["data_type"] == data_type),
        )
        if not series:
            continue
        sizes = [s for s, _, _ in series]
        times = [t for _, t, _ in series]
        errors = [e for _, _, e in series]
        plt.errorbar(sizes, times, yerr=errors, marker="o", capsize=3,
                     label=name, color=PLOT_COLORS[name])

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Input size (n)")
    plt.ylabel("Average time (seconds)")
    plt.title(f"Sorting performance: {data_type.replace('_', ' ')}")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, f"{data_type}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def complexity_summary(rows):
    """Fit each algorithm's random-data timings with the Week 1 framework
    and record the best-fit complexity, R², and doubling ratio."""
    bench = AlgorithmBenchmark()
    summary_path = os.path.join(RESULTS_DIR, "complexity_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["algorithm", "best_fit", "r_squared",
                         "avg_doubling_ratio"])
        for name in ALGORITHMS:
            results = [
                BenchmarkResult(
                    algorithm_name=name,
                    input_size=int(r["size"]),
                    average_time=float(r["avg_time"]),
                    std_deviation=float(r["std_dev"]),
                    min_time=float(r["min_time"]),
                    max_time=float(r["max_time"]),
                )
                for r in rows
                if r["algorithm"] == name and r["data_type"] == "random_data"
            ]
            analysis = bench.analyze_complexity(results, name)
            if "error" in analysis:
                print(f"{name}: {analysis['error']}")
                continue
            writer.writerow([
                name,
                analysis["best_fit_complexity"],
                f"{analysis['best_fit_r_squared']:.4f}",
                f"{analysis['average_doubling_ratio']:.3f}",
            ])
            print(f"{name:15s} best fit {analysis['best_fit_complexity']:11s} "
                  f"R²={analysis['best_fit_r_squared']:.4f}")
    print(f"Saved {summary_path}")


def main():
    run_benchmarks()
    rows = load_rows()
    for data_type in DATA_GENERATORS:
        plot_data_type(rows, data_type)
    complexity_summary(rows)
    print("Done.")


if __name__ == "__main__":
    main()
