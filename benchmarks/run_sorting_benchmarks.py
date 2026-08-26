# File: benchmarks/run_sorting_benchmarks.py
"""
Benchmark the three basic sorting algorithms and produce the charts
and data used in the performance analysis report.

Run from the project root:
    python benchmarks/run_sorting_benchmarks.py

Outputs:
    docs/images/sorting_scaling.png       - time vs input size (log-log)
    docs/images/data_type_comparison.png  - effect of input ordering at n=1000
    benchmarks/results/sorting_benchmarks.csv - raw timing data
"""
import sys
from pathlib import Path

# Allow running as a plain script from the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Use a non-interactive backend so the script runs without a display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.sorting.basic_sorts import bubble_sort, selection_sort, insertion_sort
from src.utils.benchmark import AlgorithmBenchmark

ALGORITHMS = {
    "bubble_sort": bubble_sort,
    "selection_sort": selection_sort,
    "insertion_sort": insertion_sort,
}
SIZES = [100, 200, 400, 800, 1600]  # doubling sizes make ratios easy to read
RUNS = 5
DATA_TYPES = ["sorted", "nearly_sorted", "random", "reverse"]
COMPARISON_SIZE = 1000

IMAGES_DIR = PROJECT_ROOT / "docs" / "images"
RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "results"


def run_scaling_benchmark(benchmark: AlgorithmBenchmark) -> None:
    """Time all algorithms on random data across doubling input sizes."""
    print("\n### Scaling benchmark (random data) ###")
    results = benchmark.benchmark_suite(ALGORITHMS, SIZES, ["random"], runs=RUNS)

    benchmark.plot_comparison(
        results,
        title="Basic Sorting Algorithms - Random Data",
        log_scale=True,
        save_path=str(IMAGES_DIR / "sorting_scaling.png"),
    )

    print("\n### Empirical complexity analysis ###")
    for name, algorithm_results in results.items():
        analysis = benchmark.analyze_complexity(algorithm_results, name)
        print(f"\n{name}:")
        print(f"  {analysis['interpretation']}")


def run_data_type_comparison(benchmark: AlgorithmBenchmark) -> None:
    """Compare all algorithms across input orderings at a fixed size."""
    print(f"\n### Data type comparison (n = {COMPARISON_SIZE}) ###")
    timings = {name: [] for name in ALGORITHMS}

    for data_type in DATA_TYPES:
        data = benchmark.generate_test_data(COMPARISON_SIZE, data_type, seed=42)
        for name, algorithm in ALGORITHMS.items():
            result = benchmark.time_algorithm(algorithm, data, runs=RUNS)
            timings[name].append(result.average_time)
            print(f"  {data_type:15} {name:20}: "
                  f"{result.average_time:.6f}s ± {result.std_deviation:.6f}s")

    # Grouped bar chart, log-scale y so the sorted-input times stay visible
    x = np.arange(len(DATA_TYPES))
    width = 0.25
    plt.figure(figsize=(10, 6))
    for offset, (name, times) in enumerate(timings.items()):
        plt.bar(x + (offset - 1) * width, times, width, label=name)
    plt.yscale("log")
    plt.xticks(x, DATA_TYPES)
    plt.xlabel("Input ordering", fontsize=12)
    plt.ylabel("Time (seconds, log scale)", fontsize=12)
    plt.title(f"Effect of Input Ordering (n = {COMPARISON_SIZE:,})",
              fontsize=14, fontweight="bold")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    save_path = IMAGES_DIR / "data_type_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {save_path}")


def main() -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    benchmark = AlgorithmBenchmark(warmup_runs=2)
    run_scaling_benchmark(benchmark)
    run_data_type_comparison(benchmark)
    benchmark.export_results(str(RESULTS_DIR / "sorting_benchmarks.csv"))
    print("\nDone. Charts are in docs/images/, raw data in benchmarks/results/.")


if __name__ == "__main__":
    main()
