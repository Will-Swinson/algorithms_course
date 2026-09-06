# Advanced Algorithms Course

## Description

My implementation of algorithms studied in Advanced Algorithms course,
following Chapter 1.6 ("Setting Up Your Algorithm Laboratory") of
_Advanced Algorithms: A Journey Through Computational Problem Solving_
by Dr. Moody Amakobe.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Verify the environment:

```bash
python scripts/check_environment.py
```

## Project Structure

```
src/sorting/basic_sorts.py    Bubble, selection, and insertion sort (Week 1)
src/sorting/merge_sort.py     Merge sort with merge helper (Week 2)
src/sorting/quick_sort.py     Randomized quicksort: three-way partition,
                              insertion-sort cutoff (Week 2)
src/utils/benchmark.py        Reusable benchmarking framework
tests/                        Pytest test suite
benchmarks/                   Benchmark runners + raw results (CSV, PNG)
analysis/week2_report.md      Week 2 technical report
examples/week2_demo.py        Runnable demo of all five sorts
docs/performance_analysis.md  Week 1 performance report with charts
docs/images/                  Week 1 benchmark visualizations
```

## Running Tests

```bash
pytest tests/ -v
```

## Running Benchmarks

Week 1 (basic sorts):

```bash
python benchmarks/run_sorting_benchmarks.py
```

Regenerates the charts in `docs/images/` and the raw timing data in
`benchmarks/results/sorting_benchmarks.csv`.

Week 2 (all five algorithms, 6 data types, n up to 50,000):

```bash
python benchmarks/week2_performance.py
```

Writes per-data-type plots, `comparison_table.csv`, and
`complexity_summary.csv` to `benchmarks/results/`. The full run takes
roughly an hour (the O(n²) sorts at n = 50,000 dominate); results are
appended incrementally, so an interrupted run resumes where it left off.

## Demo

```bash
python examples/week2_demo.py
```

## Current Progress

- [x] Week 1: Environment setup, basic sorting algorithms, benchmarking
      framework, test suite, and performance analysis
- [x] Week 2: Divide and conquer — merge sort, randomized quicksort,
      full 5-algorithm benchmark comparison, and technical report
- [ ] Week 3: Search algorithms

## Author

Will Swinson
