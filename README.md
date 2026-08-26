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
src/sorting/basic_sorts.py    Bubble, selection, and insertion sort
src/utils/benchmark.py        Reusable benchmarking framework
tests/                        Pytest test suite (59 tests)
benchmarks/                   Benchmark runner + raw results (CSV)
docs/performance_analysis.md  Written performance report with charts
docs/images/                  Benchmark visualizations
```

## Running Tests

```bash
pytest tests/ -v
```

## Running Benchmarks

```bash
python benchmarks/run_sorting_benchmarks.py
```

Regenerates the charts in `docs/images/` and the raw timing data in
`benchmarks/results/sorting_benchmarks.csv`.

## Current Progress

- [x] Week 1: Environment setup, basic sorting algorithms, benchmarking
      framework, test suite, and performance analysis
- [ ] Week 2: Sorting algorithms (advanced)
- [ ] Week 3: Search algorithms

## Author

Will Swinson
