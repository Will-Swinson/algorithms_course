# File: scripts/check_environment.py
"""
Environment check script for the Advanced Algorithms course.

Verifies that:
1. Python is version 3.9 or later
2. All required packages are installed and importable
3. The expected project directory structure exists

Run from the project root:
    python scripts/check_environment.py

Exits with status 0 if everything passes, 1 otherwise.
"""
import importlib
import sys
from pathlib import Path

REQUIRED_PYTHON = (3, 9)
REQUIRED_PACKAGES = ["numpy", "matplotlib", "pandas", "pytest"]
REQUIRED_DIRS = [
    "src/sorting",
    "src/searching",
    "src/graph",
    "src/dynamic_programming",
    "src/data_structures",
    "src/utils",
    "tests",
    "benchmarks",
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def check_python_version() -> bool:
    """Verify the running interpreter meets the minimum version."""
    current = sys.version_info[:2]
    ok = current >= REQUIRED_PYTHON
    status = "OK " if ok else "FAIL"
    print(f"[{status}] Python {sys.version.split()[0]} "
          f"(need >= {REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]}) "
          f"at {sys.executable}")
    return ok


def check_packages() -> bool:
    """Verify each required package imports and report its version."""
    all_ok = True
    for name in REQUIRED_PACKAGES:
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", "unknown")
            print(f"[OK ] {name} {version}")
        except ImportError as error:
            print(f"[FAIL] {name} - not importable ({error})")
            all_ok = False
    return all_ok


def check_project_structure() -> bool:
    """Verify the expected package directories exist."""
    all_ok = True
    for relative in REQUIRED_DIRS:
        path = PROJECT_ROOT / relative
        if path.is_dir():
            print(f"[OK ] {relative}/")
        else:
            print(f"[FAIL] {relative}/ - missing")
            all_ok = False
    return all_ok


def main() -> int:
    print("=" * 60)
    print("Algorithm Laboratory - Environment Check")
    print("=" * 60)

    print("\nPython version:")
    python_ok = check_python_version()

    print("\nRequired packages:")
    packages_ok = check_packages()

    print("\nProject structure:")
    structure_ok = check_project_structure()

    print("\n" + "=" * 60)
    if python_ok and packages_ok and structure_ok:
        print("Environment check PASSED - your laboratory is ready.")
        return 0
    print("Environment check FAILED - fix the items marked FAIL above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
