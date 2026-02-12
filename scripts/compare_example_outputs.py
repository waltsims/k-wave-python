from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np


def collect_hdf_files(root: Path, suffixes: tuple[str, ...]) -> dict[str, Path]:
    files: dict[str, Path] = {}
    suffix_set = {suffix.lower() for suffix in suffixes}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in suffix_set:
            files[path.relative_to(root).as_posix()] = path
    return files


def dataset_paths(hf: h5py.File) -> list[str]:
    paths: list[str] = []

    def visit(name: str, obj) -> None:
        if isinstance(obj, h5py.Dataset):
            paths.append(name)

    hf.visititems(visit)
    paths.sort()
    return paths


def arrays_equal(a: np.ndarray, b: np.ndarray) -> bool:
    try:
        return bool(np.array_equal(a, b, equal_nan=True))
    except TypeError:
        return bool(np.array_equal(a, b))


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float | None:
    if np.issubdtype(a.dtype, np.floating) or np.issubdtype(a.dtype, np.complexfloating):
        diff = np.abs(a - b)
        if diff.size == 0:
            return 0.0
        return float(np.nanmax(diff))
    return None


def compare_hdf_pair(path_a: Path, path_b: Path, max_value_mismatches: int) -> list[str]:
    issues: list[str] = []
    with h5py.File(path_a, "r") as h5_a, h5py.File(path_b, "r") as h5_b:
        datasets_a = set(dataset_paths(h5_a))
        datasets_b = set(dataset_paths(h5_b))

        only_a = sorted(datasets_a - datasets_b)
        only_b = sorted(datasets_b - datasets_a)
        if only_a:
            examples = ", ".join(only_a[:5])
            issues.append(f"datasets missing in second file: {len(only_a)} (examples: {examples})")
        if only_b:
            examples = ", ".join(only_b[:5])
            issues.append(f"datasets missing in first file: {len(only_b)} (examples: {examples})")

        mismatch_count = 0
        for dataset in sorted(datasets_a & datasets_b):
            dset_a = h5_a[dataset]
            dset_b = h5_b[dataset]

            if dset_a.shape != dset_b.shape:
                issues.append(f"shape mismatch for '{dataset}': {dset_a.shape} vs {dset_b.shape}")
                continue
            if dset_a.dtype != dset_b.dtype:
                issues.append(f"dtype mismatch for '{dataset}': {dset_a.dtype} vs {dset_b.dtype}")
                continue

            value_a = dset_a[()]
            value_b = dset_b[()]
            if arrays_equal(value_a, value_b):
                continue

            mismatch_count += 1
            diff = max_abs_diff(value_a, value_b)
            if diff is None:
                issues.append(f"value mismatch for '{dataset}'")
            else:
                issues.append(f"value mismatch for '{dataset}', max_abs_diff={diff}")

            if mismatch_count >= max_value_mismatches:
                issues.append(f"stopped after {max_value_mismatches} value mismatches in this file")
                break

    return issues


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two example output roots recursively by relative HDF filename and dataset content. "
            "Metadata and attributes are not compared."
        )
    )
    parser.add_argument("first_root", type=Path, help="First output root (e.g. /tmp/example_runs)")
    parser.add_argument("second_root", type=Path, help="Second output root (e.g. /tmp/example_runs_1)")
    parser.add_argument(
        "--suffixes",
        nargs="+",
        default=[".h5", ".hdf5", ".hdf"],
        help="HDF file suffixes to include (default: .h5 .hdf5 .hdf)",
    )
    parser.add_argument(
        "--max-value-mismatches-per-file",
        type=int,
        default=5,
        help="Max per-file value mismatch entries before truncation (default: 5)",
    )
    parser.add_argument(
        "--max-report-files",
        type=int,
        default=20,
        help="Max missing/mismatched files to print in detail (default: 20)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    first_root = args.first_root
    second_root = args.second_root

    if not first_root.exists():
        print(f"error: path does not exist: {first_root}")
        return 2
    if not second_root.exists():
        print(f"error: path does not exist: {second_root}")
        return 2

    files_a = collect_hdf_files(first_root, tuple(args.suffixes))
    files_b = collect_hdf_files(second_root, tuple(args.suffixes))

    set_a = set(files_a)
    set_b = set(files_b)

    missing_in_second = sorted(set_a - set_b)
    missing_in_first = sorted(set_b - set_a)
    common_files = sorted(set_a & set_b)

    mismatched_files: list[tuple[str, list[str]]] = []
    for relative_path in common_files:
        file_a = files_a[relative_path]
        file_b = files_b[relative_path]
        try:
            issues = compare_hdf_pair(file_a, file_b, args.max_value_mismatches_per_file)
        except Exception as error:
            issues = [f"read/compare error: {error}"]

        if issues:
            mismatched_files.append((relative_path, issues))

    print("Comparison summary")
    print(f"- first root files: {len(set_a)}")
    print(f"- second root files: {len(set_b)}")
    print(f"- common files checked: {len(common_files)}")
    print(f"- missing in second: {len(missing_in_second)}")
    print(f"- missing in first: {len(missing_in_first)}")
    print(f"- mismatched files: {len(mismatched_files)}")

    if missing_in_second:
        print("\nFiles missing in second root:")
        for relative_path in missing_in_second[: args.max_report_files]:
            print(f"  - {relative_path}")
        if len(missing_in_second) > args.max_report_files:
            print(f"  ... and {len(missing_in_second) - args.max_report_files} more")

    if missing_in_first:
        print("\nFiles missing in first root:")
        for relative_path in missing_in_first[: args.max_report_files]:
            print(f"  - {relative_path}")
        if len(missing_in_first) > args.max_report_files:
            print(f"  ... and {len(missing_in_first) - args.max_report_files} more")

    if mismatched_files:
        print("\nMismatched files:")
        for relative_path, issues in mismatched_files[: args.max_report_files]:
            print(f"  - {relative_path}")
            for issue in issues:
                print(f"      * {issue}")
        if len(mismatched_files) > args.max_report_files:
            print(f"  ... and {len(mismatched_files) - args.max_report_files} more")

    has_failures = bool(missing_in_second or missing_in_first or mismatched_files)
    if has_failures:
        return 1

    print("\nAll compared HDF files and dataset contents match 1:1.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
