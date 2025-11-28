#!/usr/bin/env python
import os
import argparse
from pathlib import Path

# Which extensions to include in the dump
INCLUDE_EXT = {
    ".py",
    ".md",
    ".txt",
    ".json",
    ".toml",
    ".yaml",
    ".yml",
}

# Default directory names to skip (by name, anywhere in the path)
DEFAULT_SKIP_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
    ".mypy_cache",
    ".pytest_cache",
    "old",
}

# Rough heuristic: average characters per token
CHARS_PER_TOKEN = 4.0


def should_skip_dir(path: Path, skip_dirs: set[str]) -> bool:
    """Return True if any part of the path is in skip_dirs."""
    return any(part in skip_dirs for part in path.parts)


def build_repo_map(root: Path, out_path: Path, skip_dirs: set[str]) -> int:
    """
    Build the repository map and write to out_path.
    Returns the total number of characters written.
    """
    total_chars = 0

    with out_path.open("w", encoding="utf-8") as f:

        def w(s: str):
            nonlocal total_chars
            f.write(s)
            total_chars += len(s)

        w(f"Repository map for: {root}\n\n")
        w("===== DIRECTORY TREE =====\n\n")

        # 1) DIRECTORY TREE
        for dirpath, dirnames, filenames in os.walk(root):
            dirpath = Path(dirpath)
            rel_dir = dirpath.relative_to(root)

            if should_skip_dir(rel_dir, skip_dirs):
                dirnames[:] = []  # don't descend further
                continue

            indent = "  " * len(rel_dir.parts)
            # rel_dir == Path('.') is a bit annoying, so special-case root
            dir_label = "." if rel_dir == Path(".") else str(rel_dir)
            w(f"{indent}{dir_label}/\n")
            for name in sorted(filenames):
                w(f"{indent}  {name}\n")

        w("\n\n===== FILE CONTENTS =====\n\n")

        # 2) FILE CONTENTS
        for dirpath, dirnames, filenames in os.walk(root):
            dirpath = Path(dirpath)
            rel_dir = dirpath.relative_to(root)

            if should_skip_dir(rel_dir, skip_dirs):
                dirnames[:] = []
                continue

            for name in sorted(filenames):
                path = dirpath / name

                # Filter by extension
                if path.suffix not in INCLUDE_EXT:
                    continue

                rel_path = path.relative_to(root)
                w(f"\n\n===== FILE: {rel_path} =====\n\n")

                try:
                    text = path.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    w("[Skipping binary or non-UTF8 file]\n")
                    continue

                w(text)

    return total_chars


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a text dump of a repository structure and contents."
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="Root directory of the repository (default: current directory).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="repo_map.txt",
        help="Output file path (default: repo_map.txt).",
    )
    parser.add_argument(
        "--skip-dir",
        action="append",
        default=[],
        help=(
            "Directory name to skip (can be used multiple times). "
            "Example: --skip-dir __pycache__ --skip-dir logs"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    root = Path(args.root).resolve()
    out_path = Path(args.output).resolve()

    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Root directory does not exist or is not a directory: {root}")

    # Merge default skip dirs with user-provided ones
    skip_dirs = set(DEFAULT_SKIP_DIRS) | set(args.skip_dir)

    total_chars = build_repo_map(root, out_path, skip_dirs)
    approx_tokens = total_chars / CHARS_PER_TOKEN

    print(f"Repo map written to: {out_path}")
    print(
        f"Approximate tokens for the whole file: ~{approx_tokens:.0f} "
        f"(assuming ~{CHARS_PER_TOKEN:.1f} chars/token)"
    )
    print(f"Skipped directory names: {sorted(skip_dirs)}")


if __name__ == "__main__":
    main()
