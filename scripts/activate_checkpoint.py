from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Activate an exported checkpoint as models/latest.")
    parser.add_argument("--source", required=True, help="Path to an exported checkpoint directory.")
    parser.add_argument(
        "--target",
        default="models/latest",
        help="Target path to expose as the active checkpoint.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy the checkpoint instead of creating a symlink.",
    )
    return parser


def remove_existing(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    source = Path(args.source).resolve()
    target = Path(args.target)

    if not source.is_dir():
        raise SystemExit(f"Checkpoint source does not exist or is not a directory: {source}")

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        remove_existing(target)

    if args.copy:
        shutil.copytree(source, target)
        print(f"copied {source} -> {target}")
    else:
        target.symlink_to(source, target_is_directory=True)
        print(f"linked {target} -> {source}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
