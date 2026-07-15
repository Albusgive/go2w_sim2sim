#!/usr/bin/env python3
"""Create a standalone archive for the generated data_collection dataset."""

from __future__ import annotations

import argparse
import json
import tarfile
import re
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default="data_collection",
        help="Dataset root directory to package (default: data_collection)",
    )
    parser.add_argument(
        "--output",
        help="Output archive path, default: go2w_data_collection_<UTC timestamp>.tar.gz",
    )
    parser.add_argument(
        "--root-name",
        default="go2w_data_collection_bundle",
        help="Top-level directory name inside the archive (default: go2w_data_collection_bundle)",
    )
    parser.add_argument(
        "--include-hidden",
        action="store_true",
        help="Include hidden files under data_collection (e.g. .collection_status)",
    )
    parser.add_argument(
        "--no-keys",
        action="store_true",
        help="Exclude collected key trajectories and keep terrain/status metadata only",
    )
    return parser.parse_args()


KEY_FILE = re.compile(r"^(?:(?:single_platform|ditch|double_platform)-cmd_linv_x_)")


def iter_files(root: Path, include_hidden: bool, include_keys: bool) -> Iterable[Path]:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if not include_hidden and any(part.startswith(".") for part in path.parts):
            continue
        if not include_keys and path.suffix == ".xml" and KEY_FILE.match(path.name):
            continue
        yield path


def default_output_path(root: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path(f"go2w_data_collection_bundle_{stamp}.tar.gz")


def main() -> int:
    args = parse_args()
    source = Path(args.source).resolve()
    if not source.is_dir():
        print(f"Source directory does not exist: {source}")
        return 2

    include_keys = not args.no_keys
    output = Path(args.output).resolve() if args.output else default_output_path(source)
    root_name = args.root_name.strip().rstrip("/")
    if not root_name:
        root_name = "go2w_data_collection_bundle"

    files = list(iter_files(source, args.include_hidden, include_keys))
    if not files:
        print(f"No files found under {source}")
        return 1

    with tarfile.open(output, "w:gz") as archive:
        generated = datetime.now(timezone.utc)
        for path in files:
            arc = f"{root_name}/{path.relative_to(source)}"
            archive.add(path, arcname=arc)

        manifest_body = {
            "source": str(source),
            "generated_utc": generated.isoformat(timespec="seconds"),
            "root_name": root_name,
            "format": "tar.gz",
            "file_count": len(files),
            "include_hidden": args.include_hidden,
            "include_keys": include_keys,
            "output": str(output),
            "version": 1,
        }

        manifest_name = f"{root_name}/PACKAGE_INFO.json"
        manifest_data = json.dumps(manifest_body, ensure_ascii=False, indent=2).encode(
            "utf-8"
        ) + b"\n"
        manifest = tarfile.TarInfo(manifest_name)
        manifest.size = len(manifest_data)
        archive.addfile(manifest, BytesIO(manifest_data))

        readme_text = (
            "Go2W data-collection dataset bundle\n"
            "================================\n"
            f"source={source}\n"
            f"generated_utc={manifest_body['generated_utc']}\n"
            f"root_name={root_name}\n"
            f"include_hidden={args.include_hidden}\n"
            f"include_keys={include_keys}\n"
            f"file_count={len(files)}\n"
        )
        readme_name = f"{root_name}/README.txt"
        readme_bytes = readme_text.encode("utf-8")
        readme = tarfile.TarInfo(readme_name)
        readme.size = len(readme_bytes)
        archive.addfile(readme, BytesIO(readme_bytes))

    print(f"Packaged {len(files)} files to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
