#!/usr/bin/env python3
"""Generate the fixed MuJoCo terrain matrix used for data collection.

The generated files are intentionally dataset-self-contained when
``data_collection/go2w/mjcf/go2w.xml`` exists. If that copy is missing, the
script falls back to the repository's canonical go2w MJCF source.
Run this script after changing terrain definitions, or pass ``--check`` in CI to
verify committed output is current.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "data_collection"
ROBOT_MJCF = REPO_ROOT / "robot" / "go2w_description" / "mjcf" / "go2w.xml"
DATA_COLLECTION_GO2W_MJCF = (
    OUTPUT_ROOT / "go2w" / "mjcf" / "go2w.xml"
)
ACTIVE_GO2W_MJCF = (
    DATA_COLLECTION_GO2W_MJCF
    if DATA_COLLECTION_GO2W_MJCF.is_file()
    else ROBOT_MJCF
)

FOOTPRINT = {
    "x_min": -3.0,
    "x_max": 7.0,
    "y_min": -5.0,
    "y_max": 5.0,
    "unit": "m",
}
COMMAND = {
    "name": "linv_x",
    "min": 0.5,
    "max": 1.0,
    "step": 0.05,
    "unit": "m/s",
}
EXPECTED_COUNTS = {
    "flat": 1,
    "single_platform": 9,
    "ditch": 7,
    "double_platform": 117,
}

# The deployed Go2W settles with its base_link roughly 0.358 m above its
# supporting surface.  Terminal detection only needs to reject a body that has
# fallen beside/between platforms, so keep a conservative margin below that
# nominal clearance instead of assuming a 0.45 m standing height.
MIN_BASE_CLEARANCE_M = 0.25
DITCH_FLOOR_BOX_DEPTH_M = 2.0


@dataclass(frozen=True)
class Geom:
    name: str
    geom_type: str
    pos: tuple[float, float, float]
    size: tuple[float, float, float]
    role: str
    material: str = "terrain_floor"


@dataclass(frozen=True)
class TerrainSpec:
    terrain_id: str
    task_name: str
    collect: bool
    description: str
    params: dict[str, float]
    terminal: dict[str, float] | None
    geoms: tuple[Geom, ...]

    @property
    def directory(self) -> Path:
        return OUTPUT_ROOT / self.task_name / self.terrain_id


def decimal_token(value: float) -> str:
    """Return a filename-safe, fixed two-decimal metre token (0.10 -> 0p10)."""

    return f"{value:.2f}".replace("-", "n").replace(".", "p")


def xml_number(value: float) -> str:
    value = 0.0 if abs(value) < 5e-10 else value
    return f"{value:.6f}".rstrip("0").rstrip(".") or "0"


def vector(values: Iterable[float]) -> str:
    return " ".join(xml_number(value) for value in values)


def plane_floor() -> Geom:
    return Geom(
        name="terrain_floor",
        geom_type="plane",
        pos=(2.0, 0.0, 0.0),
        size=(5.0, 5.0, 0.05),
        role="floor",
    )


def finite_floor() -> Geom:
    """Return an exact 10x10 m floor with its top surface at z=0."""

    return Geom(
        name="terrain_floor",
        geom_type="box",
        pos=(2.0, 0.0, -0.05),
        size=(5.0, 5.0, 0.05),
        role="floor",
    )


def flat_spec() -> TerrainSpec:
    return TerrainSpec(
        terrain_id="plane-flat-10x10m",
        task_name="flat",
        collect=False,
        description="Flat 10x10 m plane; generated for reference and excluded from collection.",
        params={},
        terminal=None,
        geoms=(plane_floor(),),
    )


def single_platform_specs() -> Iterable[TerrainSpec]:
    for height_cm in range(10, 51, 5):
        height = height_cm / 100.0
        terrain_id = (
            f"platform-h{decimal_token(height)}-w1p20-l2p00-10x10m"
        )
        yield TerrainSpec(
            terrain_id=terrain_id,
            task_name="single_platform",
            collect=True,
            description=(
                f"A {height:.2f} m high platform with its near edge 3.00 m "
                "in front of the robot."
            ),
            params={
                "height_m": height,
                "width_m": 1.2,
                "length_m": 2.0,
                "near_edge_x_m": 3.0,
            },
            terminal={
                "target_x": 4.0,
                "x_tolerance": 0.4,
                "min_base_z": height + MIN_BASE_CLEARANCE_M,
                "max_abs_y": 0.6,
                "stop_duration_s": 1.0,
            },
            geoms=(
                finite_floor(),
                Geom(
                    name="platform",
                    geom_type="box",
                    pos=(4.0, 0.0, height / 2.0),
                    size=(1.0, 0.6, height / 2.0),
                    role="platform",
                    material="terrain_obstacle",
                ),
            ),
        )


def ditch_floor_geoms(gap: float) -> tuple[Geom, ...]:
    """Cover the 10x10 footprint with deep boxes except for the ditch opening."""

    half_depth = DITCH_FLOOR_BOX_DEPTH_M / 2.0
    half_gap = gap / 2.0
    after_half_length = (4.0 - gap) / 2.0
    return (
        Geom(
            name="floor_before_ditch",
            geom_type="box",
            pos=(0.0, 0.0, -half_depth),
            size=(3.0, 5.0, half_depth),
            role="floor_before_ditch",
        ),
        Geom(
            name="floor_after_ditch",
            geom_type="box",
            pos=(5.0 + half_gap, 0.0, -half_depth),
            size=(after_half_length, 5.0, half_depth),
            role="floor_after_ditch",
        ),
        Geom(
            name="floor_ditch_negative_y",
            geom_type="box",
            pos=(3.0 + half_gap, -2.875, -half_depth),
            size=(half_gap, 2.125, half_depth),
            role="floor_beside_ditch",
        ),
        Geom(
            name="floor_ditch_positive_y",
            geom_type="box",
            pos=(3.0 + half_gap, 2.875, -half_depth),
            size=(half_gap, 2.125, half_depth),
            role="floor_beside_ditch",
        ),
    )


def ditch_specs() -> Iterable[TerrainSpec]:
    for gap_cm in range(30, 61, 5):
        gap = gap_cm / 100.0
        terrain_id = f"ditch-gap{decimal_token(gap)}-w1p50-10x10m"
        yield TerrainSpec(
            terrain_id=terrain_id,
            task_name="ditch",
            collect=True,
            description=(
                f"A {gap:.2f} m opening along +x and 1.50 m across y, with "
                "its near edge 3.00 m in front of the robot. The surrounding "
                f"floor boxes extend {DITCH_FLOOR_BOX_DEPTH_M:.2f} m downward."
            ),
            params={
                "gap_m": gap,
                "transverse_width_m": 1.5,
                "near_edge_x_m": 3.0,
                "floor_box_depth_m": DITCH_FLOOR_BOX_DEPTH_M,
            },
            terminal={
                "target_x": 4.0 + gap,
                "min_base_z": MIN_BASE_CLEARANCE_M,
                "max_abs_y": 0.75,
                "stop_duration_s": 1.0,
            },
            geoms=ditch_floor_geoms(gap),
        )


def double_platform_specs() -> Iterable[TerrainSpec]:
    for first_height_cm in range(10, 51, 5):
        first_height = first_height_cm / 100.0
        for height_delta_cm in range(10, 41, 10):
            height_delta = height_delta_cm / 100.0
            second_height = first_height + height_delta
            for gap_cm in range(10, 41, 10):
                if height_delta_cm + gap_cm > 60:
                    continue
                gap = gap_cm / 100.0
                terrain_id = (
                    f"double-platform-h1{decimal_token(first_height)}-"
                    f"h2{decimal_token(second_height)}-"
                    f"gap{decimal_token(gap)}-w1p20-l1p00-10x10m"
                )
                yield TerrainSpec(
                    terrain_id=terrain_id,
                    task_name="double_platform",
                    collect=True,
                    description=(
                        f"Two 1.00 m platforms separated by {gap:.2f} m: "
                        f"heights {first_height:.2f} m and {second_height:.2f} m."
                    ),
                    params={
                        "first_height_m": first_height,
                        "height_delta_m": height_delta,
                        "second_height_m": second_height,
                        "gap_m": gap,
                        "platform_width_m": 1.2,
                        "platform_length_m": 1.0,
                        "near_edge_x_m": 3.0,
                    },
                    terminal={
                        "target_x": 4.5 + gap,
                        "x_tolerance": 0.3,
                        "min_base_z": second_height + MIN_BASE_CLEARANCE_M,
                        "max_abs_y": 0.6,
                        "stop_duration_s": 1.0,
                    },
                    geoms=(
                        finite_floor(),
                        Geom(
                            name="first_platform",
                            geom_type="box",
                            pos=(3.5, 0.0, first_height / 2.0),
                            size=(0.5, 0.6, first_height / 2.0),
                            role="first_platform",
                            material="terrain_obstacle_low",
                        ),
                        Geom(
                            name="second_platform",
                            geom_type="box",
                            pos=(4.5 + gap, 0.0, second_height / 2.0),
                            size=(0.5, 0.6, second_height / 2.0),
                            role="second_platform",
                            material="terrain_obstacle",
                        ),
                    ),
                )


def all_specs() -> tuple[TerrainSpec, ...]:
    specs = (
        flat_spec(),
        *single_platform_specs(),
        *ditch_specs(),
        *double_platform_specs(),
    )
    counts = {
        task: sum(spec.task_name == task for spec in specs)
        for task in EXPECTED_COUNTS
    }
    if counts != EXPECTED_COUNTS:
        raise RuntimeError(f"terrain matrix count mismatch: {counts}")
    if len({spec.terrain_id for spec in specs}) != len(specs):
        raise RuntimeError("terrain identifiers are not unique")
    return specs


def terrain_xml(spec: TerrainSpec) -> str:
    include_path = os.path.relpath(ACTIVE_GO2W_MJCF, spec.directory).replace(
        os.sep, "/"
    )
    mesh_directory = os.path.relpath(
        ACTIVE_GO2W_MJCF.parent / "assets", spec.directory
    ).replace(os.sep, "/")
    root = ET.Element("mujoco", {"model": f"data collection {spec.terrain_id}"})
    ET.SubElement(root, "include", {"file": include_path})
    # The compiler from an included MJCF does not retain its meshdir under MuJoCo
    # 3.4/3.5.  Repeat the repository-relative directory after the include so it
    # wins while remaining portable when the repository is moved as a unit.
    ET.SubElement(
        root,
        "compiler",
        {"angle": "radian", "autolimits": "true", "meshdir": mesh_directory},
    )
    ET.SubElement(root, "statistic", {"center": "2 0 0.5", "extent": "6"})

    visual = ET.SubElement(root, "visual")
    ET.SubElement(
        visual,
        "headlight",
        {"diffuse": "0.7 0.7 0.7", "ambient": "0.3 0.3 0.3", "specular": "0 0 0"},
    )
    ET.SubElement(visual, "rgba", {"haze": "0.15 0.22 0.30 1"})
    ET.SubElement(visual, "global", {"azimuth": "-125", "elevation": "-20"})

    asset = ET.SubElement(root, "asset")
    ET.SubElement(
        asset,
        "texture",
        {
            "name": "terrain_checker",
            "type": "2d",
            "builtin": "checker",
            "mark": "edge",
            "rgb1": "0.22 0.30 0.38",
            "rgb2": "0.12 0.18 0.24",
            "markrgb": "0.75 0.82 0.88",
            "width": "512",
            "height": "512",
        },
    )
    ET.SubElement(
        asset,
        "material",
        {
            "name": "terrain_floor",
            "texture": "terrain_checker",
            "texuniform": "true",
            "texrepeat": "10 10",
            "reflectance": "0.05",
        },
    )
    ET.SubElement(
        asset,
        "material",
        {"name": "terrain_obstacle", "rgba": "0.42 0.62 0.82 1"},
    )
    ET.SubElement(
        asset,
        "material",
        {"name": "terrain_obstacle_low", "rgba": "0.55 0.72 0.88 1"},
    )

    defaults = ET.SubElement(root, "default")
    terrain_default = ET.SubElement(defaults, "default", {"class": "terrain_collision"})
    ET.SubElement(
        terrain_default,
        "geom",
        {
            "friction": "0.8 0.02 0.01",
            "condim": "6",
            "solimp": "0.9 0.95 0.001",
            "solref": "0.02 1",
        },
    )

    worldbody = ET.SubElement(root, "worldbody")
    ET.SubElement(
        worldbody,
        "light",
        {"pos": "2 0 5", "dir": "0 0 -1", "directional": "true"},
    )
    for geom in spec.geoms:
        ET.SubElement(
            worldbody,
            "geom",
            {
                "name": geom.name,
                "type": geom.geom_type,
                "pos": vector(geom.pos),
                "size": vector(geom.size),
                "material": geom.material,
                "class": "terrain_collision",
            },
        )

    ET.indent(root, space="  ")
    return '<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(
        root, encoding="unicode", short_empty_elements=True
    ) + "\n"


def terrain_metadata(spec: TerrainSpec) -> str:
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "terrain_id": spec.terrain_id,
        "task_name": spec.task_name,
        "collect": spec.collect,
        "description": spec.description,
        "footprint": FOOTPRINT,
        "params": spec.params,
        "command": COMMAND if spec.collect else None,
        "terminal": spec.terminal,
        "geometry": [
            {
                "name": geom.name,
                "type": geom.geom_type,
                "role": geom.role,
                "pos_m": list(geom.pos),
                "size_half_extents_m": list(geom.size),
            }
            for geom in spec.geoms
        ],
    }
    return json.dumps(clean_json_numbers(metadata), ensure_ascii=False, indent=2) + "\n"


def clean_json_numbers(value: Any) -> Any:
    """Remove binary floating-point noise from human- and machine-facing metadata."""

    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {key: clean_json_numbers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_json_numbers(item) for item in value]
    return value


def readme() -> str:
    return """# MuJoCo data-collection terrains

This directory is generated by `tools/generate_data_collection_terrains.py`.
Do not hand-edit `terrain.xml` or `terrain.json`; change the generator and rerun it.

## Coordinate and naming conventions

- The Go2W starts facing `+x`. Each nominal terrain footprint is 10 x 10 m:
  `x=[-3, 7]`, `y=[-5, 5]`.
- An obstacle's near edge is at `x=3.00 m`; width is measured across `y` and
  length/gap is measured along `x`.
- Decimal values in directory and key names use `p`, for example `0p25 = 0.25 m`.
- A collected key file is named `<task_name>-cmd_linv_x_<speed>.xml`, for example
  `single_platform-cmd_linv_x_0p50.xml`.
- Every `terrain.xml` includes the bundled Go2W model under `data_collection/go2w/mjcf/go2w.xml`
  so moving the generated dataset directory does not invalidate the model path.

## Fixed terrain matrix

| Task directory | Terrain matrix | Count | Collection |
| --- | --- | ---: | --- |
| `flat` | one `plane` (`plane-flat-10x10m`) | 1 | no |
| `single_platform` | height 0.10-0.50 m in 0.05 m steps; width 1.20 m; length 2.00 m | 9 | yes |
| `ditch` | true opening through 2.00 m-deep box slabs; gap 0.30-0.60 m along `x` in 0.05 m steps; width 1.50 m across `y` | 7 | yes |
| `double_platform` | first height 0.10-0.50 m/0.05 m; height delta and gap each 0.10-0.40 m/0.10 m, constrained by delta + gap <= 0.60 m | 117 | yes |

There are 134 terrain directories. The 133 collected terrains each use 11
`linv_x` commands from 0.50 through 1.00 m/s in 0.05 m/s steps, for 1,463
expected collection items.

Directory names encode all varying geometry. Examples:

- `platform-h0p10-w1p20-l2p00-10x10m`
- `ditch-gap0p30-w1p50-10x10m`
- `double-platform-h10p10-h20p20-gap0p10-w1p20-l1p00-10x10m`

## Metadata contract

Each `terrain.json` contains stable `task_name`, `terrain_id`, `collect`, `params`,
`command`, `terminal`, and `geometry` fields. Collector terminal conditions use
`target_x`, `min_base_z`, and `max_abs_y`. Platform tasks also use `x_tolerance`
to define a center band, while ditch tasks use `target_x` as a one-sided minimum.
After reaching the terminal region, the command is zeroed for `stop_duration_s`
and support is retained throughout the stop. MuJoCo `box` sizes in XML and
`size_half_extents_m` in JSON are half extents, as required by MJCF.

Regenerate or verify committed output with:

```bash
python3 tools/generate_data_collection_terrains.py
python3 tools/generate_data_collection_terrains.py --check
```

## Batch collection and report

After building `mujoco_data_collector`, run the complete resumable matrix and
regenerate the Markdown/radar report with four worker processes (the default):

```bash
python3 tools/recollect_all_near_reset.py --workers 4
```

The formal collection profile uses `vtm_lstm_sru` and resets recurrent
hidden/cell state once when the robot reaches 1 m before the terrain edge. Its
protocol fingerprint invalidates older no-reset results, while completed jobs
under the new profile remain resume points after interruption. Use
`tools/run_data_collection.py` directly with `--dry-run`, `--task`,
`--terrain-id`, or `--speed` to inspect or restrict the matrix. Partial
attempts, corrupt XML, and infrastructure errors are retried.

The runner optionally loads `data_collection/pid_config.json` using the schema
written by `tools/calibrate_straight_pid.py`. If the file is absent it uses the
built-in path-heading PID defaults printed by `--dry-run`. The complete explicit
configuration participates in every job fingerprint.

Calibrate one fixed PID on the flat reference terrain at 0.50, 0.75, and
1.00 m/s before a new batch with:

```bash
python3 tools/calibrate_straight_pid.py \\
  --binary mujoco/C++/build_onnx/mujoco_data_collector \\
  --workers 4
```

## Visual collection and replay UI

After building both `mujoco_data_collector` and `mujoco_key_replayer`, launch
the local trajectory browser:

```bash
python3 tools/data_collection_ui.py
```

The UI selects one exact task / terrain / speed tuple. **Visual collect** runs
the same `vtm_lstm_sru` collector and terminal logic at real-time pace in a
MuJoCo window. **Replay** restores the selected MJCF key sequence exactly; its
buttons control play/pause, frame stepping, seeking, looping, and playback rate.
Keys whose input fingerprint is stale are not replayed because they reference
the current `terrain.xml`, which may no longer match the collected geometry.
Use `python3 tools/data_collection_ui.py --check` to validate paths without
opening a window. Override `--collector-binary` and `--replay-binary` when the
build directory is nonstandard.

Generate or refresh the local Markdown report and its SVG radar charts with:

```bash
python3 tools/generate_data_collection_report.py
```

Key XMLs, `.collection_status/`, `collection_report.md`, and `report_assets/`
are intentionally local and ignored by Git.

## 独立打包数据集

生成的 `data_collection` 可直接打包为一个独立归档，包含地形、状态、密钥轨迹和
go2w 模型资源。打包后移动到新机器仍可直接重放：

```bash
python3 tools/package_data_collection_dataset.py --output /tmp/go2w-data-collection.tar.gz
```

默认打包目录名为 `go2w_data_collection_bundle`，输出到当前目录。默认行为：

- 打包 `*.xml` 键值轨迹（`single_platform/ditch/double_platform` 的
  `*-cmd_linv_x_*.xml`）；
- 不包含以 `.` 开头的隐藏文件（如 `.collection_status`）；
- 自动加入打包清单 `PACKAGE_INFO.json` 与 `README.txt`，用于在新环境快速校验。

如果你要连同收集状态一起打包，增加：

```bash
python3 tools/package_data_collection_dataset.py --include-hidden --output /tmp/go2w-data-collection-full.tar.gz
```

如果只导出地形与状态，不导出轨迹：

```bash
python3 tools/package_data_collection_dataset.py --no-keys --output /tmp/go2w-data-collection-metadata-only.tar.gz
```
"""


def expected_files(specs: Iterable[TerrainSpec]) -> dict[Path, str]:
    files = {OUTPUT_ROOT / "README.md": readme()}
    for spec in specs:
        files[spec.directory / "terrain.xml"] = terrain_xml(spec)
        files[spec.directory / "terrain.json"] = terrain_metadata(spec)
    return files


def check_files(files: dict[Path, str]) -> int:
    problems: list[str] = []
    for path, expected in files.items():
        if not path.is_file():
            problems.append(f"missing: {path.relative_to(REPO_ROOT)}")
            continue
        actual = path.read_text(encoding="utf-8")
        if actual != expected:
            problems.append(f"stale: {path.relative_to(REPO_ROOT)}")

    expected_terrain_files = {
        path.resolve()
        for path in files
        if path.name in {"terrain.xml", "terrain.json"}
    }
    if OUTPUT_ROOT.is_dir():
        actual_terrain_files = {
            path.resolve()
            for pattern in ("terrain.xml", "terrain.json")
            for path in OUTPUT_ROOT.glob(f"*/*/{pattern}")
        }
        for path in sorted(actual_terrain_files - expected_terrain_files):
            problems.append(f"unexpected: {path.relative_to(REPO_ROOT)}")

    if problems:
        print("Generated terrain files are not current:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    print(f"OK: {len(files) - 1} terrain files plus data_collection/README.md")
    return 0


def write_files(files: dict[Path, str]) -> None:
    changed = 0
    for path, content in files.items():
        if path.is_file() and path.read_text(encoding="utf-8") == content:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        changed += 1
    print(f"Generated {len(files) - 1} terrain files plus README ({changed} changed).")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify generated files without modifying them",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not ACTIVE_GO2W_MJCF.is_file():
        print(
            "Go2W MJCF not found in data_collection or robot/go2w_description: "
            f"{DATA_COLLECTION_GO2W_MJCF}, {ROBOT_MJCF}",
            file=sys.stderr,
        )
        return 2
    specs = all_specs()
    files = expected_files(specs)
    if args.check:
        return check_files(files)
    write_files(files)
    return check_files(files)


if __name__ == "__main__":
    raise SystemExit(main())
