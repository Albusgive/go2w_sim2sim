#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RECORD_ROOT="${REPO_ROOT}/split_records"

usage() {
  cat <<'EOF'
Usage:
  render_frames_to_video.sh [record_dir] [--output PATH] [--fps N] [--format NAME] [--crf N] [--max-frames N] [--keep-temp]

Examples:
  render_frames_to_video.sh
  render_frames_to_video.sh split_records/vtm_sru_20260325_230430
  render_frames_to_video.sh split_records/vtm_sru_20260325_230430 --fps 50
  render_frames_to_video.sh split_records/vtm_sru_20260325_230430 --format webm
  render_frames_to_video.sh split_records/vtm_sru_20260325_230430 --output /tmp/render.mp4 --max-frames 300

Behavior:
  - If record_dir is omitted, the script lists available recordings and prompts for an index or full path.
  - If record_dir is a number such as 3, it is treated as the indexed recording in that list.
  - If --fps or --format is omitted, the script prompts for common options.
  - The script reconstructs one video frame per recording step using render_frames.csv.
  - If a row has no image file, the previous valid render frame is reused.
  - Output defaults to <record_dir>/render_view.<format>.
EOF
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "missing required command: $1" >&2
    exit 2
  fi
}

find_latest_record_dir() {
  if [[ ! -d "${RECORD_ROOT}" ]]; then
    echo "no split_records directory found at ${RECORD_ROOT}" >&2
    exit 2
  fi

  local latest=""
  latest="$(find "${RECORD_ROOT}" -mindepth 1 -maxdepth 1 -type d \
    -exec test -f "{}/render_frames.csv" ';' -print | sort -r | head -n 1)"

  if [[ -z "${latest}" ]]; then
    echo "no recording with render_frames.csv found under ${RECORD_ROOT}" >&2
    exit 2
  fi

  printf '%s\n' "${latest}"
}

collect_record_dirs() {
  if [[ ! -d "${RECORD_ROOT}" ]]; then
    return 0
  fi

  find "${RECORD_ROOT}" -mindepth 1 -maxdepth 1 -type d \
    -exec test -f "{}/render_frames.csv" ';' -print | sort -r
}

print_record_menu() {
  local -a candidates=("$@")
  local idx=1
  local candidate=""
  echo "Available render recordings:" >&2
  for candidate in "${candidates[@]}"; do
    local rel="${candidate#${REPO_ROOT}/}"
    echo "  [${idx}] ${rel}" >&2
    idx=$((idx + 1))
  done
}

resolve_record_selection() {
  local selection="$1"
  shift
  local -a candidates=("$@")

  if [[ -z "${selection}" ]]; then
    return 1
  fi

  if [[ "${selection}" =~ ^[0-9]+$ ]]; then
    local index=$((10#${selection}))
    if (( index < 1 || index > ${#candidates[@]} )); then
      echo "selection index out of range: ${selection}" >&2
      exit 2
    fi
    printf '%s\n' "${candidates[$((index - 1))]}"
    return 0
  fi

  printf '%s\n' "${selection}"
}

read_prompt_value() {
  local prompt_text="$1"
  local value=""
  if [[ -t 0 && -r /dev/tty ]]; then
    read -r -p "${prompt_text}" value < /dev/tty
  else
    echo -n "${prompt_text}" >&2
    read -r value
  fi
  printf '%s\n' "${value}"
}

prompt_for_record_dir() {
  local -a candidates=()
  mapfile -t candidates < <(collect_record_dirs)

  if [[ ${#candidates[@]} -eq 0 ]]; then
    echo "no recording with render_frames.csv found under ${RECORD_ROOT}" >&2
    exit 2
  fi

  print_record_menu "${candidates[@]}"
  local default_selection="1"
  local selection
  selection="$(read_prompt_value "Select recording [1-${#candidates[@]}] or enter full path (default ${default_selection}): ")"
  selection="${selection:-${default_selection}}"
  resolve_record_selection "${selection}" "${candidates[@]}"
}

resolve_fps_selection() {
  local selection="$1"
  case "${selection}" in
    1) printf '25\n' ;;
    2) printf '30\n' ;;
    3) printf '50\n' ;;
    4) printf '60\n' ;;
    "" ) printf '50\n' ;;
    *)
      if [[ "${selection}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        printf '%s\n' "${selection}"
      else
        echo "invalid fps selection: ${selection}" >&2
        exit 2
      fi
      ;;
  esac
}

prompt_for_fps() {
  echo "FPS options:" >&2
  echo "  [1] 25" >&2
  echo "  [2] 30" >&2
  echo "  [3] 50" >&2
  echo "  [4] 60" >&2
  local selection
  selection="$(read_prompt_value "Select FPS [1-4] or enter a numeric value (default 3 = 50): ")"
  selection="${selection:-3}"
  resolve_fps_selection "${selection}"
}

normalize_format_name() {
  local value="${1,,}"
  case "${value}" in
    1|mp4) printf 'mp4\n' ;;
    2|mov) printf 'mov\n' ;;
    3|webm) printf 'webm\n' ;;
    4|gif) printf 'gif\n' ;;
    "" ) printf 'mp4\n' ;;
    *)
      echo "invalid format selection: ${1}" >&2
      exit 2
      ;;
  esac
}

prompt_for_format() {
  echo "Format options:" >&2
  echo "  [1] mp4" >&2
  echo "  [2] mov" >&2
  echo "  [3] webm" >&2
  echo "  [4] gif" >&2
  local selection
  selection="$(read_prompt_value "Select format [1-4] or enter mp4/mov/webm/gif (default 1 = mp4): ")"
  selection="${selection:-1}"
  normalize_format_name "${selection}"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

require_cmd ffmpeg
require_cmd python3

record_dir=""
output_path=""
fps=""
video_format=""
crf="18"
max_frames=""
keep_temp="false"

if [[ $# -gt 0 && "${1:0:1}" != "-" ]]; then
  record_dir="$1"
  shift
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output)
      output_path="${2:?missing value for --output}"
      shift 2
      ;;
    --fps)
      fps="${2:?missing value for --fps}"
      shift 2
      ;;
    --format)
      video_format="${2:?missing value for --format}"
      shift 2
      ;;
    --crf)
      crf="${2:?missing value for --crf}"
      shift 2
      ;;
    --max-frames)
      max_frames="${2:?missing value for --max-frames}"
      shift 2
      ;;
    --keep-temp)
      keep_temp="true"
      shift
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "${record_dir}" ]]; then
  record_dir="$(prompt_for_record_dir)"
elif [[ "${record_dir}" =~ ^[0-9]+$ ]]; then
  mapfile -t _render_record_candidates < <(collect_record_dirs)
  if [[ ${#_render_record_candidates[@]} -eq 0 ]]; then
    echo "no recording with render_frames.csv found under ${RECORD_ROOT}" >&2
    exit 2
  fi
  record_dir="$(resolve_record_selection "${record_dir}" "${_render_record_candidates[@]}")"
fi

if [[ -z "${fps}" ]]; then
  fps="$(prompt_for_fps)"
else
  fps="$(resolve_fps_selection "${fps}")"
fi

if [[ -z "${video_format}" ]]; then
  video_format="$(prompt_for_format)"
else
  video_format="$(normalize_format_name "${video_format}")"
fi

record_dir="$(python3 - <<'PY' "$record_dir"
from pathlib import Path
import sys
print(Path(sys.argv[1]).resolve())
PY
)"

if [[ ! -d "${record_dir}" ]]; then
  echo "record_dir does not exist: ${record_dir}" >&2
  exit 2
fi

if [[ ! -f "${record_dir}/render_frames.csv" ]]; then
  echo "render_frames.csv not found under: ${record_dir}" >&2
  exit 2
fi

if [[ -z "${output_path}" ]]; then
  output_path="${record_dir}/render_view.${video_format}"
fi

output_path="$(python3 - <<'PY' "$output_path"
from pathlib import Path
import sys
print(Path(sys.argv[1]).resolve())
PY
)"

tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/render_video_XXXXXX")"
seq_dir="${tmp_dir}/sequence"
mkdir -p "${seq_dir}"

cleanup() {
  if [[ "${keep_temp}" != "true" ]]; then
    rm -rf "${tmp_dir}"
  fi
}
trap cleanup EXIT

python3 - <<'PY' "${record_dir}" "${seq_dir}" "${max_frames}"
import csv
import os
import sys
from pathlib import Path

record_dir = Path(sys.argv[1]).resolve()
seq_dir = Path(sys.argv[2]).resolve()
max_frames_arg = sys.argv[3].strip()
max_frames = int(max_frames_arg) if max_frames_arg else None

csv_path = record_dir / "render_frames.csv"
row_count = 0
symlink_count = 0
missing_head_rows = 0
reused_rows = 0
last_valid = None

with csv_path.open("r", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        row_count += 1
        if max_frames is not None and symlink_count >= max_frames:
            break

        rel_file = (row.get("file") or "").strip()
        src = None
        if rel_file:
            candidate = (record_dir / rel_file).resolve()
            if candidate.exists():
                src = candidate
                last_valid = candidate

        if src is None and last_valid is not None:
            src = last_valid
            reused_rows += 1
        elif src is None:
            missing_head_rows += 1
            continue

        symlink_count += 1
        dst = seq_dir / f"frame_{symlink_count:08d}.jpg"
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(src, dst)

print(f"prepared_frames={symlink_count}")
print(f"skipped_leading_missing_rows={missing_head_rows}")
print(f"reused_previous_frames={reused_rows}")
PY

frame_count="$(find "${seq_dir}" -maxdepth 1 -type l -name 'frame_*.jpg' | wc -l)"
frame_count="${frame_count// /}"

if [[ "${frame_count}" == "0" ]]; then
  echo "no usable render frames found in ${record_dir}" >&2
  exit 2
fi

mkdir -p "$(dirname "${output_path}")"

echo "record_dir: ${record_dir}"
echo "output: ${output_path}"
echo "fps: ${fps}"
echo "format: ${video_format}"
echo "frames: ${frame_count}"
if [[ "${keep_temp}" == "true" ]]; then
  echo "temp_sequence_dir: ${seq_dir}"
fi

case "${video_format}" in
  mp4|mov)
    ffmpeg \
      -y \
      -hide_banner \
      -loglevel warning \
      -framerate "${fps}" \
      -i "${seq_dir}/frame_%08d.jpg" \
      -c:v libx264 \
      -crf "${crf}" \
      -pix_fmt yuv420p \
      -movflags +faststart \
      "${output_path}"
    ;;
  webm)
    ffmpeg \
      -y \
      -hide_banner \
      -loglevel warning \
      -framerate "${fps}" \
      -i "${seq_dir}/frame_%08d.jpg" \
      -c:v libvpx-vp9 \
      -crf "${crf}" \
      -b:v 0 \
      -pix_fmt yuv420p \
      "${output_path}"
    ;;
  gif)
    ffmpeg \
      -y \
      -hide_banner \
      -loglevel warning \
      -framerate "${fps}" \
      -i "${seq_dir}/frame_%08d.jpg" \
      -vf "fps=${fps},split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" \
      "${output_path}"
    ;;
  *)
    echo "unsupported format: ${video_format}" >&2
    exit 2
    ;;
esac

echo "wrote video: ${output_path}"
