#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RECORD_ROOT="${REPO_ROOT}/split_records"
LEGACY_RECORD_ROOTS=(
  "${REPO_ROOT}/build/split_records"
  "${REPO_ROOT}/build_onnx/split_records"
)
PYTHON_SCRIPT="${REPO_ROOT}/tools/play_split_record_heatmaps.py"
DEPLOY_JSON="${REPO_ROOT}/policy/vtm_sru/student_deploy.json"

usage() {
  cat <<'EOF'
Usage:
  open_split_record_heatmaps.sh [record_dir] [extra python args...]

Examples:
  open_split_record_heatmaps.sh
  open_split_record_heatmaps.sh /path/to/split_records/vtm_sru_20260325_184212
  open_split_record_heatmaps.sh --focus-mark manual_mark_0003 --window 160
  open_split_record_heatmaps.sh /path/to/record --focus-mark all --stride 2

Behavior:
  - If record_dir is omitted, the launcher uses the latest available recording.
  - If --focus-mark is omitted, the launcher defaults to --focus-mark all.
  - If --open is omitted, the launcher defaults to --open.
  - The launcher pre-generates overview viewers for all known recordings so the
    HTML UI can switch trajectories directly.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

record_dir=""
if [[ $# -gt 0 && "${1:0:1}" != "-" ]]; then
  record_dir="$1"
  shift
fi

collect_record_dirs() {
  local search_roots=()
  if [[ -d "${RECORD_ROOT}" ]]; then
    search_roots+=("${RECORD_ROOT}")
  fi
  for legacy_root in "${LEGACY_RECORD_ROOTS[@]}"; do
    if [[ -d "${legacy_root}" ]]; then
      search_roots+=("${legacy_root}")
    fi
  done

  local dirs=()
  local search_root=""
  while IFS= read -r search_root; do
    [[ -z "${search_root}" ]] && continue
    while IFS= read -r dir; do
      [[ -z "${dir}" ]] && continue
      dirs+=("${dir}")
    done < <(find "${search_root}" -mindepth 1 -maxdepth 1 -type d | sort)
  done < <(printf '%s\n' "${search_roots[@]}")

  if [[ ${#dirs[@]} -eq 0 ]]; then
    return 0
  fi

  printf '%s\n' "${dirs[@]}" | sort -r
}

select_default_record_dir() {
  local -a candidates=()
  mapfile -t candidates < <(collect_record_dirs)

  if [[ ${#candidates[@]} -eq 0 ]]; then
    echo "no recording directories found under repo split_records roots" >&2
    exit 2
  fi
  printf '%s\n' "${candidates[0]}"
}

if [[ -z "${record_dir}" ]]; then
  record_dir="$(select_default_record_dir)"
fi

if [[ ! -d "${record_dir}" ]]; then
  echo "record_dir does not exist: ${record_dir}" >&2
  exit 2
fi

has_focus_mark=false
has_open=false
for arg in "$@"; do
  if [[ "${arg}" == "--focus-mark" ]]; then
    has_focus_mark=true
  fi
  if [[ "${arg}" == "--open" ]]; then
    has_open=true
  fi
done

generate_overview_viewers() {
  local -a candidates=()
  mapfile -t candidates < <(collect_record_dirs)
  local dir=""
  for dir in "${candidates[@]}"; do
    python3 "${PYTHON_SCRIPT}" \
      "${dir}" \
      --deploy-json "${DEPLOY_JSON}" \
      --focus-mark all \
      >/dev/null
  done
}

generate_overview_viewers

cmd=(
  python3
  "${PYTHON_SCRIPT}"
  "${record_dir}"
  --deploy-json
  "${DEPLOY_JSON}"
)

if [[ "${has_focus_mark}" == false ]]; then
  cmd+=(--focus-mark all)
fi
if [[ "${has_open}" == false ]]; then
  cmd+=(--open)
fi

cmd+=("$@")

echo "record_dir: ${record_dir}"
echo "launch: ${cmd[*]}"
exec "${cmd[@]}"
