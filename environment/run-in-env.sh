#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

detect_target_os() {
  local uname_s
  uname_s="$(uname -s)"
  case "${uname_s}" in
    Darwin)
      echo "macos"
      ;;
    Linux)
      echo "linux"
      ;;
    *)
      echo ""
      ;;
  esac
}

if [[ "$#" -eq 0 ]]; then
  echo "[run-in-env] Usage: environment/run-in-env.sh <command> [args...]" >&2
  exit 1
fi

TARGET_OS="${ARC_SETUP_OS:-}"
if [[ -z "${TARGET_OS}" ]]; then
  TARGET_OS="$(detect_target_os)"
fi

case "${TARGET_OS}" in
  macos | linux)
    ;;
  "")
    echo "[run-in-env] Unsupported host OS: $(uname -s)." >&2
    exit 1
    ;;
  *)
    echo "[run-in-env] Invalid ARC_SETUP_OS='${TARGET_OS}'. Expected 'macos' or 'linux'." >&2
    exit 1
    ;;
esac

ENV_FILE="${SCRIPT_DIR}/${TARGET_OS}/environment.yml"
if [[ ! -f "${ENV_FILE}" ]]; then
  echo "[run-in-env] Missing environment spec: ${ENV_FILE}" >&2
  exit 1
fi

ENV_NAME="$(awk -F':' '/^name:/ {gsub(/ /, "", $2); print $2; exit}' "${ENV_FILE}")"
if [[ -z "${ENV_NAME}" ]]; then
  echo "[run-in-env] Could not parse environment name from ${ENV_FILE}." >&2
  exit 1
fi

if [[ "${CONDA_DEFAULT_ENV:-}" == "${ENV_NAME}" ]]; then
  exec "$@"
fi

if command -v mamba >/dev/null 2>&1; then
  RUNNER="mamba"
elif command -v conda >/dev/null 2>&1; then
  RUNNER="conda"
else
  echo "[run-in-env] Missing dependency: install mamba or conda, then run 'make env'." >&2
  exit 1
fi

if ! "${RUNNER}" env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  echo "[run-in-env] Environment '${ENV_NAME}' not found. Run 'make env' first." >&2
  exit 1
fi

exec "${RUNNER}" run -n "${ENV_NAME}" "$@"
