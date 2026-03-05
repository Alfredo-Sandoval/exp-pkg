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

TARGET_OS="${ARC_SETUP_OS:-}"
if [[ -z "${TARGET_OS}" ]]; then
  TARGET_OS="$(detect_target_os)"
fi

case "${TARGET_OS}" in
  macos | linux)
    ;;
  "")
    echo "[setup] Unsupported host OS: $(uname -s). Use environment/<os>/setup.sh directly." >&2
    exit 1
    ;;
  *)
    echo "[setup] Invalid ARC_SETUP_OS='${TARGET_OS}'. Expected 'macos' or 'linux'." >&2
    exit 1
    ;;
esac

SETUP_SCRIPT="${SCRIPT_DIR}/${TARGET_OS}/setup.sh"
if [[ ! -f "${SETUP_SCRIPT}" ]]; then
  echo "[setup] Missing setup script: ${SETUP_SCRIPT}" >&2
  exit 1
fi

echo "[setup] Using ${SETUP_SCRIPT}"
exec bash "${SETUP_SCRIPT}" "$@"
