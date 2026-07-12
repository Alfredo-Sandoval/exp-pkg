#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
HOST_CHECK=0

usage() {
  cat <<'EOF'
Usage: environment/setup.sh [--host-check]

Create or update the xpkg conda environment from environment/environment.yml,
then install this repository editably with the dev and docs extras.

Options:
  --host-check      Print host and environment metadata, then exit.
  -h, --help        Show this help text.
EOF
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --host-check)
      HOST_CHECK=1
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "[setup] Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

ENV_FILE="${SCRIPT_DIR}/environment.yml"
if [[ ! -f "${ENV_FILE}" ]]; then
  echo "[setup] Missing environment spec: ${ENV_FILE}" >&2
  exit 1
fi

ENV_NAME="$(awk -F':' '/^name:/ {gsub(/ /, "", $2); print $2; exit}' "${ENV_FILE}")"
if [[ -z "${ENV_NAME}" ]]; then
  echo "[setup] Could not parse environment name from ${ENV_FILE}. Expected 'name: <env_name>'." >&2
  exit 1
fi

if [[ "${HOST_CHECK}" -eq 1 ]]; then
  echo "host_os=$(uname -s)"
  echo "environment_spec=${ENV_FILE}"
  echo "environment_name=${ENV_NAME}"
  exit 0
fi

if command -v mamba >/dev/null 2>&1; then
  MAMBA_BIN="mamba"
elif command -v conda >/dev/null 2>&1; then
  MAMBA_BIN="conda"
else
  echo "[setup] Missing dependency: install mamba or conda." >&2
  exit 1
fi

if command -v conda >/dev/null 2>&1; then
  ENV_LIST_BIN="conda"
else
  ENV_LIST_BIN="${MAMBA_BIN}"
fi

echo "[setup] Using ${ENV_FILE}"
if "${MAMBA_BIN}" env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  "${MAMBA_BIN}" env update -n "${ENV_NAME}" -f "${ENV_FILE}" --prune -y
else
  "${MAMBA_BIN}" env create -n "${ENV_NAME}" -f "${ENV_FILE}" -y
fi

ENV_PREFIX="$(
  "${ENV_LIST_BIN}" env list --json | python -c '
import json
import pathlib
import sys

env_name = sys.argv[1]
payload = json.load(sys.stdin)
for path in payload.get("envs", []):
    candidate = pathlib.Path(path)
    if candidate.name == env_name:
        print(candidate)
        break
else:
    raise SystemExit(1)
' "${ENV_NAME}"
)"
if [[ -z "${ENV_PREFIX}" ]]; then
  echo "[setup] Could not resolve environment prefix for '${ENV_NAME}'." >&2
  exit 1
fi

UV_BIN="${ENV_PREFIX}/bin/uv"
PYTHON_BIN="${ENV_PREFIX}/bin/python"
if [[ ! -x "${UV_BIN}" ]]; then
  echo "[setup] Missing uv binary in environment: ${UV_BIN}" >&2
  exit 1
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[setup] Missing Python binary in environment: ${PYTHON_BIN}" >&2
  exit 1
fi

"${UV_BIN}" pip install --python "${PYTHON_BIN}" -e "${REPO_ROOT}[dev,docs]"

echo "[setup] Environment '${ENV_NAME}' is ready."
