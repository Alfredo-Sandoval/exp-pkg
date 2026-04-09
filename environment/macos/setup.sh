#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/environment.yml"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing environment spec: ${ENV_FILE}" >&2
  exit 1
fi

if command -v mamba >/dev/null 2>&1; then
  MAMBA_BIN="mamba"
elif command -v conda >/dev/null 2>&1; then
  MAMBA_BIN="conda"
else
  echo "Missing dependency: install mamba or conda." >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "Missing dependency: install uv." >&2
  exit 1
fi

ENV_NAME="$(awk -F':' '/^name:/ {gsub(/ /, "", $2); print $2; exit}' "${ENV_FILE}")"
if [[ -z "${ENV_NAME}" ]]; then
  echo "Could not parse environment name from ${ENV_FILE}. Expected 'name: <env_name>'." >&2
  exit 1
fi

if "${MAMBA_BIN}" env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  "${MAMBA_BIN}" env update -n "${ENV_NAME}" -f "${ENV_FILE}" --prune -y
else
  "${MAMBA_BIN}" env create -n "${ENV_NAME}" -f "${ENV_FILE}" -y
fi
"${MAMBA_BIN}" run -n "${ENV_NAME}" uv pip install -e "${REPO_ROOT}[dev,docs]"

echo "Environment '${ENV_NAME}' is ready."
