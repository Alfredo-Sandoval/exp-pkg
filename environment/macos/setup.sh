#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/environment.yml"
REQ_FILE="${SCRIPT_DIR}/requirements.txt"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

install_js_dependencies_if_present() {
  local package_json
  package_json="${REPO_ROOT}/package.json"
  if [[ ! -f "${package_json}" ]]; then
    return 0
  fi

  if [[ -f "${REPO_ROOT}/pnpm-lock.yaml" ]]; then
    if ! "${MAMBA_BIN}" run -n "${ENV_NAME}" pnpm --version >/dev/null 2>&1; then
      echo "Missing dependency in environment '${ENV_NAME}': pnpm (required by pnpm-lock.yaml)." >&2
      exit 1
    fi
    (
      cd "${REPO_ROOT}"
      "${MAMBA_BIN}" run -n "${ENV_NAME}" pnpm install --frozen-lockfile
    )
    return 0
  fi

  if [[ -f "${REPO_ROOT}/package-lock.json" ]]; then
    if ! "${MAMBA_BIN}" run -n "${ENV_NAME}" npm --version >/dev/null 2>&1; then
      echo "Missing dependency in environment '${ENV_NAME}': npm (required by package-lock.json)." >&2
      exit 1
    fi
    (
      cd "${REPO_ROOT}"
      "${MAMBA_BIN}" run -n "${ENV_NAME}" npm ci
    )
    return 0
  fi

  if [[ -f "${REPO_ROOT}/yarn.lock" ]]; then
    if ! "${MAMBA_BIN}" run -n "${ENV_NAME}" yarn --version >/dev/null 2>&1; then
      echo "Missing dependency in environment '${ENV_NAME}': yarn (required by yarn.lock)." >&2
      exit 1
    fi
    (
      cd "${REPO_ROOT}"
      "${MAMBA_BIN}" run -n "${ENV_NAME}" yarn install --frozen-lockfile
    )
    return 0
  fi

  echo "Detected package.json without lockfile. Running npm install for initial lock generation." >&2
  if ! "${MAMBA_BIN}" run -n "${ENV_NAME}" npm --version >/dev/null 2>&1; then
    echo "Missing dependency in environment '${ENV_NAME}': npm (required to install package.json dependencies)." >&2
    exit 1
  fi
  (
    cd "${REPO_ROOT}"
    "${MAMBA_BIN}" run -n "${ENV_NAME}" npm install
  )
}

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing environment spec: ${ENV_FILE}" >&2
  exit 1
fi

if [[ ! -f "${REQ_FILE}" ]]; then
  echo "Missing requirements file: ${REQ_FILE}" >&2
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
"${MAMBA_BIN}" run -n "${ENV_NAME}" uv pip install -r "${REQ_FILE}"
"${MAMBA_BIN}" run -n "${ENV_NAME}" uv pip install -e "${REPO_ROOT}"
install_js_dependencies_if_present

echo "Environment '${ENV_NAME}' is ready."
