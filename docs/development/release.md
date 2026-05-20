# Release Checklist

Use this checklist before publishing a public `exp-pkg` release.

## One-Time Package Index Setup

The repository currently does not ship GitHub Actions publishing workflows. A
maintainer with PyPI/TestPyPI project permissions must publish the checked local
distributions manually, or add a reviewed publish workflow before using Trusted
Publishing.

- PyPI project: `exp-pkg`
- TestPyPI project: `exp-pkg`

If a future GitHub Actions publish workflow is added, configure PyPI Trusted
Publishing for `Alfredo-Sandoval/exp-pkg`, protect the `pypi` GitHub
Environment with required manual approval, and verify the workflow builds and
checks distributions before publishing.

## Pre-Release Gate

Run the full local release gate against representative private data:

```bash
make release-check REAL_DATA_ROOT=../xpkg-real-data
```

If the private corpus is split, set `XPKG_REAL_DATA_MANIFEST` to the manifest
file for the intended release pass.

## TestPyPI Dry Run

Build and check local distributions:

```bash
uv build --out-dir dist --clear
uvx twine check dist/*
```

Publish to TestPyPI from a maintainer account:

```bash
uvx twine upload --repository testpypi dist/*
```

After it publishes, smoke-test in a fresh environment:

```bash
smoke_env="$(mktemp -d -t xpkg-testpypi-smoke.XXXXXX)"
trap 'rm -rf "$smoke_env"' EXIT
uv venv "$smoke_env"
"$smoke_env/bin/python" -m pip install --upgrade pip
"$smoke_env/bin/python" -m pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  exp-pkg
"$smoke_env/bin/xpkg" --help
"$smoke_env/bin/python" -c "from xpkg.services import ProjectService; assert ProjectService"
```

## PyPI Release

1. Confirm `src/xpkg/version.py` matches the intended tag.
2. Confirm `CHANGELOG.md` has a dated section for the release.
3. Build and check distributions with `uv build --out-dir dist --clear` and
   `uvx twine check dist/*`.
4. Publish the checked distributions to PyPI from a maintainer account:
   `uvx twine upload dist/*`.
5. Push a tag named `vX.Y.Z` that matches `__version__`.
6. Draft and publish a GitHub Release for that tag, attaching the checked
   wheel and sdist from `dist/`.

Keep the package files attached to the GitHub Release byte-for-byte identical
to the files uploaded to PyPI.
