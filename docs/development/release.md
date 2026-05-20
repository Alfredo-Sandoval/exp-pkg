# Release Checklist

Use this checklist before publishing a public `exp-pkg` release.

## One-Time Repository Setup

Configure PyPI Trusted Publishing for this repository:

- PyPI project: `exp-pkg`
- Repository owner: `Alfredo-Sandoval`
- Repository name: `exp-pkg`
- Workflow file: `publish.yml`
- PyPI environment: `pypi`
- TestPyPI environment: `testpypi`

Protect the `pypi` GitHub Environment with required manual approval. TestPyPI
approval is optional.

## Pre-Release Gate

Run the full local release gate against representative private data:

```bash
make release-check REAL_DATA_ROOT=/path/to/xpkg-real-data
```

If the private corpus is split, set `XPKG_REAL_DATA_MANIFEST` to the manifest
file for the intended release pass.

## TestPyPI Dry Run

Use the `Publish Python Package` workflow with `workflow_dispatch` and
`target=testpypi`.

After it publishes, smoke-test in a fresh environment:

```bash
uv venv /tmp/xpkg-testpypi-smoke
/tmp/xpkg-testpypi-smoke/bin/python -m pip install --upgrade pip
/tmp/xpkg-testpypi-smoke/bin/python -m pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  exp-pkg
/tmp/xpkg-testpypi-smoke/bin/xpkg --help
/tmp/xpkg-testpypi-smoke/bin/python -c "from xpkg.services import ProjectService; assert ProjectService"
```

## PyPI Release

1. Confirm `src/xpkg/version.py` matches the intended tag.
2. Confirm `CHANGELOG.md` has a dated section for the release.
3. Push a tag named `vX.Y.Z` that matches `__version__`.
4. Draft and publish a GitHub Release for that tag.

Publishing the GitHub Release runs `.github/workflows/publish.yml`, which:

- verifies the release tag matches `src/xpkg/version.py`
- builds the sdist and wheel
- runs `twine check`
- attaches distributions to the GitHub Release
- publishes to PyPI through Trusted Publishing

The workflow uses short-lived OIDC credentials and does not require a stored
PyPI API token.
