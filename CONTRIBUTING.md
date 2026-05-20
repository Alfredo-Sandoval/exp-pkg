# Contributing

Thanks for helping make `exp-pkg` a practical IO layer for multimodal
neuroscience experiment data.

## Setup

Use the repository setup target:

```bash
make env
```

If needed, use the fallback setup script:

```bash
bash environment/setup.sh
```

The package distribution name is `exp-pkg`. The Python import name and CLI
command are both `xpkg`.

## Quality Gates

Run the fast local gate before opening a pull request or handing off a branch:

```bash
make qa
```

Run the package and docs gate before release work:

```bash
make package-check docs-build
```

Run the full release gate with private representative data before a
TestPyPI/PyPI cut:

```bash
make release-check REAL_DATA_ROOT=../xpkg-real-data
```

## Package Boundary

Keep the core package focused on IO and workspace/session contracts:

- direct readers for lab file formats
- workspace imports for durable projects
- shared timing, event, signal, pose, video, and metadata models
- portable artifacts and machine-readable CLI output

Avoid adding analysis workflows, private lab assumptions, absolute local paths,
or mandatory downstream formats to the core package.
