# Posetta

Posetta is a focused Python package for pose-format interoperability.

The initial extraction centers on:

- the native `.siesta`/Posetta bundle format
- adapters for `SLEAP` packages
- adapters for `DeepLabCut` tracking exports

The repository uses a `src/posetta/` layout and keeps the format surface separate from the lower-level IO internals already present in the tree.
