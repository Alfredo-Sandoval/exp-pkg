# Security Policy

`exp-pkg` is an IO and packaging library for local experiment data. It does not
ship a hosted service or network daemon.

## Reporting

Please report security-sensitive issues privately to the maintainers before
opening a public issue.

If the issue is not security-sensitive, use the public issue tracker.

## Data Handling Expectations

The package should not require secrets, cloud credentials, personal machine
paths, or private lab data to run its normal test suite.

Do not commit:

- real participant, animal, or lab-private data
- API tokens or cloud credentials
- personal machine paths
- private real-data manifests
- generated release artifacts

Private representative data should stay outside the repository and be supplied
through `XPKG_REAL_DATA_ROOT` or `REAL_DATA_ROOT=...` when running release gates.
