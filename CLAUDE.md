# CLAUDE.md

Project-wide agent guidelines live in [AGENTS.md](AGENTS.md). Follow them.

## Commit messages

**Do not add Claude attributions to commits.** Commits authored with Claude's
help must not include any of:

- `Co-Authored-By: Claude ...` trailers
- `Generated with [Claude Code](...)` lines or any AI-generated badge/footer
- References to Claude, Anthropic, the model name, or the harness in the
  commit subject, body, or trailers

Commits should read as if the human author wrote them. The same rule applies
to PR titles and bodies.
