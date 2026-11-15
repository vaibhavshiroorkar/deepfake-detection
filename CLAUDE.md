# Project rules

## Pending work and versions

`docs/PENDING.md` holds open work and `docs/CHANGELOG.md` holds finished work, by version. Keep both current in every session, without being asked:

- **When planning**, add every new task, decision to make, or follow-up to the right section of `docs/PENDING.md` before starting it. That covers work for Claude and for Vaibhav. Nothing planned lives only in chat.
- **When a task is done**, move it from `docs/PENDING.md` to the Unreleased section of `docs/CHANGELOG.md`, in the same commit as the work.
- **When a task changes**, edit it in place: new estimate, new owner, new reason.
- **At the end of a stage**, cut a version: Unreleased becomes the version and date, `pyproject.toml` (and the project entry in `uv.lock`) takes the same version, and the commit is tagged `vX.Y.Z` and pushed with its tag. The next planned version is `1.0.0`, the final-year submission.
- **Before ending a turn** that changed the plan or finished work, check both files against what happened and commit them.
