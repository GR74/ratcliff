# Cognitive Society archive

Archived on 2026-10-07 at the user's request. This research is paused while the
active project focuses on the Model A / Model B simulators and the web application.

## Contents

- `cognitive_society/`: the complete Python package, experiments, and tests.
- `docs/`: society and swarm plans, results, collective-cognition documentation,
  cognitive digital twin direction, and paper specifications.
- `docs_v2/08-cognitive-society.md`: the society learning-guide chapter.
- `snapshots/`: original versions of mixed documents before their active copies
  were narrowed to the simulators and web app. These include the old top-level
  README, contributor guide, learning guides, and SBI research design.
- `manifest.json`: each moved or copied file's original path, archive path, size,
  and SHA-256 checksum at archive time.

Files were moved or copied without changing their contents. Historical paths and
commands inside these files describe the original repository layout. They may
need adjustment when reading or running material from the archive.

The 2D simulator's SBI prototype (`scripts/sbi_mvp.py`) and its validation note
remain active because they belong to scientific parameter inference. The mixed
2026-06-05 session handoff also remains as a historical record of the simulator
and application work.

## Resuming

Use the manifest's `original_path` for entries marked `move` to restore their
repository locations. Review entries marked `copy` against the current active
documents before restoring any text. Then add `cognitive_society*` to package
discovery and `cognitive_society/tests` to pytest's test paths in `pyproject.toml`.
The restored package will sit outside the archive exclusion.

The archive is retained in the repository but excluded from normal Python package
installation, default pytest collection, and the Docker build context. Existing
Model A / Model B, shared, backend, and frontend implementation files are unchanged.
