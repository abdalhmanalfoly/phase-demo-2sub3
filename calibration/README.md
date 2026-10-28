# calibration/

Calibration artifacts (git-ignored except this file and `example_synthetic/`). Layout:

```
calibration/
  intrinsics/<camera_id>__<W>x<H>.json      reusable per-camera intrinsics
  <rig_id>/<camera_id>.json                 CURRENT calibration of a camera inside a rig (what fusion reads)
  <rig_id>/history/                         replaced artifacts (never deleted automatically)
  example_synthetic/default_rig/*.json      SYNTHETIC example (flag synthetic=true; refused unless allowed)
```
Real calibration images/artifacts describe your physical setup (and may include other people's captures): keep them out of version control.
Procedure and conventions: `docs/spatial.md`. Quick check: `.venv/bin/python scripts/calibrate.py inspect`.
