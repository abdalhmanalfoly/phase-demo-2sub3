#!/usr/bin/env python3
"""SAGA 3D calibration CLI (Phase 2). Run with the project venv:  .venv/bin/python scripts/calibrate.py <command> ...

  capture      grab still frames from a USB/laptop webcam (OpenCV VideoCapture)         [hardware: untested in CI]
  intrinsics   board images of ONE camera  -> intrinsics file (K, distortion, quality)
  rig          intrinsics + static reference-board images of EVERY camera -> shared-frame calibration artifacts
  inspect      are the cameras calibrated, valid and mutually compatible?  (exit 0 only if >= 2 usable cameras)
  invalidate   mark a camera's calibration invalid (camera moved/replaced); fusion refuses it until recalibrated
  accuracy     error statistics of a fusion log against a physically measured point
  selftest     run the whole pipeline on synthetic rendered boards and print the measured errors (no hardware)
  example      write a clearly-marked SYNTHETIC example rig into calibration/example_synthetic/
See docs/spatial.md for the full operator procedure.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from host.app.config import load_settings  # noqa: E402
from host.app.spatial import calibration_tool as T  # noqa: E402
from host.app.spatial.calibration_store import (CalibrationLoadError, CalibrationStore, check_rig, load_intrinsics_file)  # noqa: E402


def _pattern(s: str) -> tuple[int, int]:
    try:
        c, r = s.lower().split("x")
        return int(c), int(r)
    except ValueError:
        raise SystemExit(f"--pattern must look like 9x6 (inner corners: columns x rows), got '{s}'")


def _common(p):
    p.add_argument("--pattern", default="9x6", help="inner corners columns x rows (use an ODD x EVEN board so the origin corner is unambiguous)")
    p.add_argument("--square-size-m", type=float, default=0.04, help="MEASURE the printed square edge, in metres")
    p.add_argument("--calibration-dir", default=None)
    p.add_argument("--rig-id", default=None)


def _store(a, settings):
    d = Path(a.calibration_dir) if a.calibration_dir else settings.calibration_root
    return CalibrationStore(d, a.rig_id or settings.fusion.spatial.rig_id)


def cmd_capture(a, settings):
    import cv2
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(a.device)
    if a.width: cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
    if a.height: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
    if not cap.isOpened():
        raise SystemExit("cannot open camera device")
    print(f"Capturing {a.count} frames, one every {a.interval}s. Move/tilt the board between frames. Ctrl-C to stop.")
    for i in range(a.count):
        time.sleep(a.interval)
        ok, frame = cap.read()
        if ok:
            path = out / f"{a.prefix}_{i:02d}.png"; cv2.imwrite(str(path), frame); print(f"  saved {path} ({frame.shape[1]}x{frame.shape[0]})")
    cap.release()


def cmd_intrinsics(a, settings):
    cols, rows = _pattern(a.pattern)
    samples, rejected, size = T.load_samples(a.images, cols, rows)
    print(f"{len(samples)} usable / {len(samples) + len(rejected)} images; rejected: {[(r.name, r.reason) for r in rejected]}")
    if size is None:
        raise SystemExit("no readable images found")
    art = T.calibrate_intrinsics(samples, size, cols, rows, a.square_size_m, a.camera_id, settings.fusion.calibration_quality, rejected)
    path = _store(a, settings).save_intrinsics(art)
    q = art.quality
    print(f"OK intrinsics for '{a.camera_id}' at {size[0]}x{size[1]} -> {path}\n  RMS {q.intrinsic_rms_px:.3f} px, coverage {q.coverage_fraction:.0%}, "
          f"tilt spread {q.tilt_spread_deg:.1f} deg, f = {art.camera_matrix[0][0]:.1f}/{art.camera_matrix[1][1]:.1f}, dist = {np.round(art.distortion, 4).tolist()}")
    for w in art.validity.warnings: print("  WARNING", w)


def cmd_rig(a, settings):
    store = _store(a, settings)
    cams = {}
    for spec in a.camera:
        try:
            cid, rest = spec.split("=", 1); intr_path, img_dir = rest.split(":", 1)
        except ValueError:
            raise SystemExit(f"--camera expects CAMERA_ID=INTRINSICS.json:REFERENCE_IMAGES_DIR, got '{spec}'")
        intr = load_intrinsics_file(intr_path)
        samples, rejected, size = T.load_samples(img_dir, intr.target.inner_corners_cols, intr.target.inner_corners_rows)
        print(f"{cid}: {len(samples)} reference images with a board; rejected {[(r.name, r.reason) for r in rejected]}")
        cams[cid] = (intr, samples)
    b2w = None
    if a.board_to_world:
        d = json.loads(Path(a.board_to_world).read_text()); b2w = (np.array(d["rotation"], float), np.array(d["translation_m"], float))
    arts = T.calibrate_rig(store.rig_id, cams, a.world_frame_id, b2w, settings.fusion.calibration_quality)
    for art in arts.values():
        print(f"OK {art.camera_id}: saved {store.save_camera(art)}  (extrinsic RMS {art.quality.extrinsic_rms_px:.3f} px)")
        for w in art.validity.warnings: print("   WARNING", w)
    chk = next(iter(arts.values())).quality.rig_check
    print("Rig cross-check (board corners triangulated from camera pairs vs known board geometry):")
    for k, v in chk["pairs"].items(): print(f"  {k}: mean {v['mean_mm']:.2f} mm, max {v['max_mm']:.2f} mm, ray angle {v['ray_angle_deg']:.1f} deg")
    print("  (" + chk["note"] + ")\nNEXT: leave the cameras and board untouched; verify with a measured point (docs/spatial.md, 'accuracy').")


def cmd_inspect(a, settings):
    store = _store(a, settings)
    rep = store.load_all()
    out = check_rig(rep.artifacts, settings.fusion.spatial.allow_synthetic_calibration)
    print(f"Rig '{store.rig_id}' in {store.root}")
    for cid, v in out["cameras"].items():
        flag = "OK " if not v["problems"] else "BAD"
        print(f" [{flag}] {cid}: {v['image']} calib={v['calibration_id']} rig_session={v['rig_calibration_id']} intrinsic_rms={v['intrinsic_rms_px']} extrinsic_rms={v['extrinsic_rms_px']}")
        for p in v["problems"] + v["warnings"]: print("        -", p)
    for name, err in rep.errors.items(): print(f" [ERR] {name}: {err}")
    print(out["summary"])
    return 0 if out["can_triangulate"] and not rep.errors else 1


def cmd_invalidate(a, settings):
    art = _store(a, settings).invalidate(a.camera_id, a.reason)
    print(f"{a.camera_id}: marked invalid ({art.validity.reason}). Recalibrate (rig command) before using fusion again.")


def cmd_accuracy(a, settings):
    xyz = [float(v) for v in a.truth.split(",")]
    print(json.dumps(T.accuracy_report(a.fusion_log, xyz, a.target), indent=2))


def _selftest_report():
    from host.app.spatial import geometry as G
    from host.app.spatial import synthetic as S
    from host.app.spatial.calibration_store import camera_model
    arts, truth = S.calibrate_synthetic_rig(noise=1.0, seed=0)
    for cid, art in arts.items():
        m, tr = camera_model(art), truth[cid]
        print(f"{cid}: intrinsic RMS {art.quality.intrinsic_rms_px:.3f}px, extrinsic RMS {art.quality.extrinsic_rms_px:.3f}px, "
              f"focal {m.K[0, 0]:.1f} (true {tr.K[0, 0]:.0f}), rotation error {G.rotation_angle_deg(m.R, tr.R):.3f} deg, "
              f"camera-centre error {np.linalg.norm(m.center - tr.center) * 1000:.2f} mm")
    return arts, truth


def cmd_selftest(a, settings):
    from host.app.spatial import synthetic as S
    print("SYNTHETIC self-test (rendered boards; proves the pipeline, NOT physical accuracy)")
    arts, truth = _selftest_report()
    print("\nEnd-to-end triangulation error with the CALIBRATED cameras (300 random points per row; 640x480, f=600 px, ~60 deg rig):")
    print(f"{'pixel noise':>12} {'depth from cams':>16} | {'mean':>8} {'p95':>8} {'max':>8}   (mm)")
    for r in S.accuracy_table(arts, truth):
        print(f"{r['noise_px']:>9.1f} px {r['mean_depth_from_cameras_m']:>13.2f} m | {r['mean_mm']:>8.2f} {r['p95_mm']:>8.2f} {r['max_mm']:>8.2f}")


def cmd_example(a, settings):
    arts, _ = _selftest_report()
    store = CalibrationStore(Path(a.out), settings.fusion.spatial.rig_id)
    for art in arts.values(): print("wrote", store.save_camera(art))
    print("These files are flagged synthetic=true and are REFUSED by fusion unless spatial.allow_synthetic_calibration is set.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("capture"); p.add_argument("--device", type=int, default=0); p.add_argument("--out", required=True)
    p.add_argument("--count", type=int, default=20); p.add_argument("--interval", type=float, default=2.0)
    p.add_argument("--width", type=int); p.add_argument("--height", type=int); p.add_argument("--prefix", default="frame"); p.set_defaults(fn=cmd_capture)
    p = sub.add_parser("intrinsics"); _common(p); p.add_argument("--camera-id", required=True); p.add_argument("--images", required=True); p.set_defaults(fn=cmd_intrinsics)
    p = sub.add_parser("rig"); _common(p); p.add_argument("--camera", action="append", required=True, metavar="ID=INTRINSICS.json:IMAGES_DIR")
    p.add_argument("--world-frame-id", default="board_frame"); p.add_argument("--board-to-world", help="JSON {rotation:3x3, translation_m:[3]}"); p.set_defaults(fn=cmd_rig)
    p = sub.add_parser("inspect"); _common(p); p.set_defaults(fn=cmd_inspect)
    p = sub.add_parser("invalidate"); _common(p); p.add_argument("--camera-id", required=True); p.add_argument("--reason", default="camera moved or replaced"); p.set_defaults(fn=cmd_invalidate)
    p = sub.add_parser("accuracy"); p.add_argument("--fusion-log", required=True); p.add_argument("--truth", required=True, help="x,y,z in metres (world frame)")
    p.add_argument("--target", default="wrist"); p.set_defaults(fn=cmd_accuracy)
    p = sub.add_parser("selftest"); p.set_defaults(fn=cmd_selftest)
    p = sub.add_parser("example"); p.add_argument("--out", default=str(ROOT / "calibration" / "example_synthetic")); p.set_defaults(fn=cmd_example)
    a = ap.parse_args(argv)
    try:
        return a.fn(a, load_settings()) or 0
    except (T.CalibrationError, CalibrationLoadError) as exc:
        print(f"ERROR {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
