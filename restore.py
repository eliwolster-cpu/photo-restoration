"""CLI entry point: ingest -> mask -> clean -> (sharpen) -> compare -> status + report.

Usage: python restore.py <image> [--mask MASK.png] [--backend telea|ns|lama]
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

from restoration.clean import CleanError, clean
from restoration.compare import compare_outside_mask
from restoration.config import ConfigError, load_config
from restoration.faces import compare_faces
from restoration.ingest import IngestError, ingest
from restoration.job import DONE, REVIEW, Job, run_job
from restoration.sharpen import sharpen

log = logging.getLogger("restoration")


def _read(path, flags=cv2.IMREAD_UNCHANGED):
    # np.fromfile + imdecode, because cv2.imread/imwrite fail on non-ASCII Windows paths
    return cv2.imdecode(np.fromfile(str(path), np.uint8), flags)


def _write_png(path, image):
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise ValueError(f"cannot encode image: {path}")
    buf.tofile(str(path))


def _bgr(image):
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image[..., :3]


def _check_face_restore(cfg):
    if not cfg.get("enabled", False):
        return
    if cfg.get("fidelity", 0) < cfg.get("min_fidelity", 0.8):
        raise CleanError(f"face_restore.fidelity {cfg.get('fidelity')} is below the minimum {cfg.get('min_fidelity')}")
    raise CleanError("face_restore is not implemented yet (CodeFormer needs a GPU/runner decision)")


def _overlapping_faces(faces, mask):
    """Indices of detected faces whose box contains any damage-mask pixel."""
    return [i for i, c in enumerate(faces.checks)
            if mask[c.box.y:c.box.y + c.box.h, c.box.x:c.box.x + c.box.w].any()]


def _decide(cfg, mask_fraction, outside, faces, mask):
    """Return (status, reasons). Any doubt -> review; a suspected face change is never done."""
    reasons = []
    if mask_fraction > cfg["max_mask_fraction"]:
        reasons.append(f"mask covers {100 * mask_fraction:.1f}% of the image (max {100 * cfg['max_mask_fraction']:.1f}%)")
    if outside.ssim < cfg["min_outside_ssim"]:
        reasons.append(f"outside-mask SSIM {outside.ssim:.3f} below {cfg['min_outside_ssim']}")
    if outside.changed_fraction > cfg["max_outside_changed_fraction"]:
        reasons.append(f"{100 * outside.changed_fraction:.2f}% of pixels changed outside the mask")
    if faces.result_count < faces.original_count:
        reasons.append(f"faces detected dropped from {faces.original_count} to {faces.result_count}")
    overlapping = _overlapping_faces(faces, mask)
    for i, c in enumerate(faces.checks):
        if i in overlapping:
            reasons.append(f"face {i}: damage mask touches the face")
        elif c.ssim < cfg["min_face_ssim"]:
            reasons.append(f"face {i}: SSIM {c.ssim:.3f} below {cfg['min_face_ssim']}")
        if c.small:
            reasons.append(f"face {i}: too small to verify")
    return (REVIEW if reasons else DONE), reasons


def _process(job, config, storage, name, manual_mask):
    original = _read(job.data["paths"]["original"])
    _check_face_restore(config.get("face_restore", {}))
    mask_in = None if manual_mask is None else _read(manual_mask, cv2.IMREAD_GRAYSCALE)
    if manual_mask is not None and mask_in is None:
        raise CleanError(f"cannot read mask: {manual_mask}")

    cleaned = clean(original, config["clean"], mask_in, name)
    tools = [f"inpaint:{cleaned.backend}"]
    result = cleaned.image
    sharpen_cfg = config.get("sharpen", {})
    if sharpen_cfg.get("enabled", False):
        result = sharpen(result, sharpen_cfg["amount"], sharpen_cfg["sigma"])
        tools.append("sharpen:unsharp")

    outside = compare_outside_mask(original, result, cleaned.mask)
    faces = compare_faces(original, result)
    status, reasons = _decide(config["decide"], cleaned.mask_fraction, outside, faces, cleaned.mask)
    face_overlap = _overlapping_faces(faces, cleaned.mask)
    if not cleaned.outside_identical:
        status, reasons = REVIEW, reasons + ["clean changed pixels outside the mask"]

    job_id = job.data["job_id"]
    results, reports = Path(storage) / "results", Path(storage) / "reports"
    results.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    result_path = results / f"{job_id}-restored.png"
    _write_png(result_path, result)
    # before | after | amplified difference (x8) with the mask outlined in red
    diff = cv2.absdiff(_bgr(original), _bgr(result))
    diff = cv2.convertScaleAbs(diff, alpha=8)
    contours, _ = cv2.findContours(cleaned.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(diff, contours, -1, (0, 0, 255), 1)
    _write_png(reports / f"{job_id}-diff.png", diff)
    _write_png(reports / f"{job_id}-before_after.png", np.hstack([_bgr(original), _bgr(result), diff]))

    report = {
        "job_id": job_id, "source_name": name, "status": status, "reasons": reasons,
        "tools": tools, "config_version": job.data["config_version"],
        "mask_fraction": cleaned.mask_fraction, "face_fidelity_weight": None,
        "face_overlap": bool(face_overlap),
        "clean_outside_mask_identical": cleaned.outside_identical,
        "outside_mask": {"ssim": outside.ssim, "changed_fraction": outside.changed_fraction},
        "faces": {"original": faces.original_count, "result": faces.result_count,
                  "ssim": [c.ssim for c in faces.checks]},
        "result": str(result_path),
    }
    (reports / f"{job_id}-restored.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("%s job=%s status=%s mask=%.2f%% face_overlap=%s tools=%s face_w=none outside_mask_identical=%s reasons=%s",
             name, job_id, status, 100 * cleaned.mask_fraction, "yes" if face_overlap else "no",
             ",".join(tools), cleaned.outside_identical, reasons)
    return status, reasons


def main(argv=None):
    parser = argparse.ArgumentParser(description="Conservatively restore a scanned photo.")
    parser.add_argument("image")
    parser.add_argument("--storage", default="storage")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--mask", help="manual damage mask (white = damage); default is automatic")
    parser.add_argument("--backend", choices=["telea", "ns", "lama"], help="override clean.backend")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    try:
        config, config_version = load_config(args.config)
        if args.backend:
            config["clean"]["backend"] = args.backend
        ingested = ingest(args.image, args.storage)
    except (ConfigError, IngestError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    job = Job.create(ingested.job_id, ingested.source_name, ingested.sha256, config_version,
                     ingested.original_path, args.storage)
    run_job(job, lambda j: _process(j, config, args.storage, ingested.source_name, args.mask), args.storage)
    if job.status == "failed":
        print(f"job {job.data['job_id']}: failed: {job.data['error']}", file=sys.stderr)
        return 1
    note = " (duplicate, already ingested)" if ingested.duplicate else ""
    print(f"job {job.data['job_id']}{note}: {job.status}" +"".join(f"\n  - {r}" for r in job.data["decision_reasons"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
