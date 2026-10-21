"""CLI entry point: ingest -> mask -> clean -> (sharpen) -> compare -> status + report.

Usage: python restore.py <image> [--mask MASK.png] [--backend telea|ns|lama]
       python restore.py <image> --engine gemini [--model ID]   (whole-photo generative restoration)
       python restore.py <image> --engine external --raw RESTORED.png   (post-process e.g. a DiffBIR output)
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
from restoration.faces import compare_faces, detect_faces
from restoration.generative import (DEFAULT_MODEL, DEFAULT_PROMPT, GenerativeError, gemini_restore,
                                     match_size, paste_original_faces)
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


def _save_outputs(storage, job_id, original, result, mask):
    """Write the result PNG and the diff / before-after images; return the result path."""
    results, reports = Path(storage) / "results", Path(storage) / "reports"
    results.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    result_path = results / f"{job_id}-restored.png"
    _write_png(result_path, result)
    # before | after | amplified difference (x8) with the mask outlined in red
    diff = cv2.convertScaleAbs(cv2.absdiff(_bgr(original), _bgr(result)), alpha=8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(diff, contours, -1, (0, 0, 255), 1)
    _write_png(reports / f"{job_id}-diff.png", diff)
    _write_png(reports / f"{job_id}-before_after.png", np.hstack([_bgr(original), _bgr(result), diff]))
    return result_path


def _process_generative(job, config, storage, name, gcfg, raw=None, engine="gemini", faces_src=None):
    """A model restores the whole photo; original faces are pasted back; structure is checked.

    raw: a result already produced elsewhere (e.g. DiffBIR on Colab); None = ask Gemini now.
    """
    original = _read(job.data["paths"]["original"])
    if original.dtype != np.uint8:
        raise GenerativeError(f"unsupported bit depth {original.dtype}; only 8-bit images are supported")
    faces_found = detect_faces(original)
    if raw is None:
        raw = gemini_restore(original, gcfg.get("prompt", DEFAULT_PROMPT), gcfg.get("model", DEFAULT_MODEL))
    aligned, resized, shift = match_size(_bgr(original), raw)
    # faces come from the original, or from faces_src (e.g. the crack-cleaned image: same pixels except the cracks)
    face_source = _bgr(original) if faces_src is None else match_size(_bgr(original), _bgr(faces_src))[0]
    result = paste_original_faces(face_source, aligned, faces_found, gcfg.get("face_pad", 0.25))

    nothing = np.zeros(original.shape[:2], np.uint8)
    structure = compare_outside_mask(_bgr(original), result, nothing)
    faces = compare_faces(_bgr(original), result)
    reasons = []
    if structure.ssim < gcfg["min_structure_ssim"]:
        reasons.append(f"structure SSIM {structure.ssim:.3f} below {gcfg['min_structure_ssim']} (objects may have changed)")
    if shift > gcfg["max_shift_px"]:
        reasons.append(f"result is shifted {shift:.1f}px from the original (max {gcfg['max_shift_px']})")
    if not faces_found:
        reasons.append("no face detected: a face the detector missed may have been redrawn")
    if faces.result_count < faces.original_count:
        reasons.append(f"faces detected dropped from {faces.original_count} to {faces.result_count}")
    status = REVIEW if reasons else DONE

    result_path = _save_outputs(storage, job.data["job_id"], _bgr(original), result, nothing)
    tools = [gcfg.get("tool_label") or f"gemini:{gcfg.get('model', DEFAULT_MODEL)}", "faces_pasted_back"]
    report = {
        "job_id": job.data["job_id"], "source_name": name, "status": status, "reasons": reasons,
        "tools": tools, "config_version": job.data["config_version"], "engine": engine,
        "model_resized_output": resized, "shift_px": shift, "faces_pasted_back": len(faces_found),
        "structure": {"ssim": structure.ssim, "changed_fraction": structure.changed_fraction},
        "faces": {"original": faces.original_count, "result": faces.result_count,
                  "ssim": [c.ssim for c in faces.checks]},
        "result": str(result_path),
    }
    if engine == "gemini":
        report["sent_to_external_service"] = "Google Gemini API"
    (Path(storage) / "reports" / f"{job.data['job_id']}-restored.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    log.info("%s job=%s status=%s engine=%s model=%s structure_ssim=%.3f shift=%.1fpx faces_pasted_back=%d reasons=%s",
             name, job.data["job_id"], status, engine, tools[0], structure.ssim, shift,
             len(faces_found), reasons)
    return status, reasons


def _process(job, config, storage, name, manual_mask):
    original = _read(job.data["paths"]["original"])
    _check_face_restore(config.get("face_restore", {}))
    mask_in = None if manual_mask is None else _read(manual_mask, cv2.IMREAD_GRAYSCALE)
    if manual_mask is not None and mask_in is None:
        raise CleanError(f"cannot read mask: {manual_mask}")

    boxes = [(f.x, f.y, f.w, f.h) for f in detect_faces(original)]
    cleaned = clean(original, config["clean"], mask_in, name, boxes)
    tools = [f"inpaint:{cleaned.backend}"] + (["stain_correction"] if cleaned.stain_fraction > 0 else [])
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
    result_path = _save_outputs(storage, job_id, original, result, cleaned.mask)

    report = {
        "job_id": job_id, "source_name": name, "status": status, "reasons": reasons,
        "tools": tools, "config_version": job.data["config_version"],
        "mask_fraction": cleaned.mask_fraction, "stain_fraction": cleaned.stain_fraction,
        "face_fidelity_weight": None,
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
    parser.add_argument("--engine", choices=["local", "gemini", "external"], default="local",
                        help="local = mask + inpainting; gemini = whole-photo generative restoration "
                             "(sends the image to Google; needs GEMINI_API_KEY in .env); "
                             "external = use a restored image made elsewhere (--raw), e.g. DiffBIR on Colab")
    parser.add_argument("--raw", help="with --engine external: the restored image to post-process")
    parser.add_argument("--faces-from", help="with --engine external: take the faces from this image instead of the original")
    parser.add_argument("--model", help="Gemini model id (default from config generative.model)")
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
    if args.engine in ("gemini", "external"):
        gcfg = dict(config.get("generative", {}))
        gcfg.setdefault("min_structure_ssim", 0.8)
        gcfg.setdefault("max_shift_px", 3)
        if args.model:
            gcfg["model"] = args.model
        raw = None
        if args.engine == "external":
            raw = _read(args.raw, cv2.IMREAD_COLOR) if args.raw else None
            if raw is None:
                print("error: --engine external needs --raw <readable image>", file=sys.stderr)
                return 1
            gcfg["tool_label"] = f"external:{Path(args.raw).name}"
        faces_src = _read(args.faces_from, cv2.IMREAD_COLOR) if args.faces_from else None
        if args.faces_from and faces_src is None:
            print("error: cannot read --faces-from image", file=sys.stderr)
            return 1
        process = lambda j: _process_generative(j, config, args.storage, ingested.source_name, gcfg, raw, args.engine, faces_src)
    else:
        process = lambda j: _process(j, config, args.storage, ingested.source_name, args.mask)
    run_job(job, process, args.storage)
    if job.status == "failed":
        print(f"job {job.data['job_id']}: failed: {job.data['error']}", file=sys.stderr)
        return 1
    note = " (duplicate, already ingested)" if ingested.duplicate else ""
    print(f"job {job.data['job_id']}{note}: {job.status}" +"".join(f"\n  - {r}" for r in job.data["decision_reasons"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
