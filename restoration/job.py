"""Job lifecycle: a JSON file per job with validated status transitions."""
import json
import logging
from datetime import datetime
from pathlib import Path

PENDING, PROCESSING, DONE, REVIEW, FAILED = "pending", "processing", "done", "review", "failed"
# The single place that defines allowed transitions.
TRANSITIONS = {
    PENDING: {PROCESSING},
    PROCESSING: {DONE, REVIEW, FAILED},
    DONE: set(),
    REVIEW: set(),
    FAILED: set(),
}


class JobError(Exception):
    """Invalid job operation (illegal transition, missing job)."""


def _now():
    return datetime.now().isoformat(timespec="seconds")


class Job:
    def __init__(self, data, jobs_dir):
        self.data = data
        self.path = Path(jobs_dir) / f"{data['job_id']}.json"

    @classmethod
    def create(cls, job_id, source_name, sha256, config_version, original_path, storage_root="storage"):
        jobs_dir = Path(storage_root) / "jobs"
        jobs_dir.mkdir(parents=True, exist_ok=True)
        now = _now()
        data = {
            "job_id": job_id, "source_name": source_name, "sha256": sha256,
            "config_version": config_version, "status": PENDING, "stages": [],
            "paths": {"original": str(original_path)}, "decision_reasons": [],
            "error": None, "created_at": now, "updated_at": now,
        }
        job = cls(data, jobs_dir)
        job.save()
        return job

    @classmethod
    def load(cls, job_id, storage_root="storage"):
        jobs_dir = Path(storage_root) / "jobs"
        path = jobs_dir / f"{job_id}.json"
        if not path.is_file():
            raise JobError(f"job not found: {job_id}")
        return cls(json.loads(path.read_text(encoding="utf-8")), jobs_dir)

    @property
    def status(self):
        return self.data["status"]

    def save(self):
        self.data["updated_at"] = _now()
        self.path.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")

    def transition(self, new_status, reasons=None, error=None, force=False):
        if new_status not in TRANSITIONS:
            raise JobError(f"unknown status: {new_status}")
        if new_status not in TRANSITIONS[self.status] and not force:
            raise JobError(f"illegal transition {self.status} -> {new_status}")
        self.data["status"] = new_status
        if reasons is not None:
            self.data["decision_reasons"] = list(reasons)
        self.data["error"] = error
        self.save()

    def add_stage(self, name, status, started_at, finished_at=None, error=None):
        self.data["stages"].append({"name": name, "status": status, "started_at": started_at,
                                    "finished_at": finished_at or _now(), "error": error})
        self.save()


def log_error(storage_root, message):
    """Append to storage/logs/restore.log."""
    logs = Path(storage_root) / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"restore.{Path(storage_root).resolve()}")
    if not logger.handlers:
        handler = logging.FileHandler(logs / "restore.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    logger.error(message)
    for h in logger.handlers:
        h.flush()


def run_job(job, process, storage_root="storage"):
    """Run process(job) with lifecycle handling.

    process returns (status, reasons) with status done/review. Any exception marks the
    job failed with the error message and writes it to the log.
    """
    job.transition(PROCESSING)
    try:
        status, reasons = process(job)
        if status not in (DONE, REVIEW):
            raise JobError(f"process returned invalid final status: {status}")
        job.transition(status, reasons=reasons)
    except Exception as e:
        message = f"{type(e).__name__}: {e}"
        log_error(storage_root, f"job {job.data['job_id']} failed: {message}")
        job.transition(FAILED, error=message)
    return job
