import logging

import pytest

from restoration.config import ConfigError, load_config
from restoration.job import DONE, FAILED, PENDING, PROCESSING, REVIEW, Job, JobError, run_job


def make_job(tmp_path):
    return Job.create("20260101-000000-abcd1234", "a.jpg", "abcd1234" * 8, "v1", "orig.jpg", tmp_path)


@pytest.fixture(autouse=True)
def _close_log_handlers():
    yield
    # Release restore.log so tmp_path can be cleaned up on Windows.
    for name, logger in list(logging.Logger.manager.loggerDict.items()):
        if name.startswith("restore.") and isinstance(logger, logging.Logger):
            for h in list(logger.handlers):
                h.close()
                logger.removeHandler(h)


def test_create_persists_pending(tmp_path):
    job = make_job(tmp_path)
    assert Job.load(job.data["job_id"], tmp_path).status == PENDING


def test_legal_transitions(tmp_path):
    job = make_job(tmp_path)
    job.transition(PROCESSING)
    job.transition(REVIEW, reasons=["small_face"])
    loaded = Job.load(job.data["job_id"], tmp_path)
    assert loaded.status == REVIEW and loaded.data["decision_reasons"] == ["small_face"]


@pytest.mark.parametrize("start,target", [(PENDING, DONE), (PENDING, FAILED), (DONE, PROCESSING), (FAILED, DONE)])
def test_illegal_transitions(tmp_path, start, target):
    job = make_job(tmp_path)
    job.data["status"] = start
    with pytest.raises(JobError):
        job.transition(target)


def test_force_allows_rerun(tmp_path):
    job = make_job(tmp_path)
    job.data["status"] = DONE
    job.transition(PROCESSING, force=True)
    assert job.status == PROCESSING


def test_exception_marks_failed_and_logs(tmp_path):
    job = make_job(tmp_path)

    def boom(_):
        raise RuntimeError("kaboom")

    run_job(job, boom, tmp_path)
    loaded = Job.load(job.data["job_id"], tmp_path)
    assert loaded.status == FAILED and "kaboom" in loaded.data["error"]
    assert "kaboom" in (tmp_path / "logs" / "restore.log").read_text(encoding="utf-8")


def test_success_sets_final_status(tmp_path):
    job = run_job(make_job(tmp_path), lambda j: (DONE, []), tmp_path)
    assert job.status == DONE


def test_invalid_process_status_fails(tmp_path):
    job = run_job(make_job(tmp_path), lambda j: (PENDING, []), tmp_path)
    assert job.status == FAILED


def test_load_missing_job(tmp_path):
    with pytest.raises(JobError):
        Job.load("nope", tmp_path)


def test_config_version_changes_with_content(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("a: 1\n")
    _, v1 = load_config(p)
    p.write_text("a: 2\n")
    cfg, v2 = load_config(p)
    assert v1 != v2 and cfg == {"a": 2}


def test_config_errors(tmp_path):
    with pytest.raises(ConfigError):
        load_config(tmp_path / "missing.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("- just\n- a list\n")
    with pytest.raises(ConfigError):
        load_config(bad)


def test_repo_config_loads():
    cfg, _ = load_config("config.yaml")
    assert cfg["ingest"]["min_side"] == 32
