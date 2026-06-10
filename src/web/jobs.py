from __future__ import annotations

import logging
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Callable, Dict, Optional


logger = logging.getLogger("uvicorn.error")


@dataclass
class Job:
    id: str
    status: str
    message: str
    created_at: str
    updated_at: str
    result: Any = field(default_factory=dict)
    progress: Any = field(default_factory=dict)
    error: str = ""
    traceback: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.id,
            "status": self.status,
            "message": self.message,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "result": self.result,
            "progress": self.progress,
            "error": self.error,
        }


class JobManager:
    def __init__(self, max_workers: int = 2):
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._jobs: Dict[str, Job] = {}
        self._lock = Lock()

    def submit(self, message: str, fn: Callable, *args, with_progress: bool = False, **kwargs) -> Dict[str, Any]:
        job_id = uuid.uuid4().hex
        now = self._now()
        job = Job(id=job_id, status="queued", message=message, created_at=now, updated_at=now)
        with self._lock:
            self._jobs[job_id] = job
        self._executor.submit(self._run, job_id, fn, args, kwargs, with_progress)
        return job.to_dict()

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            job = self._jobs.get(job_id)
            return None if job is None else job.to_dict()

    def _run(self, job_id: str, fn: Callable, args: tuple, kwargs: dict, with_progress: bool):
        self._update(job_id, status="running")
        if with_progress:
            kwargs["progress_callback"] = lambda progress: self._update(job_id, progress=progress)
        try:
            result = fn(*args, **kwargs)
        except Exception as exc:
            traceback_text = traceback.format_exc()
            logger.error("Job %s failed: %s: %s\n%s", job_id, exc.__class__.__name__, exc, traceback_text)
            print(f"[job:{job_id}] failed {exc.__class__.__name__}: {exc}\n{traceback_text}", flush=True)
            self._update(
                job_id,
                status="failed",
                error=f"{exc.__class__.__name__}: {exc}",
                traceback_text=traceback_text,
            )
            return
        self._update(job_id, status="succeeded", result=result)

    def _update(
            self,
            job_id: str,
            status: Optional[str] = None,
            result: Optional[Any] = None,
            progress: Optional[Any] = None,
            error: Optional[str] = None,
            traceback_text: Optional[str] = None,
    ):
        with self._lock:
            job = self._jobs[job_id]
            if status is not None:
                job.status = status
            if result is not None:
                job.result = result
            if progress is not None:
                job.progress = progress
            if error is not None:
                job.error = error
            if traceback_text is not None:
                job.traceback = traceback_text
            job.updated_at = self._now()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()


jobs = JobManager()
