from __future__ import annotations

import logging
import traceback
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Callable, Dict, Optional


logger = logging.getLogger("uvicorn.error")
JOB_STALE_TIMEOUT_SECONDS = 10 * 60
STALE_JOB_ERROR = "Proceso detenido por falta de progreso. Vuelve a ejecutar con un perfil ligero."


@dataclass
class Job:
    id: str
    status: str
    message: str
    created_at: str
    updated_at: str
    lock_key: str = ""
    result: Any = field(default_factory=dict)
    progress: Any = field(default_factory=dict)
    error: str = ""
    traceback: str = ""
    future: Future | None = field(default=None, repr=False, compare=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.id,
            "status": self.status,
            "message": self.message,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "lock_key": self.lock_key,
            "result": self.result,
            "progress": self.progress,
            "error": self.error,
        }


class JobManager:
    def __init__(self, max_workers: int = 2):
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._jobs: Dict[str, Job] = {}
        self._lock = Lock()

    def submit(self, message: str, fn: Callable, *args, with_progress: bool = False, lock_key: str = "", **kwargs) -> Dict[str, Any]:
        job_id = uuid.uuid4().hex
        now = self._now()
        job = Job(id=job_id, status="queued", message=message, created_at=now, updated_at=now, lock_key=str(lock_key or ""))
        with self._lock:
            self._expire_stale_jobs_locked(now)
            if lock_key:
                active = [
                    item for item in self._jobs.values()
                    if item.lock_key == lock_key and item.status in {"queued", "running"}
                ]
                if active:
                    raise RuntimeError("Ya hay un entrenamiento Mundial en ejecucion. Espera a que termine antes de iniciar otro.")
            self._jobs[job_id] = job
        future = self._executor.submit(self._run, job_id, fn, args, kwargs, with_progress)
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].future = future
        return job.to_dict()

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            self._expire_stale_jobs_locked(self._now())
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
            if job.status == "failed" and job.error == STALE_JOB_ERROR:
                return
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

    def _expire_stale_jobs_locked(self, now: str) -> None:
        now_dt = self._parse_time(now)
        for job in self._jobs.values():
            if job.status not in {"queued", "running"}:
                continue
            updated_dt = self._parse_time(job.updated_at)
            if updated_dt is None or now_dt is None:
                continue
            if (now_dt - updated_dt).total_seconds() < JOB_STALE_TIMEOUT_SECONDS:
                continue
            future = job.future
            if future is None:
                continue
            if future.running():
                continue
            if not future.done() and not future.cancel():
                continue
            if future.done() and not future.cancelled():
                continue
            job.status = "failed"
            job.error = STALE_JOB_ERROR
            job.updated_at = now

    @staticmethod
    def _parse_time(value: str) -> Optional[datetime]:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()


jobs = JobManager()
