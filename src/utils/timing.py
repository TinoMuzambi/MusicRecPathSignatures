"""Timing and enforceable process-tree RSS monitoring utilities."""

from __future__ import annotations

import json
import math
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil

from src.utils.logger_config import setup_logger


logger = setup_logger("timing")


class ResourceLimitExceeded(RuntimeError):
    """Raised when sampled process-tree RSS exceeds its declared ceiling."""

    def __init__(self, *, observed_peak_mb: float, ceiling_mb: float):
        self.observed_peak_mb = float(observed_peak_mb)
        self.ceiling_mb = float(ceiling_mb)
        super().__init__(
            "process-tree RSS ceiling exceeded: "
            f"peak={self.observed_peak_mb:.6f} MB; ceiling={self.ceiling_mb:.6f} MB"
        )


class ResourceMeasurementUnavailable(RuntimeError):
    """Raised when process-tree RSS cannot be measured completely."""

    _STAGES = frozenset(
        {
            "root_process",
            "child_enumeration",
            "root_rss",
            "child_rss",
            "sampler_contract",
        }
    )

    def __init__(self, *, stage: str, cause: BaseException):
        if stage not in self._STAGES:
            raise ValueError(f"unknown RSS measurement stage: {stage!r}")
        if not isinstance(cause, BaseException):
            raise TypeError("cause must be an exception")
        self.stage = stage
        self.cause_type = type(cause).__name__
        super().__init__(
            "process-tree RSS measurement unavailable: "
            f"stage={self.stage}; cause={self.cause_type}"
        )


def _positive_number(value: object, *, field: str, allow_none: bool = False) -> float | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a positive finite number")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{field} must be a positive finite number")
    return number


def process_tree_rss_mb(process: psutil.Process | None = None) -> float:
    """Return RSS for one process and all currently live recursive children."""

    if process is None:
        try:
            root = psutil.Process()
        except (psutil.Error, OSError, RuntimeError) as cause:
            raise ResourceMeasurementUnavailable(
                stage="root_process", cause=cause
            ) from cause
    else:
        root = process

    try:
        processes = [root, *root.children(recursive=True)]
    except (psutil.Error, OSError, RuntimeError, TypeError, ValueError) as cause:
        raise ResourceMeasurementUnavailable(
            stage="child_enumeration", cause=cause
        ) from cause

    total = 0
    seen: set[int] = set()
    for index, member in enumerate(processes):
        stage = "root_rss" if index == 0 else "child_rss"
        try:
            pid = int(member.pid)
            if pid in seen:
                continue
            seen.add(pid)
            total += int(member.memory_info().rss)
        except psutil.NoSuchProcess as cause:
            if index != 0:
                continue
            raise ResourceMeasurementUnavailable(
                stage=stage, cause=cause
            ) from cause
        except (psutil.Error, OSError, RuntimeError, TypeError, ValueError) as cause:
            raise ResourceMeasurementUnavailable(
                stage=stage, cause=cause
            ) from cause
    return float(total / (1024 * 1024))


def get_memory_usage_mb() -> float:
    """Compatibility name for current process-tree RSS in MiB."""

    return process_tree_rss_mb()


class ProcessTreeRSSMonitor:
    """Sample current and peak process-tree RSS and enforce an optional ceiling."""

    def __init__(
        self,
        *,
        rss_ceiling_mb: float | None = None,
        poll_interval_seconds: float = 0.05,
        sampler: Callable[[], float] = process_tree_rss_mb,
    ) -> None:
        self.rss_ceiling_mb = _positive_number(
            rss_ceiling_mb, field="RSS ceiling", allow_none=True
        )
        self.poll_interval_seconds = _positive_number(
            poll_interval_seconds, field="poll interval"
        )
        if not callable(sampler):
            raise TypeError("sampler must be callable")
        self._sampler = sampler
        self._current_rss_mb = 0.0
        self._peak_rss_mb = 0.0
        self._limit_exceeded = False
        self._measurement_error: ResourceMeasurementUnavailable | None = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def current_rss_mb(self) -> float:
        with self._lock:
            return self._current_rss_mb

    @property
    def peak_rss_mb(self) -> float:
        with self._lock:
            return self._peak_rss_mb

    @property
    def limit_exceeded(self) -> bool:
        with self._lock:
            return self._limit_exceeded

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _latch_measurement_error(
        self, error: ResourceMeasurementUnavailable
    ) -> ResourceMeasurementUnavailable:
        with self._lock:
            if self._measurement_error is None:
                self._measurement_error = error
            return self._measurement_error

    def _raise_if_unavailable(self) -> None:
        with self._lock:
            error = self._measurement_error
        if error is not None:
            raise error

    def _sample(self) -> float:
        self._raise_if_unavailable()
        try:
            value = self._sampler()
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("RSS sampler must return a finite non-negative number")
            rss = float(value)
            if not math.isfinite(rss) or rss < 0.0:
                raise ValueError("RSS sampler must return a finite non-negative number")
        except ResourceMeasurementUnavailable as error:
            latched = self._latch_measurement_error(error)
            if latched is error:
                raise
            raise latched from error
        except (TypeError, ValueError) as cause:
            error = ResourceMeasurementUnavailable(
                stage="sampler_contract", cause=cause
            )
            raise self._latch_measurement_error(error) from cause
        with self._lock:
            if self._measurement_error is not None:
                raise self._measurement_error
            self._current_rss_mb = rss
            self._peak_rss_mb = max(self._peak_rss_mb, rss)
            if self.rss_ceiling_mb is not None and rss > self.rss_ceiling_mb:
                self._limit_exceeded = True
        return rss

    def _raise_if_exceeded(self) -> None:
        if self.limit_exceeded:
            raise ResourceLimitExceeded(
                observed_peak_mb=self.peak_rss_mb,
                ceiling_mb=float(self.rss_ceiling_mb),
            )

    def _poll(self) -> None:
        while not self._stop_event.wait(self.poll_interval_seconds):
            try:
                self._sample()
            except ResourceMeasurementUnavailable:
                self._stop_event.set()
                return

    def start(self) -> "ProcessTreeRSSMonitor":
        if self.is_running:
            raise RuntimeError("RSS monitor is already running")
        self._stop_event.clear()
        self._sample()
        self._thread = threading.Thread(
            target=self._poll,
            name="process-tree-rss-monitor",
            daemon=True,
        )
        self._thread.start()
        return self

    def checkpoint(self) -> float:
        rss = self._sample()
        self._raise_if_exceeded()
        return rss

    def stop(self, *, check_limit: bool = True) -> float:
        thread = self._thread
        if thread is not None:
            self._stop_event.set()
            thread.join(timeout=max(1.0, self.poll_interval_seconds * 2.0))
            if thread.is_alive():
                raise RuntimeError("RSS monitor thread did not stop")
            self._thread = None
        self._raise_if_unavailable()
        rss = self._sample()
        if check_limit:
            self._raise_if_exceeded()
        return rss

    def __enter__(self) -> "ProcessTreeRSSMonitor":
        return self.start()

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self.stop(check_limit=exc_type is None)
        return False


class TimingReport:
    """Track execution time, sections, and honest process-tree RSS."""

    def __init__(
        self,
        script_name: str,
        *,
        rss_ceiling_mb: float | None = None,
        poll_interval_seconds: float = 0.05,
        rss_monitor: ProcessTreeRSSMonitor | None = None,
    ) -> None:
        self.script_name = script_name
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.start_memory: Optional[float] = None
        self.end_memory: Optional[float] = None
        self.sections: List[Dict[str, Any]] = []
        self.current_section: Optional[str] = None
        self.section_start: Optional[float] = None
        self.section_start_memory: Optional[float] = None
        if rss_monitor is not None and rss_ceiling_mb is not None:
            raise ValueError("provide either rss_monitor or RSS ceiling, not both")
        self.rss_monitor = rss_monitor or ProcessTreeRSSMonitor(
            rss_ceiling_mb=rss_ceiling_mb,
            poll_interval_seconds=poll_interval_seconds,
        )

    def start(self) -> None:
        if self.start_time is not None and self.end_time is None:
            raise RuntimeError("timing report is already running")
        self.start_time = time.time()
        self.end_time = None
        self.rss_monitor.start()
        self.start_memory = self.rss_monitor.current_rss_mb
        logger.info(
            "Started timing for %s (process-tree RSS: %.1f MB)",
            self.script_name,
            self.start_memory,
        )

    def stop(self) -> None:
        section_error: BaseException | None = None
        try:
            if self.current_section:
                self.end_section()
        except BaseException as exc:
            section_error = exc
        self.end_time = time.time()
        try:
            self.end_memory = self.rss_monitor.stop(
                check_limit=section_error is None
            )
        finally:
            if section_error is not None:
                raise section_error
        logger.info(
            "Stopped timing for %s (endpoint RSS: %.1f MB, peak RSS: %.1f MB)",
            self.script_name,
            self.end_memory,
            self.rss_monitor.peak_rss_mb,
        )

    @contextmanager
    def section(self, section_name: str):
        if self.current_section:
            self.end_section()
        self.current_section = section_name
        self.section_start = time.time()
        self.section_start_memory = self.rss_monitor.checkpoint()
        try:
            yield
        finally:
            self.end_section()

    def end_section(self) -> None:
        if self.current_section is not None and self.section_start is not None:
            section_name = self.current_section
            section_start = self.section_start
            start_memory = self.section_start_memory or 0.0
            self.current_section = None
            self.section_start = None
            self.section_start_memory = None
            duration = time.time() - section_start
            end_memory = self.rss_monitor.checkpoint()
            self.sections.append(
                {
                    "section": section_name,
                    "duration_seconds": duration,
                    "duration_formatted": self._format_duration(duration),
                    "memory_mb": end_memory,
                    "memory_delta_mb": end_memory - start_memory,
                    "process_tree_peak_rss_mb": self.rss_monitor.peak_rss_mb,
                }
            )

    @property
    def total_duration(self) -> Optional[float]:
        if self.start_time is not None and self.end_time is not None:
            return self.end_time - self.start_time
        return None

    @staticmethod
    def _format_duration(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.2f}s"
        if seconds < 3600:
            minutes = int(seconds // 60)
            return f"{minutes}m {seconds % 60:.2f}s"
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m {seconds % 60:.2f}s"

    def to_dict(self) -> Dict[str, Any]:
        total = self.total_duration
        memory_delta = (
            self.end_memory - self.start_memory
            if self.start_memory is not None and self.end_memory is not None
            else None
        )
        peak = self.rss_monitor.peak_rss_mb
        return {
            "script_name": self.script_name,
            "start_time": datetime.fromtimestamp(self.start_time).isoformat()
            if self.start_time is not None
            else None,
            "end_time": datetime.fromtimestamp(self.end_time).isoformat()
            if self.end_time is not None
            else None,
            "total_duration_seconds": total,
            "total_duration_formatted": self._format_duration(total)
            if total is not None
            else None,
            "start_memory_mb": self.start_memory,
            "end_memory_mb": self.end_memory,
            "endpoint_process_tree_rss_mb": self.end_memory,
            "memory_delta_mb": memory_delta,
            "process_tree_peak_rss_mb": peak,
            "peak_memory_mb": peak,
            "rss_ceiling_mb": self.rss_monitor.rss_ceiling_mb,
            "rss_limit_exceeded": self.rss_monitor.limit_exceeded,
            "sections": self.sections,
            "section_count": len(self.sections),
        }

    def to_markdown(self) -> str:
        report = self.to_dict()
        total = report["total_duration_seconds"]
        lines = [
            f"# Execution Timing Report: {self.script_name}",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            "",
            f"- **Total Execution Time:** {report['total_duration_formatted'] or 'N/A'}",
            f"- **Number of Sections:** {len(self.sections)}",
            f"- **Start Process-tree RSS:** {self.start_memory:.1f} MB"
            if self.start_memory is not None
            else "- **Start Process-tree RSS:** N/A",
            f"- **Endpoint Process-tree RSS:** {self.end_memory:.1f} MB"
            if self.end_memory is not None
            else "- **Endpoint Process-tree RSS:** N/A",
            f"- **Peak Process-tree RSS:** {self.rss_monitor.peak_rss_mb:.1f} MB",
            f"- **RSS Ceiling:** {self.rss_monitor.rss_ceiling_mb:.1f} MB"
            if self.rss_monitor.rss_ceiling_mb is not None
            else "- **RSS Ceiling:** not set",
            f"- **RSS Limit Exceeded:** {self.rss_monitor.limit_exceeded}",
            "",
        ]
        if self.sections:
            lines.extend(
                [
                    "## Section Breakdown",
                    "",
                    "| Section | Duration | Endpoint RSS (MB) | Peak RSS (MB) |",
                    "|---------|----------|-------------------|---------------|",
                ]
            )
            for section in self.sections:
                lines.append(
                    f"| {section['section']} | {section['duration_formatted']} | "
                    f"{section['memory_mb']:.1f} | "
                    f"{section['process_tree_peak_rss_mb']:.1f} |"
                )
            if total is not None and total > 0:
                lines.extend(["", "### Section Percentages", ""])
                for section in self.sections:
                    percentage = section["duration_seconds"] / total * 100
                    lines.append(
                        f"- {section['section']}: {percentage:.1f}%"
                    )
        return "\n".join(lines)

    def save_report(self, output_dir: Path, format: str = "both") -> None:
        if format not in {"json", "markdown", "both"}:
            raise ValueError("report format must be 'json', 'markdown', or 'both'")
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        base_name = self.script_name.replace(".py", "").replace("_", "-")
        if format in {"json", "both"}:
            with (output / f"{base_name}_timing.json").open(
                "w", encoding="utf-8"
            ) as handle:
                json.dump(self.to_dict(), handle, indent=2)
        if format in {"markdown", "both"}:
            (output / f"{base_name}_timing.md").write_text(
                self.to_markdown(), encoding="utf-8"
            )

    def print_summary(self) -> None:
        report = self.to_dict()
        print("\n" + "=" * 60)
        print(f"Timing Report: {self.script_name}")
        print("=" * 60)
        print(f"Total Execution Time: {report['total_duration_formatted'] or 'N/A'}")
        print(f"Number of Sections: {len(self.sections)}")
        print(f"Peak Process-tree RSS: {self.rss_monitor.peak_rss_mb:.1f} MB")
        if self.end_memory is not None:
            print(f"Endpoint Process-tree RSS: {self.end_memory:.1f} MB")
        for section in self.sections:
            print(f"  - {section['section']}: {section['duration_formatted']}")
        print("=" * 60 + "\n")
