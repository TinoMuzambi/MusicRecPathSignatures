"""
Timing utilities for tracking script execution time and section durations.

This module provides a context manager and utilities for tracking execution time
of scripts and their key sections, generating reports suitable for dissertation
documentation.
"""

from __future__ import annotations

import time
import psutil
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional
import json
from datetime import datetime

from src.utils.logger_config import setup_logger


logger = setup_logger("timing")


def get_memory_usage_mb() -> float:
    """Get current memory usage in MB."""
    try:
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except (OSError, AttributeError, RuntimeError):  # psutil may fail in some environments
        return 0.0


class TimingReport:
    """Track and report execution times for scripts and sections."""

    def __init__(self, script_name: str):
        """
        Initialize timing report.

        Args:
            script_name: Name of the script being timed
        """
        self.script_name = script_name
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.start_memory: Optional[float] = None
        self.end_memory: Optional[float] = None
        self.sections: List[Dict[str, float]] = []
        self.current_section: Optional[str] = None
        self.section_start: Optional[float] = None
        self.section_start_memory: Optional[float] = None

    def start(self) -> None:
        """Start timing the script."""
        self.start_time = time.time()
        self.start_memory = get_memory_usage_mb()
        logger.info(
            "Started timing for %s (Memory: %.1f MB)",
            self.script_name,
            self.start_memory,
        )

    def stop(self) -> None:
        """Stop timing the script."""
        if self.current_section:
            self.end_section()
        self.end_time = time.time()
        self.end_memory = get_memory_usage_mb()
        memory_delta = (
            (self.end_memory - self.start_memory) if self.start_memory else 0.0
        )
        logger.info(
            "Stopped timing for %s (Memory: %.1f MB, Delta: %.1f MB)",
            self.script_name,
            self.end_memory,
            memory_delta,
        )

    @contextmanager
    def section(self, section_name: str):
        """
        Context manager for timing a section.

        Args:
            section_name: Name of the section being timed

        Example:
            with timing.section("Data Loading"):
                load_data()
        """
        if self.current_section:
            self.end_section()
        self.current_section = section_name
        self.section_start = time.time()
        self.section_start_memory = get_memory_usage_mb()
        logger.debug(
            "Started section: %s (Memory: %.1f MB)",
            section_name,
            self.section_start_memory,
        )
        try:
            yield
        finally:
            self.end_section()

    def end_section(self) -> None:
        """End the current section timing."""
        if self.current_section and self.section_start:
            duration = time.time() - self.section_start
            end_memory = get_memory_usage_mb()
            memory_delta = (
                (end_memory - self.section_start_memory)
                if self.section_start_memory
                else 0.0
            )

            self.sections.append(
                {
                    "section": self.current_section,
                    "duration_seconds": duration,
                    "duration_formatted": self._format_duration(duration),
                    "memory_mb": end_memory,
                    "memory_delta_mb": memory_delta,
                }
            )
            logger.debug(
                "Section '%s' completed in %s (Memory: %.1f MB, Delta: %.1f MB)",
                self.current_section,
                self._format_duration(duration),
                end_memory,
                memory_delta,
            )
            self.current_section = None
            self.section_start = None
            self.section_start_memory = None

    @property
    def total_duration(self) -> Optional[float]:
        """Get total execution time in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None

    def _format_duration(self, seconds: float) -> str:
        """
        Format duration in human-readable format.

        Args:
            seconds: Duration in seconds

        Returns:
            Formatted string (e.g., "2h 15m 30s" or "45.2s")
        """
        if seconds < 60:
            return f"{seconds:.2f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes}m {secs:.2f}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = seconds % 60
            return f"{hours}h {minutes}m {secs:.2f}s"

    def to_dict(self) -> Dict:
        """
        Convert timing report to dictionary.

        Returns:
            Dictionary with timing information
        """
        total = self.total_duration
        memory_delta = (
            (self.end_memory - self.start_memory)
            if (self.start_memory and self.end_memory)
            else None
        )
        return {
            "script_name": self.script_name,
            "start_time": (
                datetime.fromtimestamp(self.start_time).isoformat()
                if self.start_time
                else None
            ),
            "end_time": (
                datetime.fromtimestamp(self.end_time).isoformat()
                if self.end_time
                else None
            ),
            "total_duration_seconds": total,
            "total_duration_formatted": self._format_duration(total) if total else None,
            "start_memory_mb": self.start_memory,
            "end_memory_mb": self.end_memory,
            "memory_delta_mb": memory_delta,
            "peak_memory_mb": max(
                [s.get("memory_mb", 0) for s in self.sections] + [self.end_memory or 0]
            ),
            "sections": self.sections,
            "section_count": len(self.sections),
        }

    def to_markdown(self) -> str:
        """
        Convert timing report to markdown format.

        Returns:
            Markdown formatted report
        """
        total = self.total_duration
        memory_delta = (
            (self.end_memory - self.start_memory)
            if (self.start_memory and self.end_memory)
            else None
        )
        peak_memory = max(
            [s.get("memory_mb", 0) for s in self.sections] + [self.end_memory or 0]
        )

        lines = [
            f"# Execution Timing Report: {self.script_name}",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            "",
            f"- **Total Execution Time:** {self._format_duration(total) if total else 'N/A'}",
            f"- **Number of Sections:** {len(self.sections)}",
            (
                f"- **Start Memory:** {self.start_memory:.1f} MB"
                if self.start_memory
                else "- **Start Memory:** N/A"
            ),
            (
                f"- **End Memory:** {self.end_memory:.1f} MB"
                if self.end_memory
                else "- **End Memory:** N/A"
            ),
            (
                f"- **Memory Delta:** {memory_delta:+.1f} MB"
                if memory_delta is not None
                else "- **Memory Delta:** N/A"
            ),
            f"- **Peak Memory:** {peak_memory:.1f} MB",
            "",
        ]

        if self.start_time and self.end_time:
            lines.extend(
                [
                    f"- **Start Time:** {datetime.fromtimestamp(self.start_time).strftime('%Y-%m-%d %H:%M:%S')}",
                    f"- **End Time:** {datetime.fromtimestamp(self.end_time).strftime('%Y-%m-%d %H:%M:%S')}",
                    "",
                ]
            )

        if self.sections:
            lines.extend(
                [
                    "## Section Breakdown",
                    "",
                    "| Section | Duration | Memory (MB) | Memory Delta (MB) |",
                    "|---------|----------|-------------|-------------------|",
                ]
            )
            for sec in self.sections:
                memory_mb = sec.get("memory_mb", 0)
                memory_delta = sec.get("memory_delta_mb", 0)
                lines.append(
                    f"| {sec['section']} | {sec['duration_formatted']} | {memory_mb:.1f} | {memory_delta:+.1f} |"
                )

            # Calculate percentages if total is available
            if total and total > 0:
                lines.extend(
                    [
                        "",
                        "### Section Percentages",
                        "",
                        "| Section | Duration | Percentage |",
                        "|---------|----------|------------|",
                    ]
                )
                for sec in self.sections:
                    pct = (sec["duration_seconds"] / total) * 100
                    lines.append(
                        f"| {sec['section']} | {sec['duration_formatted']} | {pct:.1f}% |"
                    )

        return "\n".join(lines)

    def save_report(self, output_dir: Path, report_format: str = "both") -> None:
        """
        Save timing report to files.

        Args:
            output_dir: Directory to save reports
            format: Format to save ('json', 'markdown', or 'both')
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        base_name = self.script_name.replace(".py", "").replace("_", "-")

        if report_format in ("json", "both"):
            json_path = output_dir / f"{base_name}_timing.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2)
            logger.info("Saved timing report (JSON) to %s", json_path)

        if report_format in ("markdown", "both"):
            md_path = output_dir / f"{base_name}_timing.md"
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(self.to_markdown())
            logger.info("Saved timing report (Markdown) to %s", md_path)

    def print_summary(self) -> None:
        """Print a summary of the timing report to console."""
        total = self.total_duration
        memory_delta = (
            (self.end_memory - self.start_memory)
            if (self.start_memory and self.end_memory)
            else None
        )
        peak_memory = max(
            [s.get("memory_mb", 0) for s in self.sections] + [self.end_memory or 0]
        )

        print("\n" + "=" * 60)
        print(f"Timing Report: {self.script_name}")
        print("=" * 60)
        print(
            f"Total Execution Time: {self._format_duration(total) if total else 'N/A'}"
        )
        print(f"Number of Sections: {len(self.sections)}")
        if self.start_memory:
            print(f"Start Memory: {self.start_memory:.1f} MB")
        if self.end_memory:
            print(f"End Memory: {self.end_memory:.1f} MB")
        if memory_delta is not None:
            print(f"Memory Delta: {memory_delta:+.1f} MB")
        print(f"Peak Memory: {peak_memory:.1f} MB")
        if self.sections:
            print("\nSection Breakdown:")
            for sec in self.sections:
                pct = (
                    (sec["duration_seconds"] / total * 100)
                    if total and total > 0
                    else 0
                )
                memory_mb = sec.get("memory_mb", 0)
                memory_delta_sec = sec.get("memory_delta_mb", 0)
                print(
                    f"  - {sec['section']}: {sec['duration_formatted']} ({pct:.1f}%) "
                    f"[Memory: {memory_mb:.1f} MB, Delta: {memory_delta_sec:+.1f} MB]"
                )
        print("=" * 60 + "\n")
