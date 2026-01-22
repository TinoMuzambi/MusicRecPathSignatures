"""
Tests for timing utilities.

This module tests the TimingReport class and memory usage functions.
"""

import time
from src.utils.timing import TimingReport, get_memory_usage_mb


class TestGetMemoryUsageMb:
    """Test get_memory_usage_mb function."""

    def test_get_memory_usage_mb_returns_float(self):
        """Test that get_memory_usage_mb returns a float."""
        memory = get_memory_usage_mb()

        assert isinstance(memory, float)
        assert memory >= 0


class TestTimingReport:
    """Test TimingReport class."""

    def test_initialization(self):
        """Test TimingReport initialization."""
        report = TimingReport("test_script.py")

        assert report.script_name == "test_script.py"
        assert report.start_time is None
        assert report.end_time is None
        assert report.sections == []
        assert report.current_section is None

    def test_start_stop(self):
        """Test start and stop methods."""
        report = TimingReport("test_script.py")
        report.start()

        assert report.start_time is not None
        assert report.start_memory is not None

        time.sleep(0.1)  # Small delay to ensure time difference

        report.stop()

        assert report.end_time is not None
        assert report.end_memory is not None
        assert report.total_duration is not None
        assert report.total_duration > 0

    def test_section_context_manager(self):
        """Test section context manager."""
        report = TimingReport("test_script.py")
        report.start()

        with report.section("test_section"):
            time.sleep(0.05)

        report.stop()

        assert len(report.sections) == 1
        assert report.sections[0]["section"] == "test_section"
        assert report.sections[0]["duration_seconds"] > 0
        assert "duration_formatted" in report.sections[0]

    def test_multiple_sections(self):
        """Test multiple sections."""
        report = TimingReport("test_script.py")
        report.start()

        with report.section("section1"):
            time.sleep(0.02)

        with report.section("section2"):
            time.sleep(0.02)

        report.stop()

        assert len(report.sections) == 2
        assert report.sections[0]["section"] == "section1"
        assert report.sections[1]["section"] == "section2"

    def test_format_duration(self):
        """Test duration formatting."""
        report = TimingReport("test_script.py")

        # Test seconds format
        formatted = report._format_duration(45.5)
        assert "s" in formatted

        # Test minutes format
        formatted = report._format_duration(125.5)
        assert "m" in formatted

        # Test hours format
        formatted = report._format_duration(3665.5)
        assert "h" in formatted

    def test_to_dict(self):
        """Test conversion to dictionary."""
        report = TimingReport("test_script.py")
        report.start()

        with report.section("test_section"):
            time.sleep(0.01)

        report.stop()

        report_dict = report.to_dict()

        assert isinstance(report_dict, dict)
        assert "script_name" in report_dict
        assert "total_duration_seconds" in report_dict
        assert "total_duration_formatted" in report_dict
        assert "sections" in report_dict
        assert report_dict["script_name"] == "test_script.py"

    def test_to_markdown(self):
        """Test conversion to markdown."""
        report = TimingReport("test_script.py")
        report.start()

        with report.section("test_section"):
            time.sleep(0.01)

        report.stop()

        markdown = report.to_markdown()

        assert isinstance(markdown, str)
        assert "# Execution Timing Report" in markdown
        assert "test_script.py" in markdown
        assert "test_section" in markdown

    def test_save_report(self, tmp_path):
        """Test saving report to files."""
        report = TimingReport("test_script.py")
        report.start()

        with report.section("test_section"):
            time.sleep(0.01)

        report.stop()

        report.save_report(tmp_path, format="both")

        json_path = tmp_path / "test-script_timing.json"
        md_path = tmp_path / "test-script_timing.md"

        assert json_path.exists()
        assert md_path.exists()

        # Verify JSON can be loaded
        import json

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert "script_name" in data

        # Verify markdown content
        with open(md_path, "r", encoding="utf-8") as f:
            content = f.read()
            assert "Execution Timing Report" in content

    def test_save_report_json_only(self, tmp_path):
        """Test saving report as JSON only."""
        report = TimingReport("test_script.py")
        report.start()
        report.stop()

        report.save_report(tmp_path, format="json")

        json_path = tmp_path / "test-script_timing.json"
        md_path = tmp_path / "test-script_timing.md"

        assert json_path.exists()
        assert not md_path.exists()

    def test_save_report_markdown_only(self, tmp_path):
        """Test saving report as Markdown only."""
        report = TimingReport("test_script.py")
        report.start()
        report.stop()

        report.save_report(tmp_path, format="markdown")

        json_path = tmp_path / "test-script_timing.json"
        md_path = tmp_path / "test-script_timing.md"

        assert not json_path.exists()
        assert md_path.exists()

    def test_print_summary(self, capsys):
        """Test printing summary."""
        report = TimingReport("test_script.py")
        report.start()

        with report.section("test_section"):
            time.sleep(0.01)

        report.stop()

        report.print_summary()

        captured = capsys.readouterr()
        assert "Timing Report" in captured.out
        assert "test_script.py" in captured.out
        assert "test_section" in captured.out

    def test_total_duration_property(self):
        """Test total_duration property."""
        report = TimingReport("test_script.py")

        # Before start/stop, should be None
        assert report.total_duration is None

        report.start()
        time.sleep(0.01)
        report.stop()

        assert report.total_duration is not None
        assert report.total_duration > 0

    def test_nested_sections(self):
        """Test that ending a section properly handles nested calls."""
        report = TimingReport("test_script.py")
        report.start()

        # Manually start a section
        report.current_section = "section1"
        report.section_start = time.time()

        # Start another section (should end the previous one)
        with report.section("section2"):
            time.sleep(0.01)

        report.stop()

        # Should have both sections
        assert len(report.sections) == 2
        assert report.sections[0]["section"] == "section1"
        assert report.sections[1]["section"] == "section2"

