"""time-series-explanation 的接线。"""
from __future__ import annotations

import sys

from .base import TaskProfile
from ..evaluation import _scalarize


class TimeSeriesExplanationProfile(TaskProfile):
    def grade_argv(self, workspace, slot, summary_path, extra):
        return [
            sys.executable,
            str(workspace / "grading" / "grade.py"),
            "--logs", str(workspace / "logs"),
            "--json-out", str(summary_path),
            "--no-markdown",
            "--real-explainer", slot,
            "--synthetic-explainer", slot,
            *extra,
        ]

    def scalarize(self, summary):
        return _scalarize(summary)
