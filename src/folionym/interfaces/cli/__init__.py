"""Command-line interface entry points for Folionym.

Only the supported console launchers are re-exported here. Parser and runtime
helpers remain internal to this interface package.
"""

from .main import main, run_doctor_checks

__all__ = ["main", "run_doctor_checks"]
