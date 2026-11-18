"""
=============================================================================
🌱 CONVENIENCE IMPORT WRAPPER: severity_calculator.py
=============================================================================
Provides clean standard Python imports for 03_severity_calculator.py:
    from src.severity_calculator import LeafSeverityCalculator, calculate_leaf_severity, SeverityResult
=============================================================================
"""

import sys
import importlib
import importlib.util
from pathlib import Path

# Load 03_severity_calculator dynamically
_module_path = Path(__file__).resolve().parent / "03_severity_calculator.py"
_spec = importlib.util.spec_from_file_location("03_severity_calculator", str(_module_path))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

LeafSeverityCalculator = _mod.LeafSeverityCalculator
SeverityResult = _mod.SeverityResult
calculate_leaf_severity = _mod.calculate_leaf_severity
SEVERITY_LEVELS = _mod.SEVERITY_LEVELS

__all__ = [
    "LeafSeverityCalculator",
    "SeverityResult",
    "calculate_leaf_severity",
    "SEVERITY_LEVELS",
]
