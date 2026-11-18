"""
=============================================================================
🌱 CONVENIENCE IMPORT WRAPPER: knowledge_base.py
=============================================================================
Provides clean standard Python imports for 04_knowledge_base.py:
    from src.knowledge_base import TreatmentKnowledgeBase
=============================================================================
"""

import sys
import importlib
import importlib.util
from pathlib import Path

_module_path = Path(__file__).resolve().parent / "04_knowledge_base.py"
_spec = importlib.util.spec_from_file_location("04_knowledge_base", str(_module_path))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

TreatmentKnowledgeBase = _mod.TreatmentKnowledgeBase
CONDITIONS_DATA = _mod.CONDITIONS_DATA
DB_PATH = _mod.DB_PATH

__all__ = [
    "TreatmentKnowledgeBase",
    "CONDITIONS_DATA",
    "DB_PATH",
]
