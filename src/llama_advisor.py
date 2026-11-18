"""
=============================================================================
🌱 CONVENIENCE IMPORT WRAPPER: llama_advisor.py
=============================================================================
Provides clean standard Python imports for 05_llama_advisor.py:
    from src.llama_advisor import LlamaCropAdvisor, generate_crop_advisory, AdvisorResponse
=============================================================================
"""

import sys
import importlib
import importlib.util
from pathlib import Path

_module_path = Path(__file__).resolve().parent / "05_llama_advisor.py"
_spec = importlib.util.spec_from_file_location("05_llama_advisor", str(_module_path))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

LlamaCropAdvisor = _mod.LlamaCropAdvisor
generate_crop_advisory = _mod.generate_crop_advisory
AdvisorResponse = _mod.AdvisorResponse
SYSTEM_PROMPT = _mod.SYSTEM_PROMPT

__all__ = [
    "LlamaCropAdvisor",
    "generate_crop_advisory",
    "AdvisorResponse",
    "SYSTEM_PROMPT",
]
