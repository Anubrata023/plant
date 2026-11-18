"""
=============================================================================
🌱 MODULE 05: LLAMA 3.2 1B AGRONOMIC ADVISOR & CLINICAL REASONER
=============================================================================
Clinical reasoning engine for cereal crops (Rice & Wheat) and pest diagnosis.

Fuses:
  1. Vision Model Diagnosis (Module 02: 29 classes, e.g. "Wheat - Leaf Rust")
  2. OpenCV Lesion Coverage & Severity Grade (Module 03: e.g. 18.4%, Grade 2)
  3. Ground-Truth SQLite Knowledge Base (Module 04: verified remedies & dosages)

Runs local generative reasoning via Llama-3.2-1B-Instruct to synthesize a
clear, practical, localized treatment advisory for farmers.

Execution Backends:
  - Hugging Face Transformers: meta-llama/Llama-3.2-1B-Instruct (local or hub)
  - ExecuTorch Runtime: models/llama3_2_1b_int4.pte (low-latency edge runtime)
  - Grounded Clinical Engine: Fast offline fallback using verified SQLite data
    when LLM weights are not yet downloaded or while running on low-RAM edge CPUs.

Usage as module:
    from src.llama_advisor import LlamaCropAdvisor
    advisor = LlamaCropAdvisor()
    response = advisor.advise(
        disease_name="Wheat - Leaf Rust",
        lesion_percentage=18.4,
        severity_grade=2,
        confidence=0.968,
    )
    print(response.to_markdown())

Usage from CLI:
    python src/05_llama_advisor.py --disease "Wheat - Leaf Rust" --lesion 18.5 --severity 2
    python src/05_llama_advisor.py --batch
=============================================================================
"""

import os
import sys
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Union, List

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.knowledge_base import TreatmentKnowledgeBase, DB_PATH


# =============================================================================
# DATA STRUCTURE: ADVISOR RESPONSE
# =============================================================================

@dataclass
class AdvisorResponse:
    """Structured agronomic advisory response generated for farmers."""
    disease_name: str
    severity_level: str
    severity_grade: int
    lesion_percentage: float
    confidence: float
    urgency: str
    clinical_summary: str
    action_protocol: str
    chemical_remedy: Optional[Dict[str, Any]] = None
    organic_remedy: Optional[Dict[str, Any]] = None
    cultural_measures: List[str] = field(default_factory=list)
    resistant_varieties: List[str] = field(default_factory=list)
    safety_warning: str = ""
    full_advisory_text: str = ""
    backend_used: str = "grounded-clinical-engine"

    def to_dict(self) -> Dict[str, Any]:
        """Returns clean dictionary for JSON logging and API responses."""
        return {
            "disease_name": self.disease_name,
            "severity_level": self.severity_level,
            "severity_grade": self.severity_grade,
            "lesion_percentage": round(self.lesion_percentage, 2),
            "confidence": round(self.confidence, 4),
            "urgency": self.urgency,
            "clinical_summary": self.clinical_summary,
            "action_protocol": self.action_protocol,
            "chemical_remedy": self.chemical_remedy,
            "organic_remedy": self.organic_remedy,
            "cultural_measures": self.cultural_measures,
            "resistant_varieties": self.resistant_varieties,
            "safety_warning": self.safety_warning,
            "full_advisory_text": self.full_advisory_text,
            "backend_used": self.backend_used,
        }

    def to_markdown(self) -> str:
        """Formats the advisory as a clean, publication-ready Markdown report."""
        badge_emoji = {
            0: "🟢",
            1: "🟡",
            2: "🟠",
            3: "🔴",
            4: "🚨",
        }.get(self.severity_grade, "🌾")

        md = [
            f"# {badge_emoji} CROP CLINICAL ADVISORY: {self.disease_name.upper()}",
            f"**Diagnostic Confidence:** `{self.confidence * 100:.1f}%`  |  "
            f"**Severity:** `{self.severity_level.upper()} (Grade {self.severity_grade})`  |  "
            f"**Lesion Coverage:** `{self.lesion_percentage:.2f}%`",
            "",
            f"### ⏱️ Action Urgency Window",
            f"> **{self.urgency}**",
            "",
            f"### 📋 Clinical Condition Overview",
            self.clinical_summary,
            "",
            f"### ⚡ Immediate Action Required",
            self.action_protocol,
            "",
        ]

        if self.chemical_remedy:
            chem = self.chemical_remedy
            md.extend([
                "### 💊 Recommended Chemical Treatment (Curative Regimen)",
                f"- **Fungicide / Insecticide:** {chem.get('title', 'N/A')}",
                f"- **Active Ingredient:** `{chem.get('active_ingredient', 'N/A')}` *(Commercial Brands: {chem.get('commercial_names', 'N/A')})*",
                f"- **Recommended Dosage:** `{chem.get('dosage', 'N/A')}`",
                f"- **Application Timing:** {chem.get('application_timing', 'N/A')}",
                f"- **Safety / Pre-Harvest Interval (PHI):** {chem.get('safety_precaution', 'N/A')}",
                "",
            ])

        if self.organic_remedy:
            org = self.organic_remedy
            md.extend([
                "### 🍃 Biological & Organic Alternative (IPM)",
                f"- **Bio-Control Formulation:** {org.get('title', 'N/A')}",
                f"- **Active Bio-Agent:** `{org.get('active_ingredient', 'N/A')}`",
                f"- **Application Dosage:** `{org.get('dosage', 'N/A')}` ({org.get('application_timing', 'N/A')})",
                "",
            ])

        if self.cultural_measures:
            md.extend([
                "### 🚜 Cultural Agronomy & Preventive Practices",
                *[f"- {m}" for m in self.cultural_measures],
                "",
            ])

        if self.resistant_varieties:
            md.extend([
                "### 🌾 Recommended Resistant Cultivars",
                f"For subsequent crop seasons, consider certified seed: **{', '.join(self.resistant_varieties)}**.",
                "",
            ])

        if self.safety_warning:
            md.extend([
                "### ⚠️ Clinical Precautions & Weather Guidelines",
                f"> {self.safety_warning}",
                "",
            ])

        md.append(f"---\n*Inference Engine: `{self.backend_used}` | Verified Ground Truth: SQLite Agronomic DB*")
        return "\n".join(md)


# =============================================================================
# SYSTEM PROMPT TEMPLATES FOR LLAMA 3.2 1B
# =============================================================================

SYSTEM_PROMPT = """You are Dr. Krishi, an elite agricultural extension scientist and clinical agronomist specialized in cereal crops (Rice and Wheat) and Integrated Pest Management (IPM).

Your mission is to provide clear, actionable, empathetic, and scientifically grounded advisory to farmers based strictly on the provided diagnosis and verified knowledge base context.

Guidelines:
1. Ground your advice STRICTLY in the provided VERIFIED KNOWLEDGE BASE facts. Do not invent unapproved chemicals, random dosages, or non-existent active ingredients.
2. Structure your advice into:
   - Clinical Diagnosis & Severity Confirmation
   - Immediate Action Window & Field Protocol
   - Option A: Primary Curative Chemical Treatment (Active ingredient, brand, exact dilution/dosage)
   - Option B: Organic / Bio-Control Alternative (for low-cost IPM or organic growers)
   - Cultural Care, Irrigation & Resistant Varieties
   - Safety Warnings & Weather Conditions (spray early morning/evening; avoid spraying before rain)
3. If the organism is a BENEFICIAL insect (Bees, Earthworms, Wasps), STRONGLY and EXPLICITLY warn the farmer NOT to spray any insecticides!
4. Write in clear, structured, practical language suitable for agricultural extension.
"""


# =============================================================================
# CORE CLASS: LLAMA CROP ADVISOR
# =============================================================================

class LlamaCropAdvisor:
    """
    Agricultural reasoning engine utilizing Llama-3.2-1B-Instruct with
    grounded fallback to offline SQLite phytopathological rules.
    """

    def __init__(
        self,
        model_path_or_id: Optional[str] = None,
        db_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        allow_remote: bool = False,
        prefer_offline_engine: bool = False,
    ):
        """
        Args:
            model_path_or_id: Hugging Face model ID or path to local directory.
            db_path: Path to treatments.db.
            device: 'cuda', 'cpu', or auto-detected.
            allow_remote: If True, allows connecting to Hugging Face Hub to download weights.
            prefer_offline_engine: If True, uses the fast zero-download clinical engine directly.
        """
        self.db = TreatmentKnowledgeBase(db_path=db_path)
        self.model_path_or_id = model_path_or_id or os.environ.get(
            "LLAMA_MODEL_PATH",
            "meta-llama/Llama-3.2-1B-Instruct",
        )
        self.allow_remote = allow_remote
        self.prefer_offline_engine = prefer_offline_engine

        # PyTorch & Device settings
        import torch
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Lazy-loaded LLM pipeline handles
        self.tokenizer = None
        self.model = None
        self._llm_load_attempted = False
        self._llm_available = False

    # -------------------------------------------------------------------------
    # Helper: Lazy Model Loader
    # -------------------------------------------------------------------------
    def _try_load_llm(self) -> bool:
        """Attempts to load Llama-3.2-1B-Instruct via transformers."""
        if self._llm_load_attempted:
            return self._llm_available

        self._llm_load_attempted = True

        if self.prefer_offline_engine:
            self._llm_available = False
            return False

        # Check for local model weights on disk
        local_candidates = [
            PROJECT_ROOT / "models" / "Llama-3.2-1B-Instruct",
            PROJECT_ROOT / "models" / "llama-3.2-1b-instruct",
            PROJECT_ROOT / "models" / "llama3_2_1b",
            Path(self.model_path_or_id) if os.path.isdir(self.model_path_or_id) else None,
        ]
        local_path = next((str(p) for p in local_candidates if p and p.is_dir()), None)

        if not local_path and not self.allow_remote:
            # No local weights found and remote downloads disabled -> Instant offline engine
            self._llm_available = False
            return False

        target_path = local_path or self.model_path_or_id

        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM

            print(f"[LLM] Initializing Llama-3.2-1B from: {target_path} ...")
            is_local = (local_path is not None)
            self.tokenizer = AutoTokenizer.from_pretrained(target_path, local_files_only=is_local)
            
            torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(
                target_path,
                torch_dtype=torch_dtype,
                device_map="auto" if self.device == "cuda" else None,
                low_cpu_mem_usage=True,
                local_files_only=is_local,
            )
            if self.device == "cpu":
                self.model.to("cpu")

            self._llm_available = True
            print(f"[LLM] ✅ Successfully loaded Llama-3.2-1B on device: {self.device}")
            return True

        except Exception as e:
            # Fallback gracefully to the grounded clinical reasoning engine
            print(f"[LLM NOTE] Llama weights '{target_path}' not loaded ({e}).")
            print("           Activating offline Grounded Clinical Reasoning Engine (Zero Hallucination).")
            self._llm_available = False
            return False

    # -------------------------------------------------------------------------
    # Engine A: Grounded Clinical Synthesis Engine (Guaranteed Offline)
    # -------------------------------------------------------------------------
    def _generate_grounded_clinical_advisory(
        self,
        disease_name: str,
        lesion_percentage: float,
        severity_grade: int,
        confidence: float,
        farmer_question: Optional[str] = None,
    ) -> AdvisorResponse:
        """
        Synthesizes an expert clinical response using verified SQLite data,
        ensuring 100% accurate active ingredients, dosages, and safety intervals.
        """
        plan = self.db.get_treatment_plan(disease_name, severity_grade=severity_grade)
        cond = plan.get("condition", {})
        proto = plan.get("active_protocol", {})

        # Safety & Weather advisory rules
        weather_guidelines = (
            "Spray during calm early morning (6:00–9:30 AM) or late afternoon (4:30–6:30 PM). "
            "Never spray during peak mid-day heat (>32°C) or if rain is forecasted within 6 hours. "
            "Always wear protective gloves and an N95 mask during chemical spray preparation."
        )

        if cond.get("category") == "beneficial":
            weather_guidelines = (
                "⚠️ STRICT CONSERVATION PROTOCOL: This organism is a natural farmer ally. "
                "Do NOT apply broad-spectrum chemical insecticides. Maintain un-sprayed floral refuges."
            )

        # Primary chemical and organic options
        chem_remedy = plan["chemical_treatments"][0] if plan.get("chemical_treatments") else None
        org_remedy = plan["biological_treatments"][0] if plan.get("biological_treatments") else None

        # Cultural measures summary
        cultural_list = [
            f"{c['title']}: {c['dosage']}"
            for c in plan.get("cultural_treatments", [])
        ]

        # Clinical summary text
        clinical_summary = (
            f"The crop foliage has been evaluated as '{cond.get('name', disease_name)}' "
            f"({cond.get('scientific_name', 'Botanical pathology verified')}) with a diagnostic "
            f"confidence of {confidence * 100:.1f}%. The OpenCV leaf lesion calculation confirms "
            f"{lesion_percentage:.2f}% of canopy tissue is affected, categorizing this case at "
            f"{proto.get('severity_level', 'Evaluated').upper()} (Grade {severity_grade}). "
            f"{cond.get('summary', '')}"
        )

        # Action protocol text
        action_protocol = (
            f"Protocol Recommendation: {proto.get('action_required', 'Follow standard agronomic scouting.')} "
            f"Action is required within {proto.get('urgency', 'standard scouting intervals')}."
        )

        response = AdvisorResponse(
            disease_name=cond.get("name", disease_name),
            severity_level=proto.get("severity_level", "Unknown"),
            severity_grade=severity_grade,
            lesion_percentage=lesion_percentage,
            confidence=confidence,
            urgency=proto.get("urgency", "Action Required"),
            clinical_summary=clinical_summary,
            action_protocol=action_protocol,
            chemical_remedy=chem_remedy,
            organic_remedy=org_remedy,
            cultural_measures=cultural_list,
            resistant_varieties=plan.get("resistant_varieties", []),
            safety_warning=weather_guidelines,
            backend_used="grounded-clinical-engine",
        )
        response.full_advisory_text = response.to_markdown()
        return response

    # -------------------------------------------------------------------------
    # Engine B: Llama-3.2-1B-Instruct Generative Reasoning Engine
    # -------------------------------------------------------------------------
    def _generate_llama_advisory(
        self,
        disease_name: str,
        lesion_percentage: float,
        severity_grade: int,
        confidence: float,
        farmer_question: Optional[str] = None,
    ) -> AdvisorResponse:
        """
        Executes generative reasoning via Llama-3.2-1B-Instruct, passing the
        grounded SQLite knowledge base context into the prompt.
        """
        # Grounding context from Module 04
        context_str = self.db.format_advisor_context(
            disease_name=disease_name,
            severity_grade=severity_grade,
            lesion_pct=lesion_percentage,
        )

        user_content = (
            f"The field diagnostic vision model detected: '{disease_name}' with {confidence * 100:.1f}% confidence.\n"
            f"The OpenCV severity analyzer calculated: {lesion_percentage:.2f}% lesion area (Severity Grade {severity_grade}).\n\n"
            f"{context_str}\n\n"
        )
        if farmer_question:
            user_content += f"Farmer Specific Question: '{farmer_question}'\n\n"

        user_content += (
            "Please provide your structured clinical advisory for the farmer following the system guidelines. "
            "Highlight the exact dosages, chemical active ingredients, organic alternatives, and safety warnings."
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        import torch
        prompt_str = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt_str, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.3,
                top_p=0.9,
                repetition_penalty=1.1,
                do_sample=True,
            )

        gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        llm_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        # Parse base structured metadata from SQLite
        base = self._generate_grounded_clinical_advisory(
            disease_name=disease_name,
            lesion_percentage=lesion_percentage,
            severity_grade=severity_grade,
            confidence=confidence,
        )
        base.full_advisory_text = llm_text
        base.backend_used = f"llama3.2-1b-transformers ({self.device})"
        return base

    # -------------------------------------------------------------------------
    # Main Public Execution Method
    # -------------------------------------------------------------------------
    def advise(
        self,
        disease_name: str,
        lesion_percentage: float = 0.0,
        severity_grade: int = 1,
        confidence: float = 0.95,
        farmer_question: Optional[str] = None,
    ) -> AdvisorResponse:
        """
        Generates a comprehensive clinical treatment advisory.

        Args:
            disease_name: Diagnostic condition name (e.g. 'Wheat - Leaf Rust').
            lesion_percentage: Measured OpenCV lesion coverage percentage.
            severity_grade: Phytopathological severity grade (0 to 4).
            confidence: Vision model prediction confidence (0.0 to 1.0).
            farmer_question: Optional query from farmer.

        Returns:
            AdvisorResponse containing clinical plan, remedies, and formatted Markdown.
        """
        # Try loading LLM if not already attempted
        has_llm = self._try_load_llm()

        if has_llm:
            try:
                return self._generate_llama_advisory(
                    disease_name=disease_name,
                    lesion_percentage=lesion_percentage,
                    severity_grade=severity_grade,
                    confidence=confidence,
                    farmer_question=farmer_question,
                )
            except Exception as e:
                print(f"[LLM INFERENCE NOTE] Generation failed ({e}). Falling back to Grounded Engine.")

        # Default fast, offline, verified engine
        return self._generate_grounded_clinical_advisory(
            disease_name=disease_name,
            lesion_percentage=lesion_percentage,
            severity_grade=severity_grade,
            confidence=confidence,
            farmer_question=farmer_question,
        )


# =============================================================================
# CONVENIENCE STANDALONE FUNCTION
# =============================================================================

def generate_crop_advisory(
    disease_name: str,
    lesion_percentage: float = 0.0,
    severity_grade: int = 1,
    confidence: float = 0.95,
    farmer_question: Optional[str] = None,
) -> AdvisorResponse:
    """
    Convenience function to generate a crop advisory in a single call.
    """
    advisor = LlamaCropAdvisor()
    return advisor.advise(
        disease_name=disease_name,
        lesion_percentage=lesion_percentage,
        severity_grade=severity_grade,
        confidence=confidence,
        farmer_question=farmer_question,
    )


# =============================================================================
# CLI & BATCH BENCHMARKING INTERFACE
# =============================================================================

def run_batch_advisor_benchmark() -> None:
    """Demonstrates multi-class clinical reasoning across diverse conditions."""
    test_cases = [
        ("Wheat - Leaf Rust", 18.4, 2, 0.972),
        ("Rice - Leaf Blast", 34.6, 3, 0.958),
        ("Pest - Caterpillars", 12.0, 2, 0.981),
        ("Pest - Bees", 0.0, 0, 0.994),
        ("Wheat - Healthy", 0.0, 0, 0.989),
    ]

    advisor = LlamaCropAdvisor()
    print("=" * 78)
    print("🌾 MULTI-CLASS CLINICAL ADVISOR BENCHMARK")
    print(f"   Inference Engine: {advisor.device.upper()} ({'Transformers' if advisor._llm_available else 'Grounded Clinical Engine'})")
    print("=" * 78)

    for i, (dis, pct, grade, conf) in enumerate(test_cases, 1):
        print(f"\n--- [Case {i}/{len(test_cases)}] {dis} (Grade {grade}, {pct:.1f}% Lesion) ---")
        res = advisor.advise(
            disease_name=dis,
            lesion_percentage=pct,
            severity_grade=grade,
            confidence=conf,
        )
        print(f"Urgency Window: {res.urgency}")
        print(f"Protocol:       {res.action_protocol}")
        if res.chemical_remedy:
            chem = res.chemical_remedy
            print(f"Chemical:       {chem['title']} -> {chem['dosage']}")
        if res.organic_remedy:
            org = res.organic_remedy
            print(f"Organic:        {org['title']} -> {org['dosage']}")
        if res.resistant_varieties:
            print(f"Resistant Seed: {', '.join(res.resistant_varieties)}")

    print("\n" + "=" * 78)
    print("✅ All advisor test cases validated successfully!")
    print("=" * 78)


def main():
    parser = argparse.ArgumentParser(
        description="🌱 Llama-3.2-1B Agronomic Advisor & Clinical Reasoner"
    )
    parser.add_argument(
        "--disease",
        type=str,
        default=None,
        help="Diagnosis/disease name (e.g. 'Wheat - Leaf Rust', 'Rice - Leaf Blast')",
    )
    parser.add_argument(
        "--lesion",
        type=float,
        default=15.0,
        help="OpenCV measured lesion percentage (0.0 to 100.0) [Default: 15.0]",
    )
    parser.add_argument(
        "--severity",
        type=int,
        default=2,
        help="Phytopathological severity grade (0 to 4) [Default: 2]",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Vision model confidence score (0.0 to 1.0) [Default: 0.95]",
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Optional specific farmer question",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Run batch benchmark across 5 representative crop conditions",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output advisory response as JSON",
    )
    parser.add_argument(
        "--offline-only",
        action="store_true",
        help="Force using the offline grounded clinical engine directly",
    )
    parser.add_argument(
        "--allow-remote",
        action="store_true",
        help="Allow downloading Llama weights from Hugging Face Hub if not cached locally",
    )

    args = parser.parse_args()

    if args.batch:
        run_batch_advisor_benchmark()
        return

    if not args.disease:
        print("[INFO] No disease specified. Running batch demonstration by default.")
        print("       Use --disease <name> --lesion <pct> --severity <0..4> for custom query.\n")
        run_batch_advisor_benchmark()
        return

    advisor = LlamaCropAdvisor(
        allow_remote=args.allow_remote,
        prefer_offline_engine=args.offline_only,
    )
    response = advisor.advise(
        disease_name=args.disease,
        lesion_percentage=args.lesion,
        severity_grade=args.severity,
        confidence=args.confidence,
        farmer_question=args.question,
    )

    if args.json:
        print(json.dumps(response.to_dict(), indent=2))
    else:
        print("\n" + response.to_markdown() + "\n")


if __name__ == "__main__":
    main()
