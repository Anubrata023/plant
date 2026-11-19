"""
=============================================================================
🌱 COMPLETE END-TO-END DIAGNOSTIC PIPELINE: VISION + SEVERITY + LLAMA
=============================================================================
Orchestrates the entire clinical diagnostic pipeline in a single call:
  1. Deep Learning Vision Model (EfficientNet-B0): Predicts disease/pest class & confidence.
  2. OpenCV Severity Engine: Quantifies exact leaf lesion % & IRRI/Cobb severity grade.
  3. Offline SQLite Knowledge Base: Retrieves verified remedies, dosages & safety intervals.
  4. Llama-3.2-1B Agronomic Advisor: Synthesizes personalized clinical guidance for farmers.

Usage:
    python src/diagnose.py --image path/to/leaf.jpg
    python src/diagnose.py --image path/to/leaf.jpg --output-dir output/
=============================================================================
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Union

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.severity_calculator import LeafSeverityCalculator, SeverityResult
from src.knowledge_base import TreatmentKnowledgeBase
from src.llama_advisor import LlamaCropAdvisor, AdvisorResponse


# =============================================================================
# UNIFIED CROP DIAGNOSTIC PIPELINE
# =============================================================================

class CropDiagnosticPipeline:
    """
    End-to-end engine linking Vision Classifier, OpenCV Severity Engine,
    SQLite Knowledge Base, and Llama 3.2 1B Agronomic Reasoner.
    """

    def __init__(
        self,
        vision_model_path: Optional[str] = None,
        classes_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.classes_path = Path(classes_path or (PROJECT_ROOT / "data" / "class_names.json"))
        self.vision_model_path = Path(vision_model_path or (PROJECT_ROOT / "models" / "best_efficientnet_b0.pth"))
        
        # Load class mappings
        with open(self.classes_path, "r", encoding="utf-8") as f:
            raw_classes = json.load(f)
        self.class_names = {int(k): v for k, v in raw_classes.items()}
        self.num_classes = len(self.class_names)

        # Device determination (safe CPU fallback for RTX 5050 pending sm_120 torch wheel)
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cpu")
            if torch.cuda.is_available():
                try:
                    _ = torch.zeros(1, device="cuda")
                    self.device = torch.device("cuda")
                except Exception:
                    self.device = torch.device("cpu")

        # Initialize Vision Model
        self.vision_model = self._load_vision_model()
        self.vision_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Initialize Subsystems
        self.severity_calculator = LeafSeverityCalculator()
        self.knowledge_base = TreatmentKnowledgeBase()
        self.llama_advisor = LlamaCropAdvisor(device=str(self.device))

    def _load_vision_model(self) -> nn.Module:
        """Loads trained PyTorch EfficientNet-B0 classifier."""
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, self.num_classes)

        if self.vision_model_path.exists():
            ckpt = torch.load(self.vision_model_path, map_location=self.device, weights_only=False)
            state_dict = ckpt.get("state_dict", ckpt)
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
        else:
            raise FileNotFoundError(f"Vision model weights not found at: {self.vision_model_path}")

        return model

    def run_diagnosis(
        self,
        image_path: Union[str, Path],
        farmer_question: Optional[str] = None,
        save_visualizations: bool = True,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        Executes complete pipeline: Image -> Vision -> Severity -> SQLite -> Llama Advisor.
        """
        img_path = Path(image_path)
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        # ---------------------------------------------------------------------
        # STEP 1: Vision Model Inference (EfficientNet-B0)
        # ---------------------------------------------------------------------
        raw_pil = Image.open(img_path).convert("RGB")
        tensor_inp = self.vision_transforms(raw_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.vision_model(tensor_inp)
            probs = torch.softmax(logits, dim=1)[0]
            top_id = int(torch.argmax(probs).item())
            confidence = float(probs[top_id].item())
            predicted_class = self.class_names[top_id]

            # Stage 1: Crop / Specimen Group Identification (Wheat vs Rice vs Pest)
            crop_scores = {"Wheat": 0.0, "Rice": 0.0, "Pest": 0.0}
            for idx, p in enumerate(probs):
                cname = self.class_names[idx]
                if cname.startswith("Wheat"):
                    crop_scores["Wheat"] += float(p)
                elif cname.startswith("Rice"):
                    crop_scores["Rice"] += float(p)
                elif cname.startswith("Pest"):
                    crop_scores["Pest"] += float(p)

            detected_crop = max(crop_scores, key=crop_scores.get)
            crop_confidence = crop_scores[detected_crop]

            # Stage 2: Specific Condition / Disease Identification
            if " - " in predicted_class:
                crop_prefix, condition_name = predicted_class.split(" - ", 1)
            else:
                crop_prefix, condition_name = detected_crop, predicted_class

            is_healthy = "healthy" in condition_name.lower()

            # Candidate conditions within the identified crop group
            crop_candidates = []
            for idx, p in enumerate(probs):
                cname = self.class_names[idx]
                if cname.startswith(detected_crop):
                    c_cond = cname.split(" - ", 1)[1] if " - " in cname else cname
                    crop_candidates.append({
                        "condition": c_cond,
                        "full_class": cname,
                        "confidence": round(float(p.item()), 4),
                    })
            crop_candidates.sort(key=lambda x: x["confidence"], reverse=True)

            # Top-3 predictions overall for clinical certainty
            top3_probs, top3_ids = torch.topk(probs, k=min(3, self.num_classes))
            top3 = [
                {"class_name": self.class_names[int(idx.item())], "confidence": round(float(p.item()), 4)}
                for p, idx in zip(top3_probs, top3_ids)
            ]

        # ---------------------------------------------------------------------
        # STEP 2: OpenCV Severity & Lesion Surface Area Quantification
        # ---------------------------------------------------------------------
        severity_result = self.severity_calculator.calculate_severity(
            image_input=str(img_path),
            disease_name=predicted_class,
            generate_canvas=True,
        )

        # ---------------------------------------------------------------------
        # STEP 3: Llama-3.2-1B Clinical Advisory Reasoning
        # ---------------------------------------------------------------------
        advisory_response = self.llama_advisor.advise(
            disease_name=predicted_class,
            lesion_percentage=severity_result.lesion_percentage,
            severity_grade=severity_result.severity_grade,
            confidence=confidence,
            farmer_question=farmer_question,
        )

        # ---------------------------------------------------------------------
        # STEP 4: Save Diagnostic Outputs (Optional)
        # ---------------------------------------------------------------------
        saved_paths = {}
        if save_visualizations:
            out_folder = Path(output_dir or (PROJECT_ROOT / "models" / "diagnosis_outputs"))
            out_folder.mkdir(parents=True, exist_ok=True)
            stem = img_path.stem

            overlay_file = out_folder / f"{stem}_annotated.jpg"
            canvas_file = out_folder / f"{stem}_diagnostic_canvas.jpg"

            severity_result.save(overlay_file, save_canvas=False)
            severity_result.save(canvas_file, save_canvas=True)
            saved_paths["overlay"] = str(overlay_file)
            saved_paths["canvas"] = str(canvas_file)

        return {
            "image_path": str(img_path),
            "vision": {
                "predicted_class": predicted_class,
                "confidence": round(confidence, 4),
                "detected_crop": detected_crop,
                "crop_confidence": round(crop_confidence, 4),
                "crop_scores": {k: round(v, 4) for k, v in crop_scores.items()},
                "condition_name": condition_name,
                "is_healthy": is_healthy,
                "top_crop_candidates": crop_candidates[:4],
                "top3": top3,
            },
            "severity": {
                "lesion_percentage": severity_result.lesion_percentage,
                "severity_level": severity_result.severity_level,
                "severity_grade": severity_result.severity_grade,
                "leaf_pixels": severity_result.total_leaf_pixels,
                "lesion_pixels": severity_result.lesion_pixels,
                "color_hex": severity_result.color_hex,
            },
            "advisory": advisory_response.to_dict(),
            "saved_visualizations": saved_paths,
            "markdown_report": advisory_response.to_markdown(),
        }


# =============================================================================
# CLI ENTRYPOINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="🌾 Complete Diagnostic Pipeline: Vision + Severity + Llama Advisor"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to crop foliage image file",
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Optional specific farmer question",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save visual overlays and 4-panel diagnostic canvas",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output complete diagnostic results as JSON",
    )

    args = parser.parse_args()

    pipeline = CropDiagnosticPipeline()
    result = pipeline.run_diagnosis(
        image_path=args.image,
        farmer_question=args.question,
        save_visualizations=True,
        output_dir=args.output_dir,
    )

    if args.json:
        # Strip out long markdown for clean JSON display
        clean_res = result.copy()
        print(json.dumps(clean_res, indent=2))
    else:
        vis = result["vision"]
        sev = result["severity"]
        adv = result["advisory"]

        crop_icon = "🌾" if vis["detected_crop"] == "Wheat" else ("🍚" if vis["detected_crop"] == "Rice" else "🐛")
        health_status = "✅ Healthy Foliage" if vis["is_healthy"] else "⚠️ Pathological Foliage Detected"

        print("\n" + "=" * 76)
        print("🌾 TWO-STAGE CROP HEALTH DIAGNOSIS & CLINICAL REPORT")
        print("=" * 76)
        print(f"📁 Image:                  {Path(result['image_path']).name}")
        print(f"1️⃣ Stage 1 (Crop Type):    {crop_icon} {vis['detected_crop']} (Certainty: {vis['crop_confidence']*100:.1f}%)")
        print(f"   Distribution:           Wheat: {vis['crop_scores']['Wheat']*100:.1f}% | Rice: {vis['crop_scores']['Rice']*100:.1f}% | Pest: {vis['crop_scores']['Pest']*100:.1f}%")
        print(f"2️⃣ Stage 2 (Pathology):    {vis['condition_name']} [{health_status}] (Certainty: {vis['confidence']*100:.1f}%)")
        print(f"📐 Stage 3 (Lesion Area):  {sev['lesion_percentage']:.2f}% | Severity: {sev['severity_level'].upper()} (Grade {sev['severity_grade']})")
        print(f"⏱️ Urgency Window:         {adv['urgency']}")
        print(f"⚡ Action Required:        {adv['action_protocol']}")

        if adv.get("chemical_remedy"):
            chem = adv["chemical_remedy"]
            print(f"\n💊 Curative Chemical:   {chem['title']} ({chem['commercial_names']})")
            print(f"   Dosage:              {chem['dosage']} ({chem['application_timing']})")
            print(f"   Safety/PHI:          {chem['safety_precaution']}")

        if adv.get("organic_remedy"):
            org = adv["organic_remedy"]
            print(f"\n🍃 Bio-Alternative:     {org['title']} -> {org['dosage']}")

        if adv.get("resistant_varieties"):
            print(f"\n🌾 Recommended Seeds:   {', '.join(adv['resistant_varieties'])}")

        if result.get("saved_visualizations"):
            print("\n🖼️ Saved Diagnostic Visualizations:")
            for k, v in result["saved_visualizations"].items():
                print(f"   • {k.capitalize()}: {v}")

        print("=" * 76 + "\n")


if __name__ == "__main__":
    main()
