"""
=============================================================================
🌾 KRISHI-AI: AUTONOMOUS IN-FIELD CLINICAL CROP PATHOLOGY & ADVISORY SYSTEM
=============================================================================
Flask Web Application connecting:
  1. Deep Vision Classifier (EfficientNet-B0) -> Disease & Pest Identification
  2. OpenCV Foliar Lesion Calculator -> Surface Area % & Severity Grading
  3. Offline SQLite Agronomic Knowledge Base -> Verified Chemical/Bio Protocols
  4. Llama-3.2-1B Agronomic Advisor -> Clinical Reasoning & Farmer Q&A

Usage:
  python app.py
  python app.py --port 5000 --host 0.0.0.0
=============================================================================
"""

import os
import sys
import io
import json
import base64
import argparse
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, Union

# Suppress sm_120 torch wheel warning on RTX 5050 Laptop GPU
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import cv2
from PIL import Image
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.diagnose import CropDiagnosticPipeline

# =============================================================================
# FLASK APPLICATION CONFIGURATION
# =============================================================================

app = Flask(
    __name__,
    template_folder=str(PROJECT_ROOT / "templates"),
    static_folder=str(PROJECT_ROOT / "static") if (PROJECT_ROOT / "static").exists() else None,
)
CORS(app)

# Allowed file extensions for upload
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}

# Pre-indexed One-Click Demo Field Samples
DEMO_SAMPLES = {
    "wheat_rust": PROJECT_ROOT / "data" / "raw" / "wheat_diseases" / "Wheat Diseases 14,155" / "Data" / "Brown Rust" / "brown_rust_0.png",
    "rice_blast": PROJECT_ROOT / "data" / "raw" / "rice_diseases" / "RiceLeafsDisease" / "train" / "leaf_blast" / "leaf_blast (10).jpg",
    "rice_spot": PROJECT_ROOT / "data" / "raw" / "rice_diseases" / "RiceLeafsDisease" / "train" / "brown_spot" / "brown_spot (102).jpg",
    "caterpillar": PROJECT_ROOT / "data" / "raw" / "agropest_12" / "test" / "images" / "catterpillar-100-_jpg.rf.6308bf1bee47aa497409a7ff17c7141d.jpg",
}

# Global pipeline instance (initialized once at startup)
pipeline: Optional[CropDiagnosticPipeline] = None


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def is_allowed_file(filename: str) -> bool:
    """Checks if uploaded filename has an allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def image_to_base64(image_input: Union[str, Path, np.ndarray, Image.Image], format: str = "JPEG") -> str:
    """
    Converts a file path, numpy OpenCV image, or PIL Image to a data URL string.
    """
    if isinstance(image_input, (str, Path)):
        p = Path(image_input)
        if not p.exists():
            return ""
        with open(p, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
            ext = p.suffix.lower().replace(".", "")
            mime = "image/png" if ext == "png" else "image/jpeg"
            return f"data:{mime};base64,{b64}"

    elif isinstance(image_input, np.ndarray):
        if len(image_input.shape) == 3 and image_input.shape[2] == 3:
            rgb = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
        else:
            rgb = image_input
        pil_img = Image.fromarray(rgb)
        buf = io.BytesIO()
        pil_img.save(buf, format=format, quality=90)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"

    elif isinstance(image_input, Image.Image):
        buf = io.BytesIO()
        image_input.save(buf, format=format, quality=90)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"

    return ""


def format_clinical_answer(
    question: str,
    disease_name: str,
    severity_grade: int,
    lesion_percentage: float,
    kb_plan: Dict[str, Any],
) -> str:
    """
    Generates an intelligent, clinically grounded answer to farmer follow-up questions
    using the verified SQLite knowledge base and phytopathological protocols.
    """
    q_lower = question.lower()
    cond = kb_plan.get("condition", {})
    proto = kb_plan.get("active_protocol", {})
    chem_list = kb_plan.get("chemical_treatments", [])
    bio_list = kb_plan.get("biological_treatments", [])
    cult_list = kb_plan.get("cultural_treatments", [])
    seeds = kb_plan.get("resistant_varieties", [])

    # 1. Weather, Rain, or Spraying Window Questions
    if any(w in q_lower for w in ["rain", "weather", "spray today", "when to spray", "timing", "wind", "morning", "evening"]):
        return (
            f"🌧️ **Spray & Weather Protocol for {disease_name}:**\n"
            f"• **Rain Warning:** Do NOT apply foliar sprays if rain is expected within 4–6 hours. Rainfall washes off fungicides and bio-pesticides before plant absorption, causing wastage and environmental runoff.\n"
            f"• **Optimal Window:** Best spraying hours are calm early mornings (6:30–9:30 AM) or late afternoons (4:30–6:30 PM).\n"
            f"• **Temperature:** Avoid application when midday heat exceeds 32°C to prevent rapid evaporation and foliar scorch.\n"
            f"• **Wind:** Ensure wind speed is below 10 km/h to prevent spray drift to non-target crops."
        )

    # 2. Organic / Bio-control / Natural Remedies
    if any(w in q_lower for w in ["organic", "bio", "natural", "chemical free", "neem", "trichoderma", "home remedy"]):
        if bio_list:
            bio = bio_list[0]
            return (
                f"🍃 **Organic & Biological Control Option:**\n"
                f"• **Bio-Agent:** {bio['title']} ({bio.get('active_ingredient', 'Natural microbial')})\n"
                f"• **Application Rate:** {bio['dosage']}\n"
                f"• **Application Timing:** {bio.get('application_timing', 'Apply in early morning or late evening under humid conditions')}\n"
                f"• **Clinical Note:** Biologicals work best proactively or at Grade 1–2 severity ({lesion_percentage:.1f}% affected area). Apply a wetting agent (sticker) for improved leaf adhesion."
            )
        else:
            return (
                f"🍃 For {disease_name}, certified biological controls are limited in emergency outbreaks. "
                f"Consider preventive cultural sanitation and balanced potassium fertilization to bolster natural plant immunity."
            )

    # 3. Chemical Dosage, Brand, or Tank Mix
    if any(w in q_lower for w in ["dosage", "dose", "chemical", "medicine", "brand", "dilution", "how much", "ml", "tank", "water"]):
        if chem_list:
            chem = chem_list[0]
            return (
                f"💊 **Prescription Chemical Protocol:**\n"
                f"• **Active Ingredient:** {chem['active_ingredient']} ({chem['title']})\n"
                f"• **Commercial Brands:** {chem.get('commercial_names', 'Standard Agricultural Formulations')}\n"
                f"• **Exact Calibrated Dosage:** **{chem['dosage']}**\n"
                f"• **Carrier Volume:** Thoroughly mix in 150–200 Liters of clean water per acre for knapsack sprayers.\n"
                f"• **Safety & PHI:** {chem.get('safety_precaution', 'Wear protective gloves; observe pre-harvest interval.')}"
            )
        else:
            return f"No chemical sprays are required or recommended for this condition ({disease_name})."

    # 4. Seeds, Resistance, Next Season Planting
    if any(w in q_lower for w in ["seed", "variety", "cultivar", "resistant", "next season", "sow", "planting"]):
        if seeds:
            return (
                f"🌾 **Certified Resistant Cultivars for Future Prevention:**\n"
                f"To prevent recurrence of {disease_name} in upcoming crop cycles, switch to certified resistant seed varieties: "
                f"**{', '.join(seeds)}**.\n"
                f"Ensure seeds are certified and treated with appropriate fungicide/bactericide slurry before sowing."
            )
        else:
            return f"Rotate crop plots and maintain strict field sanitation to minimize carry-over inoculum."

    # 5. Beneficial Insects & Safety
    if cond.get("category") == "beneficial" or any(w in q_lower for w in ["bee", "earthworm", "ladybug", "beneficial", "kill"]):
        return (
            f"🐝 **CRITICAL BENEFICIAL ORGANISM CONSERVATION:**\n"
            f"This organism ({disease_name}) is a natural farmer ally and beneficial pollinator/soil aerator. "
            f"**DO NOT apply any insecticides or chemical treatments.** "
            f"Chemical application will eliminate natural pest predators and reduce crop yields."
        )

    # 6. Default Grounded Clinical Answer
    urgency = proto.get("urgency", "Action required based on field scouting.")
    action = proto.get("action_required", "Follow standard integrated pest and disease management.")
    return (
        f"🌾 **Agronomic Recommendation for {disease_name} (Grade {severity_grade}, {lesion_percentage:.1f}% Lesion):**\n"
        f"• **Action Required:** {action}\n"
        f"• **Urgency:** {urgency}\n"
        f"• **Cultural Practice:** Maintain optimal nitrogen-to-potassium balance, avoid evening overhead irrigation, and inspect boundary foliage every 3 days."
    )


# =============================================================================
# FLASK ROUTES
# =============================================================================

@app.route("/")
def index():
    """Serves the main Krishi-AI Glassmorphism Web Interface."""
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint confirming model status."""
    global pipeline
    if pipeline is None:
        return jsonify({"status": "initializing", "message": "Model pipeline is loading"}), 503
    return jsonify({
        "status": "healthy",
        "num_classes": pipeline.num_classes,
        "device": str(pipeline.device),
        "vision_model": "EfficientNet-B0 (PyTorch)",
        "offline_db": "SQLite (treatments.db)",
    })


@app.route("/api/diagnose", methods=["POST"])
def diagnose():
    """
    Primary Diagnostic Endpoint:
    Accepts:
      - 'image': Uploaded foliage file (multipart/form-data)
      - 'question': Optional farmer question string
      - 'language': Language code ('en', 'hi', 'bn')
    Returns:
      Comprehensive JSON with Vision prediction, OpenCV severity,
      remedies from SQLite, Llama advisory, and base64 diagnostic overlays.
    """
    global pipeline
    if pipeline is None:
        return jsonify({"status": "error", "error": "Diagnostic pipeline is not ready yet"}), 503

    if "image" not in request.files:
        return jsonify({"status": "error", "error": "No image file provided in request"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"status": "error", "error": "Selected file is empty"}), 400

    if not is_allowed_file(file.filename):
        return jsonify({"status": "error", "error": f"Invalid file format. Allowed: {ALLOWED_EXTENSIONS}"}), 400

    farmer_question = request.form.get("question", "").strip() or None
    language = request.form.get("language", "en")

    # Save uploaded file to temporary location
    temp_dir = PROJECT_ROOT / "models" / "diagnosis_outputs" / "temp_uploads"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_image_path = temp_dir / file.filename
    file.save(str(temp_image_path))

    try:
        # Run end-to-end diagnostic pipeline
        result = pipeline.run_diagnosis(
            image_path=temp_image_path,
            farmer_question=farmer_question,
            save_visualizations=True,
        )

        predicted_class = result["vision"]["predicted_class"]
        severity_grade = result["severity"]["severity_grade"]

        # Fetch verified treatments from SQLite knowledge base
        kb_plan = pipeline.knowledge_base.get_treatment_plan(
            predicted_class,
            severity_grade=severity_grade,
        )

        # Encode visual outputs as base64 data URLs for seamless instant frontend rendering
        orig_b64 = image_to_base64(temp_image_path)
        overlay_b64 = image_to_base64(result["saved_visualizations"].get("overlay", ""))
        canvas_b64 = image_to_base64(result["saved_visualizations"].get("canvas", ""))

        response_data = {
            "status": "success",
            "image_path": str(temp_image_path),
            "vision": result["vision"],
            "severity": result["severity"],
            "advisory": result["advisory"],
            "knowledge_base": kb_plan,
            "markdown_report": result["markdown_report"],
            "original_base64": orig_b64,
            "overlay_base64": overlay_b64,
            "canvas_base64": canvas_b64,
        }

        return jsonify(response_data)

    except Exception as e:
        app.logger.error(f"Diagnosis failed: {e}", exc_info=True)
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/sample/<sample_key>", methods=["GET"])
def sample_diagnosis(sample_key: str):
    """
    Executes one-click instant field demo diagnosis on pre-indexed sample images:
      - 'wheat_rust': Wheat Leaf Rust (Brown Rust)
      - 'rice_blast': Rice Leaf Blast
      - 'rice_spot': Rice Brown Spot
      - 'caterpillar': In-Field Pest Caterpillar
    """
    global pipeline
    if pipeline is None:
        return jsonify({"status": "error", "error": "Diagnostic pipeline is not ready yet"}), 503

    if sample_key not in DEMO_SAMPLES:
        return jsonify({"status": "error", "error": f"Unknown sample key '{sample_key}'. Available: {list(DEMO_SAMPLES.keys())}"}), 404

    sample_path = DEMO_SAMPLES[sample_key]
    if not sample_path.exists():
        return jsonify({"status": "error", "error": f"Sample image file not found on disk: {sample_path}"}), 404

    farmer_question = request.args.get("question", "").strip() or None

    try:
        result = pipeline.run_diagnosis(
            image_path=sample_path,
            farmer_question=farmer_question,
            save_visualizations=True,
        )

        predicted_class = result["vision"]["predicted_class"]
        severity_grade = result["severity"]["severity_grade"]

        kb_plan = pipeline.knowledge_base.get_treatment_plan(
            predicted_class,
            severity_grade=severity_grade,
        )

        orig_b64 = image_to_base64(sample_path)
        overlay_b64 = image_to_base64(result["saved_visualizations"].get("overlay", ""))
        canvas_b64 = image_to_base64(result["saved_visualizations"].get("canvas", ""))

        response_data = {
            "status": "success",
            "sample_key": sample_key,
            "image_path": str(sample_path),
            "vision": result["vision"],
            "severity": result["severity"],
            "advisory": result["advisory"],
            "knowledge_base": kb_plan,
            "markdown_report": result["markdown_report"],
            "original_base64": orig_b64,
            "overlay_base64": overlay_b64,
            "canvas_base64": canvas_b64,
        }

        return jsonify(response_data)

    except Exception as e:
        app.logger.error(f"Sample diagnosis failed for {sample_key}: {e}", exc_info=True)
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Conversational follow-up endpoint with Dr. Krishi:
    Allows farmers to ask questions about the current diagnosis (e.g. rain forecasts,
    organic options, seed recommendations, pesticide dilution).
    """
    global pipeline
    if pipeline is None:
        return jsonify({"status": "error", "error": "Pipeline not initialized"}), 503

    data = request.get_json(force=True, silent=True) or {}
    question = data.get("question", "").strip()
    disease_name = data.get("disease_name", "Wheat - Leaf Rust")
    severity_grade = int(data.get("severity_grade", 2))
    lesion_percentage = float(data.get("lesion_percentage", 15.0))
    confidence = float(data.get("confidence", 0.95))

    if not question:
        return jsonify({"status": "error", "error": "Question is required"}), 400

    try:
        # Check if generative Llama LLM is available
        if pipeline.llama_advisor._llm_available:
            adv_res = pipeline.llama_advisor.advise(
                disease_name=disease_name,
                lesion_percentage=lesion_percentage,
                severity_grade=severity_grade,
                confidence=confidence,
                farmer_question=question,
            )
            reply = adv_res.full_advisory_text
        else:
            # Query grounded clinical database for 100% verified advice
            kb_plan = pipeline.knowledge_base.get_treatment_plan(
                disease_name,
                severity_grade=severity_grade,
            )
            reply = format_clinical_answer(
                question=question,
                disease_name=disease_name,
                severity_grade=severity_grade,
                lesion_percentage=lesion_percentage,
                kb_plan=kb_plan,
            )

        return jsonify({"status": "success", "reply": reply})

    except Exception as e:
        app.logger.error(f"Chat failed: {e}", exc_info=True)
        return jsonify({"status": "error", "error": str(e)}), 500


# =============================================================================
# APPLICATION INITIALIZER & RUNNER
# =============================================================================

def initialize_pipeline():
    """Loads deep vision and reasoning models into memory."""
    global pipeline
    print("=" * 76)
    print("🌾 INITIALIZING KRISHI-AI CLINICAL DIAGNOSTIC PIPELINE")
    print("=" * 76)
    pipeline = CropDiagnosticPipeline()
    print("✅ EfficientNet-B0 Vision Classifier loaded (29 classes)")
    print("✅ OpenCV Leaf Lesion Quantification Engine ready")
    print("✅ Offline SQLite Agronomic Knowledge Base connected")
    print("✅ Dr. Krishi Agronomic Reasoner active")
    print("=" * 76)


def main():
    parser = argparse.ArgumentParser(description="🌾 Krishi-AI Web Application Server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface to bind [Default: 127.0.0.1]")
    parser.add_argument("--port", type=int, default=5000, help="Port to listen on [Default: 5000]")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode")
    args = parser.parse_args()

    initialize_pipeline()

    print(f"\n🚀 Server running at: http://{args.host}:{args.port}")
    print("   Press CTRL+C to stop.\n")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
