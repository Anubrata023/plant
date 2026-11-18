"""
=============================================================================
🌱 MODULE 03: LEAF LESION & SEVERITY CALCULATOR (OpenCV)
=============================================================================
Computer Vision engine for quantifying foliar disease severity and lesion
surface area in cereal crops (Rice & Wheat) and pest defoliation damage.

Methodology:
  1. Multi-Space Leaf Canopy Segmentation:
     Extracts plant foliage from field backgrounds using hybrid HSV color
     filtering (Hue 20-88), CIE L*a*b* green-yellow channel analysis, and
     Excess Green (ExG) vegetation indexing with morphological refinement.
  2. Disease-Aware Lesion Detection:
     Specialized diagnostic filters for:
       - Necrotic spots (Rice Brown Spot, Rice Blast, Wheat Blights, Tan Spot)
       - Rust pustules (Wheat Leaf Rust, Stripe/Yellow Rust, Stem Rust)
       - Powdery mildew (Whitish-grey fungal mycelium)
       - Smut & Black Point (Black fungal spore masses)
       - Pest defoliation & chewing damage
       - Healthy foliage (clean zero-lesion baseline)
  3. Phytopathological Severity Classification:
     Standardized against IRRI SES (Standard Evaluation System for Rice) and
     the Modified Cobb / USDA scale for cereal rusts:
       - Grade 0: Healthy     (0.00%)
       - Grade 1: Mild        (0.01% - 10.00%)
       - Grade 2: Moderate    (10.01% - 25.00%)
       - Grade 3: Severe      (25.01% - 50.00%)
       - Grade 4: Critical    (> 50.00%)
  4. Diagnostic Visualizations:
     Generates alpha-blended lesion overlays, contour outlines, Heads-Up
     Display (HUD) metric banners, and 4-panel diagnostic inspection canvases.

Usage:
    from src.severity_calculator import LeafSeverityCalculator, calculate_leaf_severity
    result = calculate_leaf_severity("leaf.jpg", disease_name="Wheat - Leaf Rust")
    print(result.severity_level, f"{result.lesion_percentage:.2f}%")

CLI Usage:
    python src/03_severity_calculator.py --image path/to/leaf.jpg --disease "Wheat - Leaf Rust"
    python src/03_severity_calculator.py --batch 5
=============================================================================
"""

import os
import sys
import json
import base64
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Union, Optional, Tuple, Dict, Any

import cv2
import numpy as np
from PIL import Image


# =============================================================================
# AGRONOMIC SEVERITY SCALES & COLOR DEFINITIONS
# =============================================================================

SEVERITY_LEVELS = {
    0: {
        "level": "Healthy",
        "grade": 0,
        "range": (0.0, 0.0),
        "color_bgr": (50, 180, 50),     # Forest Green
        "color_hex": "#2E7D32",
        "urgency": "No intervention required. Routine prophylactic monitoring.",
        "description": "Foliage exhibits vigorous chlorophyll distribution with no identifiable lesions.",
    },
    1: {
        "level": "Mild",
        "grade": 1,
        "range": (0.01, 10.00),
        "color_bgr": (40, 215, 245),    # Golden Amber
        "color_hex": "#FBC02D",
        "urgency": "Early localized infection. Prevent spore propagation; consider bio-fungicides or neem extract.",
        "description": "Foliage presents minor localized lesions/pustules (<10% area). Early curative window active.",
    },
    2: {
        "level": "Moderate",
        "grade": 2,
        "range": (10.01, 25.00),
        "color_bgr": (0, 140, 255),     # Deep Orange
        "color_hex": "#F57C00",
        "urgency": "Targeted chemical treatment recommended within 3 to 5 days to prevent secondary spread.",
        "description": "Established foliar lesions covering 10-25% of canopy. Photosynthetic efficiency impaired.",
    },
    3: {
        "level": "Severe",
        "grade": 3,
        "range": (25.01, 50.00),
        "color_bgr": (30, 40, 225),     # Bright Red
        "color_hex": "#D32F2F",
        "urgency": "Urgent systemic fungicide or broad-spectrum insecticide required within 24 to 48 hours.",
        "description": "Extensive chlorosis and tissue necrosis covering 25-50% of leaf blade. Imminent yield loss risk.",
    },
    4: {
        "level": "Critical",
        "grade": 4,
        "range": (50.01, 100.00),
        "color_bgr": (20, 20, 160),     # Dark Crimson
        "color_hex": "#B71C1C",
        "urgency": "Emergency intervention required. Canopy necrosis catastrophic; isolate affected plot.",
        "description": "Over 50% canopy blighted or destroyed. High risk of complete lodging or field collapse.",
    },
}


# =============================================================================
# DATA STRUCTURE: SEVERITY RESULT
# =============================================================================

@dataclass
class SeverityResult:
    """Encapsulates comprehensive severity metrics and diagnostic visualizations."""
    lesion_percentage: float
    severity_level: str
    severity_grade: int
    total_leaf_pixels: int
    lesion_pixels: int
    healthy_pixels: int
    disease_name: Optional[str] = None
    urgency: str = ""
    description: str = ""
    color_hex: str = "#2E7D32"
    color_bgr: Tuple[int, int, int] = (50, 180, 50)
    leaf_mask: Optional[np.ndarray] = None
    lesion_mask: Optional[np.ndarray] = None
    visual_overlay: Optional[np.ndarray] = None
    diagnostic_canvas: Optional[np.ndarray] = None

    def to_dict(self, include_images: bool = False) -> Dict[str, Any]:
        """Returns clean dictionary serialization for JSON logging and API responses."""
        data = {
            "disease_name": self.disease_name,
            "lesion_percentage": round(self.lesion_percentage, 2),
            "severity_level": self.severity_level,
            "severity_grade": self.severity_grade,
            "total_leaf_pixels": int(self.total_leaf_pixels),
            "lesion_pixels": int(self.lesion_pixels),
            "healthy_pixels": int(self.healthy_pixels),
            "urgency": self.urgency,
            "description": self.description,
            "color_hex": self.color_hex,
        }
        if include_images and self.visual_overlay is not None:
            data["visual_overlay_base64"] = self.to_base64_overlay()
        return data

    def to_base64_overlay(self, quality: int = 85) -> str:
        """Encodes visual overlay to base64 JPEG data URL for HTML/Web UI embedding."""
        if self.visual_overlay is None:
            return ""
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        success, buffer = cv2.imencode(".jpg", self.visual_overlay, encode_params)
        if not success:
            return ""
        b64_str = base64.b64encode(buffer).decode("utf-8")
        return f"data:image/jpeg;base64,{b64_str}"

    def to_base64_canvas(self, quality: int = 85) -> str:
        """Encodes 4-panel diagnostic canvas to base64 JPEG data URL."""
        if self.diagnostic_canvas is None:
            return ""
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        success, buffer = cv2.imencode(".jpg", self.diagnostic_canvas, encode_params)
        if not success:
            return ""
        b64_str = base64.b64encode(buffer).decode("utf-8")
        return f"data:image/jpeg;base64,{b64_str}"

    def save(self, output_path: Union[str, Path], save_canvas: bool = False) -> str:
        """Saves annotated visual overlay (or canvas) to disk."""
        target_path = Path(output_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        img_to_save = self.diagnostic_canvas if (save_canvas and self.diagnostic_canvas is not None) else self.visual_overlay
        if img_to_save is not None:
            cv2.imwrite(str(target_path), img_to_save)
            return str(target_path)
        return ""


# =============================================================================
# CORE ENGINE: LEAF SEVERITY CALCULATOR
# =============================================================================

class LeafSeverityCalculator:
    """
    Robust OpenCV engine for in-field leaf lesion quantification.
    
    Processes field imagery of Rice and Wheat leaves and crop pests,
    segmenting healthy canopy tissue from disease symptoms (rust pustules,
    necrotic spots, fungal mildew, chlorosis, and insect defoliation).
    """

    def __init__(
        self,
        min_leaf_area: int = 350,
        min_lesion_area: int = 8,
        default_max_dim: int = 1200,
    ):
        """
        Args:
            min_leaf_area: Minimum connected pixel count to qualify as leaf tissue.
            min_lesion_area: Minimum connected pixel count to qualify as a lesion.
            default_max_dim: Target maximum dimension for processing to maintain real-time speed.
        """
        self.min_leaf_area = min_leaf_area
        self.min_lesion_area = min_lesion_area
        self.default_max_dim = default_max_dim

    # -------------------------------------------------------------------------
    # Helper: Image Loading & Preprocessing
    # -------------------------------------------------------------------------
    def _load_image(self, image_input: Union[str, Path, np.ndarray, Image.Image]) -> np.ndarray:
        """Converts diverse image inputs (file path, PIL Image, numpy array) to OpenCV BGR."""
        if isinstance(image_input, (str, Path)):
            img_path = str(image_input)
            if not os.path.isfile(img_path):
                raise FileNotFoundError(f"Input image not found: {img_path}")
            img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Failed to decode image from path: {img_path}")
            return img

        elif isinstance(image_input, Image.Image):
            rgb = np.array(image_input.convert("RGB"))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:
                return cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
            elif image_input.ndim == 3 and image_input.shape[2] == 4:
                return cv2.cvtColor(image_input, cv2.COLOR_BGRA2BGR)
            elif image_input.ndim == 3 and image_input.shape[2] == 3:
                return image_input.copy()
            else:
                raise ValueError(f"Unsupported numpy image array shape: {image_input.shape}")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

    def _resize_if_needed(self, img: np.ndarray) -> Tuple[np.ndarray, float]:
        """Downscales very large high-res photos to maintain real-time speed while preserving aspect ratio."""
        h, w = img.shape[:2]
        max_dim = max(h, w)
        if max_dim > self.default_max_dim:
            scale = self.default_max_dim / float(max_dim)
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            return resized, scale
        return img, 1.0

    # -------------------------------------------------------------------------
    # Step 1: Leaf Canopy Segmentation (Foreground vs Field Background)
    # -------------------------------------------------------------------------
    def segment_leaf(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Extracts foliage canopy tissue from in-field backgrounds (soil, sky, shadows, hands).
        
        Uses a combination of:
          - HSV Plant Range: Hue 20-88, Saturation 25-255, Value 25-245
          - CIE L*a*b* Leaf Tissue: a* < 132 (green component), b* > 115 (yellow component)
          - Excess Green Index (ExG): 2*G - R - B > 0
        Filters out sky (blue/cyan), pure white specular glare, and dark shadows.
        """
        h, w = img_bgr.shape[:2]
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        L, a_channel, b_channel = cv2.split(lab)
        b_img, g_img, r_img = cv2.split(img_bgr.astype(np.float32))

        # 1. Broad vegetation mask in HSV (captures green, yellow-green, olive, lime)
        hsv_veg = cv2.inRange(hsv, (20, 25, 25), (88, 255, 245))

        # 2. CIE L*a*b* green-yellow vegetation mask
        lab_veg = (a_channel < 132) & (b_channel > 115) & (L > 25) & (L < 245)

        # 3. Excess Green vegetation index (ExG = 2G - R - B)
        exg = 2.0 * g_img - r_img - b_img
        exg_veg = exg > 3.0

        # 4. Background Exclusions:
        sky_mask = (hsv[:, :, 0] >= 95) & (hsv[:, :, 0] <= 135) & (hsv[:, :, 1] >= 40)
        shadow_mask = (hsv[:, :, 2] < 22) | (L < 22)

        # Combine candidate foliage indicators
        candidate_mask = (hsv_veg | lab_veg | exg_veg) & (~sky_mask) & (~shadow_mask)
        candidate_mask = candidate_mask.astype(np.uint8) * 255

        # Morphological refinement
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(candidate_mask, cv2.MORPH_CLOSE, kernel_close)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_open)

        # Filter out tiny disconnected background blobs
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        leaf_mask = np.zeros((h, w), dtype=np.uint8)

        total_pixels = h * w
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= self.min_leaf_area:
                cv2.drawContours(leaf_mask, [cnt], -1, 255, -1)

        # Fallback for macro/close-up photographs where leaf fills virtually 100% of the image
        leaf_pixel_count = np.count_nonzero(leaf_mask)
        if leaf_pixel_count < (0.05 * total_pixels):
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, otsu_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            if np.mean(gray[otsu_mask == 255]) < np.mean(gray[otsu_mask == 0]):
                otsu_mask = cv2.bitwise_not(otsu_mask)
            leaf_mask = otsu_mask

        return leaf_mask

    # -------------------------------------------------------------------------
    # Step 2: Disease-Aware Lesion Detection Inside the Leaf Mask
    # -------------------------------------------------------------------------
    def segment_lesions(
        self,
        img_bgr: np.ndarray,
        leaf_mask: np.ndarray,
        disease_name: Optional[str] = None,
    ) -> np.ndarray:
        """
        Isolates diseased lesions within the segmented leaf mask.
        
        Specialized detection pipelines:
          - Healthy: Returns clean zero mask (Grade 0).
          - Powdery Mildew: Whitish/grey mycelium on foliage.
          - Rusts (Leaf, Stripe, Stem): Orange/amber pustules and chlorotic borders.
          - Necrotic spots (Brown Spot, Blast, Blight, Tan Spot, Septoria): Dark brown/reddish centers.
          - Smut & Black Point: Deep black/dark brown fungal spore masses.
          - Pests: Foliar bite margins, holes, and necrotic feeding scars.
        """
        h, w = img_bgr.shape[:2]
        lesion_mask = np.zeros((h, w), dtype=np.uint8)

        # 1. Healthy class override
        if disease_name and "healthy" in disease_name.lower():
            return lesion_mask

        # If leaf mask is essentially empty, nothing to evaluate
        if np.count_nonzero(leaf_mask) < self.min_leaf_area:
            return lesion_mask

        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        L, a_channel, b_channel = cv2.split(lab)
        b_img, g_img, r_img = cv2.split(img_bgr.astype(np.float32))

        # Disease context parsing
        d_lower = disease_name.lower() if disease_name else ""
        is_powdery = ("powdery" in d_lower or "mildew" in d_lower)
        is_rust = ("rust" in d_lower)
        is_smut_or_black = ("smut" in d_lower or "black" in d_lower)
        is_spot_or_blight = (
            "spot" in d_lower or "blast" in d_lower or "blight" in d_lower or
            "septoria" in d_lower or "rot" in d_lower
        )
        is_pest = ("pest" in d_lower)

        # ---------------------------------------------------------------------
        # Detection Component A: Rust Pustules (Orange, Reddish-Brown)
        # ---------------------------------------------------------------------
        rust_pustules = (
            (hsv[:, :, 0] >= 6) & (hsv[:, :, 0] <= 21) &
            (hsv[:, :, 1] >= 55) &
            (hsv[:, :, 2] >= 35) &
            (leaf_mask > 0)
        )

        # ---------------------------------------------------------------------
        # Detection Component B: Necrotic Lesions (Dark brown / Reddish / ExR)
        # ---------------------------------------------------------------------
        exr = 1.4 * r_img - g_img
        dark_necrotic = (
            (hsv[:, :, 2] < 75) & (hsv[:, :, 2] > 15) &
            (hsv[:, :, 0] < 22) &
            (leaf_mask > 0)
        )
        reddish_necrotic = (
            ((hsv[:, :, 0] <= 6) | (hsv[:, :, 0] >= 172)) &
            (a_channel > 133) &
            (leaf_mask > 0)
        )
        exr_necrotic = (
            (exr > 38.0) &
            (a_channel > 130) &
            (leaf_mask > 0)
        )
        necrotic_union = dark_necrotic | reddish_necrotic | exr_necrotic

        # ---------------------------------------------------------------------
        # Detection Component C: Powdery Mildew (White/Grey Mycelium on leaf)
        # ---------------------------------------------------------------------
        powdery_mildew = (
            (L > 155) &
            (hsv[:, :, 1] < 60) &
            (leaf_mask > 0)
        )

        # ---------------------------------------------------------------------
        # Detection Component D: Black Point / Loose Smut Spore Masses
        # ---------------------------------------------------------------------
        black_smut = (
            (hsv[:, :, 2] < 50) &
            (L < 50) &
            (leaf_mask > 0)
        )

        # ---------------------------------------------------------------------
        # Assemble Disease-Aware Candidate Lesion Mask
        # ---------------------------------------------------------------------
        if is_powdery:
            raw_lesion = powdery_mildew | necrotic_union
        elif is_rust:
            raw_lesion = rust_pustules | necrotic_union
        elif is_smut_or_black:
            raw_lesion = black_smut | dark_necrotic
        elif is_spot_or_blight:
            raw_lesion = necrotic_union | rust_pustules
        elif is_pest:
            raw_lesion = necrotic_union | rust_pustules
        else:
            # General fallback: necrotic lesions + rust pustules + black spore masses
            # (Powdery mildew is excluded from general mode to prevent overexposed leaf glare false positives)
            raw_lesion = necrotic_union | rust_pustules | black_smut

        # Ensure lesions strictly reside inside the detected leaf mask
        raw_lesion = (raw_lesion & (leaf_mask > 0)).astype(np.uint8) * 255

        # Morphological noise filtering: remove single-pixel false positives
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        filtered = cv2.morphologyEx(raw_lesion, cv2.MORPH_OPEN, kernel_clean)

        # Filter out micro-contours below minimum lesion area
        contours, _ = cv2.findContours(filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= self.min_lesion_area:
                cv2.drawContours(lesion_mask, [cnt], -1, 255, -1)

        return lesion_mask

    # -------------------------------------------------------------------------
    # Step 3: Severity Metric Computation & Grade Classification
    # -------------------------------------------------------------------------
    @staticmethod
    def classify_severity(lesion_percentage: float, disease_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Maps numerical lesion percentage to IRRI SES / Modified Cobb agronomic grades.
        """
        if disease_name and "healthy" in disease_name.lower():
            info = SEVERITY_LEVELS[0].copy()
            info["lesion_percentage"] = 0.0
            return info

        pct = max(0.0, min(100.0, float(lesion_percentage)))

        if pct <= 0.05:
            grade = 0
        elif pct <= 10.0:
            grade = 1
        elif pct <= 25.0:
            grade = 2
        elif pct <= 50.0:
            grade = 3
        else:
            grade = 4

        info = SEVERITY_LEVELS[grade].copy()
        info["lesion_percentage"] = pct
        return info

    # -------------------------------------------------------------------------
    # Step 4: Visual Overlay & Diagnostic HUD Generation
    # -------------------------------------------------------------------------
    def generate_visual_overlay(
        self,
        img_bgr: np.ndarray,
        leaf_mask: np.ndarray,
        lesion_mask: np.ndarray,
        severity_info: Dict[str, Any],
        disease_name: Optional[str] = None,
    ) -> np.ndarray:
        """
        Creates an annotated visual overlay featuring:
          - Semi-transparent lesion color wash (coral/red alpha blend)
          - Lesion boundary contours (bright yellow/amber)
          - Leaf perimeter outline (subtle emerald green)
          - Heads-Up Display (HUD) banner with severity metrics & grade badge
        """
        overlay = img_bgr.copy()
        h, w = img_bgr.shape[:2]

        # 1. Highlight Lesion Zones with Alpha-Blended Fill
        if np.count_nonzero(lesion_mask) > 0:
            color_bgr = severity_info.get("color_bgr", (30, 40, 225))
            colored_fill = np.zeros_like(img_bgr)
            colored_fill[:] = color_bgr

            alpha = 0.45
            blended = cv2.addWeighted(colored_fill, alpha, overlay, 1.0 - alpha, 0)
            overlay = np.where(lesion_mask[:, :, None] > 0, blended, overlay)

            # Lesion boundary contours
            lesion_contours, _ = cv2.findContours(lesion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, lesion_contours, -1, (0, 240, 255), max(1, int(w / 600)))

        # 2. Outline Leaf Canopy Boundary (subtle green)
        if np.count_nonzero(leaf_mask) > 0:
            leaf_contours, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, leaf_contours, -1, (46, 204, 113), max(1, int(w / 800)))

        # 3. Draw Heads-Up Display (HUD) Diagnostic Banner
        hud_height = max(50, int(h * 0.12))
        hud = overlay.copy()

        # Dark frosted banner at the top
        cv2.rectangle(hud, (0, 0), (w, hud_height), (18, 22, 28), -1)
        cv2.addWeighted(hud, 0.82, overlay, 0.18, 0, overlay)

        # Dynamic font scaling based on image width
        font_scale_main = max(0.45, min(1.1, w / 850.0))
        font_scale_sub = max(0.35, min(0.85, w / 1100.0))
        thickness_main = max(1, int(font_scale_main * 2))
        thickness_sub = max(1, int(font_scale_sub * 1.5))
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Severity Badge Color Block
        badge_color = severity_info.get("color_bgr", (50, 180, 50))
        badge_w = max(14, int(w * 0.02))
        cv2.rectangle(overlay, (8, 8), (8 + badge_w, hud_height - 8), badge_color, -1)

        # Title: Disease Name / Diagnosis
        display_title = disease_name if disease_name else "Leaf Health Assessment"
        title_x = 18 + badge_w
        title_y = int(hud_height * 0.44)
        cv2.putText(
            overlay,
            display_title,
            (title_x, title_y),
            font,
            font_scale_main,
            (255, 255, 255),
            thickness_main,
            cv2.LINE_AA,
        )

        # Subtitle: Severity Grade & Affected Area %
        level_name = severity_info.get("level", "Unknown").upper()
        grade_num = severity_info.get("grade", 0)
        pct_val = severity_info.get("lesion_percentage", 0.0)

        metrics_text = f"Severity: {level_name} (Grade {grade_num})  |  Lesion Area: {pct_val:.2f}%"
        metrics_y = int(hud_height * 0.82)
        cv2.putText(
            overlay,
            metrics_text,
            (title_x, metrics_y),
            font,
            font_scale_sub,
            (210, 225, 235),
            thickness_sub,
            cv2.LINE_AA,
        )

        # Right-aligned summary badge
        status_tag = f"[{level_name}]"
        (tag_w, _), _ = cv2.getTextSize(status_tag, font, font_scale_main, thickness_main)
        tag_x = max(title_x + 50, w - tag_w - 18)
        cv2.putText(
            overlay,
            status_tag,
            (tag_x, title_y),
            font,
            font_scale_main,
            badge_color,
            thickness_main,
            cv2.LINE_AA,
        )

        return overlay

    # -------------------------------------------------------------------------
    # Step 5: 4-Panel Diagnostic Canvas Generation
    # -------------------------------------------------------------------------
    def generate_diagnostic_canvas(
        self,
        img_bgr: np.ndarray,
        leaf_mask: np.ndarray,
        lesion_mask: np.ndarray,
        visual_overlay: np.ndarray,
        target_dim: int = 512,
    ) -> np.ndarray:
        """
        Assembles a side-by-side 4-panel diagnostic inspection canvas:
          Panel (0,0): Original Field Image
          Panel (0,1): Segmented Leaf Canopy (Green)
          Panel (1,0): Lesion Heatmap / Detected Hotspots (Red)
          Panel (1,1): Final Inspection Overlay with HUD
        """
        def prep_panel(panel_img: np.ndarray, title: str) -> np.ndarray:
            p = cv2.resize(panel_img, (target_dim, target_dim), interpolation=cv2.INTER_AREA)
            cv2.rectangle(p, (0, 0), (target_dim, 32), (20, 24, 30), -1)
            cv2.putText(p, title, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            return p

        # 1. Original
        p1 = prep_panel(img_bgr, "1. Input Field Image")

        # 2. Leaf Mask in green
        leaf_vis = np.zeros_like(img_bgr)
        leaf_vis[leaf_mask > 0] = (46, 204, 113)
        blended_leaf = cv2.addWeighted(leaf_vis, 0.5, img_bgr, 0.5, 0)
        p2 = prep_panel(blended_leaf, "2. Segmented Leaf Canopy")

        # 3. Lesion Mask in fiery amber/red
        lesion_vis = np.zeros_like(img_bgr)
        lesion_vis[lesion_mask > 0] = (0, 69, 255)
        blended_lesion = cv2.addWeighted(lesion_vis, 0.7, img_bgr, 0.3, 0)
        p3 = prep_panel(blended_lesion, "3. Detected Lesion Hotspots")

        # 4. Diagnostic Overlay
        p4 = prep_panel(visual_overlay, "4. Phytopathological HUD")

        # Assemble 2x2 grid
        top_row = np.hstack([p1, p2])
        bottom_row = np.hstack([p3, p4])
        canvas = np.vstack([top_row, bottom_row])
        return canvas

    # -------------------------------------------------------------------------
    # Main Public Execution Method
    # -------------------------------------------------------------------------
    def calculate_severity(
        self,
        image_input: Union[str, Path, np.ndarray, Image.Image],
        disease_name: Optional[str] = None,
        generate_canvas: bool = True,
    ) -> SeverityResult:
        """
        Executes complete leaf lesion quantification on an input image.

        Args:
            image_input: File path, PIL Image, or OpenCV BGR array.
            disease_name: Optional disease name (e.g. "Wheat - Leaf Rust", "Rice - Brown Spot").
            generate_canvas: Whether to assemble the 4-panel diagnostic comparison image.

        Returns:
            SeverityResult containing metrics, masks, overlay, and agronomic guidance.
        """
        img_bgr = self._load_image(image_input)
        img_work, scale = self._resize_if_needed(img_bgr)

        # 1. Segment Leaf Foreground
        leaf_mask = self.segment_leaf(img_work)
        total_leaf_pixels = int(np.count_nonzero(leaf_mask))

        # 2. Segment Lesions
        lesion_mask = self.segment_lesions(img_work, leaf_mask, disease_name=disease_name)
        lesion_pixels = int(np.count_nonzero(lesion_mask))

        # 3. Compute Area & Severity Metrics
        if total_leaf_pixels > 0:
            lesion_pct = (lesion_pixels / float(total_leaf_pixels)) * 100.0
        else:
            non_black = int(np.count_nonzero(cv2.cvtColor(img_work, cv2.COLOR_BGR2GRAY) > 10))
            total_leaf_pixels = max(non_black, 1)
            lesion_pct = (lesion_pixels / float(total_leaf_pixels)) * 100.0

        lesion_pct = min(100.0, max(0.0, lesion_pct))
        healthy_pixels = max(0, total_leaf_pixels - lesion_pixels)

        # Agronomic classification
        severity_info = self.classify_severity(lesion_pct, disease_name=disease_name)

        # If image was resized, scale masks back to original resolution for visual overlay
        if scale != 1.0:
            orig_h, orig_w = img_bgr.shape[:2]
            leaf_mask_orig = cv2.resize(leaf_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            lesion_mask_orig = cv2.resize(lesion_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        else:
            leaf_mask_orig = leaf_mask
            lesion_mask_orig = lesion_mask

        # 4. Generate Visual Overlay
        visual_overlay = self.generate_visual_overlay(
            img_bgr=img_bgr,
            leaf_mask=leaf_mask_orig,
            lesion_mask=lesion_mask_orig,
            severity_info=severity_info,
            disease_name=disease_name,
        )

        # 5. Generate 4-Panel Diagnostic Canvas
        diagnostic_canvas = None
        if generate_canvas:
            diagnostic_canvas = self.generate_diagnostic_canvas(
                img_bgr=img_bgr,
                leaf_mask=leaf_mask_orig,
                lesion_mask=lesion_mask_orig,
                visual_overlay=visual_overlay,
            )

        return SeverityResult(
            lesion_percentage=round(severity_info["lesion_percentage"], 2),
            severity_level=severity_info["level"],
            severity_grade=severity_info["grade"],
            total_leaf_pixels=total_leaf_pixels,
            lesion_pixels=lesion_pixels,
            healthy_pixels=healthy_pixels,
            disease_name=disease_name,
            urgency=severity_info["urgency"],
            description=severity_info["description"],
            color_hex=severity_info["color_hex"],
            color_bgr=severity_info["color_bgr"],
            leaf_mask=leaf_mask_orig,
            lesion_mask=lesion_mask_orig,
            visual_overlay=visual_overlay,
            diagnostic_canvas=diagnostic_canvas,
        )


# =============================================================================
# CONVENIENCE STANDALONE FUNCTION
# =============================================================================

def calculate_leaf_severity(
    image_input: Union[str, Path, np.ndarray, Image.Image],
    disease_name: Optional[str] = None,
    output_path: Optional[Union[str, Path]] = None,
    generate_canvas: bool = True,
) -> SeverityResult:
    """
    Convenience function to evaluate leaf lesion severity in a single call.

    Args:
        image_input: Path to image, PIL Image, or OpenCV BGR array.
        disease_name: Optional disease name to specialize detection logic.
        output_path: Optional path to save annotated visual overlay image.
        generate_canvas: Whether to generate the 4-panel diagnostic canvas.

    Returns:
        SeverityResult dataclass instance.
    """
    calculator = LeafSeverityCalculator()
    result = calculator.calculate_severity(
        image_input=image_input,
        disease_name=disease_name,
        generate_canvas=generate_canvas,
    )
    if output_path is not None:
        result.save(output_path)
    return result


# =============================================================================
# CLI & BATCH BENCHMARKING INTERFACE
# =============================================================================

def run_batch_evaluation(n_samples: int = 5) -> None:
    """
    Validates the severity calculator against diverse authentic in-field
    images from unified_index.csv across Rice, Wheat, and Pests.
    """
    project_root = Path(__file__).resolve().parent.parent
    csv_path = project_root / "data" / "unified_index.csv"

    if not csv_path.exists():
        print(f"[ERROR] unified_index.csv not found at: {csv_path}")
        return

    import pandas as pd
    df = pd.read_csv(csv_path)

    target_classes = [
        "Rice - Brown Spot",
        "Wheat - Leaf Rust",
        "Rice - Healthy",
        "Wheat - Powdery Mildew",
        "Wheat - Stripe Rust (Yellow Rust)",
    ]

    samples = []
    for cls in target_classes:
        sub = df[df["label_name"] == cls]
        if not sub.empty:
            samples.append(sub.sample(n=1, random_state=42).iloc[0])

    if len(samples) < n_samples:
        extra = df.sample(n=(n_samples - len(samples)), random_state=42)
        for _, row in extra.iterrows():
            samples.append(row)

    print("=" * 78)
    print("🌾 IN-FIELD SEVERITY CALCULATOR BATCH BENCHMARK")
    print("=" * 78)

    calculator = LeafSeverityCalculator()
    output_dir = project_root / "models" / "severity_benchmark_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, row in enumerate(samples[:n_samples], 1):
        img_path = row["image_path"]
        label = row["label_name"]

        if not os.path.isfile(img_path):
            print(f"[{i}/{n_samples}] SKIP: Image file not found -> {img_path}")
            continue

        res = calculator.calculate_severity(img_path, disease_name=label)
        out_vis = output_dir / f"sample_{i}_overlay.jpg"
        out_canvas = output_dir / f"sample_{i}_canvas.jpg"
        res.save(out_vis, save_canvas=False)
        res.save(out_canvas, save_canvas=True)

        print(f"\n[{i}/{n_samples}] Diagnosis: {label}")
        print(f"   Image: {Path(img_path).name}")
        print(f"   Lesion Coverage: {res.lesion_percentage:6.2f}%")
        print(f"   Severity Grade:  Grade {res.severity_grade} ({res.severity_level})")
        print(f"   Leaf Pixels:     {res.total_leaf_pixels:,} | Lesion Pixels: {res.lesion_pixels:,}")
        print(f"   Agronomic Note:  {res.urgency}")
        print(f"   Saved Overlay:   {out_vis}")

    print("\n" + "=" * 78)
    print(f"✅ Batch benchmark finished! Diagnostic overlays saved to:\n   {output_dir}")
    print("=" * 78)


def main():
    parser = argparse.ArgumentParser(
        description="🌾 OpenCV Leaf Lesion & Severity Calculator for Rice, Wheat & Pests"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to input field image (JPEG/PNG)",
    )
    parser.add_argument(
        "--disease",
        type=str,
        default=None,
        help="Diagnosis/disease name (e.g. 'Wheat - Leaf Rust', 'Rice - Brown Spot', 'Rice - Healthy')",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save annotated inspection overlay image",
    )
    parser.add_argument(
        "--canvas",
        type=str,
        default=None,
        help="Path to save 4-panel diagnostic comparison canvas",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print diagnostic output formatted as JSON to stdout",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Run batch evaluation on N samples from data/unified_index.csv",
    )

    args = parser.parse_args()

    # Batch evaluation mode
    if args.batch is not None:
        run_batch_evaluation(n_samples=args.batch)
        return

    # Single image mode
    if args.image is None:
        print("[INFO] No input image provided. Running batch benchmark of 5 samples by default.")
        print("       Use --image <path> --disease <name> to analyze a specific leaf photo.\n")
        run_batch_evaluation(n_samples=5)
        return

    calculator = LeafSeverityCalculator()
    result = calculator.calculate_severity(
        image_input=args.image,
        disease_name=args.disease,
        generate_canvas=(args.canvas is not None),
    )

    if args.output:
        result.save(args.output, save_canvas=False)
        print(f"[SAVED] Overlay saved to: {args.output}")

    if args.canvas:
        result.save(args.canvas, save_canvas=True)
        print(f"[SAVED] Diagnostic canvas saved to: {args.canvas}")

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print("\n" + "=" * 65)
        print("🌿 CROP LEAF SEVERITY DIAGNOSIS REPORT")
        print("=" * 65)
        if result.disease_name:
            print(f"Diagnosed Condition: {result.disease_name}")
        print(f"Severity Stage:      {result.severity_level.upper()} (Grade {result.severity_grade})")
        print(f"Lesion Surface Area: {result.lesion_percentage:.2f}%")
        print(f"Total Leaf Pixels:   {result.total_leaf_pixels:,}")
        print(f"Lesion Pixels:       {result.lesion_pixels:,}")
        print(f"Healthy Pixels:      {result.healthy_pixels:,}")
        print(f"Clinical Guidance:   {result.description}")
        print(f"Action Urgency:      {result.urgency}")
        print(f"UI Badge Color:      {result.color_hex}")
        print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
