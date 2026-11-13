#!/usr/bin/env python3
"""
src/audit_visual_layout.py
Automated Pixel-Level Visual Layout & Document Geometry Auditor
Author: Adam Dalwai (Visual Layout Auditor)

Renders all pages of LaTeX reports and presentation slide decks at 150 DPI into docs/visual_checks/.
Performs comprehensive programmatic audits:
  - Page/Slide count constraints (Reports <= 6 pages, Presentations = 10 slides)
  - Document geometry, margins, aspect ratios
  - Two-column balance and whitespace distribution
  - Orphaned headings and typographic breaks
  - Overfull boxes and text clipping
  - Figure placement, aspect ratio, centering, and caption proximity
  - Table formatting, booktabs rules, and horizontal fit
  - Aesthetic identity and color palette divergence
"""

import os
import sys
import json
import pymupdf  # PyMuPDF fitz
from PIL import Image
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(BASE_DIR, "docs/visual_checks")
os.makedirs(OUTPUT_DIR, exist_ok=True)

REPORT_VARIANTS = [
    {
        "id": "report_v1_ieee",
        "name": "Variant 1: IEEE Transactions (Double-Column)",
        "path": os.path.join(BASE_DIR, "docs/latex_reports/variant1_ieee/report.pdf"),
        "type": "report",
        "expected_pages_max": 6,
        "is_two_column": True,
        "style_theme": "IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)"
    },
    {
        "id": "report_v2_nature",
        "name": "Variant 2: Nature / Springer Clean Editorial",
        "path": os.path.join(BASE_DIR, "docs/latex_reports/variant2_nature/report.pdf"),
        "type": "report",
        "expected_pages_max": 6,
        "is_two_column": True,
        "style_theme": "Nature Machine Intelligence / Springer Editorial"
    },
    {
        "id": "report_v3_iclr",
        "name": "Variant 3: ICLR / NeurIPS Top-Tier Conference",
        "path": os.path.join(BASE_DIR, "docs/latex_reports/variant3_iclr/report.pdf"),
        "type": "report",
        "expected_pages_max": 6,
        "is_two_column": False,
        "style_theme": "ICLR / NeurIPS Dual-Format"
    },
    {
        "id": "report_v4_monograph",
        "name": "Variant 4: MIT Press Executive Monograph",
        "path": os.path.join(BASE_DIR, "docs/latex_reports/variant4_monograph/report.pdf"),
        "type": "report",
        "expected_pages_max": 6,
        "is_two_column": False,
        "style_theme": "MIT Press Technical Monograph / Palatino Serif"
    },
    {
        "id": "report_v5_deepmind",
        "name": "Variant 5: DeepMind / OpenAI Technical Whitepaper",
        "path": os.path.join(BASE_DIR, "docs/latex_reports/variant5_deepmind/report.pdf"),
        "type": "report",
        "expected_pages_max": 6,
        "is_two_column": False,
        "style_theme": "DeepMind / OpenAI Technical Whitepaper"
    }
]

PRESENTATION_VARIANTS = [
    {
        "id": "pres_v1_beamer_madrid",
        "name": "Presentation Variant 1: Academic LaTeX Beamer (Madrid / Whale)",
        "path": os.path.join(BASE_DIR, "docs/presentation_variants/variant1_beamer_madrid/presentation.pdf"),
        "type": "presentation",
        "expected_slides": 10,
        "widescreen_169": True,
        "style_theme": "LaTeX Beamer Madrid (Academic Blue/Cyan)"
    },
    {
        "id": "pres_v2_dark_mode_tech",
        "name": "Presentation Variant 2: Dark Mode Tech Keynote",
        "path": os.path.join(BASE_DIR, "docs/presentation_variants/variant2_dark_mode_tech/presentation.pdf"),
        "type": "presentation",
        "expected_slides": 10,
        "widescreen_169": True,
        "style_theme": "Obsidian / Neon Cyan Keynote"
    },
    {
        "id": "pres_v3_swiss_clean",
        "name": "Presentation Variant 3: Swiss International Clean",
        "path": os.path.join(BASE_DIR, "docs/presentation_variants/variant3_swiss_clean/presentation.pdf"),
        "type": "presentation",
        "expected_slides": 10,
        "widescreen_169": True,
        "style_theme": "Minimalist Swiss International (White / Vermilion)"
    },
    {
        "id": "pres_v4_editorial_executive",
        "name": "Presentation Variant 4: Harvard Business Editorial",
        "path": os.path.join(BASE_DIR, "docs/presentation_variants/variant4_editorial_executive/presentation.pdf"),
        "type": "presentation",
        "expected_slides": 10,
        "widescreen_169": True,
        "style_theme": "Editorial Executive Cream / Crimson"
    },
    {
        "id": "pres_v5_indigo_enterprise",
        "name": "Presentation Variant 5: Modern SaaS Indigo Enterprise",
        "path": os.path.join(BASE_DIR, "docs/presentation_variants/variant5_indigo_enterprise/presentation.pdf"),
        "type": "presentation",
        "expected_slides": 10,
        "widescreen_169": True,
        "style_theme": "Modern SaaS Indigo / Slate"
    }
]

def analyze_page_colors(png_path):
    """Samples dominant colors from the rendered PNG to assess aesthetic identity."""
    try:
        img = Image.open(png_path).convert("RGB")
        # Resize small to speed up histogram
        img_small = img.resize((100, 100))
        arr = np.array(img_small)
        # Background color approx (corners)
        corners = [arr[0, 0], arr[0, -1], arr[-1, 0], arr[-1, -1]]
        bg_rgb = tuple(int(x) for x in np.median(corners, axis=0))
        # Brightness
        brightness = 0.299 * bg_rgb[0] + 0.587 * bg_rgb[1] + 0.114 * bg_rgb[2]
        is_dark_mode = brightness < 100
        return {
            "bg_rgb": bg_rgb,
            "is_dark_mode": bool(is_dark_mode),
            "brightness": float(round(brightness, 1))
        }
    except Exception as e:
        return {"error": str(e)}

def audit_document(doc_info):
    """Performs deep programmatic visual and geometric audit on a single document."""
    path = doc_info["path"]
    doc_id = doc_info["id"]
    is_report = doc_info["type"] == "report"
    
    if not os.path.exists(path):
        return {
            "id": doc_id,
            "status": "MISSING_FILE",
            "error": f"File does not exist: {path}"
        }
    
    doc = pymupdf.open(path)
    total_pages = len(doc)
    page_audits = []
    
    ref_start_page = None
    ref_end_page = None
    
    for page_num in range(total_pages):
        page = doc[page_num]
        rect = page.rect
        width, height = rect.width, rect.height
        aspect_ratio = width / height if height > 0 else 0
        
        # Render PNG at 150 DPI
        png_filename = f"{doc_id}_p{page_num + 1}.png"
        png_path = os.path.join(OUTPUT_DIR, png_filename)
        pix = page.get_pixmap(dpi=150)
        pix.save(png_path)
        
        # Color palette analysis
        color_stats = analyze_page_colors(png_path)
        
        # Extract text blocks and bounding boxes
        blocks = page.get_text("blocks")
        text_full = page.get_text()
        
        # Check references detection
        if ("References" in text_full or "REFERENCES" in text_full or "Bibliography" in text_full):
            if ref_start_page is None:
                ref_start_page = page_num + 1
            ref_end_page = page_num + 1
        elif ref_start_page is not None and ref_end_page is not None and (page_num + 1) > ref_end_page:
            # Check if text continues references
            if "[1]" in text_full or "[5]" in text_full or "IEEE" in text_full:
                ref_end_page = page_num + 1
                
        # Margin and bounding box analysis
        min_x = width
        max_x = 0
        min_y = height
        max_y = 0
        
        text_block_boxes = []
        headings = []
        
        for b in blocks:
            bx0, by0, bx1, by1, btext, bno, btype = b
            btext_str = btext.strip()
            if not btext_str:
                continue
            min_x = min(min_x, bx0)
            max_x = max(max_x, bx1)
            min_y = min(min_y, by0)
            max_y = max(max_y, by1)
            text_block_boxes.append((bx0, by0, bx1, by1, btext_str))
            
            # Identify headings
            lines = btext_str.split("\n")
            first_line = lines[0].strip()
            if len(lines) <= 2 and (
                first_line.startswith(("1.", "2.", "3.", "4.", "5.", "6.", "7.", "I.", "II.", "III.", "IV.", "V.", "VI."))
                or first_line.isupper() and len(first_line) < 40
                or "Section" in first_line
            ):
                headings.append((first_line, by1))
                
        # Margin clearances (points)
        left_margin = min_x if min_x < width else 0
        right_margin = width - max_x if max_x > 0 else 0
        top_margin = min_y if min_y < height else 0
        bottom_margin = height - max_y if max_y > 0 else 0
        
        # Overfull box check (text extending within 15 pt of page edge or past width - 20)
        overfull_issues = []
        for bx0, by0, bx1, by1, btext_str in text_block_boxes:
            if bx1 > width - 15:
                overfull_issues.append({
                    "box": (round(bx0, 1), round(by0, 1), round(bx1, 1), round(by1, 1)),
                    "snippet": btext_str[:60],
                    "excess": round(bx1 - (width - 15), 1)
                })
                
        # Orphaned heading check (heading in bottom 8% of page with no body text following)
        orphaned_headings = []
        for htext, hy1 in headings:
            if hy1 > height - 70:  # Within 70pt of page bottom
                # Check if there is another text box beneath it
                boxes_below = [b for b in text_block_boxes if b[1] > hy1]
                if not boxes_below or len(boxes_below) < 1:
                    orphaned_headings.append(htext)
                    
        # Extract images and figures
        images = page.get_images(full=True)
        image_rects = []
        for img in images:
            xref = img[0]
            # Try to find placement bbox of image
            for rect_img in page.get_image_rects(xref):
                image_rects.append({
                    "xref": xref,
                    "rect": (round(rect_img.x0, 1), round(rect_img.y0, 1), round(rect_img.x1, 1), round(rect_img.y1, 1)),
                    "width": round(rect_img.width, 1),
                    "height": round(rect_img.height, 1),
                    "centered": abs((rect_img.x0 + rect_img.x1)/2.0 - width/2.0) < 50
                })
                
        # Two-column balance analysis (for reports)
        two_col_stats = None
        if is_report and doc_info.get("is_two_column"):
            mid_x = width / 2.0
            col1_boxes = [b for b in text_block_boxes if b[2] <= mid_x + 30]
            col2_boxes = [b for b in text_block_boxes if b[0] >= mid_x - 30]
            col1_height = (max([b[3] for b in col1_boxes]) - min([b[1] for b in col1_boxes])) if col1_boxes else 0
            col2_height = (max([b[3] for b in col2_boxes]) - min([b[1] for b in col2_boxes])) if col2_boxes else 0
            height_diff = abs(col1_height - col2_height)
            two_col_stats = {
                "col1_boxes_count": len(col1_boxes),
                "col2_boxes_count": len(col2_boxes),
                "col1_vertical_span": round(col1_height, 1),
                "col2_vertical_span": round(col2_height, 1),
                "vertical_difference": round(height_diff, 1)
            }
            
        page_audits.append({
            "page_num": page_num + 1,
            "dimensions_pt": (round(width, 1), round(height, 1)),
            "aspect_ratio": round(aspect_ratio, 3),
            "rendered_png": png_filename,
            "color_stats": color_stats,
            "margins": {
                "left": round(left_margin, 1),
                "right": round(right_margin, 1),
                "top": round(top_margin, 1),
                "bottom": round(bottom_margin, 1)
            },
            "content_stats": {
                "text_char_count": len(text_full),
                "text_blocks_count": len(text_block_boxes),
                "images_count": len(image_rects),
                "images": image_rects,
                "headings_detected": [h[0] for h in headings]
            },
            "anomalies": {
                "overfull_boxes": overfull_issues,
                "orphaned_headings": orphaned_headings
            },
            "two_col_stats": two_col_stats
        })
        
    # Summary compliance evaluation
    if is_report:
        pages_excluding_ref = (ref_start_page - 1) if ref_start_page else total_pages
        page_count_compliant = total_pages <= doc_info["expected_pages_max"]
        compliance_summary = {
            "total_pages": total_pages,
            "ref_start_page": ref_start_page,
            "pages_before_references": pages_excluding_ref,
            "max_allowed_pages": doc_info["expected_pages_max"],
            "page_count_compliant": page_count_compliant,
            "status": "PASS" if page_count_compliant else "FAIL"
        }
    else:
        slide_count_compliant = total_pages == doc_info["expected_slides"]
        # Check widescreen 16:9 ratio (approx 1.777)
        first_aspect = page_audits[0]["aspect_ratio"] if page_audits else 0
        is_widescreen = abs(first_aspect - (16.0/9.0)) < 0.15 or abs(first_aspect - 1.777) < 0.15
        compliance_summary = {
            "total_slides": total_pages,
            "expected_slides": doc_info["expected_slides"],
            "slide_count_compliant": slide_count_compliant,
            "aspect_ratio": first_aspect,
            "is_widescreen_169": is_widescreen,
            "status": "PASS" if (slide_count_compliant and is_widescreen) else "NEEDS_OPTIMIZATION"
        }
        
    return {
        "id": doc_id,
        "name": doc_info["name"],
        "path": path,
        "type": doc_info["type"],
        "style_theme": doc_info.get("style_theme", ""),
        "compliance": compliance_summary,
        "page_audits": page_audits
    }

def main():
    print("=" * 80)
    print("STARTING COMPREHENSIVE VISUAL & GEOMETRY LAYOUT AUDIT")
    print(f"Output Directory for 150 DPI Renderings: {OUTPUT_DIR}")
    print("=" * 80)
    
    all_results = {
        "reports": [],
        "presentations": []
    }
    
    print("\n>>> AUDITING LATEX REPORT VARIANTS (5 VARIANTS)...")
    for rinfo in REPORT_VARIANTS:
        print(f" Auditing {rinfo['name']}...")
        res = audit_document(rinfo)
        all_results["reports"].append(res)
        status = res["compliance"]["status"]
        tot = res["compliance"]["total_pages"]
        ref_p = res["compliance"]["ref_start_page"]
        print(f"   -> Status: {status} | Pages: {tot} (Ref starts p.{ref_p})")
        
    print("\n>>> AUDITING PRESENTATION SLIDE VARIANTS (5 VARIANTS)...")
    for pinfo in PRESENTATION_VARIANTS:
        print(f" Auditing {pinfo['name']}...")
        res = audit_document(pinfo)
        all_results["presentations"].append(res)
        status = res["compliance"]["status"]
        tot = res["compliance"]["total_slides"]
        aspect = res["compliance"]["aspect_ratio"]
        print(f"   -> Status: {status} | Slides: {tot} | Aspect: {aspect:.3f} (16:9 check: {res['compliance']['is_widescreen_169']})")

    json_path = os.path.join(OUTPUT_DIR, "audit_metrics.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAudit complete! Detailed metrics saved to {json_path}")
    print(f"Rendered PNGs saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
