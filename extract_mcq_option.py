"""
Extract MCQs (question + options) from a selected PDF in ./Subject2 and write a
website-compatible JSON file into ./website/data.

Correct answer detection rule:
- Pick the option row that contains a green checkmark/tick icon on the right.

This script is designed to run locally or on RunPod.

Usage:
  python extract_mcq_option.py
  python extract_mcq_option.py --pdf-index 2
    python extract_mcq_option.py --pdf "Subject2/COMS_finale_dedup.pdf"

Output:
  website/data/<pdf_stem>.json

JSON schema matches the website (see website/js/app.js):
[
  {"id": 1, "question": "...", "options": ["..."], "correct": 2},
  ...
]
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import cv2
import pytesseract
from pdf2image import convert_from_path
from PIL import Image


# =========================
# Defaults (repo-relative)
# =========================
REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_SUBJECT_DIR = REPO_ROOT / "Subject"
DEFAULT_SUBJECT_DIR = REPO_ROOT / "Subject2"
DEFAULT_WEBSITE_DATA_DIR = REPO_ROOT / "website" / "data"

# OCR / processing
DPI = 250
BATCH_SIZE = 30
NUM_WORKERS = 8
SIMILARITY_THRESHOLD = 0.88

# Tesseract
TESSERACT_CONFIG = "--oem 3 --psm 6 -l eng"


@dataclass(frozen=True)
class Mcq:
    question: str
    options: List[str]
    correct: int


def list_pdfs(subject_dir: Path) -> List[Path]:
    if not subject_dir.exists():
        return []
    return sorted([p for p in subject_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"], key=lambda p: p.name.lower())


def display_pdf_list(pdf_files: List[Path]) -> None:
    print("\n" + "=" * 60)
    print("AVAILABLE PDF FILES (Subject2 folder)")
    print("=" * 60)
    if not pdf_files:
        print("No PDF files found.")
        return
    for i, pdf in enumerate(pdf_files, start=1):
        print(f"  {i:>3}) {pdf.name}")
    print("=" * 60)


def prompt_pdf_index(max_index: int) -> int:
    while True:
        choice = input(f"Enter the PDF number to process (1-{max_index}, or 'q' to quit): ").strip().lower()
        if choice == "q":
            raise SystemExit(0)
        if not choice.isdigit():
            print("Please enter a valid number.")
            continue
        idx = int(choice)
        if not (1 <= idx <= max_index):
            print(f"Please enter a number between 1 and {max_index}.")
            continue
        return idx


# =========================
# OCR + parsing
# =========================

def preprocess_image_for_ocr(pil_image: Image.Image) -> Image.Image:
    img = np.array(pil_image)

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img

    height, width = gray.shape
    if width < 1500:
        scale = 1500 / width
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    gray = cv2.fastNlMeansDenoising(gray, h=12)

    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    gray = cv2.filter2D(gray, -1, kernel)

    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 8)
    return Image.fromarray(binary)


def extract_text_with_confidence(page_image: Image.Image) -> Tuple[str, float]:
    processed = preprocess_image_for_ocr(page_image)

    data = pytesseract.image_to_data(
        processed,
        lang="eng",
        config=TESSERACT_CONFIG,
        output_type=pytesseract.Output.DICT,
    )

    confidences = [int(c) for c, t in zip(data["conf"], data["text"]) if c != "-1" and str(t).strip()]
    avg_confidence = (sum(confidences) / len(confidences)) if confidences else 0.0

    text = pytesseract.image_to_string(processed, lang="eng", config=TESSERACT_CONFIG)
    return text, avg_confidence


def clean_text(text: str) -> str:
    patterns = [
        r"\d{1,2}:\d{2}\s*(AM|PM|am|pm)?",
        r"(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d+",
        r"PT\d*\s*\d+\.\d+",
        r"\d{3}\s+PT\d?",
        r"©.*?(Con|Tota|@|&)",
        r"<>\s*Results",
        r"Results",
        r"Multiple Choice",
        r"^\s*Question\s+\d+\s*$",
        r"Pevesys",
        r"Page\s+\d+",
        r"www\.\S+",
        r"http\S+",
    ]

    for pattern in patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.MULTILINE)

    text = re.sub(r"\n\s*\n", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def parse_mcq(raw_text: str, correct_idx: int) -> Optional[Mcq]:
    text = clean_text(raw_text)
    lines = [l.strip() for l in text.split("\n") if l.strip() and len(l.strip()) > 2]

    filtered: List[str] = []
    for line in lines:
        if any(x in line.lower() for x in ["results", "multiple choice", "pevesys", "page "]):
            continue
        if re.match(r"^[\d\s\.\-\(\)]+$", line):
            continue
        if len(line) < 4:
            continue
        filtered.append(line)

    if len(filtered) < 3:
        return None

    question = ""
    options: List[str] = []
    question_found = False
    option_pattern = re.compile(r"^[aAbBcCdD][\.\)\:\s]+(.+)", re.IGNORECASE)

    for line in filtered:
        opt_match = option_pattern.match(line)
        if opt_match:
            question_found = True
            opt_text = opt_match.group(1).strip()
            if opt_text and len(opt_text) > 2:
                options.append(opt_text)
        elif not question_found:
            if re.search(r"question\s*\d+", line, re.IGNORECASE):
                match = re.search(r"question\s*\d+[\.\:\)\s]*(.+)", line, re.IGNORECASE)
                if match and match.group(1).strip():
                    question += " " + match.group(1).strip()
                continue
            question += " " + line

    if len(options) < 2:
        options = []
        for i, line in enumerate(filtered):
            if i >= 2:
                opt = re.sub(r"^[a-d][\.\)\:\s]+", "", line, flags=re.IGNORECASE)
                if opt and len(opt) > 2:
                    options.append(opt)
            else:
                if not question:
                    question = line
                else:
                    question += " " + line

    question = " ".join(question.split()).strip()
    options = [" ".join(o.split()).strip() for o in options[:4] if len(o.strip()) > 2]

    unique_options: List[str] = []
    for opt in options:
        if opt not in unique_options:
            unique_options.append(opt)
    options = unique_options

    if len(question) < 15 or len(options) < 2:
        return None

    if correct_idx < 0 or correct_idx >= len(options):
        correct_idx = 0

    return Mcq(question=question, options=options, correct=correct_idx)


# =========================
# Correct option detection
# =========================

def _green_pixel_count(bgr_img: np.ndarray) -> int:
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)

    green_ranges = [
        (np.array([35, 100, 100]), np.array([85, 255, 255])),
        (np.array([35, 50, 50]), np.array([85, 255, 200])),
        (np.array([70, 80, 80]), np.array([100, 255, 255])),
        (np.array([30, 100, 100]), np.array([50, 255, 255])),
    ]

    mask_total = None
    for lower, upper in green_ranges:
        mask = cv2.inRange(hsv, lower, upper)
        mask_total = mask if mask_total is None else cv2.bitwise_or(mask_total, mask)

    kernel = np.ones((3, 3), np.uint8)
    mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_CLOSE, kernel)
    mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_OPEN, kernel)

    return int(cv2.countNonZero(mask_total))


def detect_correct_option_by_right_icon(page_image: Image.Image) -> int:
    """
    Detect the option index (0-3) by looking for a green checkmark icon
    on the right side of the option rows.

    Returns -1 if not found.
    """
    rgb = np.array(page_image)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    height, width = bgr.shape[:2]

    # Option vertical bands (mirrors runpod/extract_mcq.py mapping)
    bands = [
        (0.28, 0.40),
        (0.40, 0.52),
        (0.52, 0.65),
        (0.65, 0.85),
    ]

    # Right-side area where ✓ / X icons appear
    x0 = int(width * 0.72)
    x1 = int(width * 0.97)

    scores: List[int] = []
    for y0r, y1r in bands:
        y0 = int(height * y0r)
        y1 = int(height * y1r)
        crop = bgr[y0:y1, x0:x1]
        scores.append(_green_pixel_count(crop))

    best_idx = int(np.argmax(scores)) if scores else -1
    best_score = scores[best_idx] if best_idx >= 0 else 0

    # Threshold: require some minimum green pixels
    # (tuned conservatively; adjust if needed)
    if best_score >= 150:
        return best_idx

    # Fallback: sometimes the PDF marks the correct option via a wider green/teal
    # element that isn't fully inside the right-side crop (or the icon isn't green).
    return detect_correct_option_by_page_green(page_image)


def detect_correct_option_by_page_green(page_image: Image.Image) -> int:
    """Fallback green detection over the entire page.

    Mirrors the existing RunPod script strategy: find the most prominent green/teal
    contour and map its Y position to option A-D.
    """
    rgb = np.array(page_image)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    height, width = bgr.shape[:2]

    green_ranges = [
        (np.array([35, 100, 100]), np.array([85, 255, 255])),
        (np.array([35, 50, 50]), np.array([85, 255, 200])),
        (np.array([70, 80, 80]), np.array([100, 255, 255])),
        (np.array([30, 100, 100]), np.array([50, 255, 255])),
    ]

    contours_all = []
    for lower, upper in green_ranges:
        mask = cv2.inRange(hsv, lower, upper)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_all.extend(contours)

    if not contours_all:
        return -1

    candidates = []
    for cnt in contours_all:
        area = cv2.contourArea(cnt)
        if area < 200:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        center_y = y + h / 2
        center_x = x + w / 2
        rel_y = center_y / height
        rel_x = center_x / width

        if rel_y < 0.20 or rel_y > 0.90:
            continue

        aspect = w / (h + 1)
        is_checkish = (0.5 < aspect < 2.0) and (500 < area < 10000)
        is_highlight = (aspect > 3.0) and (area > 2000)
        if is_checkish or is_highlight:
            candidates.append((area, rel_y, rel_x))

    if not candidates:
        return -1

    candidates.sort(key=lambda t: t[0], reverse=True)
    _area, rel_y, _rel_x = candidates[0]

    if rel_y < 0.28:
        return -1
    if rel_y < 0.40:
        return 0
    if rel_y < 0.52:
        return 1
    if rel_y < 0.65:
        return 2
    if rel_y < 0.85:
        return 3
    return -1


# =========================
# Dedup helpers
# =========================

def get_signature(question: str) -> str:
    text = question.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = " ".join(text.split())
    return text[:200]


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def is_duplicate(sig: str, seen: List[str], threshold: float) -> bool:
    for prev in seen:
        if similarity(sig, prev) >= threshold:
            return True
    return False


# =========================
# PDF processing
# =========================

def _get_page_count(pdf_path: Path) -> int:
    from PyPDF2 import PdfReader

    reader = PdfReader(str(pdf_path))
    return len(reader.pages)


def extract_mcqs_from_pdf(
    pdf_path: Path,
    *,
    poppler_path: Optional[str],
    dpi: int,
    batch_size: int,
    num_workers: int,
    dedup_threshold: float,
) -> List[Mcq]:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    total_pages = _get_page_count(pdf_path)
    print(f"\nProcessing: {pdf_path.name}")
    print(f"  Total pages: {total_pages}")

    mcqs: List[Mcq] = []
    seen_sigs: List[str] = []

    for start in range(1, total_pages + 1, batch_size):
        end = min(start + batch_size - 1, total_pages)
        pages = convert_from_path(
            str(pdf_path),
            dpi=dpi,
            first_page=start,
            last_page=end,
            poppler_path=poppler_path,
        )

        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            futures = {}
            for i, page_img in enumerate(pages):
                page_num = start + i
                futures[ex.submit(_process_page, page_img)] = page_num

            for fut in as_completed(futures):
                page_num = futures[fut]
                mcq = fut.result()
                if mcq is None:
                    continue

                sig = get_signature(mcq.question)
                if is_duplicate(sig, seen_sigs, dedup_threshold):
                    continue

                seen_sigs.append(sig)
                mcqs.append(mcq)

        # Help GC in long runs
        del pages

        print(f"  Pages {start}-{end}: total MCQs so far = {len(mcqs)}")

    return mcqs


def _process_page(page_img: Image.Image) -> Optional[Mcq]:
    text, _confidence = extract_text_with_confidence(page_img)
    correct_idx = detect_correct_option_by_right_icon(page_img)
    mcq = parse_mcq(text, correct_idx)
    return mcq


def write_website_json(mcqs: List[Mcq], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = []
    for idx, mcq in enumerate(mcqs, start=1):
        payload.append(
            {
                "question": mcq.question,
                "options": mcq.options,
                "correct": mcq.correct,
                "id": idx,
            }
        )

    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract MCQs (with options + correct ✓) into website/data JSON")

    parser.add_argument("--subject-dir", type=str, default=str(DEFAULT_SUBJECT_DIR), help="Folder containing PDFs (default: ./Subject2)")
    parser.add_argument("--website-data-dir", type=str, default=str(DEFAULT_WEBSITE_DATA_DIR), help="Output folder (default: ./website/data)")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--pdf", type=str, help="Path to a specific PDF")
    group.add_argument("--pdf-index", type=int, help="Select PDF by index from Subject folder (1-based)")

    parser.add_argument("--dpi", type=int, default=DPI)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--dedup-threshold", type=float, default=SIMILARITY_THRESHOLD)

    parser.add_argument(
        "--poppler-path",
        type=str,
        default=os.environ.get("POPPLER_PATH"),
        help="Optional Poppler bin path (mainly for Windows). Can also set POPPLER_PATH env var.",
    )

    args = parser.parse_args()

    subject_dir = Path(args.subject_dir)
    website_data_dir = Path(args.website_data_dir)

    if args.pdf:
        pdf_path = Path(args.pdf)
    else:
        pdfs = list_pdfs(subject_dir)
        display_pdf_list(pdfs)
        if not pdfs:
            raise SystemExit(f"No PDFs found in: {subject_dir}")

        if args.pdf_index is None:
            idx = prompt_pdf_index(len(pdfs))
        else:
            idx = args.pdf_index
            if idx < 1 or idx > len(pdfs):
                raise SystemExit(f"--pdf-index must be between 1 and {len(pdfs)}")

        pdf_path = pdfs[idx - 1]

    if not pdf_path.exists():
        raise SystemExit(f"PDF not found: {pdf_path}")

    mcqs = extract_mcqs_from_pdf(
        pdf_path,
        poppler_path=args.poppler_path,
        dpi=args.dpi,
        batch_size=args.batch_size,
        num_workers=args.workers,
        dedup_threshold=args.dedup_threshold,
    )

    out_path = website_data_dir / f"{pdf_path.stem}.json"
    write_website_json(mcqs, out_path)

    print("\n" + "=" * 60)
    print("DONE")
    print(f"  Extracted MCQs: {len(mcqs)}")
    print(f"  Output JSON: {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
