"""
MCQ Data Extraction Script for RunPod
Optimized for fast extraction on powerful GPU servers
Run this on RunPod after uploading your PDFs
"""

import os
import re
import json
from pathlib import Path
from collections import defaultdict
from difflib import SequenceMatcher
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytesseract
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance
import numpy as np
import cv2

# ============================================================
# CONFIGURATION
# ============================================================
SUBJECT_DIR = "/workspace/pdfs"
OUTPUT_DIR = "/workspace/output"
WEBSITE_DATA_DIR = "/workspace/MCQ/website/data"  # Also save to website folder

# Subject full names (add more as needed)
SUBJECT_NAMES = {
    "COMS": "Communications",
    "HPL": "Human Performance & Limitations",
    "OPS": "Flight Operations",
    "RNAV": "Radio Navigation",
    # Add more subjects here
}

# Processing settings
DPI = 200  # Higher = better quality but slower
BATCH_SIZE = 50  # Pages per batch (increase for more RAM)
NUM_WORKERS = 8  # Parallel workers for OCR
SIMILARITY_THRESHOLD = 0.92  # For duplicate detection (higher = less aggressive)

# ============================================================


def preprocess_image(pil_image):
    """Preprocess image for better OCR."""
    img = np.array(pil_image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # Denoise
    gray = cv2.fastNlMeansDenoising(gray, h=10)
    
    # Binary threshold
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return Image.fromarray(binary)


def extract_text_from_page(page_image):
    """Extract text using Tesseract OCR."""
    processed = preprocess_image(page_image)
    config = '--oem 3 --psm 6'
    text = pytesseract.image_to_string(processed, lang='eng', config=config)
    return text


def detect_correct_answer(page_image):
    """Detect correct answer by finding teal/green highlight."""
    img_array = np.array(page_image)
    img_hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    
    # Teal/cyan color range
    lower_teal = np.array([70, 80, 80])
    upper_teal = np.array([100, 255, 255])
    mask = cv2.inRange(img_hsv, lower_teal, upper_teal)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # Try green range
        lower_green = np.array([35, 80, 80])
        upper_green = np.array([75, 255, 255])
        mask = cv2.inRange(img_hsv, lower_green, upper_green)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return -1
    
    # Find largest wide contour (option bar)
    best_contour = None
    max_area = 0
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area and area > 1000:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > h * 2:
                max_area = area
                best_contour = cnt
    
    if best_contour is None:
        return -1
    
    x, y, w, h = cv2.boundingRect(best_contour)
    img_height = img_array.shape[0]
    relative_y = (y + h / 2) / img_height
    
    if relative_y < 0.28:
        return -1
    elif relative_y < 0.38:
        return 0
    elif relative_y < 0.48:
        return 1
    elif relative_y < 0.58:
        return 2
    elif relative_y < 0.72:
        return 3
    return -1


def clean_text(text):
    """Remove OCR artifacts."""
    patterns = [
        r'\d{1,2}:\d{2}\s*(AM|PM|am|pm)?',
        r'(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d+',
        r'PT2\s*\d+\.\d+',
        r'062\s+PT2',
        r'©.*?(Con|Tota|@|&)',
        r'<>\s*Results',
        r'Results',
        r'Multiple Choice',
        r'^\s*Question\s+\d+\s*$',
    ]
    
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    return text.strip()


def parse_mcq(raw_text, correct_idx=-1):
    """Parse question and options from text."""
    text = clean_text(raw_text)
    lines = [l.strip() for l in text.split('\n') if l.strip() and len(l.strip()) > 2]
    
    # Filter noise
    filtered = []
    for line in lines:
        if any(x in line.lower() for x in ['results', 'multiple choice', 'pevesys']):
            continue
        if re.match(r'^[\d\s\.\-\(\)]+$', line):
            continue
        if len(line) < 4:
            continue
        filtered.append(line)
    
    if len(filtered) < 2:
        return None
    
    question = ""
    options = []
    question_found = False
    
    for line in filtered:
        if not question_found:
            if re.search(r'question\s*\d+', line, re.IGNORECASE):
                match = re.search(r'question\s*\d+\s*[\(\)]?\s*(.*)', line, re.IGNORECASE)
                if match and match.group(1).strip():
                    question = match.group(1).strip()
                continue
            
            question += " " + line
            
            if line.rstrip().endswith('...') or line.rstrip().endswith('?'):
                question_found = True
        else:
            opt = re.sub(r'^[a-d][\.\)\s]+', '', line.strip(), flags=re.IGNORECASE)
            if opt and len(opt) > 2:
                options.append(opt)
    
    # Fallback parsing
    if not question_found and len(filtered) >= 3:
        for i, line in enumerate(filtered):
            if line.rstrip().endswith('...') or line.rstrip().endswith('?'):
                question = ' '.join(filtered[:i+1])
                options = [re.sub(r'^[a-d][\.\)\s]+', '', l, flags=re.IGNORECASE) 
                          for l in filtered[i+1:]]
                break
        
        if not question and len(filtered) >= 5:
            question = ' '.join(filtered[:2])
            options = filtered[2:]
    
    question = ' '.join(clean_text(question).split())
    options = [clean_text(o) for o in options[:4] if len(clean_text(o)) > 2]
    
    if len(question) < 15 or len(options) < 2:
        return None
    
    if correct_idx < 0 or correct_idx >= len(options):
        correct_idx = 0
    
    return {
        "question": question,
        "options": options,
        "correct": correct_idx
    }


def process_page(args):
    """Process a single page (for parallel execution)."""
    page_img, page_num = args
    text = extract_text_from_page(page_img)
    correct_idx = detect_correct_answer(page_img)
    mcq = parse_mcq(text, correct_idx)
    return page_num, mcq


def get_signature(question):
    """Create signature for duplicate detection."""
    text = question.lower()
    # Remove common OCR noise from beginning
    noise_prefixes = ['cae', 'crew', 'training', 'sigma', 'web', 'https', 'http', 'www']
    for prefix in noise_prefixes:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join(text.split())
    # Use more of the question for better uniqueness (200 chars instead of 100)
    return text[:200]


def similarity(s1, s2):
    """Calculate string similarity."""
    # If strings are very short, require exact match
    if len(s1) < 30 or len(s2) < 30:
        return 1.0 if s1 == s2 else 0.0
    return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()


def process_pdf(pdf_path):
    """Process entire PDF and extract MCQs."""
    from PyPDF2 import PdfReader
    
    print(f"\n{'='*60}")
    print(f"Processing: {os.path.basename(pdf_path)}")
    
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)
    print(f"  Total pages: {total_pages}")
    
    mcqs = []
    seen_sigs = []
    
    for start in range(0, total_pages, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total_pages)
        print(f"  Processing pages {start+1}-{end}...")
        
        pages = convert_from_path(
            pdf_path,
            dpi=DPI,
            first_page=start + 1,
            last_page=end
        )
        
        # Process pages in parallel
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = [executor.submit(process_page, (img, start + i + 1)) 
                      for i, img in enumerate(pages)]
            
            for future in as_completed(futures):
                page_num, mcq = future.result()
                
                if mcq:
                    sig = get_signature(mcq['question'])
                    
                    is_dup = any(similarity(sig, s) >= SIMILARITY_THRESHOLD for s in seen_sigs)
                    
                    if not is_dup and len(sig) > 15:
                        seen_sigs.append(sig)
                        mcq['id'] = len(mcqs) + 1
                        mcqs.append(mcq)
        
        del pages
    
    print(f"  Extracted: {len(mcqs)} unique questions")
    return mcqs


def main():
    print("="*60)
    print("MCQ DATA EXTRACTION - RunPod Edition")
    print("="*60)
    
    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(WEBSITE_DATA_DIR, exist_ok=True)
    
    # Find PDFs
    pdf_files = list(Path(SUBJECT_DIR).glob("*.pdf"))
    
    if not pdf_files:
        print(f"\nERROR: No PDF files found in {SUBJECT_DIR}")
        print("Please upload your PDF files to that directory.")
        return
    
    print(f"\nFound {len(pdf_files)} PDF files:")
    for pdf in pdf_files:
        print(f"  - {pdf.name}")
    
    subjects = {}
    
    for pdf_path in pdf_files:
        subject_code = pdf_path.stem
        subject_name = SUBJECT_NAMES.get(subject_code, subject_code)
        
        mcqs = process_pdf(str(pdf_path))
        
        if mcqs:
            output_file = os.path.join(OUTPUT_DIR, f"{subject_code}.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(mcqs, f, indent=2, ensure_ascii=False)
            
            # Also save to website data folder
            website_file = os.path.join(WEBSITE_DATA_DIR, f"{subject_code}.json")
            with open(website_file, 'w', encoding='utf-8') as f:
                json.dump(mcqs, f, indent=2, ensure_ascii=False)
            
            subjects[subject_code] = {
                "code": subject_code,
                "name": subject_name,
                "questionCount": len(mcqs),
                "file": f"{subject_code}.json"
            }
    
    # Save index to both locations
    index_file = os.path.join(OUTPUT_DIR, "subjects.json")
    with open(index_file, 'w', encoding='utf-8') as f:
        json.dump(subjects, f, indent=2, ensure_ascii=False)
    
    website_index = os.path.join(WEBSITE_DATA_DIR, "subjects.json")
    with open(website_index, 'w', encoding='utf-8') as f:
        json.dump(subjects, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*60)
    print("EXTRACTION COMPLETE!")
    print("="*60)
    total = 0
    for code, info in subjects.items():
        print(f"  {code}: {info['questionCount']} questions")
        total += info['questionCount']
    print(f"\n  TOTAL: {total} questions")
    print(f"\nOutput saved to: {OUTPUT_DIR}")
    print("Download the JSON files and upload to your website's /data folder")


if __name__ == "__main__":
    main()
