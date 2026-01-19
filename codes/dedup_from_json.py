"""
MCQ PDF Deduplication Tool - JSON Based
- Extracts questions from PDF pages and stores in JSON
- Detects duplicate questions using fuzzy matching
- Creates deduplicated PDF with unique questions only
- Optimized for GPU (RTX 5090) and high-RAM systems
"""

import os
import re
import sys
import json
import shutil
from pathlib import Path
from collections import defaultdict
from difflib import SequenceMatcher
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

import pytesseract
from pdf2image import convert_from_path
from PIL import Image

# Allow large images (disable decompression bomb protection - we trust the source)
Image.MAX_IMAGE_PIXELS = None

# For PDF manipulation
try:
    from PyPDF2 import PdfReader, PdfWriter
except ImportError:
    print("Installing PyPDF2...")
    os.system(f'"{sys.executable}" -m pip install PyPDF2')
    from PyPDF2 import PdfReader, PdfWriter

# Poppler path (used by pdf2image). If not found, pdf2image will rely on PATH.
_DEFAULT_POPPLER_PATH = Path(os.path.expandvars(r"%LOCALAPPDATA%\poppler\poppler-25.12.0\Library\bin"))
POPPLER_PATH = str(_DEFAULT_POPPLER_PATH) if _DEFAULT_POPPLER_PATH.exists() else None

# Similarity threshold (0.0 to 1.0) - questions with similarity >= this are considered duplicates
SIMILARITY_THRESHOLD = 0.85

# ============== RUNPOD / GPU CONFIGURATION ==============
# Number of parallel workers for processing pages
NUM_WORKERS = 8

# Batch size - number of pages to process in memory at once
# With 92GB RAM, we can handle large batches
BATCH_SIZE = 50

# DPI for OCR - higher = better quality but slower
OCR_DPI = 200

# Tesseract configuration for GPU optimization
# Use LSTM engine (more accurate, can use GPU if compiled with OpenCL)
TESSERACT_CONFIG = '--oem 1 --psm 6'
# ========================================================


def extract_text_from_page(page_image):
    """Extract text from a page image using OCR."""
    gray = page_image.convert('L')
    text = pytesseract.image_to_string(gray, lang='eng', config=TESSERACT_CONFIG)
    return text


def clean_text(text):
    """Clean and normalize text for comparison."""
    # Convert to lowercase
    text = text.lower()
    
    # Remove common OCR artifacts and noise
    text = re.sub(r'[^\w\s\.\?\,\(\)\-\']', ' ', text)
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    # Remove common header/footer elements
    noise_patterns = [
        r'\d{1,2}:\d{2}\s*(am|pm)?',  # Time
        r'sat\s+dec\s+\d+',  # Date
        r'pt2\s+\d+\.\d+',  # Exam code
        r'question\s+\d+',  # Question number
        r'multiple\s+choice',
        r'results',
        r'^\d+$',  # Standalone numbers
    ]
    
    for pattern in noise_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Clean up again
    text = ' '.join(text.split())
    
    return text.strip()


def extract_question_only(full_text):
    """
    Extract just the question text, ignoring options.
    Questions typically end with '...' or '?'
    """
    lines = full_text.strip().split('\n')
    lines = [line.strip() for line in lines if line.strip()]
    
    question_text = ""
    found_question_start = False
    
    for line in lines:
        line_lower = line.lower()
        
        # Skip header elements
        if any(x in line_lower for x in ['pt2', 'results', 'multiple choice']):
            continue
        if re.match(r'^\d{1,2}:\d{2}', line):  # Time
            continue
        if re.match(r'^question\s+\d+', line_lower):
            found_question_start = True
            continue
            
        if found_question_start or len(line) > 30:
            # This might be the question
            question_text += " " + line
            
            # Check if question ends here (ends with ... or ?)
            if line.rstrip().endswith('...') or line.rstrip().endswith('?'):
                break
    
    # If we didn't find a clear question ending, take the longest line as question
    if not question_text.strip():
        # Find the longest meaningful line (likely the question)
        meaningful_lines = [l for l in lines if len(l) > 20]
        if meaningful_lines:
            # Look for lines ending with ... or ?
            for l in meaningful_lines:
                if l.rstrip().endswith('...') or l.rstrip().endswith('?'):
                    question_text = l
                    break
            if not question_text:
                question_text = max(meaningful_lines, key=len)
    
    return clean_text(question_text)


def similarity(str1, str2):
    """Calculate similarity ratio between two strings."""
    if not str1 or not str2:
        return 0.0
    return SequenceMatcher(None, str1, str2).ratio()


def get_pdf_files(pdf_dir):
    """Get all PDF files from the directory."""
    if not pdf_dir.exists():
        return []
    
    pdf_files = sorted(
        [p for p in pdf_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"],
        key=lambda p: p.name.lower(),
    )
    return pdf_files


def display_pdf_list(pdf_files):
    """Display all available PDFs with numbers."""
    print("\n" + "=" * 60)
    print("AVAILABLE PDF FILES")
    print("=" * 60)
    
    if not pdf_files:
        print("No PDF files found!")
        return
    
    for i, pdf_file in enumerate(pdf_files, start=1):
        print(f"  {i:>3}) {pdf_file.name}")
    
    print("=" * 60)


def select_pdf(pdf_files):
    """Let user select a PDF by number."""
    while True:
        choice = input("\nEnter the PDF number to process (or 'q' to quit): ").strip().lower()
        
        if choice == 'q':
            return None
        
        if not choice.isdigit():
            print("Please enter a valid number.")
            continue
        
        idx = int(choice)
        if not (1 <= idx <= len(pdf_files)):
            print(f"Please enter a number between 1 and {len(pdf_files)}.")
            continue
        
        return pdf_files[idx - 1]


def get_pdf_page_count(pdf_path):
    """Get the total number of pages in a PDF without loading all pages."""
    reader = PdfReader(str(pdf_path))
    return len(reader.pages)


def process_single_page(args):
    """Process a single page - used for parallel processing."""
    pdf_path, page_num, dpi, poppler_path = args
    
    try:
        # Convert single page to image
        page_images = convert_from_path(
            str(pdf_path), 
            dpi=dpi, 
            first_page=page_num, 
            last_page=page_num,
            thread_count=1,
            poppler_path=poppler_path
        )
        
        if page_images:
            page_img = page_images[0]
            
            # Extract full text and question
            full_text = extract_text_from_page(page_img)
            question_text = extract_question_only(full_text)
            
            result = {
                "page_number": page_num,
                "question_text": question_text,
                "raw_text_preview": full_text[:500] if len(full_text) > 500 else full_text
            }
            
            # Free memory
            del page_images
            del page_img
            
            return result
    except Exception as e:
        print(f"  Error processing page {page_num}: {e}")
        return {
            "page_number": page_num,
            "question_text": "",
            "raw_text_preview": f"ERROR: {str(e)}"
        }
    
    return None


def process_batch(pdf_path, start_page, end_page, dpi, poppler_path):
    """Process a batch of pages at once (for high-RAM systems)."""
    results = []
    
    try:
        # Convert batch of pages to images
        page_images = convert_from_path(
            str(pdf_path), 
            dpi=dpi, 
            first_page=start_page, 
            last_page=end_page,
            thread_count=NUM_WORKERS,
            poppler_path=poppler_path
        )
        
        for i, page_img in enumerate(page_images):
            page_num = start_page + i
            
            # Extract full text and question
            full_text = extract_text_from_page(page_img)
            question_text = extract_question_only(full_text)
            
            result = {
                "page_number": page_num,
                "question_text": question_text,
                "raw_text_preview": full_text[:500] if len(full_text) > 500 else full_text
            }
            results.append(result)
        
        # Free memory
        del page_images
        
    except Exception as e:
        print(f"  Error processing batch {start_page}-{end_page}: {e}")
    
    return results


def scan_and_collect_questions(pdf_path, json_folder, dpi=OCR_DPI, use_parallel=True):
    """
    Scan PDF and collect all questions with their page numbers.
    Store the result in a JSON file.
    Uses parallel processing for speed on high-spec systems.
    
    Returns: (json_file_path, questions_data)
    """
    print(f"\nScanning PDF: {pdf_path.name}")
    
    # Get total pages without loading all images
    total_pages = get_pdf_page_count(pdf_path)
    print(f"Total pages: {total_pages}")
    print(f"Using {NUM_WORKERS} workers, batch size: {BATCH_SIZE}")
    
    # Store questions data
    questions_data = {
        "source_pdf": str(pdf_path),
        "pdf_name": pdf_path.name,
        "total_pages": total_pages,
        "scan_date": datetime.now().isoformat(),
        "questions": []
    }
    
    all_results = []
    
    if use_parallel and total_pages > BATCH_SIZE:
        # Process in batches for high-RAM systems
        print(f"\nExtracting questions using batch processing ({BATCH_SIZE} pages per batch)...")
        
        for batch_start in range(1, total_pages + 1, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE - 1, total_pages)
            print(f"  Processing pages {batch_start}-{batch_end}/{total_pages}...")
            
            batch_results = process_batch(pdf_path, batch_start, batch_end, dpi, POPPLER_PATH)
            all_results.extend(batch_results)
            
            # Progress update
            progress = (batch_end / total_pages) * 100
            print(f"  Progress: {progress:.1f}%")
    else:
        # Use ThreadPoolExecutor for parallel page processing
        print(f"\nExtracting questions using {NUM_WORKERS} parallel workers...")
        
        # Prepare arguments for each page
        page_args = [(pdf_path, page_num, dpi, POPPLER_PATH) for page_num in range(1, total_pages + 1)]
        
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            # Submit all tasks
            future_to_page = {executor.submit(process_single_page, args): args[1] for args in page_args}
            
            completed = 0
            for future in as_completed(future_to_page):
                page_num = future_to_page[future]
                result = future.result()
                if result:
                    all_results.append(result)
                
                completed += 1
                if completed % 50 == 0 or completed == total_pages:
                    print(f"  Processed {completed}/{total_pages} pages ({(completed/total_pages)*100:.1f}%)")
    
    # Sort results by page number
    all_results.sort(key=lambda x: x["page_number"])
    questions_data["questions"] = all_results
    
    # Save to JSON file
    json_folder.mkdir(parents=True, exist_ok=True)
    json_filename = f"{pdf_path.stem}_questions.json"
    json_path = json_folder / json_filename
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(questions_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nQuestions saved to: {json_path}")
    print(f"Total questions extracted: {len(questions_data['questions'])}")
    
    return json_path, questions_data


def find_duplicate_questions(questions_data, threshold=SIMILARITY_THRESHOLD):
    """
    Find similar/duplicate questions from the extracted data.
    
    Returns: (duplicate_page_numbers, duplicate_details)
    """
    questions = questions_data["questions"]
    total_questions = len(questions)
    
    print(f"\nChecking for duplicate questions (threshold: {threshold:.0%})...")
    
    # Track duplicates
    duplicate_pages = set()  # Pages that are duplicates (to be removed)
    duplicate_details = defaultdict(list)  # original_page -> [(dup_page, similarity), ...]
    
    for i, q1 in enumerate(questions):
        page_num_1 = q1["page_number"]
        q1_text = q1["question_text"]
        
        # Skip if this page is already marked as duplicate
        if page_num_1 in duplicate_pages:
            continue
        
        if (i + 1) % 100 == 0:
            print(f"  Comparing page {page_num_1}/{total_questions}...")
        
        # Compare with all subsequent questions
        for j in range(i + 1, len(questions)):
            q2 = questions[j]
            page_num_2 = q2["page_number"]
            q2_text = q2["question_text"]
            
            # Skip if already marked as duplicate
            if page_num_2 in duplicate_pages:
                continue
            
            # Calculate similarity
            sim_score = similarity(q1_text, q2_text)
            
            if sim_score >= threshold:
                duplicate_pages.add(page_num_2)
                duplicate_details[page_num_1].append({
                    "duplicate_page": page_num_2,
                    "similarity": round(sim_score, 4),
                    "duplicate_question": q2_text[:100] + "..." if len(q2_text) > 100 else q2_text
                })
    
    print(f"\nDuplicate detection complete!")
    print(f"  Unique questions: {total_questions - len(duplicate_pages)}")
    print(f"  Duplicate pages found: {len(duplicate_pages)}")
    
    return list(sorted(duplicate_pages)), duplicate_details


def save_duplicate_report(questions_data, duplicate_pages, duplicate_details, json_folder):
    """Save duplicate detection report to JSON."""
    pdf_name = questions_data["pdf_name"]
    
    report_data = {
        "source_pdf": questions_data["source_pdf"],
        "pdf_name": pdf_name,
        "total_pages": questions_data["total_pages"],
        "unique_pages": questions_data["total_pages"] - len(duplicate_pages),
        "duplicate_pages_count": len(duplicate_pages),
        "analysis_date": datetime.now().isoformat(),
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "duplicate_page_numbers": duplicate_pages,
        "duplicate_details": {str(k): v for k, v in duplicate_details.items()}
    }
    
    report_filename = f"{Path(pdf_name).stem}_duplicates_report.json"
    report_path = json_folder / report_filename
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nDuplicate report saved to: {report_path}")
    
    return report_path


def create_deduplicated_pdf(source_pdf_path, duplicate_pages, output_pdf_path):
    """
    Create a new PDF with duplicate pages removed.
    
    Args:
        source_pdf_path: Path to the original PDF
        duplicate_pages: List of page numbers to remove (1-indexed)
        output_pdf_path: Path for the output deduplicated PDF
    """
    print(f"\nCreating deduplicated PDF...")
    print(f"  Removing {len(duplicate_pages)} duplicate pages...")
    
    reader = PdfReader(str(source_pdf_path))
    writer = PdfWriter()
    
    total_pages = len(reader.pages)
    duplicate_pages_set = set(duplicate_pages)
    
    # Add only non-duplicate pages
    pages_kept = []
    for page_num in range(1, total_pages + 1):
        if page_num not in duplicate_pages_set:
            writer.add_page(reader.pages[page_num - 1])
            pages_kept.append(page_num)
    
    # Save the deduplicated PDF
    output_pdf_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_pdf_path, 'wb') as f:
        writer.write(f)
    
    print(f"\nDeduplicated PDF saved to: {output_pdf_path}")
    print(f"  Original pages: {total_pages}")
    print(f"  Pages kept: {len(pages_kept)}")
    print(f"  Pages removed: {len(duplicate_pages)}")
    
    return pages_kept


def print_summary(total_pages, unique_count, duplicate_count):
    """Print a summary of the deduplication process."""
    print("\n" + "=" * 60)
    print("DEDUPLICATION SUMMARY")
    print("=" * 60)
    print(f"  Total pages in original PDF:    {total_pages}")
    print(f"  Unique questions (kept):        {unique_count}")
    print(f"  Duplicate pages (removed):      {duplicate_count}")
    if total_pages > 0:
        reduction = (duplicate_count / total_pages) * 100
        print(f"  Size reduction:                 {reduction:.1f}%")
    print("=" * 60)


def main():
    # Try to locate Tesseract automatically
    current_cmd = getattr(pytesseract.pytesseract, "tesseract_cmd", "tesseract")
    if not shutil.which(str(current_cmd)):
        tesseract_on_path = shutil.which("tesseract")
        if tesseract_on_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_on_path
        else:
            print(
                "Warning: Tesseract OCR binary not found on PATH. "
                "Install Tesseract or set pytesseract.pytesseract.tesseract_cmd."
            )
    
    # Set up paths
    workspace_root = Path(__file__).resolve().parents[1]
    pdf_dir = workspace_root / "pdfs"
    json_folder = workspace_root / "JSON"
    output_dir = workspace_root / "output"
    
    print("\n" + "=" * 60)
    print("MCQ PDF DEDUPLICATION TOOL - JSON BASED")
    print("=" * 60)
    print(f"PDF Folder: {pdf_dir}")
    print(f"JSON Folder: {json_folder}")
    print(f"Output Folder: {output_dir}")
    
    # Step 1: Read PDF folder and display available PDFs
    pdf_files = get_pdf_files(pdf_dir)
    
    if not pdf_files:
        print(f"\nError: No PDF files found in {pdf_dir}")
        return
    
    display_pdf_list(pdf_files)
    
    # Step 2: User selects a PDF
    selected_pdf = select_pdf(pdf_files)
    
    if selected_pdf is None:
        print("Operation cancelled.")
        return
    
    print(f"\nSelected PDF: {selected_pdf.name}")
    
    # Step 3 & 4: Scan and collect questions, store in JSON
    json_path, questions_data = scan_and_collect_questions(selected_pdf, json_folder)
    
    # Step 5: Find duplicate questions
    duplicate_pages, duplicate_details = find_duplicate_questions(questions_data)
    
    # Save duplicate report to JSON
    save_duplicate_report(questions_data, duplicate_pages, duplicate_details, json_folder)
    
    # Display duplicate info
    if duplicate_pages:
        print(f"\nDuplicate pages to be removed: {duplicate_pages[:20]}{'...' if len(duplicate_pages) > 20 else ''}")
        
        # Show some duplicate details
        if duplicate_details:
            print("\nSample duplicate groups:")
            for i, (orig_page, dups) in enumerate(list(duplicate_details.items())[:5]):
                print(f"  Page {orig_page} has {len(dups)} duplicate(s): ", end="")
                dup_pages = [d['duplicate_page'] for d in dups]
                print(f"{dup_pages[:5]}{'...' if len(dup_pages) > 5 else ''}")
    else:
        print("\nNo duplicates found! All questions are unique.")
        return
    
    # Step 6: Get output PDF name from user and create deduplicated PDF
    print("\n" + "-" * 60)
    output_name = input("Enter the name for the deduplicated PDF (e.g., unique_questions.pdf): ").strip()
    
    if not output_name:
        output_name = f"{selected_pdf.stem}_unique.pdf"
    
    if not output_name.lower().endswith('.pdf'):
        output_name += '.pdf'
    
    output_pdf_path = output_dir / output_name
    
    # Confirm before proceeding
    print(f"\nOutput PDF will be saved to: {output_pdf_path}")
    confirm = input("Proceed with creating deduplicated PDF? (y/n): ").strip().lower()
    
    if confirm != 'y':
        print("Operation cancelled. Questions and duplicate report have been saved to JSON folder.")
        return
    
    # Create the deduplicated PDF
    pages_kept = create_deduplicated_pdf(selected_pdf, duplicate_pages, output_pdf_path)
    
    # Print summary
    print_summary(
        total_pages=questions_data["total_pages"],
        unique_count=len(pages_kept),
        duplicate_count=len(duplicate_pages)
    )
    
    print("\nDone! Files created:")
    print(f"  1. Questions JSON: {json_folder / f'{selected_pdf.stem}_questions.json'}")
    print(f"  2. Duplicates Report: {json_folder / f'{selected_pdf.stem}_duplicates_report.json'}")
    print(f"  3. Deduplicated PDF: {output_pdf_path}")


if __name__ == "__main__":
    main()
