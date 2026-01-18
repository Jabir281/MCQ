#!/bin/bash
# ============================================
# MCQ Extraction - One Command Setup
# Run this single script on RunPod!
# ============================================

set -e  # Exit on any error

echo "============================================"
echo "MCQ EXTRACTION SETUP"
echo "============================================"

# Step 1: Install system dependencies
echo ""
echo "[1/5] Installing system dependencies..."
apt-get update -qq
apt-get install -y -qq tesseract-ocr poppler-utils

# Step 2: Install Python dependencies
echo ""
echo "[2/5] Installing Python packages..."
pip install -q pdf2image pytesseract opencv-python numpy Pillow PyPDF2 gdown

# Step 3: Create directories
echo ""
echo "[3/5] Creating directories..."
mkdir -p /workspace/pdfs
mkdir -p /workspace/output

# Step 4: Download PDFs from Google Drive using gdown
echo ""
echo "[4/5] Downloading PDF files from Google Drive..."

# Google Drive File IDs
COMS_ID="1h8vTjKWRVEiEsNWMqu7cwj6FgUN8qOp0"
HPL_ID="1h4ik5KHom6QSNrHdKRa2b1U5wHJhfDzV"
OPS_ID="15Q8wI2GW6tGSK07ILZIdf3BTmuXOgouO"
RNAV_ID="107zwdXgeghdqHxw2udlh5r-3DjtQ03VX"

echo "  Downloading COMS.pdf..."
gdown --id $COMS_ID -O /workspace/pdfs/COMS.pdf --quiet

echo "  Downloading HPL.pdf..."
gdown --id $HPL_ID -O /workspace/pdfs/HPL.pdf --quiet

echo "  Downloading OPS.pdf..."
gdown --id $OPS_ID -O /workspace/pdfs/OPS.pdf --quiet

echo "  Downloading RNAV.pdf..."
gdown --id $RNAV_ID -O /workspace/pdfs/RNAV.pdf --quiet

echo ""
echo "  Verifying downloads..."
ls -lh /workspace/pdfs/

# Check if files are valid (not empty or too small)
for pdf in /workspace/pdfs/*.pdf; do
    size=$(stat -f%z "$pdf" 2>/dev/null || stat -c%s "$pdf" 2>/dev/null)
    if [ "$size" -lt 100000 ]; then
        echo "  ❌ ERROR: $pdf seems corrupted (too small: $size bytes)"
        echo "  Please check Google Drive sharing permissions (must be 'Anyone with link')"
        exit 1
    fi
    echo "  ✅ $(basename $pdf): $size bytes"
done

# Step 5: Run extraction
echo ""
echo "[5/5] Starting MCQ extraction..."
echo "============================================"
python /workspace/MCQ/extract_mcq.py

echo ""
echo "============================================"
echo "VERIFYING EXTRACTION QUALITY..."
echo "============================================"
python /workspace/MCQ/verify_extraction.py

echo ""
echo "============================================"
echo "OUTPUT FILES:"
echo "============================================"
ls -la /workspace/output/

echo ""
echo "============================================"
echo "NEXT STEPS:"
echo "============================================"
echo "If extraction looks good, push to GitHub:"
echo ""
echo "  cd /workspace/MCQ && git add . && git commit -m 'Add extracted data' && git push"
echo ""
echo "If there are issues, you can:"
echo "  1. Check samples: python verify_extraction.py RNAV"
echo "  2. Re-run extraction: python extract_mcq.py"
echo "============================================"
