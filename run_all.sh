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
apt-get install -y -qq tesseract-ocr poppler-utils wget

# Step 2: Install Python dependencies
echo ""
echo "[2/5] Installing Python packages..."
pip install -q pdf2image pytesseract opencv-python numpy Pillow PyPDF2

# Step 3: Create directories
echo ""
echo "[3/5] Creating directories..."
mkdir -p /workspace/pdfs
mkdir -p /workspace/output

# Step 4: Download PDFs from Google Drive
echo ""
echo "[4/5] Downloading PDF files from Google Drive..."

download_gdrive() {
    FILE_ID=$1
    OUTPUT=$2
    echo "  Downloading $OUTPUT..."
    wget --quiet --load-cookies /tmp/cookies.txt \
        "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='$FILE_ID -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$FILE_ID" \
        -O "$OUTPUT" 2>/dev/null || wget --quiet --no-check-certificate "https://docs.google.com/uc?export=download&id=$FILE_ID" -O "$OUTPUT"
    rm -f /tmp/cookies.txt
}

# Google Drive File IDs
COMS_ID="1h8vTjKWRVEiEsNWMqu7cwj6FgUN8qOp0"
HPL_ID="1h4ik5KHom6QSNrHdKRa2b1U5wHJhfDzV"
OPS_ID="15Q8wI2GW6tGSK07ILZIdf3BTmuXOgouO"
RNAV_ID="107zwdXgeghdqHxw2udlh5r-3DjtQ03VX"

download_gdrive $COMS_ID "/workspace/pdfs/COMS.pdf"
download_gdrive $HPL_ID "/workspace/pdfs/HPL.pdf"
download_gdrive $OPS_ID "/workspace/pdfs/OPS.pdf"
download_gdrive $RNAV_ID "/workspace/pdfs/RNAV.pdf"

echo "  Downloads complete!"
ls -la /workspace/pdfs/

# Step 5: Run extraction
echo ""
echo "[5/5] Starting MCQ extraction..."
echo "============================================"
python /workspace/MCQ/extract_mcq.py

echo ""
echo "============================================"
echo "DONE! Output files are in /workspace/output/"
echo "============================================"
ls -la /workspace/output/
