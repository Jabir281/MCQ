# Aviation MCQ Test Website

Practice for your ATPL/CPL exams with this MCQ question bank.

## Quick Start on RunPod

**One command to run everything:**

```bash
cd /workspace && git clone https://github.com/Jabir281/MCQ.git && chmod +x MCQ/run_all.sh && ./MCQ/run_all.sh
```

That's it! The script will:
1. Install all dependencies
2. Download PDFs from Google Drive
3. Extract questions using OCR
4. Generate JSON files for the website

## After Extraction

Push the updated data to GitHub:
```bash
cd /workspace/MCQ
git add .
git commit -m "Add extracted MCQ data"
git push
```

## Deploy to Hostinger

1. Download/clone this repo
2. Upload the `website/` folder contents to your Hostinger `public_html/`
3. Done!

## Project Structure

```
MCQ/
├── run_all.sh          # One-click setup script for RunPod
├── extract_mcq.py      # MCQ extraction script
├── website/
│   ├── index.html      # Main page
│   ├── css/style.css   # Styling
│   ├── js/app.js       # Quiz logic
│   └── data/           # JSON question files (generated)
└── README.md
```

## Adding New Subjects

1. Add Google Drive file ID to `run_all.sh`
2. Add subject name to `extract_mcq.py` SUBJECT_NAMES dict
3. Run extraction again
