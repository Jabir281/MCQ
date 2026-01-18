"""
Verify MCQ Extraction Quality
Run this after extraction to check if data looks correct
"""

import json
import os
import sys

OUTPUT_DIR = "/workspace/output"
WEBSITE_DIR = "/workspace/MCQ/website/data"

def check_quality():
    print("=" * 60)
    print("MCQ EXTRACTION QUALITY CHECK")
    print("=" * 60)
    
    # Check if files exist
    subjects_file = os.path.join(OUTPUT_DIR, "subjects.json")
    if not os.path.exists(subjects_file):
        print("\n❌ ERROR: subjects.json not found!")
        print("   Extraction may have failed. Check for errors above.")
        return False
    
    with open(subjects_file, 'r', encoding='utf-8') as f:
        subjects = json.load(f)
    
    print(f"\n✅ Found {len(subjects)} subjects\n")
    
    all_good = True
    
    for code, info in subjects.items():
        print("-" * 60)
        print(f"📚 {code} - {info['name']}")
        print(f"   Questions: {info['questionCount']}")
        
        # Load questions
        q_file = os.path.join(OUTPUT_DIR, info['file'])
        if not os.path.exists(q_file):
            print(f"   ❌ File not found: {info['file']}")
            all_good = False
            continue
        
        with open(q_file, 'r', encoding='utf-8') as f:
            questions = json.load(f)
        
        if len(questions) == 0:
            print("   ❌ No questions extracted!")
            all_good = False
            continue
        
        # Check quality metrics
        good_questions = 0
        bad_questions = []
        
        for i, q in enumerate(questions[:50]):  # Check first 50
            # Quality checks
            is_good = True
            issues = []
            
            # Question length check
            if len(q['question']) < 20:
                is_good = False
                issues.append("question too short")
            
            # Options check
            if len(q['options']) < 2:
                is_good = False
                issues.append(f"only {len(q['options'])} options")
            
            # Check for OCR garbage
            garbage_indicators = ['<>', 'Results', 'PT2', '062', 'Ooms', 'Pevesys']
            for garbage in garbage_indicators:
                if garbage in q['question'] or any(garbage in opt for opt in q['options']):
                    is_good = False
                    issues.append(f"contains '{garbage}'")
                    break
            
            # Check if options are too short
            short_opts = sum(1 for opt in q['options'] if len(opt) < 5)
            if short_opts >= 2:
                is_good = False
                issues.append("options too short")
            
            if is_good:
                good_questions += 1
            else:
                if len(bad_questions) < 3:  # Store first 3 bad examples
                    bad_questions.append((i+1, q, issues))
        
        quality_pct = (good_questions / min(50, len(questions))) * 100
        
        if quality_pct >= 80:
            print(f"   ✅ Quality: {quality_pct:.0f}% good (checked first 50)")
        elif quality_pct >= 50:
            print(f"   ⚠️  Quality: {quality_pct:.0f}% good (some issues)")
            all_good = False
        else:
            print(f"   ❌ Quality: {quality_pct:.0f}% good (POOR)")
            all_good = False
        
        # Show sample good question
        print(f"\n   📝 Sample Question (#{questions[0]['id']}):")
        print(f"      Q: {questions[0]['question'][:100]}...")
        print(f"      Options: {len(questions[0]['options'])}")
        for j, opt in enumerate(questions[0]['options'][:4]):
            marker = "→" if j == questions[0]['correct'] else " "
            print(f"        {marker} {opt[:60]}{'...' if len(opt) > 60 else ''}")
        
        # Show bad examples if any
        if bad_questions:
            print(f"\n   ⚠️  Problem examples:")
            for idx, q, issues in bad_questions[:2]:
                print(f"      #{idx}: {', '.join(issues)}")
                print(f"         Q: {q['question'][:80]}...")
    
    print("\n" + "=" * 60)
    
    if all_good:
        print("✅ EXTRACTION LOOKS GOOD!")
        print("   You can proceed to push to GitHub.")
    else:
        print("⚠️  SOME ISSUES DETECTED")
        print("   Review the problems above.")
        print("   You may want to:")
        print("   1. Try increasing DPI in extract_mcq.py (line ~25)")
        print("   2. Check if PDFs downloaded correctly")
        print("   3. Run extraction again")
    
    print("=" * 60)
    
    return all_good


def show_random_samples(subject_code, count=5):
    """Show random sample questions from a subject"""
    import random
    
    q_file = os.path.join(OUTPUT_DIR, f"{subject_code}.json")
    if not os.path.exists(q_file):
        print(f"File not found: {q_file}")
        return
    
    with open(q_file, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    
    samples = random.sample(questions, min(count, len(questions)))
    
    print(f"\n{'='*60}")
    print(f"Random samples from {subject_code}")
    print(f"{'='*60}")
    
    for q in samples:
        print(f"\n📝 Question #{q['id']}:")
        print(f"   {q['question']}")
        print(f"   Options:")
        for i, opt in enumerate(q['options']):
            marker = "✓" if i == q['correct'] else " "
            print(f"     {marker} {['A','B','C','D'][i]}. {opt}")
        print()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Show samples for specific subject
        show_random_samples(sys.argv[1].upper(), 5)
    else:
        # Run quality check
        check_quality()
