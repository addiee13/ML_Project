#!/usr/bin/env python3
"""
Master Execution Script for Machine Learning Course Projects
Project 1: Spam Email Detection (Supervised Learning)
Project 2: Dry Bean Clustering (Unsupervised Learning)

Reproduces all experiments, tables, figures, predictions, and reports.
"""

import os
import sys
import subprocess
import time

def main():
    start_time = time.time()
    print("=================================================================")
    print("Starting ML Course Projects End-to-End Pipeline")
    print("=================================================================")
    
    # Ensure directories exist
    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("outputs/tables", exist_ok=True)
    os.makedirs("docs", exist_ok=True)
    
    # 1. Run Project 1: Spam Email Detection
    print("\n>>> [1/4] Executing Project 1: Spam Email Detection Pipeline...")
    spam_script = os.path.join("src", "spam_detection", "train_predict.py")
    if os.path.exists(spam_script):
        subprocess.run([sys.executable, spam_script], check=True)
        print(">>> Project 1 completed successfully.")
    else:
        print(f"Error: {spam_script} not found.")
        sys.exit(1)
        
    # 2. Run Project 2: Dry Bean Clustering
    print("\n>>> [2/4] Executing Project 2: Dry Bean Clustering Pipeline...")
    bean_script = os.path.join("src", "dry_bean", "run_dry_bean.py")
    if os.path.exists(bean_script):
        subprocess.run([sys.executable, bean_script], check=True)
        print(">>> Project 2 completed successfully.")
    else:
        print(f"Error: {bean_script} not found.")
        sys.exit(1)

    # 3. Generate Publication-Quality Report PDF
    print("\n>>> [3/4] Compiling Academic Report PDF (docs/report.pdf)...")
    report_script = os.path.join("src", "generate_report_pdf.py")
    if os.path.exists(report_script):
        subprocess.run([sys.executable, report_script], check=True)
        print(">>> Report PDF compiled successfully.")
    else:
        print(f"Warning: {report_script} not found.")

    # 4. Generate Presentation Slides (PPTX & PDF)
    print("\n>>> [4/4] Building Presentation Deck (PPTX & PDF)...")
    pres_script = os.path.join("src", "generate_presentation.py")
    if os.path.exists(pres_script):
        subprocess.run([sys.executable, pres_script], check=True)
        print(">>> Presentation slides generated successfully.")
    else:
        print(f"Warning: {pres_script} not found.")

    elapsed = time.time() - start_time
    print(f"\n=================================================================")
    print(f"Pipeline finished in {elapsed:.2f} seconds.")
    print("=================================================================")

if __name__ == "__main__":
    main()
