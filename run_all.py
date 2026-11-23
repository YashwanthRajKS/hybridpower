"""Run the whole pipeline: data -> forecast -> SMOTE alert -> dispatch comparison -> RAG self-test.
Usage (VS Code terminal, inside this folder):   python run_all.py
Everything is written to the 'outputs' folder."""
import os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

STEPS = [
    ("1  Generate synthetic data",           "step1_generate_data.py"),
    ("2-3 Baseline + AI forecaster",          "step2_3_forecast.py"),
    ("3b SMOTE forecast-miss alert",          "step3b_smote_alert.py"),
    ("4-5 Diesel dispatch, with / without AI", "step4_5_dispatch.py"),
    ("7  RAG knowledge base + self-test",      "step7_rag.py"),
]
for title, script in STEPS:
    print("\n" + "=" * 78 + f"\nSTEP {title}\n" + "=" * 78, flush=True)
    t0 = time.time()
    r = subprocess.run([sys.executable, os.path.join(HERE, "steps", script)], cwd=OUT)
    if r.returncode != 0:
        print(f"\n!! {script} failed. Read the error above (or send it to your AI helper).")
        sys.exit(r.returncode)
    print(f"(done in {time.time() - t0:.0f} s)")
print("\nAll steps finished. Look in the 'outputs' folder for the CSV files and PNG pictures.")
print("To talk to the RAG assistant run:   python chat.py")
