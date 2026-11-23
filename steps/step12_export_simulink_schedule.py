from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs"
OUTPUT.mkdir(exist_ok=True)


def generate_simulink_schedules():
    # Compressed 2.5-second Simulink cloud-drop scenario derived from our AI dispatch rules:
    # - t = 0.00 to 0.40 s: Clear sky (1000 W/m2), renewable forecast >= load -> Diesel OFF (0)
    # - t = 0.40 s: AI forecaster detects upcoming cloud drop (irradiance drops at t = 0.46 s)
    #               and pre-triggers Diesel Breaker ON (1) so backup is synchronized in time.
    # - Reactive (No-AI): Misses the drop at t = 0.46 s and only closes breaker late at t = 0.95 s.

    t_grid = np.round(np.arange(0.0, 2.501, 0.01), 3)

    # AI Predictive Control: 0 before 0.40 s, 1 from 0.40 s onward
    ai_breaker_cmd = (t_grid >= 0.40).astype(float)

    # Reactive Baseline (No-AI): 0 before 0.95 s (delayed response), 1 from 0.95 s onward
    reactive_breaker_cmd = (t_grid >= 0.95).astype(float)

    # Irradiance profile matching the cloud-drop test (1000 W/m2 -> 300 W/m2 at t = 0.46 s)
    irradiance_wm2 = np.where(t_grid < 0.46, 1000.0, 300.0)

    df_ai = pd.DataFrame({
        "time_s": t_grid,
        "breaker_cmd": ai_breaker_cmd,
        "irradiance_wm2": irradiance_wm2,
    })

    df_react = pd.DataFrame({
        "time_s": t_grid,
        "breaker_cmd": reactive_breaker_cmd,
        "irradiance_wm2": irradiance_wm2,
    })

    ai_path = OUTPUT / "simulink_ai_breaker_schedule.csv"
    react_path = OUTPUT / "simulink_reactive_breaker_schedule.csv"

    df_ai.to_csv(ai_path, index=False)
    df_react.to_csv(react_path, index=False)

    print("=" * 65)
    print("STEP 12: SIMULINK BREAKER SCHEDULES EXPORTED")
    print("=" * 65)
    print(f"1. AI Predictive Schedule:      {ai_path}")
    print("   -> Diesel Breaker closes at t = 0.40 s (ahead of t = 0.46 s solar drop)")
    print(f"2. Reactive Baseline Schedule:  {react_path}")
    print("   -> Diesel Breaker closes late at t = 0.95 s (simulating start-up lag)")
    print("\nPreview (around t = 0.38 s to 0.50 s):")
    print(df_ai[(df_ai["time_s"] >= 0.38) & (df_ai["time_s"] <= 0.50)].to_string(index=False))


if __name__ == "__main__":
    generate_simulink_schedules()