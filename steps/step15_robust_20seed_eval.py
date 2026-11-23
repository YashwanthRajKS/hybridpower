from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)


def simulate_microgrid_seed(seed: int, days: int = 14):
    """Generates a 15-min hybrid PV-Wind-Diesel profile for a given random seed and compares AI vs Reactive dispatch."""
    rng = np.random.default_rng(seed)
    steps = days * 96  # 15-minute intervals
    t_idx = np.arange(steps)
    hour_of_day = (t_idx % 96) / 4.0

    # Solar PV (100 kW peak) with random cloud transients
    solar_base = np.maximum(0.0, np.sin(np.pi * (hour_of_day - 6.0) / 12.0)) ** 1.6
    cloud_factor = rng.uniform(0.35, 1.05, size=steps)
    # Inject sudden multi-step cloud drops
    for _ in range(days * 2):
        start = int(rng.integers(24, steps - 8))
        cloud_factor[start : start + 4] *= rng.uniform(0.25, 0.55)
    pv_kw = np.clip(100.0 * solar_base * cloud_factor, 0.0, 100.0)

    # Wind (120 kW rated) with autoregressive gusts & lulls
    wind_noise = rng.normal(0.0, 7.5, size=steps)
    wind_series = np.zeros(steps)
    wind_series[0] = rng.uniform(35.0, 70.0)
    for i in range(1, steps):
        wind_series[i] = np.clip(0.88 * wind_series[i - 1] + 0.12 * 52.0 + wind_noise[i], 0.0, 120.0)
    wind_kw = wind_series

    # Load demand (45 - 88 kW diurnal curve + noise)
    load_kw = 54.0 + 18.0 * np.sin(2 * np.pi * (hour_of_day - 8.0) / 24.0) + rng.normal(0.0, 3.5, size=steps)
    load_kw = np.clip(load_kw, 35.0, 95.0)

    ren_true = pv_kw + wind_kw
    # 30-min ahead (2-step) noisy forecast
    ren_forecast_30m = np.roll(ren_true, -2) + rng.normal(0.0, 4.5, size=steps)
    ren_forecast_30m[-2:] = ren_true[-2:]
    ren_forecast_30m = np.clip(ren_forecast_30m, 0.0, 220.0)

    # Simulate Reactive (No-AI) vs AI Predictive Dispatch (2-step = 30 min warm-up delay, 4-step = 1h min run)
    def run_policy(predictive: bool):
        unserved_kwh = 0.0
        diesel_kwh = 0.0
        starts = 0
        warmup_rem = 0
        min_run_rem = 0
        online = False

        for i in range(steps):
            net_now = load_kw[i] - ren_true[i]
            if predictive:
                net_ahead = load_kw[min(i + 2, steps - 1)] - 0.95 * ren_forecast_30m[i]
                trigger = (net_ahead > 2.0) or (net_now > 2.0)
            else:
                trigger = net_now > 2.0

            if not online and warmup_rem == 0 and trigger:
                warmup_rem = 2  # 30-minute warm-up
                starts += 1
            elif warmup_rem > 0:
                warmup_rem -= 1
                if warmup_rem == 0:
                    online = True
                    min_run_rem = 4  # 1-hour minimum run time

            if online:
                if min_run_rem > 0:
                    min_run_rem -= 1
                elif not trigger:
                    online = False

            diesel_out = max(30.0, min(100.0, net_now)) if online else 0.0
            deficit = max(0.0, load_kw[i] - (ren_true[i] + diesel_out))
            unserved_kwh += deficit * 0.25
            diesel_kwh += diesel_out * 0.25

        return unserved_kwh, diesel_kwh, starts

    react_unserved, react_diesel, react_starts = run_policy(predictive=False)
    ai_unserved, ai_diesel, ai_starts = run_policy(predictive=True)
    reduction_pct = ((react_unserved - ai_unserved) / max(react_unserved, 1e-6)) * 100.0

    return {
        "seed": seed,
        "reactive_unserved_kwh": round(react_unserved, 2),
        "ai_unserved_kwh": round(ai_unserved, 2),
        "unserved_reduction_pct": round(reduction_pct, 2),
        "reactive_diesel_kwh": round(react_diesel, 2),
        "ai_diesel_kwh": round(ai_diesel, 2),
        "ai_improved": bool(ai_unserved < react_unserved),
    }


def main():
    results = [simulate_microgrid_seed(seed) for seed in range(1, 21)]
    df = pd.DataFrame(results)
    out_csv = OUTPUTS / "step15_20seed_robustness.csv"
    df.to_csv(out_csv, index=False)

    mean_red = df["unserved_reduction_pct"].mean()
    std_red = df["unserved_reduction_pct"].std()
    wins = int(df["ai_improved"].sum())

    print("=== Supporting Task 7: 20-Seed Statistical Robustness Benchmark ===")
    print(df.to_string(index=False))
    print(f"\nAI Outperformed Reactive Baseline in: {wins}/20 seeds ({wins * 5}%)")
    print(f"Unserved Energy Reduction: {mean_red:.2f}% +/- {std_red:.2f}% (Min: {df['unserved_reduction_pct'].min():.2f}%, Max: {df['unserved_reduction_pct'].max():.2f}%)")
    print(f"Saved report -> {out_csv}")


if __name__ == "__main__":
    main()