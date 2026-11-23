import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "steps"))

from thingspeak_bridge import to_power_units, decide, PU_BASE, RATED_KW
from step15_robust_20seed_eval import simulate_microgrid_seed


def test_thingspeak_per_unit_calibration():
    raw = pd.DataFrame({
        "time": pd.date_range("2026-01-01", periods=4, freq="15min"),
        "field2": [PU_BASE["pv"], 0.0, PU_BASE["pv"] * 0.5, 0.0],
        "field4": [0.0, PU_BASE["wind"], 0.0, 0.0],
        "field6": [0.0, 0.0, 0.0, PU_BASE["diesel"]],
    })
    cal = to_power_units(raw)
    assert round(cal["pv_kw"].iloc[0], 1) == RATED_KW["pv"]
    assert round(cal["wind_kw"].iloc[1], 1) == RATED_KW["wind"]
    assert round(cal["diesel_kw"].iloc[3], 1) == RATED_KW["diesel"]


def test_dispatch_decision_triggers_on_shortfall():
    low_ren = pd.DataFrame({
        "time": pd.date_range("2026-01-01", periods=4, freq="15min"),
        "pv_kw": [10.0, 8.0, 5.0, 2.0],
        "wind_kw": [15.0, 12.0, 10.0, 8.0],
        "renewable_kw": [25.0, 20.0, 15.0, 10.0],
    })
    cmd, expected, shortfall = decide(low_ren)
    assert cmd == 1
    assert shortfall > 2.0


def test_ai_reduces_unserved_energy():
    res = simulate_microgrid_seed(seed=7, days=7)
    assert res["ai_unserved_kwh"] < res["reactive_unserved_kwh"]
    assert res["unserved_reduction_pct"] > 30.0