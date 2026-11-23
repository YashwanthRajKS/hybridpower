import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / "dashboard" / ".env")

CHANNEL_ID = os.getenv("THINGSPEAK_CHANNEL_ID", "3570460").strip()
READ_KEY = os.getenv("THINGSPEAK_READ_KEY", "").strip()
CMD_WRITE_KEY = os.getenv("THINGSPEAK_CMD_WRITE_KEY", "").strip()

# Exact Simulink Mux -> ThingSpeak field mapping & per-unit scaling
FIELD_MAP = {"pv": "field2", "wind": "field4", "diesel": "field6"}
PU_BASE = {"pv": 6.0e5, "wind": 1.8e6, "diesel": 2.0e7}
RATED_KW = {"pv": 100.0, "wind": 120.0, "diesel": 100.0}
LOAD_KW = 60.0


def read_feeds(results=20):
    url = f"https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.json?results={results}"
    if READ_KEY:
        url += "&api_key=" + urllib.parse.quote(READ_KEY)
    with urllib.request.urlopen(url, timeout=20) as r:
        data = json.loads(r.read().decode())
    df = pd.DataFrame(data.get("feeds", []))
    if df.empty:
        return df
    df["time"] = pd.to_datetime(df["created_at"])
    for c in [c for c in df.columns if c.startswith("field")]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df


def write_command(value: int, expected_kw: float, shortfall_kw: float):
    """Writes AI diesel command (field1) + forecast metrics to a 2nd ThingSpeak command channel."""
    params = {
        "api_key": CMD_WRITE_KEY,
        "field1": int(value),
        "field2": round(float(expected_kw), 2),
        "field3": round(float(shortfall_kw), 2),
    }
    url = "https://api.thingspeak.com/update?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=20) as r:
        return int(r.read().decode().strip())


def to_power_units(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"time": df["time"]})
    for src, f in FIELD_MAP.items():
        raw_col = df[f] if f in df.columns else 0.0
        out[src + "_kw"] = (raw_col / PU_BASE[src]).clip(0, 1.2) * RATED_KW[src]
    out["renewable_kw"] = out["pv_kw"] + out["wind_kw"]
    return out


def decide(p: pd.DataFrame):
    ren = p["renewable_kw"].values
    trend = ren[-1] - ren[max(0, len(ren) - 4)]
    expected = max(0.0, ren[-1] + trend)
    shortfall = LOAD_KW - 0.95 * expected
    return int(shortfall > 2.0), expected, shortfall


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true", help="replay data from outputs/microgrid_data.csv")
    args = ap.parse_args()

    if args.mock:
        d = pd.read_csv(ROOT / "outputs" / "microgrid_data.csv", parse_dates=["time"])
        i0 = 96 * 5 + 64
        w = d.iloc[i0 : i0 + 8]
        p = pd.DataFrame({"time": w["time"], "pv_kw": w["pv_kw"], "wind_kw": w["wind_kw"]})
        p["renewable_kw"] = p["pv_kw"] + p["wind_kw"]
        print("MOCK MODE - last 8 samples (kW):")
        print(p.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    else:
        raw = read_feeds(20)
        if raw.empty:
            sys.exit("No feeds returned from ThingSpeak. Check THINGSPEAK_READ_KEY in dashboard/.env.")
        p = to_power_units(raw)
        print(f"LIVE THINGSPEAK CHANNEL {CHANNEL_ID} - last 8 calibrated samples (kW):")
        print(p.tail(8).to_string(index=False, float_format=lambda v: f"{v:.1f}"))

    cmd, expected, shortfall = decide(p)
    print(f"\nExpected renewable power soon: {expected:.1f} kW | Load: {LOAD_KW:.1f} kW | Shortfall: {shortfall:.1f} kW")
    print("AI DISPATCH DECISION:", "START DIESEL (1)" if cmd else "DIESEL NOT NEEDED (0)")

    if not args.mock and CMD_WRITE_KEY and "PASTE_YOUR" not in CMD_WRITE_KEY:
        entry_num = write_command(cmd, expected, shortfall)
        print(f"Command written to 2nd ThingSpeak channel -> Entry #{entry_num}")