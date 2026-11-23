# STEP 7: RAG assistant for the hybrid microgrid
#   1) turn the simulation into a text event log (the knowledge base)
#   2) index it (TF-IDF vectors + date/time filter)
#   3) retrieve the evidence for a question
#   4) build a grounded prompt for any LLM (ChatGPT, Claude, Gemini, ...)
# Needs in the same folder: dispatch_steps.csv (from step4_5_dispatch.py) and microgrid_data.csv
# Use in Colab:   %run step7_rag.py      then      ask("Why did the diesel start on 16 February at 16:30?")

import os, re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

START_DELAY_MIN, MIN_RUN_H, SAFE_AI = 30, 1, 0.95      # must match step4_5_dispatch.py
RATED_KW, MIN_LOAD_PCT = 100, 30

# ======================= 1. BUILD THE KNOWLEDGE BASE =======================
steps = pd.read_csv("dispatch_steps.csv", parse_dates=["time"])
raw = pd.read_csv("microgrid_data.csv", parse_dates=["time"])[["time", "irradiance_wm2", "wind_ms"]]
steps = steps.merge(raw, on="time", how="left").reset_index(drop=True)

docs = []
def add(kind, ts, text):
    docs.append({"id": f"{kind[:3].upper()}-{len(docs):04d}", "kind": kind, "ts": ts, "text": text})

def when(t):
    return f"{t:%d %B %Y} at {t:%H:%M}"

# ---- policy and assumption documents (answer "how / why does it work" questions) ----
policy = [
    f"Diesel dispatch rule with AI: every 15 minutes the system looks at the forecast of PV plus wind power for {START_DELAY_MIN} minutes ahead. "
    f"If the load minus {SAFE_AI:.0%} of the forecast renewable power is more than 2 kW, the diesel generator is ordered to start.",
    f"Start-up delay: the diesel generator needs {START_DELAY_MIN} minutes between the start order and being online. During this time any shortfall is unserved load.",
    f"Minimum run time: once started, the diesel runs at least {MIN_RUN_H} hour to avoid switching on and off repeatedly. It stops when the forecast shows no shortfall.",
    f"Diesel size and loading: the diesel is rated {RATED_KW} kW and when running its output is at least {MIN_LOAD_PCT} percent of the rating.",
    "Reactive strategy (no AI): the diesel is ordered only after a shortfall has been observed, so the start-up delay always causes some unserved load.",
    "Unserved energy means load in kWh that neither PV, wind nor diesel could supply, mostly during the diesel start-up delay or when the forecast under-predicted a drop.",
    "Forecast model: PV and wind output one hour ahead are predicted from recent weather, recent power and the time of day. PV uses a random forest; wind uses persistence (same as now) because the model was not better for wind.",
    "Data: the weather, load and power series are synthetic, generated from physical models of PV and wind; they are not measured data.",
    "Fuel model: fuel in litres per hour equals 0.246 times the rated kW plus 0.08415 times the output kW, a typical generic diesel curve. CO2 is 2.68 kg per litre.",
]
for t in policy:
    add("policy", None, "Policy and assumption: " + t)

# ---- events from the AI strategy and from the reactive strategy ----
def log_strategy(prefix, label, on, kw, uns, forecast_note):
    on = steps[on].values
    prev = np.r_[0, on[:-1]]
    start_idx = None
    for i in range(len(steps)):
        t = steps.time[i]
        if on[i] == 1 and prev[i] == 0:
            start_idx = i
            fc = steps.ai_forecast_kw[i]
            if prefix == "ai":
                add("start", t, f"{label} - diesel START event: on {when(t)} the diesel generator came online. "
                    f"The start order was given {START_DELAY_MIN} minutes earlier, at {t - pd.Timedelta(minutes=START_DELAY_MIN):%H:%M}, "
                    f"because the forecast for {t:%H:%M} showed PV and wind of {fc:.0f} kW against a load of {steps.load_kw[i]:.0f} kW "
                    f"(forecast shortfall {max(0, steps.load_kw[i] - fc):.0f} kW). Actual conditions then: irradiance {steps.irradiance_wm2[i]:.0f} W/m2, "
                    f"wind speed {steps.wind_ms[i]:.1f} m/s, actual PV and wind {steps.renewable_kw[i]:.0f} kW.")
            else:
                add("start", t, f"{label} - diesel START event: on {when(t)} the diesel generator came online, ordered only after a shortfall was seen "
                    f"{START_DELAY_MIN} minutes earlier. Load {steps.load_kw[i]:.0f} kW, PV and wind {steps.renewable_kw[i]:.0f} kW, "
                    f"irradiance {steps.irradiance_wm2[i]:.0f} W/m2, wind speed {steps.wind_ms[i]:.1f} m/s.")
        if on[i] == 0 and prev[i] == 1 and start_idx is not None:
            hours = (i - start_idx) * 0.25
            kwh = kw_series[start_idx:i].sum() * 0.25
            add("stop", t, f"{label} - diesel STOP event: on {when(t)} the diesel stopped after running {hours:.2f} hours and delivering {kwh:.0f} kWh. "
                f"Load {steps.load_kw[i]:.0f} kW, PV and wind {steps.renewable_kw[i]:.0f} kW. {forecast_note}")
            start_idx = None
    # unserved periods (consecutive steps with unserved load)
    u = uns.values > 0.05
    i = 0
    while i < len(u):
        if u[i]:
            j = i
            while j + 1 < len(u) and u[j + 1]:
                j += 1
            t0, t1 = steps.time[i], steps.time[j] + pd.Timedelta(minutes=15)
            e = uns.values[i:j + 1].sum()
            peak = (uns.values[i:j + 1] / 0.25).max()
            diesel_on = on[i] == 1
            reason = ("the diesel was still starting up (start-up delay)" if not diesel_on else "the diesel was running but at its limit")
            if prefix == "ai" and not diesel_on and steps.load_kw[i] - SAFE_AI * steps.ai_forecast_kw[i] <= 2:
                reason += f", and the forecast ({steps.ai_forecast_kw[i]:.0f} kW) had not predicted the shortfall (actual PV and wind {steps.renewable_kw[i]:.0f} kW)"
            add("unserved", steps.time[i], f"{label} - UNSERVED LOAD event: {e:.1f} kWh of load was unserved on {t0:%d %B %Y} between {t0:%H:%M} and {t1:%H:%M}, "
                f"peak shortfall {peak:.0f} kW, because {reason}. Irradiance {steps.irradiance_wm2[i]:.0f} W/m2, wind {steps.wind_ms[i]:.1f} m/s.")
            i = j + 1
        else:
            i += 1

kw_series = steps.ai_diesel_kw.values
log_strategy("ai", "AI strategy", "ai_diesel_on", steps.ai_diesel_kw, steps.ai_unserved_kwh, "The forecast showed no shortfall.")
kw_series = steps.react_diesel_kw.values
log_strategy("re", "No-AI strategy", "react_diesel_on", steps.react_diesel_kw, steps.react_unserved_kwh, "The shortfall had ended.")

# ---- daily summaries ----
steps["date"] = steps.time.dt.date
for d, g in steps.groupby("date"):
    t = pd.Timestamp(d)
    mae = (g.ai_forecast_kw - g.renewable_kw).abs().mean()
    for prefix, label in [("ai", "AI strategy"), ("react", "No-AI strategy")]:
        on, kw, uns = g[f"{prefix}_diesel_on"], g[f"{prefix}_diesel_kw"], g[f"{prefix}_unserved_kwh"]
        starts = int(((on.values[1:] == 1) & (on.values[:-1] == 0)).sum())
        add("summary", t, f"Daily summary {label} for {t:%d %B %Y}: load energy {g.load_kw.sum() * 0.25:.0f} kWh, PV and wind energy {g.renewable_kw.sum() * 0.25:.0f} kWh, "
            f"diesel energy {kw.sum() * 0.25:.0f} kWh, diesel run time {on.sum() * 0.25:.1f} hours, diesel starts {starts}, unserved energy {uns.sum():.1f} kWh. "
            f"Average forecast error that day {mae:.1f} kW.")
print(f"Knowledge base built: {len(docs)} entries "
      f"({pd.Series([d['kind'] for d in docs]).value_counts().to_dict()})")

# ======================= 2. INDEX =======================
texts = [d["text"] for d in docs]
vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
M = vec.fit_transform(texts)

MONTHS = {m.lower(): i for i, m in enumerate(["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"], 1)}

def parse_when(q):
    """find a date (and optional time) in the question"""
    ql = q.lower()
    date = None
    m = re.search(r"(\d{1,2})(?:st|nd|rd|th)?\s+(" + "|".join(MONTHS) + r")", ql) or None
    if m:
        date = pd.Timestamp(2026, MONTHS[m.group(2)], int(m.group(1)))
    else:
        m = re.search(r"(" + "|".join(MONTHS) + r")\s+(\d{1,2})", ql)
        if m:
            date = pd.Timestamp(2026, MONTHS[m.group(1)], int(m.group(2)))
        else:
            m = re.search(r"(2026)-(\d{2})-(\d{2})", ql)
            if m:
                date = pd.Timestamp(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    tm = re.search(r"\b(\d{1,2}):(\d{2})\b", q)
    if date is not None and tm:
        date = date + pd.Timedelta(hours=int(tm.group(1)), minutes=int(tm.group(2)))
        return date, True
    return date, False

# ======================= 3. RETRIEVE =======================
def retrieve(q, k=5):
    sim = cosine_similarity(vec.transform([q]), M).ravel()
    target, has_time = parse_when(q)
    if target is not None:
        for i, d in enumerate(docs):
            if d["ts"] is None:
                continue
            if d["ts"].date() == target.date():
                sim[i] += 0.30                                   # right day
                if has_time:
                    gap = abs((d["ts"] - target).total_seconds()) / 60
                    if gap <= 75:
                        sim[i] += 0.5 * (1 - gap / 75)           # close in time
            elif d["kind"] != "policy":
                sim[i] *= 0.3                                    # wrong day: push down
    # question type -> prefer matching kind of entry
    ql = q.lower()
    kinds = []
    if re.search(r"\bstart|started|came online|turn(ed)? on|switch(ed)? on", ql): kinds.append("start")
    if re.search(r"\bstop|stopped|turn(ed)? off|switch(ed)? off", ql): kinds.append("stop")
    if re.search(r"unserved|blackout|shortfall|not supplied|load shed", ql): kinds.append("unserved")
    if re.search(r"total|summary|daily|run time|run hours|how much|how many|per day|average", ql): kinds.append("summary")
    if re.search(r"\brule|policy|assumption|how does|how is|what is|data|synthetic|fuel model|forecast model", ql): kinds.append("policy")
    want_noai = bool(re.search(r"\bno[- ]ai\b|without ai|reactive|no forecast", ql))
    want_ai = (not want_noai) and bool(re.search(r"\bai\b|forecast", ql))
    for i, d in enumerate(docs):
        if d["kind"] in kinds:
            sim[i] += 0.5 if d["kind"] == "summary" else 0.35
        is_noai = "No-AI strategy" in d["text"][:60]
        is_ai = (not is_noai) and "AI strategy" in d["text"][:60]
        if want_noai and is_noai: sim[i] += 0.25
        if want_ai and is_ai: sim[i] += 0.25
    order = np.argsort(-sim)[:k]
    return [(docs[i], float(sim[i])) for i in order]

# ======================= 4. ASK =======================
def build_prompt(q, hits):
    ev = "\n".join(f"[{n}] ({d['id']}) {d['text']}" for n, (d, s) in enumerate(hits, 1))
    return ("You are the operator assistant of a hybrid PV-wind-diesel microgrid. Answer ONLY from the evidence below. "
            "If the evidence does not contain the answer, say you do not know. Quote times and numbers exactly, "
            "mention the evidence numbers like [1], and keep the answer under 120 words.\n\n"
            f"EVIDENCE:\n{ev}\n\nQUESTION: {q}\nANSWER:")

def call_llm(prompt):
    """Optional: only used if you set ANTHROPIC_API_KEY. Untested here; otherwise copy the prompt into any chat assistant."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    try:
        import anthropic
        msg = anthropic.Anthropic(api_key=key).messages.create(
            model="claude-sonnet-5-5", max_tokens=400, messages=[{"role": "user", "content": prompt}])
        return msg.content[0].text
    except Exception as e:
        return f"(LLM call failed: {e})"

def ask(q, k=5, show_prompt=True):
    hits = retrieve(q, k)
    print("QUESTION:", q)
    print("\nEVIDENCE FOUND:")
    for n, (d, s) in enumerate(hits, 1):
        print(f" [{n}] {d['id']}  score {s:.2f}\n     {d['text'][:230]}{'...' if len(d['text']) > 230 else ''}")
    prompt = build_prompt(q, hits)
    ans = call_llm(prompt)
    if ans:
        print("\nANSWER (LLM):\n" + ans)
    elif show_prompt:
        print("\n--- No API key set. Copy everything between the lines into ChatGPT / Claude / Gemini to get the answer ---")
        print(prompt)
        print("--- end of prompt ---")
    return hits

# ======================= 5. RETRIEVAL TEST (a number for your report) =======================
def retrieval_test(seed=1, k=3):
    rng = np.random.default_rng(seed)
    tests = []
    starts = [d for d in docs if d["kind"] == "start" and d["text"].startswith("AI strategy")]
    for d in rng.permutation(starts)[:10]:
        t = d["ts"]
        tests.append((f"Why did the diesel start on {t:%d %B} at {t:%H:%M} under the AI strategy?", d["id"]))
    uns = [d for d in docs if d["kind"] == "unserved" and d["text"].startswith("AI strategy")]
    for d in rng.permutation(uns)[:5]:
        t = d["ts"]
        tests.append((f"Was any load unserved around {t:%H:%M} on {t:%d %B} with the AI strategy?", d["id"]))
    sums = [d for d in docs if d["kind"] == "summary" and "AI strategy" in d["text"]]
    for d in rng.permutation(sums)[:5]:
        t = d["ts"]
        tests.append((f"What was the diesel run time and unserved energy on {t:%d %B} for the AI strategy?", d["id"]))
    hit = {1: 0, 3: 0}
    for q, exp in tests:
        ids = [d["id"] for d, s in retrieve(q, k)]
        hit[1] += exp == ids[0]
        hit[3] += exp in ids
    print(f"\nRetrieval test on {len(tests)} automatic questions: right entry ranked first in "
          f"{100 * hit[1] / len(tests):.0f}% and within the top 3 in {100 * hit[3] / len(tests):.0f}% of cases.")

if __name__ == "__main__" or "get_ipython" in globals():
    retrieval_test()
    example = [d for d in docs if d["kind"] == "start" and d["text"].startswith("AI strategy")][2]
    t = example["ts"]
    print("\n" + "=" * 70 + "\nEXAMPLE\n" + "=" * 70)
    ask(f"Why did the diesel start on {t:%d %B} at {t:%H:%M}?")
