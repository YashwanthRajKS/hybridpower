# Hybrid microgrid: AI forecasting, SMOTE alert and RAG assistant (runs locally in VS Code)

## One-time setup (Windows)
1. Install Python 3.10 or newer from python.org (tick "Add Python to PATH").
2. Install VS Code and its "Python" extension. Open this folder (File > Open Folder).
3. Open a terminal in VS Code (Terminal > New Terminal) and run:
       python -m venv .venv
       .venv\Scripts\activate
       pip install -r requirements.txt
   (Optional: `pip install imbalanced-learn` to use the standard SMOTE library.)

## Run everything
    python run_all.py        (about 2 to 4 minutes)
    python chat.py           (ask the RAG assistant; type quit to stop)

## Questions to ask the RAG assistant
After running `python chat.py`, type a question at the `Your question>` prompt and press Enter. These are examples; use dates and times that appear in your generated data:

### Diesel events and comparisons
    Why did the diesel start on 16 February at 16:45 under the AI strategy?
    Why did the diesel start on 16 February at 17:15 under the No-AI strategy?
    Was any load unserved around 16:30 on 16 February with the AI strategy?
    When did the diesel stop on 16 February under the AI strategy?
    Compare the AI and No-AI strategies on 16 February.

### Daily results
    What was the diesel run time and unserved energy on 16 February for the AI strategy?
    How much diesel energy was used on 16 February by the No-AI strategy?
    How many times did the diesel start on 16 February?

### General project rules and assumptions
    How does the AI diesel dispatch rule work?
    What is the diesel generator start-up delay?
    What is the minimum diesel run time?
    What does unserved energy mean?
    Is the project data measured or synthetic?
    How does the renewable power forecast work?

The assistant prints retrieved evidence and a ready-made answer prompt in the terminal. By default, it does not generate the final answer itself: copy the prompt between the markers into ChatGPT, Claude, or Gemini. Type `quit` to exit. The RAG assistant answers from its text knowledge base; for charts such as `alert_plot.png`, open the image in an image-capable tool instead.

## What gets created in the outputs folder
| File | What it is |
|---|---|
| microgrid_data.csv, data_preview.png | 60 days of synthetic weather, load, PV and wind power |
| metrics.csv, forecast_plot.png | baseline versus AI forecast errors |
| alert_metrics.csv, alert_plot.png | SMOTE forecast-miss alert: precision, recall, F1 |
| dispatch_summary.csv, sensitivity.csv, dispatch_*.png, sensitivity.png | diesel strategies compared |
| dispatch_steps.csv | step-by-step record (feeds the RAG knowledge base) |

## Honest limits
* The data is synthetic (generated from physical models), not measured.
* SMOTE is applied to the forecast-miss CLASSIFIER only, never to the regression forecaster, and only to training data.
* The ThingSpeak bridge (steps/thingspeak_bridge.py) network functions are untested; its --mock mode is tested.
* The Simulink model is not yet driven by the AI decision (that is the planned Step 6).

# AI-Driven Hybrid PV–Wind–Diesel Microgrid Operator Dashboard

An end-to-end **Predictive AI Dispatch, Closed-Loop Simulink Validation, Retrieval-Augmented Generation (RAG) Assistant, and Cloud Telemetry Platform** for an off-grid Hybrid PV–Wind–Diesel Microgrid located in **Bengaluru, India**.

* **Live Cloud Dashboard:** [https://hybrid-microgrid-dashboard.onrender.com](https://hybrid-microgrid-dashboard.onrender.com)
* **GitHub Repository:** [YashwanthRajKS/hybridpower](https://github.com/YashwanthRajKS/hybridpower)
* **IoT Telemetry Feed:** ThingSpeak Channel `3570460`
* **Cloud Database:** MongoDB Atlas (`M0` Free Tier — `hybridpower-microgrid / Cluster0`)

---

## 1. Executive Summary & Validated KPIs

| Metric | Validated Result | Benchmark / Details |
| :--- | :--- | :--- |
| **Unserved Energy Reduction** | **61.5%** | `52.4 kWh` (Predictive AI) vs. `136.1 kWh` (Reactive No-AI) on 1,440h NASA POWER Bengaluru dataset |
| **Closed-Loop Simulink Lag** | **0.00 s** | Eliminates `0.49 s` reactive diesel startup deficit via 30-minute look-ahead pre-warming |
| **SMOTE Shortfall Alert Recall** | **0.667** | `PR-AUC 0.686` calibrated on a `2.0 kW` NASA POWER deficit threshold |
| **Semantic RAG Top-3 Recall** | **85.0%** | `148` indexed dispatch & rulebook chunks using `384-D` MiniLM embeddings |
| **Cloud Hosting & Database Cost** | **$0 / month** | Render Free Web Service (`Node.js`) + MongoDB Atlas `M0` Free Cluster |

---

## 2. End-to-End System Architecture

1. **Data Ingestion & Forecasting Layer (`NASA POWER + ML`)**
   * Processes **1,440 hours** (60 days) of real-world solar irradiance, wind speed, and load demand data for **Bengaluru, India**.
   * Uses **SMOTE-balanced classification** (`0.667` recall, `0.686` PR-AUC) and **30-minute look-ahead forecasting** to detect renewable shortfalls (`> 2.0 kW` deficit after a `5%` safety margin) before they occur.
2. **Closed-Loop Control & Simulink Simulation Layer**
   * Enforces physical generator constraints: **1-hour (4 × 15-min) minimum diesel runtime** to prevent cylinder glazing/short-cycling and **30-minute (2 × 15-min) warm-up startup lag**.
   * Eliminates the `0.49 s` reactive deficit down to **`0.00 s` closed-loop lag**.
3. **384-D Semantic RAG Operator Assistant (`Python + Node.js Cloud Fallback`)**
   * **Local Mode:** Queries `384-dimensional` MiniLM embeddings over `148` dispatch log and system rulebook chunks.
   * **Render Cloud Mode (`queryRagCloudFallback`):** Executes in-memory JavaScript retrieval directly over `outputs/rag_docs_cache.json` and the operational rulebook for zero-latency serverless operation without Python runtime overhead.
4. **IoT Telemetry & Cloud Operator Backend (`Node.js + Express + ThingSpeak + MongoDB Atlas`)**
   * **ThingSpeak Bridge:** Fetches real-time sensor/telemetry feeds from **ThingSpeak Channel `3570460`**.
   * **Persistent Audit Logging:** Saves every operator query, grounded RAG response, backend engine tag, and UTC timestamp to MongoDB Atlas (`test.operatorauditlogs`).

---

## 3. What Has Been Completed & Updated

### Core Engineering & AI Pipeline
- [x] **NASA POWER 1,440h Benchmark & Dispatch Engine:** Built and validated predictive vs. reactive dispatch logs (`61.5%` unserved energy reduction).
- [x] **Closed-Loop Simulink Validation:** Verified `0.00 s` deficit lag against `0.49 s` reactive baseline.
- [x] **SMOTE Shortfall Classifier:** Tuned imbalanced shortfall detection (`0.667` recall, `0.686` PR-AUC).
- [x] **384-D RAG Knowledge Base:** Chunked `148` dispatch events and generator rules into `outputs/rag_docs_cache.json`.

### Recent Cloud Migration & Production Updates
- [x] **Replaced Jenkins with Render Native Deployment:** Removed `Jenkinsfile` and added `render.yaml` (`cd dashboard && npm install` / `node dashboard/server.js`) for automatic GitHub `main` branch deployments at `$0/month`.
- [x] **Serverless Cloud RAG Fallback (`dashboard/server.js`):** Added `queryRagCloudFallback` so the **Ask Microgrid AI Operator Assistant** runs natively in Node.js on Render without spawning heavy Python ML processes.
- [x] **MongoDB Atlas Integration (`M0` Free Cluster):** Provisioned `Cluster0` under `hybridpower-microgrid`, configured database user credentials, whitelisted `0.0.0.0/0` network access, and wired `MONGODB_URI` into Render Environment Variables.
- [x] **End-to-End Live Verification:** Confirmed **MongoDB Atlas Connected** status badge on the live Render URL and verified real-time document writes into `Cluster0 > test > operatorauditlogs`.

---

## 4. What Still Needs to Be Done (Final Project Wrap-Up)

- [ ] **Commit & Push This `README.md`:** Save this documentation to `microgrid_ai/README.md` and push to `origin main`.
- [ ] **Verify Live ThingSpeak Sensor Feed Display:** Confirm live telemetry packets from ThingSpeak Channel `3570460` are actively populating in the dashboard telemetry table during demo hours.
- [ ] **Generate 10-Slide Defense Presentation (`PPTX`):** Build the final defense deck covering Problem Statement, Architecture, NASA POWER Results, Simulink Lag Comparison, RAG Assistant, and Live Cloud Demo.
- [ ] **Final Project Report & Viva Preparation:** Capture screenshots of the live Render dashboard, MongoDB Atlas `operatorauditlogs` documents, and Simulink waveforms for the final thesis/project report.

---

## 5. Repository Structure

```text
microgrid_ai/
├── dashboard/
│   ├── public/              # Frontend HTML/CSS/JS (KPI cards, 72h Chart.js, RAG Chat UI)
│   ├── server.js            # Express backend, ThingSpeak bridge, Cloud RAG fallback, Mongoose audit logger
│   ├── package.json         # Node.js dependencies (express, mongoose, dotenv, cors)
│   └── .env                 # Local environment secrets (ignored by Git)
├── outputs/
│   ├── rag_docs_cache.json  # Pre-indexed 148-chunk dispatch & rulebook knowledge base
│   └── ...                  # NASA POWER dispatch results & evaluation metrics
├── render.yaml              # Render Infrastructure-as-Code deployment config
└── README.md                # End-to-end project documentation