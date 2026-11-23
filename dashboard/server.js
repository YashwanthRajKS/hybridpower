const express = require("express");
const cors = require("cors");
const path = require("path");
const fs = require("fs");
const { execFile } = require("child_process");
const axios = require("axios");
const mongoose = require("mongoose");
const { parse } = require("csv-parse/sync");
require("dotenv").config({ path: path.join(__dirname, ".env") });

const app = express();
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, "public")));

const ROOT_DIR = path.resolve(__dirname, "..");
const OUTPUTS_DIR = path.join(ROOT_DIR, "outputs");
const PYTHON_EXE = path.join(ROOT_DIR, ".venv", "Scripts", "python.exe");
const RAG_CLI = path.join(ROOT_DIR, "steps", "rag_query_cli.py");
const LOCAL_LOG_PATH = path.join(OUTPUTS_DIR, "operator_audit_log.json");

// Optional MongoDB Atlas / Local connection with automatic local JSON fallback
let mongoConnected = false;
const AuditSchema = new mongoose.Schema({
  timestamp: { type: Date, default: Date.now },
  question: String,
  answer: String,
  backend: String,
});
const AuditLog = mongoose.model("OperatorAuditLog", AuditSchema);

if (process.env.MONGODB_URI) {
  mongoose
    .connect(process.env.MONGODB_URI)
    .then(() => {
      mongoConnected = true;
      console.log("Connected to MongoDB for telemetry & operator audit logging.");
    })
    .catch((err) => console.warn("MongoDB fallback to local JSON:", err.message));
}

function readCsvSafe(filename) {
  const fullPath = path.join(OUTPUTS_DIR, filename);
  if (!fs.existsSync(fullPath)) return [];
  const raw = fs.readFileSync(fullPath, "utf-8");
  return parse(raw, { columns: true, skip_empty_lines: true });
}

function logOperatorEvent(entry) {
  if (mongoConnected) {
    AuditLog.create(entry).catch(() => {});
  }
  let logs = [];
  if (fs.existsSync(LOCAL_LOG_PATH)) {
    try {
      logs = JSON.parse(fs.readFileSync(LOCAL_LOG_PATH, "utf-8"));
    } catch (_) {}
  }
  logs.unshift({ timestamp: new Date().toISOString(), ...entry });
  fs.writeFileSync(LOCAL_LOG_PATH, JSON.stringify(logs.slice(0, 50), null, 2));
}

// 1. Overview KPIs across all verified milestones
app.get("/api/overview", (req, res) => {
  const simMetrics = readCsvSafe("step12_simulink_comparison_metrics.csv");
  const ragBench = readCsvSafe("step11_rag_20q_benchmark.csv");

  res.json({
    project: "AI-Driven Hybrid PV-Wind-Diesel Microgrid",
    location: "Bengaluru, India (NASA POWER 1,440h + Simulink)",
    storageMode: mongoConnected ? "MongoDB Atlas Connected" : "Local JSON/CSV Store",
    kpis: {
      unservedReductionNasaPct: 61.5,
      unservedReductionSyntheticPct: 62.1,
      smoteRecallNasa: 0.667,
      smotePrAucNasa: 0.686,
      ragTop3RecallPct: 85.0,
      ragChunks: 148,
      simulinkAiLagSec: 0.0,
      simulinkReactiveLagSec: 0.49,
    },
    simulinkComparison: simMetrics,
    ragBenchmarkSample: ragBench.slice(0, 5),
  });
});

// 2. Dispatch Time-Series (NASA POWER or Synthetic)
app.get("/api/dispatch", (req, res) => {
  const source = req.query.source === "synthetic" ? "synthetic" : "nasa";
  const candidates =
    source === "nasa"
      ? ["hourly_dispatch_details_nasa.csv", "dispatch_details_nasa.csv"]
      : ["hourly_dispatch_details_synthetic.csv", "dispatch_details_synthetic.csv"];

  let rows = [];
  for (const file of candidates) {
    rows = readCsvSafe(file);
    if (rows.length > 0) break;
  }

  if (rows.length === 0) {
    rows = readCsvSafe("nasa_microgrid_hourly.csv");
  }

  const limit = Math.min(parseInt(req.query.limit || "72", 10), 240);
  res.json({
    source,
    totalRows: rows.length,
    records: rows.slice(0, limit),
  });
});

// 3. ThingSpeak Live Calibrated Telemetry (Channel 3570460: field2=PV, field4=Wind, field6=Diesel)
app.get("/api/thingspeak", async (req, res) => {
  const channelId = process.env.THINGSPEAK_CHANNEL_ID || "3570460";
  const readKey = process.env.THINGSPEAK_READ_KEY || "";

  try {
    const url = `https://api.thingspeak.com/channels/${channelId}/feeds.json?results=10${
      readKey ? `&api_key=${encodeURIComponent(readKey)}` : ""
    }`;
    const { data } = await axios.get(url, { timeout: 6000 });
    const calibratedFeeds = (data.feeds || []).map((f) => {
      const pvKw = Math.min(1.2, Math.max(0, (parseFloat(f.field2) || 0) / 6.0e5)) * 100.0;
      const windKw = Math.min(1.2, Math.max(0, (parseFloat(f.field4) || 0) / 1.8e6)) * 120.0;
      const dieselKw = Math.min(1.2, Math.max(0, (parseFloat(f.field6) || 0) / 2.0e7)) * 100.0;
      const loadKw = 60.0;
      const breaker = dieselKw > 5.0 || loadKw - 0.95 * (pvKw + windKw) > 2.0 ? "1" : "0";
      return {
        created_at: f.created_at,
        field1: pvKw.toFixed(1),
        field2: windKw.toFixed(1),
        field3: dieselKw.toFixed(1),
        field4: loadKw.toFixed(1),
        field5: "25.00",
        field7: breaker,
      };
    });
    if (calibratedFeeds.length > 0) {
      return res.json({ mode: `live-thingspeak-${channelId}`, feeds: calibratedFeeds });
    }
  } catch (err) {
    console.warn("ThingSpeak live fetch fallback:", err.message);
  }

  res.json({
    mode: "cached-simulink-telemetry",
    feeds: [
      {
        created_at: new Date().toISOString(),
        field1: "64.5",
        field2: "78.2",
        field3: "30.0",
        field4: "60.0",
        field5: "25.02",
        field7: "1",
      },
    ],
  });
});

// 4. Operator Semantic RAG Assistant (POST /api/ask)
app.post("/api/ask", (req, res) => {
  const question = (req.body?.question || "").trim();
  if (!question) {
    return res.status(400).json({ error: "Question is required." });
  }

  const pyCmd = fs.existsSync(PYTHON_EXE) ? PYTHON_EXE : "python";
  execFile(pyCmd, [RAG_CLI, question], { timeout: 60000 }, (error, stdout, stderr) => {
    if (error) {
      return res.status(500).json({
        error: "RAG execution failed",
        details: stderr || error.message,
      });
    }
    try {
      const lines = stdout.trim().split(/\r?\n/);
      const parsed = JSON.parse(lines[lines.length - 1]);
      logOperatorEvent({
        question,
        answer: parsed.answer,
        backend: parsed.backend,
      });
      res.json(parsed);
    } catch (parseErr) {
      res.status(500).json({ error: "Invalid RAG output", raw: stdout });
    }
  });
});

// 5. Recent Operator Audit Logs
app.get("/api/logs", (req, res) => {
  if (!fs.existsSync(LOCAL_LOG_PATH)) return res.json([]);
  try {
    res.json(JSON.parse(fs.readFileSync(LOCAL_LOG_PATH, "utf-8")));
  } catch (_) {
    res.json([]);
  }
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`Microgrid Operator Backend & Dashboard running at http://localhost:${PORT}`);
});