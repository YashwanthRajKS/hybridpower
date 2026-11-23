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
