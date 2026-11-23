"""Ask the RAG assistant questions (type 'quit' to stop). Run after run_all.py.
It shows the evidence it found and a ready-made prompt. Paste that prompt into ChatGPT / Claude / Gemini.
If you set the ANTHROPIC_API_KEY environment variable and installed 'anthropic', it asks the model itself."""
import os, runpy, io, contextlib

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.join(HERE, "outputs"))
with contextlib.redirect_stdout(io.StringIO()):
    ns = runpy.run_path(os.path.join(HERE, "steps", "step7_rag.py"), run_name="lib")
print(f"RAG assistant ready ({len(ns['docs'])} knowledge entries). Example: Why did the diesel start on 16 February at 16:45?")
while True:
    q = input("\nYour question> ").strip()
    if q.lower() in ("quit", "exit", "q"):
        break
    if q:
        ns["ask"](q)
