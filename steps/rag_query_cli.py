import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import json
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "steps"))

DOCS_CACHE = OUTPUTS / "rag_docs_cache.json"
EMB_CACHE = OUTPUTS / "rag_embeddings_cache.npz"


def load_or_build_cache():
    from step11_rag_upgrade import build_knowledge_base, MicrogridVectorStore

    if DOCS_CACHE.exists() and EMB_CACHE.exists():
        docs = json.loads(DOCS_CACHE.read_text(encoding="utf-8"))
        data = np.load(EMB_CACHE)
        return docs, data["embeddings"], str(data["backend"])

    docs = build_knowledge_base()
    store = MicrogridVectorStore(docs)
    embeddings = np.asarray(store.embeddings, dtype=np.float32)
    backend = getattr(store, "backend_mode", "dense-all-MiniLM-L6-v2")

    DOCS_CACHE.write_text(json.dumps(docs, ensure_ascii=False, default=str), encoding="utf-8")
    np.savez_compressed(EMB_CACHE, embeddings=embeddings, backend=np.array(backend))
    return docs, embeddings, backend


def query_rag(question: str, top_k: int = 4):
    docs, doc_embeddings, backend = load_or_build_cache()

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    q_emb = model.encode([question], normalize_embeddings=True, show_progress_bar=False)[0]

    norms = np.linalg.norm(doc_embeddings, axis=1, keepdims=True) + 1e-12
    normed_docs = doc_embeddings / norms
    scores = normed_docs @ q_emb

    q_lower = question.lower()
    boosted_scores = scores.copy()
    if any(w in q_lower for w in ["why", "reason", "cause", "rule", "minimum", "start", "stop"]):
        for i, d in enumerate(docs):
            if d.get("category") in ("policy", "system_rules", "dispatch_rules", "rules"):
                boosted_scores[i] += 0.12

    top_idx = np.argsort(boosted_scores)[::-1][:top_k]

    evidence_list = []
    top_snippets = []
    for rank, idx in enumerate(top_idx, start=1):
        doc = docs[int(idx)]
        score = float(scores[int(idx)])
        evidence_list.append({
            "rank": rank,
            "id": doc.get("id", f"doc_{idx}"),
            "category": doc.get("category", "general"),
            "score": round(score, 3),
            "text": doc["text"],
        })
        if rank <= 3:
            top_snippets.append(f"[{rank}] {doc['text']}")

    return {
        "question": question,
        "backend": f"{backend} (cached)",
        "answer": " ".join(top_snippets),
        "evidence": evidence_list,
    }


def main():
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        print(json.dumps({"error": "No question provided"}))
        return

    result = query_rag(question)
    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()