import json
import time 
from datetime import datetime, timezone
from pathlib import Path
from parse_pdf import parse_pdf
from build_index import build_databases

RESULTS_DIR = Path(__file__).resolve().parent / "evaluation" / "results"

def build_knowledge_base(force = False):
    RESULTS_DIR.mkdir(parents=True, exist_ok = True)

    print("[1] Starting PDF parsing...", flush=True)
    start = time.perf_counter()
    parsed_pdfs = parse_pdf()
    parsing_end = time.perf_counter()
    print("[2] PDF parsing finished.", flush=True)
    
    print("[3] Starting index building...", flush=True)
    indexed_pdfs = build_databases(force = force)
    indexing_end = time.perf_counter()
    print("[4] Index building finished.", flush=True)

    metrics = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "new_pdf_count": len(set(parsed_pdfs) & set(indexed_pdfs)),
        "indexed_pdf_count": len(indexed_pdfs),
        "parsing_chunking_seconds": round(parsing_end - start, 3),
        "embedding_indexing_seconds": round(indexing_end - parsing_end, 3),
        "total_build_seconds": round(indexing_end - start, 3)
    }

    result_path = RESULTS_DIR / "kb_build_metrics.jsonl"

    with open(result_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(metrics) + "\n")

    print("\nKnowledge Base Building Metrics")
    print(json.dumps(metrics, indent=2))

    return metrics


if __name__ == "__main__":
    build_knowledge_base(force = False)