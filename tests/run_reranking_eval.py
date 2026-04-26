#!/usr/bin/env python3
"""
Run reranking evaluation across three modes on a benchmark YAML file.

Metrics recorded per (query, mode):
  - SubqueryCoverage@K
  - RelevantChunkRecall@K
  - Precision@K
  - PageDiversity@K
  - ConceptDiversity@K (keyword coverage in selected chunks)
  - LatencyTotalMs
  - LatencyRerankMs
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import yaml

DEFAULT_MODES = ("none", "cross_encoder", "adaptive_multi_hop")


@dataclass
class EvalRow:
    benchmark_id: str
    mode: str
    question: str
    top_k_used: int
    selected_chunk_ids: List[int]
    selected_pages: List[int]
    relevant_chunk_recall_at_k: Optional[float]
    precision_at_k: Optional[float]
    subquery_coverage_at_k: Optional[float]
    page_diversity_at_k: int
    concept_diversity_at_k: Optional[float]
    latency_total_ms: float
    latency_rerank_ms: float
    answer_text: str


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _is_subquery_covered(subquery: str, chunk_text: str) -> bool:
    sq_tokens = set(_tokenize(subquery))
    if not sq_tokens:
        return False
    chunk_tokens = set(_tokenize(chunk_text))
    overlap = len(sq_tokens & chunk_tokens)
    # lightweight lexical support gate
    return overlap >= max(2, int(0.25 * len(sq_tokens)))


def _compute_subquery_coverage(subqueries: Sequence[str], selected_chunks: Sequence[str]) -> Optional[float]:
    if not subqueries:
        return None
    covered = 0
    for sq in subqueries:
        if any(_is_subquery_covered(sq, chunk) for chunk in selected_chunks):
            covered += 1
    return covered / len(subqueries)


def _compute_precision_recall(
    ideal_chunk_ids: Sequence[int],
    actual_chunk_ids: Sequence[int],
) -> Tuple[Optional[float], Optional[float]]:
    if not actual_chunk_ids:
        if ideal_chunk_ids:
            return 0.0, 0.0
        return None, None
    if not ideal_chunk_ids:
        return None, None

    ideal = set(int(x) for x in ideal_chunk_ids)
    actual = [int(x) for x in actual_chunk_ids]

    def is_relaxed_match(chunk_id: int) -> bool:
        return any(abs(chunk_id - ideal_id) <= 1 for ideal_id in ideal)

    matched_actual_count = sum(1 for x in actual if is_relaxed_match(x))
    precision = matched_actual_count / len(actual) if actual else 0.0

    covered_ideal: Set[int] = set()
    for ideal_id in ideal:
        if any(abs(actual_id - ideal_id) <= 1 for actual_id in actual):
            covered_ideal.add(ideal_id)
    recall = len(covered_ideal) / len(ideal) if ideal else None
    return precision, recall


def _compute_concept_diversity(keywords: Sequence[str], selected_chunks: Sequence[str]) -> Optional[float]:
    if not keywords:
        return None
    combined = "\n".join(selected_chunks).lower()
    matched = sum(1 for kw in keywords if kw.lower() in combined)
    return matched / len(keywords)


def _build_pipeline(cfg: RAGConfig, artifacts_dir: Path, index_prefix: str):
    from src.retriever import (
        BM25Retriever,
        FAISSRetriever,
        IndexKeywordRetriever,
        load_artifacts,
    )
    from src.ranking.ranker import EnsembleRanker

    faiss_index, bm25_index, chunks, sources, metadata = load_artifacts(
        artifacts_dir=artifacts_dir,
        index_prefix=index_prefix,
    )

    retrievers = [
        FAISSRetriever(faiss_index, cfg.embed_model),
        BM25Retriever(bm25_index),
    ]
    if cfg.ranker_weights.get("index_keywords", 0) > 0:
        retrievers.append(IndexKeywordRetriever(cfg.extracted_index_path, cfg.page_to_chunk_map_path))

    ranker = EnsembleRanker(
        ensemble_method=cfg.ensemble_method,
        weights=cfg.ranker_weights,
        rrf_k=int(cfg.rrf_k),
    )
    return retrievers, ranker, chunks, metadata, sources


def _evaluate_one(
    benchmark: Dict[str, Any],
    mode: str,
    cfg: RAGConfig,
    retrievers,
    ranker: EnsembleRanker,
    chunks: List[str],
    metadata: List[Dict[str, Any]],
    *,
    top_k: int,
    pool_k: int,
    skip_generation: bool,
) -> EvalRow:
    from src.ranking.reranker import rerank_indices
    from src.retriever import get_page_numbers

    question = benchmark["question"]

    t0 = time.perf_counter()

    # Candidate pool size for retrieval before reranking.
    pool_n = max(pool_k, top_k)
    raw_scores: Dict[str, Dict[int, float]] = {}
    for retriever in retrievers:
        raw_scores[retriever.name] = retriever.get_scores(question, pool_n, chunks)

    ordered, _scores = ranker.rank(raw_scores=raw_scores)
    topk_idxs = ordered[:pool_n]
    ranked_chunks = [chunks[i] for i in topk_idxs]

    page_nums = get_page_numbers(topk_idxs, metadata)
    local_pages = [
        (page_nums.get(idx, [1]) if isinstance(page_nums.get(idx, [1]), list) else [page_nums.get(idx, 1)])
        for idx in topk_idxs
    ]

    t_rerank_start = time.perf_counter()
    local_order = rerank_indices(
        question,
        ranked_chunks,
        mode=mode,
        top_n=top_k,
        chunk_pages=local_pages,
    )
    rerank_ms = (time.perf_counter() - t_rerank_start) * 1000.0

    if local_order:
        topk_idxs = [topk_idxs[i] for i in local_order if i < len(topk_idxs)]
        ranked_chunks = [ranked_chunks[i] for i in local_order if i < len(ranked_chunks)]
    topk_idxs = topk_idxs[:top_k]
    ranked_chunks = ranked_chunks[:top_k]

    if skip_generation:
        ans = ""
    else:
        from src.generator import answer
        stream_iter = answer(
            question,
            ranked_chunks,
            cfg.gen_model,
            max_tokens=cfg.max_gen_tokens,
            system_prompt_mode=cfg.system_prompt_mode,
        )
        ans = "".join(stream_iter).strip()

    total_ms = (time.perf_counter() - t0) * 1000.0

    selected_page_set = set()
    for idx in topk_idxs:
        pages = page_nums.get(idx, [1]) or [1]
        if isinstance(pages, int):
            pages = [pages]
        selected_page_set.update(int(p) for p in pages)

    precision, recall = _compute_precision_recall(
        benchmark.get("ideal_retrieved_chunks", []) or [],
        topk_idxs,
    )
    subquery_coverage = _compute_subquery_coverage(
        benchmark.get("subqueries", []) or [],
        ranked_chunks,
    )
    concept_diversity = _compute_concept_diversity(
        benchmark.get("keywords", []) or [],
        ranked_chunks,
    )

    return EvalRow(
        benchmark_id=benchmark.get("id", "unknown"),
        mode=mode,
        question=question,
        top_k_used=top_k,
        selected_chunk_ids=[int(x) for x in topk_idxs],
        selected_pages=sorted(selected_page_set),
        relevant_chunk_recall_at_k=recall,
        precision_at_k=precision,
        subquery_coverage_at_k=subquery_coverage,
        page_diversity_at_k=len(selected_page_set),
        concept_diversity_at_k=concept_diversity,
        latency_total_ms=total_ms,
        latency_rerank_ms=rerank_ms,
        answer_text=ans,
    )


def _summarize(rows: Sequence[EvalRow]) -> Dict[str, Any]:
    by_mode: Dict[str, List[EvalRow]] = {}
    for row in rows:
        by_mode.setdefault(row.mode, []).append(row)

    summary: Dict[str, Any] = {}
    for mode, mode_rows in by_mode.items():
        n = len(mode_rows) or 1
        def avg(field: str) -> Optional[float]:
            vals = [getattr(r, field) for r in mode_rows if getattr(r, field) is not None]
            return (sum(vals) / len(vals)) if vals else None

        summary[mode] = {
            "num_queries": len(mode_rows),
            "avg_relevant_chunk_recall_at_k": avg("relevant_chunk_recall_at_k"),
            "avg_precision_at_k": avg("precision_at_k"),
            "avg_subquery_coverage_at_k": avg("subquery_coverage_at_k"),
            "avg_page_diversity_at_k": avg("page_diversity_at_k"),
            "avg_concept_diversity_at_k": avg("concept_diversity_at_k"),
            "avg_latency_total_ms": sum(r.latency_total_ms for r in mode_rows) / n,
            "avg_latency_rerank_ms": sum(r.latency_rerank_ms for r in mode_rows) / n,
        }
    return summary


def main():
    from src.config import RAGConfig

    parser = argparse.ArgumentParser(description="Run reranking evaluation across modes.")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config yaml.")
    parser.add_argument("--benchmarks", default="tests/benchmarks_reranking.yaml", help="Path to benchmark yaml.")
    parser.add_argument("--index-prefix", default="textbook_index", help="Index prefix for artifacts.")
    parser.add_argument("--modes", nargs="+", default=list(DEFAULT_MODES), help="Rerank modes to evaluate.")
    parser.add_argument("--top-k", type=int, default=None, help="Override selected top-k for comparison.")
    parser.add_argument("--pool-k", type=int, default=None, help="Candidate pool size to retrieve before reranking.")
    parser.add_argument("--skip-generation", action="store_true", help="Skip answer generation for faster runs.")
    parser.add_argument("--out", default="tests/results/reranking_eval_results.json", help="Output JSON path.")
    args = parser.parse_args()

    cfg = RAGConfig.from_yaml(args.config)
    top_k = args.top_k if args.top_k is not None else int(cfg.top_k)
    pool_k = args.pool_k if args.pool_k is not None else int(cfg.num_candidates)
    pool_k = max(pool_k, top_k)

    artifacts_dir = cfg.get_artifacts_directory()
    index_prefix = args.index_prefix
    retrievers, ranker, chunks, metadata, _sources = _build_pipeline(cfg, artifacts_dir, index_prefix)

    with open(args.benchmarks, "r") as f:
        data = yaml.safe_load(f) or {}
    benchmarks = data.get("benchmarks", [])

    rows: List[EvalRow] = []
    for b in benchmarks:
        bid = b.get("id", "unknown")
        print(f"\n--- Benchmark: {bid} ---")
        for mode in args.modes:
            print(f"  Running mode={mode} ...")
            row = _evaluate_one(
                benchmark=b,
                mode=mode,
                cfg=cfg,
                retrievers=retrievers,
                ranker=ranker,
                chunks=chunks,
                metadata=metadata,
                top_k=top_k,
                pool_k=pool_k,
                skip_generation=args.skip_generation,
            )
            rows.append(row)
            print(
                f"    done | recall={row.relevant_chunk_recall_at_k} "
                f"precision={row.precision_at_k} subq_cov={row.subquery_coverage_at_k} "
                f"lat_total_ms={row.latency_total_ms:.1f} lat_rerank_ms={row.latency_rerank_ms:.1f}"
            )

    summary = _summarize(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "config_path": args.config,
            "benchmarks_path": args.benchmarks,
            "modes": args.modes,
            "top_k": top_k,
            "pool_k": pool_k,
            "skip_generation": args.skip_generation,
        },
        "summary_by_mode": summary,
        "rows": [asdict(r) for r in rows],
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print("\n=== Summary ===")
    for mode, s in summary.items():
        print(f"{mode}: {json.dumps(s, indent=2)}")
    print(f"\nSaved results to: {out_path}")


if __name__ == "__main__":
    main()
