import importlib
import sys
import types

import numpy as np
import pytest


@pytest.fixture()
def eval_mod(monkeypatch):
    """
    Ensure optional heavy dependency is stubbed before importing
    tests/run_reranking_eval.py helpers.
    """
    if "sentence_transformers" not in sys.modules:
        stub_module = types.ModuleType("sentence_transformers")

        class _DummyCrossEncoder:
            def __init__(self, *args, **kwargs):
                pass

            def predict(self, pairs, show_progress_bar=False):
                return np.zeros(len(pairs), dtype=float)

        stub_module.CrossEncoder = _DummyCrossEncoder
        monkeypatch.setitem(sys.modules, "sentence_transformers", stub_module)

    mod = importlib.import_module("tests.run_reranking_eval")
    return mod


def test_compute_subquery_coverage_empty_subqueries(eval_mod):
    assert eval_mod._compute_subquery_coverage([], ["any chunk"]) is None


def test_compute_subquery_coverage_partial(eval_mod):
    subqueries = [
        "strict 2PL deadlock behavior",
        "snapshot isolation write skew",
    ]
    selected_chunks = [
        "Strict 2PL can deadlock and needs detection.",
        "Unrelated text on buffer replacement.",
    ]
    score = eval_mod._compute_subquery_coverage(subqueries, selected_chunks)
    assert score == pytest.approx(0.5)


def test_compute_subquery_coverage_full(eval_mod):
    subqueries = [
        "strict 2PL deadlock behavior",
        "snapshot isolation write skew",
    ]
    selected_chunks = [
        "Strict 2PL can deadlock due to circular wait.",
        "Snapshot isolation may allow write skew anomalies.",
    ]
    score = eval_mod._compute_subquery_coverage(subqueries, selected_chunks)
    assert score == pytest.approx(1.0)


def test_compute_precision_recall_no_actual_and_no_ideal(eval_mod):
    precision, recall = eval_mod._compute_precision_recall([], [])
    assert precision is None
    assert recall is None


def test_compute_precision_recall_no_actual_with_ideal(eval_mod):
    precision, recall = eval_mod._compute_precision_recall([10, 20], [])
    assert precision == 0.0
    assert recall == 0.0


def test_compute_precision_recall_standard_case(eval_mod):
    precision, recall = eval_mod._compute_precision_recall([1, 2, 3], [3, 4, 1, 7])
    # Relaxed matching (+/- 1 chunk id):
    # actual 3 (exact), 4 (near 3), 1 (exact), 7 (miss)
    assert precision == pytest.approx(3 / 4)
    # all ideal chunks are covered under +/-1 (2 is near 1 or 3)
    assert recall == pytest.approx(1.0)


def test_compute_precision_recall_relaxed_window(eval_mod):
    precision, recall = eval_mod._compute_precision_recall([10, 20], [9, 21, 30])
    # 9 matches 10, 21 matches 20, 30 misses
    assert precision == pytest.approx(2 / 3)
    assert recall == pytest.approx(1.0)


def test_compute_concept_diversity_no_keywords(eval_mod):
    score = eval_mod._compute_concept_diversity([], ["some chunk text"])
    assert score is None


def test_compute_concept_diversity_case_insensitive(eval_mod):
    keywords = ["2PC", "Deadlock", "Recovery", "Phantoms"]
    selected_chunks = [
        "Two-phase commit (2pc) may block under coordinator failure.",
        "DeadLock handling and RECOVERY behavior are discussed.",
    ]
    score = eval_mod._compute_concept_diversity(keywords, selected_chunks)
    assert score == pytest.approx(3 / 4)
