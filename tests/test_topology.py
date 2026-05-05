import numpy as np

from pca_sphere_projection.topology import (
    build_spherical_knn_graph,
    detect_branchpoints,
    linear_vs_branching_score,
)


def _arc_points(n=40):
    theta = np.linspace(-1.0, 1.0, n)
    coords = np.column_stack([np.cos(theta), np.sin(theta), np.zeros(n)])
    pseudotime = theta - theta.min()
    return coords, pseudotime


def test_build_spherical_knn_graph_has_k_edges_per_row():
    coords, _ = _arc_points()
    graph = build_spherical_knn_graph(coords, k=4)

    assert graph.shape == (40, 40)
    assert np.all(np.diff(graph.indptr) == 4)
    assert np.all(graph.data > 0)


def test_detect_branchpoints_returns_scores_and_gradients():
    coords, pseudotime = _arc_points()
    result = detect_branchpoints(coords, pseudotime, k=5)

    assert result["score"].shape == (40,)
    assert result["gradient"].shape == coords.shape
    assert result["p_value"] is None
    assert np.all(result["score"] >= 0.0)
    assert np.all(result["score"] <= 1.0)


def test_linear_vs_branching_score_returns_decision_fields():
    coords, pseudotime = _arc_points()
    result = linear_vs_branching_score(coords, pseudotime, k=5)

    assert result["decision"] in {"linear", "branching"}
    assert np.isfinite(result["linear_logL"])
    assert "bayes_factor" in result
