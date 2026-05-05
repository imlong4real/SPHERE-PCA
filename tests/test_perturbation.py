import numpy as np
import pandas as pd
import pytest

from pca_sphere_projection.perturbation import (
    cluster_genes_by_perturbation_signature,
    compute_gene_perturbation_vectors,
    decompose_perturbation_vectors,
    identify_sensitive_cells,
    perturb_gene_expression,
    project_expression_to_pca,
    rank_genes_by_perturbation_magnitude,
)


@pytest.fixture
def small_data():
    rng = np.random.default_rng(42)
    n_cells, n_genes = 30, 5
    X = rng.normal(0.0, 1.0, size=(n_cells, n_genes))
    gene_names = [f"g{i}" for i in range(n_genes)]
    pca_components = np.eye(3, n_genes)  # picks the first 3 genes as PCs
    pca_mean = X.mean(axis=0)
    return X, gene_names, pca_components, pca_mean


def test_perturb_zero_changes_only_target(small_data):
    X, gene_names, _, _ = small_data
    df = pd.DataFrame(X, columns=gene_names)

    out = perturb_gene_expression(df, gene_names, "g2", mode="zero")

    assert (out["g2"] == 0).all()
    for g in gene_names:
        if g != "g2":
            assert np.allclose(out[g].values, df[g].values)


def test_perturb_double_changes_only_target(small_data):
    X, gene_names, _, _ = small_data

    out = perturb_gene_expression(X, gene_names, "g0", mode="double")

    assert np.allclose(out[:, 0], 2.0 * X[:, 0])
    assert np.allclose(out[:, 1:], X[:, 1:])


def test_perturb_does_not_mutate_input(small_data):
    X, gene_names, _, _ = small_data
    X_copy = X.copy()

    _ = perturb_gene_expression(X, gene_names, "g0", mode="zero")

    assert np.array_equal(X, X_copy)


def test_projection_dimensions(small_data):
    X, _, pca_components, pca_mean = small_data

    Q = project_expression_to_pca(X, pca_components, pca_mean)

    assert Q.shape == (X.shape[0], pca_components.shape[0])


def test_projection_matches_manual_formula(small_data):
    X, _, pca_components, pca_mean = small_data

    Q = project_expression_to_pca(X, pca_components, pca_mean)
    expected = (X - pca_mean) @ pca_components.T

    assert np.allclose(Q, expected)


def test_perturbation_vector_shapes(small_data):
    X, gene_names, pca_components, pca_mean = small_data

    res = compute_gene_perturbation_vectors(
        X, gene_names, ["g0", "g1"], pca_components, pca_mean
    )

    for g in ["g0", "g1"]:
        for key in (
            "original",
            "zeroed",
            "doubled",
            "delta_zero",
            "delta_double",
            "delta_double_minus_zero",
        ):
            assert res[g][key].shape == (X.shape[0], pca_components.shape[0])


def test_linear_projection_zero_double_average_equals_original(small_data):
    """
    With purely linear PCA projection, ``(zeroed + doubled) / 2`` reconstructs
    the original PC scores exactly. The "asymmetry" metric should therefore
    be ~ 0 in the linear regime.
    """
    X, gene_names, pca_components, pca_mean = small_data

    res = compute_gene_perturbation_vectors(
        X, gene_names, gene_names[:3], pca_components, pca_mean,
        normalize_to_sphere=False,
    )
    rank = rank_genes_by_perturbation_magnitude(res, metric="asymmetry")

    assert (rank["score"].abs() < 1e-9).all()


def test_ranking_largest_variance_gene_is_top_under_identity_loadings(small_data):
    X, gene_names, pca_components, pca_mean = small_data
    X = X.copy()
    X[:, 0] *= 10.0  # blow up g0 variance
    pca_mean = X.mean(axis=0)

    res = compute_gene_perturbation_vectors(
        X, gene_names, gene_names[:3], pca_components, pca_mean
    )
    rank = rank_genes_by_perturbation_magnitude(res, metric="symmetric_range")

    assert rank.iloc[0]["gene"] == "g0"


def test_unknown_gene_raises(small_data):
    X, gene_names, _, _ = small_data
    with pytest.raises(ValueError, match="not in"):
        perturb_gene_expression(X, gene_names, "fake_gene", mode="zero")


def test_unknown_mode_raises(small_data):
    X, gene_names, _, _ = small_data
    with pytest.raises(ValueError, match="unknown mode"):
        perturb_gene_expression(X, gene_names, "g0", mode="not_a_mode")


def test_compute_with_unknown_gene_raises(small_data):
    X, gene_names, pca_components, pca_mean = small_data
    with pytest.raises(ValueError, match="not in"):
        compute_gene_perturbation_vectors(
            X, gene_names, ["nope"], pca_components, pca_mean
        )


def test_decomposition_reconstructs_inputs():
    rng = np.random.default_rng(0)
    coords = rng.normal(size=(20, 3))
    coords = coords / np.linalg.norm(coords, axis=1, keepdims=True)
    vectors = rng.normal(size=(20, 3)) * 0.05

    out = decompose_perturbation_vectors(coords, vectors)

    rebuilt = out["radial"][:, None] * coords + out["tangent"]
    assert np.allclose(rebuilt, vectors, atol=1e-12)
    # tangent is perpendicular to radial
    assert np.allclose(np.einsum("ij,ij->i", out["tangent"], coords), 0.0, atol=1e-12)
    # theta + phi components account for full tangent magnitude
    assert np.allclose(
        out["theta_component"] ** 2 + out["phi_component"] ** 2,
        out["tangent_magnitude"] ** 2,
        atol=1e-12,
    )


def test_decomposition_pure_radial_has_zero_tangent():
    coords = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    vectors = 0.1 * coords
    out = decompose_perturbation_vectors(coords, vectors)
    assert np.allclose(out["tangent"], 0.0, atol=1e-12)
    assert np.allclose(out["radial"], [0.1, 0.1, 0.1])


def test_identify_sensitive_cells_returns_top_indices(small_data):
    X, gene_names, pca_components, pca_mean = small_data
    res = compute_gene_perturbation_vectors(
        X, gene_names, ["g0"], pca_components, pca_mean
    )
    top = identify_sensitive_cells(res, "g0", top_n=5)
    assert top.shape == (5,)
    # the magnitudes of the returned indices are non-increasing
    v = res["g0"]["doubled"] - res["g0"]["zeroed"]
    mags = np.linalg.norm(v, axis=1)[top]
    assert np.all(np.diff(mags) <= 1e-12)


def test_cluster_signature_self_overlap_equals_count(small_data):
    X, gene_names, pca_components, pca_mean = small_data
    res = compute_gene_perturbation_vectors(
        X, gene_names, gene_names[:3], pca_components, pca_mean
    )

    out = cluster_genes_by_perturbation_signature(
        res, threshold=0.0, mode="any_perturbed"
    )

    diag = np.diag(out["shared_counts"].values)
    expected = out["bitmap"].astype(int).sum(axis=1)
    assert np.array_equal(diag, expected)


def test_normalize_to_sphere_outputs_unit_vectors(small_data):
    X, gene_names, pca_components, pca_mean = small_data
    res = compute_gene_perturbation_vectors(
        X, gene_names, ["g0"], pca_components, pca_mean,
        normalize_to_sphere=True,
    )
    for key in ("original", "zeroed", "doubled"):
        norms = np.linalg.norm(res["g0"][key], axis=1)
        assert np.allclose(norms, 1.0, atol=1e-10)
