import numpy as np

from pca_sphere_projection.comparison import (
    conserved_stripe_test,
    procrustes_align_spheres,
    spherical_replicate_residual,
)


def _labelled_points():
    coords = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ])
    labels = np.array(["a", "b", "c", "d", "e", "f"])
    return coords, labels


def test_procrustes_align_spheres_identity_has_zero_residual():
    coords, labels = _labelled_points()
    result = procrustes_align_spheres(coords, coords, labels, labels)

    assert np.allclose(result["R"], np.eye(3))
    assert result["rmse"] < 1e-10
    assert set(result["anchor_residuals"]) == set(labels)


def test_spherical_replicate_residual_reports_per_label_rmse():
    coords, labels = _labelled_points()
    result = spherical_replicate_residual(coords, labels, coords, labels)

    assert result["global_rmse"] < 1e-10
    assert set(result["per_label_rmse"]) == set(labels)


def test_conserved_stripe_test_finds_matching_label_sets():
    coords, labels = _labelled_points()
    result = conserved_stripe_test(
        coords,
        labels,
        coords,
        labels,
        stripes=[(-10.0, 10.0), (80.0, 100.0)],
        n_perm=5,
        rng=np.random.default_rng(3),
    )

    assert result["jaccard"] == [1.0, 1.0]
    assert len(result["p_value"]) == 2
