import numpy as np

from pca_sphere_projection.sphere_stats import (
    fit_great_circle,
    geodesic_gradient,
    horseshoe_null_test,
    spherical_anisotropy,
    spherical_kde,
    stripe_strength_score,
)


def _equator_points(n=60):
    theta = np.linspace(-np.pi, np.pi, n, endpoint=False)
    return np.column_stack([np.cos(theta), np.sin(theta), np.zeros(n)])


def test_fit_great_circle_recovers_equator():
    coords = _equator_points()
    result = fit_great_circle(coords)

    assert abs(abs(result["normal"][2]) - 1.0) < 1e-10
    assert result["mean_residual_radians"] < 1e-10
    assert result["r_squared"] > 0.99


def test_stripe_strength_is_high_for_longitudinally_concentrated_points():
    uniform = _equator_points(72)
    clustered = np.tile(np.array([[1.0, 0.0, 0.0]]), (72, 1))

    uniform_score = stripe_strength_score(uniform, n_bins=12)["score"]
    clustered_score = stripe_strength_score(clustered, n_bins=12)["score"]

    assert uniform_score < 0.05
    assert clustered_score > 0.95


def test_spherical_kde_returns_density_on_unit_eval_grid():
    coords = _equator_points(20)
    density, eval_points = spherical_kde(coords, kappa=5.0, m_grid=50)

    assert density.shape == (50,)
    assert eval_points.shape == (50, 3)
    assert np.all(density > 0)
    assert np.allclose(np.linalg.norm(eval_points, axis=1), 1.0)


def test_geodesic_gradient_is_tangent_to_sphere():
    coords = _equator_points(40)
    scalar = np.arctan2(coords[:, 1], coords[:, 0])
    grads = geodesic_gradient(coords, scalar, k=5)

    assert grads.shape == coords.shape
    assert np.allclose(np.einsum("ij,ij->i", grads, coords), 0.0, atol=1e-10)


def test_spherical_anisotropy_reports_ordered_eigenvalues():
    coords = _equator_points(60)
    result = spherical_anisotropy(coords)

    assert result["lambda1"] >= result["lambda2"] >= result["lambda3"]
    assert result["lambda3"] < 1e-10
    assert result["planar"] > 0.9


def test_horseshoe_null_test_returns_expected_keys():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(12, 5))
    result = horseshoe_null_test(X, n_perm=3, rng=rng)

    assert set(result) == {"observed", "null_distribution", "p_value", "effect_size_z"}
    assert result["null_distribution"].shape == (3,)
    assert 0.0 <= result["p_value"] <= 1.0
