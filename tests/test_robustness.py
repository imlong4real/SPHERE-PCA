import numpy as np
import pandas as pd

from pca_sphere_projection.robustness import (
    compare_manual_vs_great_circle,
    pc_coordinate_quality_summary,
    root_sensitivity_analysis,
    rotation_robustness_analysis,
)


def _ring_data(n=80, label="A"):
    rng = np.random.default_rng(0)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = np.cos(theta) + 0.05 * rng.standard_normal(n)
    y = np.sin(theta) + 0.05 * rng.standard_normal(n)
    z = 0.05 * rng.standard_normal(n)
    return pd.DataFrame({"PC1": x, "PC2": y, "PC3": z, "celltype": label})


def test_root_sensitivity_returns_one_row_per_root():
    df = _ring_data()
    out = root_sensitivity_analysis(
        df, candidate_roots=["A"], root_column="celltype",
    )
    assert set(out.columns) == {"root", "metric", "n_cells"}
    assert len(out) == 1
    assert out["n_cells"].iloc[0] == len(df)


def test_root_sensitivity_marks_missing_roots_as_nan():
    df = _ring_data()
    out = root_sensitivity_analysis(
        df, candidate_roots=["A", "missing_root"], root_column="celltype",
    )
    miss = out[out["root"] == "missing_root"].iloc[0]
    assert miss["n_cells"] == 0
    assert np.isnan(miss["metric"])


def test_rotation_robustness_returns_distribution_keys():
    df = _ring_data()
    out = rotation_robustness_analysis(
        df, base_euler_angles=[0, 0, 0],
        perturbation_degrees=2.0, n_perturbations=5,
    )
    assert set(out) == {"base_metric", "perturbed_metrics", "mean", "std", "cv"}
    assert out["perturbed_metrics"].shape == (5,)
    assert np.isfinite(out["base_metric"])


def test_pc_coordinate_quality_summary_shape():
    df = _ring_data()
    out = pc_coordinate_quality_summary(df)
    expected = {
        "n_cells", "n_pcs", "pc_variances", "variance_ratio_top_to_bottom",
        "norm_min", "norm_median", "norm_max", "frac_zero_norm",
        "great_circle_r_squared", "great_circle_normal",
        "spherical_anisotropy_summary",
    }
    assert expected <= set(out)
    assert out["n_cells"] == len(df)


def test_compare_manual_vs_great_circle_returns_metrics():
    df = _ring_data()
    out = compare_manual_vs_great_circle(df, euler_angles=[0, 0, 0])
    assert set(out) >= {"manual_metric", "gc_metric", "gc_rotation",
                        "great_circle_r_squared"}
    assert out["gc_rotation"].shape == (3, 3)
    # Great-circle alignment of a noisy ring should give near-perfect r-squared.
    assert out["great_circle_r_squared"] > 0.95
