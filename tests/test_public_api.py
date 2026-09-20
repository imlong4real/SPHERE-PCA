import sys

import numpy as np
import pytest

import sphere_pca
from sphere_pca import cli


@pytest.fixture
def pcs():
    return np.array(
        [
            [1.0, 0.2, 0.1],
            [0.9, 0.1, 0.3],
            [0.8, -0.1, 0.2],
            [-0.4, 1.2, 0.5],
            [-0.8, -0.4, 1.1],
            [0.2, -1.1, -0.7],
        ]
    )


def test_transform_preserves_radius_and_projects_to_unit_sphere(pcs):
    result = sphere_pca.transform(pcs, root_mask=np.arange(len(pcs)) < 3)

    assert np.allclose(result.r, np.linalg.norm(pcs, axis=1))
    assert np.allclose(np.linalg.norm(result.unit_vectors, axis=1), 1.0)
    assert np.allclose(np.linalg.norm(result.aligned_vectors, axis=1), 1.0)


def test_root_mean_direction_is_aligned_to_north(pcs):
    root_mask = np.arange(len(pcs)) < 3
    result = sphere_pca.transform(pcs, root_mask=root_mask)
    root_direction = result.aligned_vectors[root_mask].mean(axis=0)
    root_direction /= np.linalg.norm(root_direction)

    assert np.allclose(root_direction, [0.0, 0.0, 1.0], atol=1e-12)
    assert np.allclose(
        result.root_centroid,
        result.unit_vectors[root_mask].mean(axis=0),
    )


def test_fixed_root_centroid_matches_equivalent_root_mask(pcs):
    root_mask = np.arange(len(pcs)) < 3
    centroid = (pcs / np.linalg.norm(pcs, axis=1, keepdims=True))[root_mask].mean(axis=0)

    from_mask = sphere_pca.transform(pcs, root_mask=root_mask)
    from_centroid = sphere_pca.transform(pcs, root_centroid=centroid)

    assert np.allclose(from_centroid.coordinates, from_mask.coordinates)
    assert np.allclose(from_centroid.aligned_vectors, from_mask.aligned_vectors)
    assert np.allclose(from_centroid.rotation_matrix, from_mask.rotation_matrix)
    assert np.array_equal(from_centroid.root_centroid, centroid)
    assert np.allclose(
        from_centroid.root_direction,
        centroid / np.linalg.norm(centroid),
    )


def test_fixed_root_centroid_is_independent_of_label_filtering(pcs):
    centroid = np.array([0.2, -0.7, 0.4])
    first = sphere_pca.fit(
        pcs,
        root_centroid=centroid,
        labels=np.array(["root", "root", "other", "other", "other", "other"]),
    )
    second = sphere_pca.fit(
        pcs,
        root_centroid=centroid,
        labels=np.array(["other"] * len(pcs)),
    )

    assert np.array_equal(first.coordinates, second.coordinates)
    assert np.array_equal(first.rotation_matrix, second.rotation_matrix)


def test_root_centroid_scale_does_not_change_coordinates_or_radii(pcs):
    centroid = np.array([0.2, -0.7, 0.4])
    first = sphere_pca.transform(pcs, root_centroid=centroid)
    scaled = sphere_pca.transform(pcs, root_centroid=10.0 * centroid)

    assert np.allclose(first.coordinates, scaled.coordinates)
    assert np.array_equal(first.r, np.linalg.norm(pcs, axis=1))
    assert np.array_equal(scaled.r, first.r)
    assert np.array_equal(first.root_centroid, centroid)
    assert np.array_equal(scaled.root_centroid, 10.0 * centroid)


@pytest.mark.parametrize(
    ("centroid", "message"),
    [
        ([1.0, 2.0], "length 3"),
        ([[1.0, 2.0, 3.0]], "length 3"),
        ([np.nan, 0.0, 1.0], "NaN or infinite"),
        ([np.inf, 0.0, 1.0], "NaN or infinite"),
        ([0.0, 0.0, 0.0], "non-zero magnitude"),
    ],
)
def test_invalid_root_centroids_are_rejected(pcs, centroid, message):
    with pytest.raises(ValueError, match=message):
        sphere_pca.transform(pcs, root_centroid=centroid)


def test_root_centroid_is_mutually_exclusive_with_root_selection(pcs):
    root_mask = np.arange(len(pcs)) < 3
    centroid = np.array([0.2, -0.7, 0.4])

    with pytest.raises(ValueError, match="either root_mask or root_centroid"):
        sphere_pca.transform(
            pcs,
            root_mask=root_mask,
            root_centroid=centroid,
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        sphere_pca.fit(
            pcs,
            root="root",
            labels=np.where(root_mask, "root", "other"),
            root_centroid=centroid,
        )


def test_coordinate_ranges(pcs):
    result = sphere_pca.transform(pcs)

    assert np.all((0.0 <= result.theta) & (result.theta <= np.pi))
    assert np.all((-np.pi <= result.phi) & (result.phi <= np.pi))


def test_transform_is_deterministic(pcs):
    root_mask = np.arange(len(pcs)) < 3
    first = sphere_pca.transform(pcs, root_mask=root_mask)
    second = sphere_pca.transform(pcs, root_mask=root_mask)

    assert np.array_equal(first.theta, second.theta)
    assert np.array_equal(first.phi, second.phi)
    assert np.array_equal(first.r, second.r)
    assert np.array_equal(first.rotation_matrix, second.rotation_matrix)


@pytest.mark.parametrize(
    "bad_vector",
    ([0.0, 0.0, 0.0], [1e-14, 0.0, 0.0]),
)
def test_zero_and_near_zero_vectors_are_rejected(pcs, bad_vector):
    bad = pcs.copy()
    bad[2] = bad_vector

    with pytest.raises(ValueError, match="zero or near-zero"):
        sphere_pca.transform(bad)


def test_antipodal_root_alignment_is_well_defined():
    pcs = np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
    result = sphere_pca.transform(pcs, root_mask=np.array([True, False]))

    assert np.allclose(result.aligned_vectors[0], [0.0, 0.0, 1.0])
    assert np.isclose(np.linalg.det(result.rotation_matrix), 1.0)


def test_fit_matrix_matches_direct_transform():
    rng = np.random.default_rng(42)
    matrix = rng.normal(size=(24, 7))
    labels = np.array(["root"] * 5 + ["other"] * 19)

    fitted = sphere_pca.fit(matrix, root="root", labels=labels)
    direct = sphere_pca.transform(fitted.pcs, root_mask=labels == "root")

    assert fitted.pca_components.shape == (3, 7)
    assert np.allclose(fitted.theta, direct.theta)
    assert np.allclose(fitted.phi, direct.phi)
    assert np.allclose(fitted.r, direct.r)


def test_fit_is_deterministic():
    rng = np.random.default_rng(17)
    matrix = rng.normal(size=(30, 8))
    root_mask = np.arange(30) < 6

    first = sphere_pca.fit(matrix, root_mask=root_mask, random_state=23)
    second = sphere_pca.fit(matrix, root_mask=root_mask, random_state=23)

    assert np.array_equal(first.coordinates, second.coordinates)
    assert np.array_equal(first.pca_components, second.pca_components)


def test_anndata_wrapper_matches_direct_coordinates():
    anndata = pytest.importorskip("anndata")
    import pandas as pd

    rng = np.random.default_rng(7)
    pcs = rng.normal(size=(12, 3))
    root_mask = np.arange(12) < 4
    adata = anndata.AnnData(
        X=rng.normal(size=(12, 6)),
        obs=pd.DataFrame(
            {"cell_type": np.where(root_mask, "stem", "mature")},
            index=[f"cell_{i}" for i in range(12)],
        ),
    )
    adata.obsm["X_pca"] = pcs

    wrapped = sphere_pca.fit(
        adata,
        root="stem",
        root_key="cell_type",
        write_back=True,
    )
    direct = sphere_pca.transform(pcs, root_mask=root_mask)
    fixed = sphere_pca.fit(adata, root_centroid=direct.root_centroid)

    assert np.allclose(wrapped.coordinates, direct.coordinates)
    assert np.allclose(fixed.coordinates, direct.coordinates)
    assert np.array_equal(fixed.root_centroid, direct.root_centroid)
    assert np.allclose(adata.obsm["X_sphere_pca"], direct.aligned_vectors)
    for column in ("sphere_pca_theta", "sphere_pca_phi", "sphere_pca_r"):
        assert column in adata.obs


def test_root_mask_must_be_boolean(pcs):
    with pytest.raises(TypeError, match="boolean"):
        sphere_pca.transform(pcs, root_mask=np.arange(len(pcs)))


def test_dashboard_cli_reports_optional_install(monkeypatch):
    monkeypatch.setattr(cli, "_missing_app_dependencies", lambda: ["pandas"])

    with pytest.raises(SystemExit, match=r"sphere-pca\[app\]"):
        cli.main()


def test_dashboard_cli_help_does_not_require_optional_dependencies(
    monkeypatch, capsys
):
    monkeypatch.setattr(cli.sys, "argv", ["sphere-trace", "--help"])
    monkeypatch.setattr(
        cli,
        "_missing_app_dependencies",
        lambda: pytest.fail("dependency check should not run for --help"),
    )

    cli.main()

    assert "usage: sphere-trace" in capsys.readouterr().out


def test_dashboard_launcher_uses_current_python(monkeypatch):
    import subprocess

    from pca_sphere_projection import app

    called = {}
    monkeypatch.setattr(app, "_ensure_streamlit", lambda: None)
    monkeypatch.setattr(app.sys, "argv", ["sphere-trace", "--help"])
    monkeypatch.delenv("STREAMLIT_SERVER_RUN_ON_SAVE", raising=False)
    monkeypatch.delenv("STREAMLIT_RUNTIME", raising=False)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda args, check: called.update(args=args, check=check),
    )

    app.main()

    assert called["args"][:4] == [sys.executable, "-m", "streamlit", "run"]
    assert called["args"][-1] == "--help"
    assert called["check"] is False
