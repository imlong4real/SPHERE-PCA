"""Preserved manuscript-analysis namespace.

The lightweight public API is :mod:`sphere_pca`. This historical namespace
retains its original numerical implementation and requires the ``legacy``
optional dependency set.
"""

from importlib import import_module


_LEGACY_IMPORTS = {
    "matplotlib.pyplot": "matplotlib",
    "pandas": "pandas",
    "plotly.graph_objects": "plotly",
    "scipy.stats": "scipy",
    "seaborn": "seaborn",
    "statsmodels.stats.multitest": "statsmodels",
}
_missing_legacy_dependencies = []
for _module, _distribution in _LEGACY_IMPORTS.items():
    try:
        import_module(_module)
    except ModuleNotFoundError:
        if _distribution not in _missing_legacy_dependencies:
            _missing_legacy_dependencies.append(_distribution)

if _missing_legacy_dependencies:
    _missing = ", ".join(sorted(_missing_legacy_dependencies))
    raise ImportError(
        "The preserved pca_sphere_projection namespace requires optional "
        f"legacy dependencies (missing: {_missing}). Install them with "
        "`python -m pip install 'sphere-pca[legacy]'`. The lightweight "
        "public API remains available as `import sphere_pca`."
    )

del _LEGACY_IMPORTS, _missing_legacy_dependencies, _module, _distribution

from .comparison import (
    conserved_stripe_test,
    procrustes_align_spheres,
    spherical_replicate_residual,
)
from .core import (
    align_to_north_pole,
    apply_euler_rotation,
    equirectangular_projection,
    find_and_visualize_correlated_genes,
    statistical_test_on_lineages,
    visualize_globe,
    visualize_lineage_correlation,
)
from .perturbation import (
    cluster_genes_by_perturbation_signature,
    compute_gene_perturbation_vectors,
    decompose_perturbation_vectors,
    identify_sensitive_cells,
    perturb_gene_expression,
    plot_perturbation_vector_field,
    project_expression_to_pca,
    rank_genes_by_perturbation_magnitude,
)
from .robustness import (
    compare_manual_vs_great_circle,
    pc_coordinate_quality_summary,
    root_sensitivity_analysis,
    rotation_robustness_analysis,
)
from .sphere_stats import (
    fit_great_circle,
    geodesic_gradient,
    horseshoe_null_test,
    spherical_anisotropy,
    spherical_kde,
    stripe_strength_score,
)
from .stripe import (
    detect_spherical_stripes,
    multi_stripe_strength_score,
    local_stripe_strength,
    plot_detected_stripes,
)
from .pc_robustness import (
    compute_scree_summary,
    estimate_intrinsic_dimension,
    compare_pc_count_spherical_geometry,
    compare_top3_vs_random3_pcs,
    compare_hvg_vs_all_gene_pc_geometry,
)
from .topology import (
    build_spherical_knn_graph,
    detect_branchpoints,
    linear_vs_branching_score,
)
from .io import (
    ExpressionData,
    load_expression_matrix,
    load_rds_expression,
    load_bz2_klein_dataset,
    load_celegan,
    load_uc_epi,
    load_hesc,
    load_brca_atlas,
    match_expression_to_pc_csv,
)
from .preprocessing import (
    PCAResult,
    normalize_log1p,
    select_hvgs,
    filter_genes,
    fit_pca_embedding,
    compare_hvg_vs_all_gene_pca,
)
from .entropy import (
    compute_transcriptional_entropy,
    compute_cytotrace_proxy,
    compute_scent_proxy,
    test_entropy_geodesic_gradient,
)
from .gene_geometry import (
    compute_gene_geodesic_gradients,
    compute_per_cell_gene_geodesic_gradient,
    decompose_gene_gradient_theta_phi,
    decompose_gene_gradient_relative_to_stripes,
    rank_stripe_boundary_genes,
    rank_stripe_boundary_genes_per_cell,
    rank_along_trajectory_genes,
    rank_along_trajectory_genes_per_cell,
    filter_housekeeping_and_low_specificity_genes,
    plot_gene_gradient_field,
    plot_gene_gradient_on_sphere,
    plot_gene_gradient_equirectangular,
)
from .known_regulators import (
    REGULATOR_SETS,
    get_regulator_set,
    annotate_overlap,
)

__all__ = [
    "align_to_north_pole",
    "apply_euler_rotation",
    "equirectangular_projection",
    "visualize_globe",
    "visualize_lineage_correlation",
    "statistical_test_on_lineages",
    "find_and_visualize_correlated_genes",
    "stripe_strength_score",
    "spherical_kde",
    "geodesic_gradient",
    "fit_great_circle",
    "spherical_anisotropy",
    "horseshoe_null_test",
    "detect_spherical_stripes",
    "multi_stripe_strength_score",
    "local_stripe_strength",
    "plot_detected_stripes",
    "compute_scree_summary",
    "estimate_intrinsic_dimension",
    "compare_pc_count_spherical_geometry",
    "compare_top3_vs_random3_pcs",
    "compare_hvg_vs_all_gene_pc_geometry",
    "build_spherical_knn_graph",
    "detect_branchpoints",
    "linear_vs_branching_score",
    "procrustes_align_spheres",
    "conserved_stripe_test",
    "spherical_replicate_residual",
    "project_expression_to_pca",
    "perturb_gene_expression",
    "compute_gene_perturbation_vectors",
    "rank_genes_by_perturbation_magnitude",
    "decompose_perturbation_vectors",
    "identify_sensitive_cells",
    "plot_perturbation_vector_field",
    "cluster_genes_by_perturbation_signature",
    "root_sensitivity_analysis",
    "rotation_robustness_analysis",
    "pc_coordinate_quality_summary",
    "compare_manual_vs_great_circle",
    # raw-expression validation
    "ExpressionData",
    "load_expression_matrix",
    "load_rds_expression",
    "load_bz2_klein_dataset",
    "load_celegan",
    "load_uc_epi",
    "load_hesc",
    "load_brca_atlas",
    "match_expression_to_pc_csv",
    "PCAResult",
    "normalize_log1p",
    "select_hvgs",
    "filter_genes",
    "fit_pca_embedding",
    "compare_hvg_vs_all_gene_pca",
    "compute_transcriptional_entropy",
    "compute_cytotrace_proxy",
    "compute_scent_proxy",
    "test_entropy_geodesic_gradient",
    "compute_gene_geodesic_gradients",
    "compute_per_cell_gene_geodesic_gradient",
    "decompose_gene_gradient_theta_phi",
    "decompose_gene_gradient_relative_to_stripes",
    "rank_stripe_boundary_genes",
    "rank_stripe_boundary_genes_per_cell",
    "rank_along_trajectory_genes",
    "rank_along_trajectory_genes_per_cell",
    "filter_housekeeping_and_low_specificity_genes",
    "plot_gene_gradient_field",
    "plot_gene_gradient_on_sphere",
    "plot_gene_gradient_equirectangular",
    "REGULATOR_SETS",
    "get_regulator_set",
    "annotate_overlap",
]
