from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pca_sphere_projection.app_processing import default_root_cluster, run_projection_pipeline, validate_input_dataframe
from pca_sphere_projection.app_report import save_html_report


def main() -> None:
    repo = REPO_ROOT
    input_path = repo / "examples" / "celegan_pca.csv"
    output_path = repo / "outputs" / "celegan_pca_interactive_report.html"

    df = pd.read_csv(input_path)
    df = validate_input_dataframe(df)
    root = default_root_cluster(df)
    result = run_projection_pipeline(
        df,
        root_cluster=root,
        compute_topology=True,
        max_topology_cells=2500,
    )
    save_html_report(result, output_path, "celegan_pca.csv")

    print(f"Smoke test passed for {input_path}")
    print(f"Rows processed: {len(result.data)}")
    print(f"Root cluster: {result.root_cluster}")
    print(f"Stripe strength: {result.metrics['stripe_strength']:.4f}")
    print(f"Great-circle residual (deg): {result.metrics['great_circle_mean_residual_deg']:.4f}")
    print(f"Report written to {output_path}")


if __name__ == "__main__":
    main()
