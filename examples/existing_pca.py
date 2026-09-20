"""Tutorial 1: transform existing PC1-PC3 coordinates."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import sphere_pca


rng = np.random.default_rng(4)
progress = np.linspace(0.15, 2.5, 80)
branch = np.repeat([-1.0, 1.0], 40)
pcs = np.column_stack(
    (
        np.sin(progress),
        0.35 * branch * np.sin(progress) + rng.normal(0, 0.04, progress.size),
        np.cos(progress),
    )
)
root_mask = progress < 0.35

result = sphere_pca.transform(pcs, root_mask=root_mask)
print(result.coordinates[:5])  # columns: theta, phi, r (angles in radians)

sphere_pca.plot(result, color=branch, projection="equirectangular")
plt.tight_layout()
output = Path(__file__).with_name("existing_pca.png")
plt.savefig(output, dpi=160)
print(f"Saved {output}")
