import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
import plotly.graph_objects as go
from scipy.stats import pearsonr, shapiro, ttest_ind, ks_2samp, spearmanr
from statsmodels.stats.multitest import multipletests

def align_to_north_pole(data, pcs_columns=['PC1', 'PC2', 'PC3'], cluster_column='cluster', root_node=None):
    """
    Aligns the first three PCs of a dataset to the north pole based on a selected root node (centroid of biological prior).
    
    Args:
        data (pd.DataFrame): A DataFrame containing PCA-reduced data.
        pcs_columns (list of str): Column names for the first three principal components. Default is ['PC1', 'PC2', 'PC3'].
        cluster_column (str): The name of the column containing cluster annotations.
        root_node (str or list): The cluster ID or centroid coordinates to align to the north pole. If a cluster ID is provided,
                                 the centroid of the cluster will be used. If a list of coordinates is provided, these will
                                 be used directly.
    
    Returns:
        pd.DataFrame: A DataFrame with the rotated PC coordinates projected onto the unit sphere.
    """
    if root_node is None:
        raise ValueError("A root node (cluster ID or centroid coordinates) must be provided.")
    
    # 
    points = data[pcs_columns].values
    
    # 
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    # 
    if isinstance(root_node, str):
        if cluster_column not in data.columns:
            raise ValueError(f"Cluster column '{cluster_column}' not found in the DataFrame.")
        
        if root_node not in data[cluster_column].unique():
            raise ValueError(f"Cluster '{root_node}' not found in the '{cluster_column}' column.")
        
        centroid = data[data[cluster_column] == root_node][pcs_columns].mean().values
    elif isinstance(root_node, (list, np.ndarray)):
        centroid = np.array(root_node)
    else:
        raise TypeError("The root node must be a cluster ID (str) or a list/array of centroid coordinates.")
    
    # 
    norm_centroid = np.linalg.norm(centroid)
    if norm_centroid == 0:
        raise ValueError("Centroid has zero magnitude, cannot perform rotation.")
    
    north_pole = np.array([0, 0, 1])
    rotation_axis = np.cross(centroid, north_pole)
    rotation_angle = np.arccos(np.dot(centroid, north_pole) / norm_centroid)
    
    # 
    if np.isclose(rotation_angle, 0):
        print("The centroid is already aligned with the north pole. No rotation performed.")
        return data[pcs_columns]
    
    # Rodrigues' rotation formula
    K = np.array([
        [0, -rotation_axis[2], rotation_axis[1]],
        [rotation_axis[2], 0, -rotation_axis[0]],
        [-rotation_axis[1], rotation_axis[0], 0]
    ])
    I = np.identity(3)
    R = I + np.sin(rotation_angle) * K + (1 - np.cos(rotation_angle)) * np.dot(K, K)
    
    # 
    rotated_points = np.dot(points, R.T)
    
    # 
    rotated_data = data.copy()
    for i, col in enumerate(pcs_columns):
        rotated_data[col] = rotated_points[:, i]
    
    return rotated_data

def apply_euler_rotation(data, pcs_columns=['PC1', 'PC2', 'PC3'], rotation_angles=[-50, 0, -20], degrees=True):
    """
    Applies Euler rotation to fine-tune data alignment for visualization, ensuring stripes align with longitudinal lines.
    
    Args:
        data (pd.DataFrame): A DataFrame containing the rotated PCA-transformed data.
        pcs_columns (list of str): Column names for the first three principal components to be rotated.
        rotation_angles (list or array): Angles for the Euler rotation (in degrees or radians).
        degrees (bool): If True, the rotation angles are in degrees. Otherwise, they are in radians.
    
    Returns:
        pd.DataFrame: A DataFrame with fine-tuned PC coordinates normalized onto the unit sphere. 
                      The output columns are prefixed with 'rotated_'.
    """
    # 
    points = data[pcs_columns].values

    # 
    euler_rotation = R.from_euler('xyz', rotation_angles, degrees=degrees)
    rotated_points = euler_rotation.apply(points)

    # 
    norms = np.linalg.norm(rotated_points, axis=1, keepdims=True)
    normalized_points = rotated_points / norms

    # 
    fine_tuned_data = data.copy()
    for i, col in enumerate(pcs_columns):
        fine_tuned_data[f'rotated_{col}'] = normalized_points[:, i]

    return fine_tuned_data

def equirectangular_projection(data, 
                                pcs_columns=['rotated_PC1', 'rotated_PC2', 'rotated_PC3'], 
                                cluster_column='cluster', 
                                categories=None, 
                                selected_celltypes=None, 
                                colormap='Blues', 
                                figsize=(12, 6), 
                                point_size=5, 
                                alpha=0.7, 
                                xlim=(-180, 180), 
                                ylim=(-70, 70)):
    """
    Creates an equirectangular projection to visualize 3D globe in 2D.

    Args:
        data (pd.DataFrame): A DataFrame containing the rotated PCA data with columns for 3D coordinates.
        pcs_columns (list of str): Column names for the rotated PC coordinates. Default is ['rotated_PC1', 'rotated_PC2', 'rotated_PC3'].
        cluster_column (str): Column name for the cell type or cluster annotations.
        categories (list): List of categories to create a custom color ramp (if None, unique clusters will be used).
        selected_celltypes (list): List of specific cell types or clusters to visualize (if None, visualizes all).
        colormap (str): The base colormap for the visualization. Default is 'Blues'.
        figsize (tuple): Figure size. Default is (12, 6).
        point_size (int): Size of the points in the scatterplot. Default is 5.
        alpha (float): Transparency of the points. Default is 0.7.
        xlim (tuple): Limits for the x-axis (longitude). Default is (-180, 180).
        ylim (tuple): Limits for the y-axis (latitude). Default is (-70, 70).

    Returns:
        None: Displays a 2D scatter plot.
    """
    # 
    if selected_celltypes is not None:
        data = data[data[cluster_column].isin(selected_celltypes)]
    
    # 
    def cartesian_to_spherical(x, y, z):
        longitude = np.arctan2(y, x)
        latitude = np.arcsin(z / np.sqrt(x**2 + y**2 + z**2))
        return longitude, latitude

    longitudes, latitudes = cartesian_to_spherical(data[pcs_columns[0]], data[pcs_columns[1]], data[pcs_columns[2]])
    longitudes = np.degrees(longitudes)
    latitudes = np.degrees(latitudes)
    
    # 
    data['Longitude'] = longitudes
    data['Latitude'] = latitudes
    
    # 
    if categories is None:
        categories = sorted(data[cluster_column].unique())
    color_ramp = plt.cm.get_cmap(colormap)
    custom_palette = color_ramp(np.linspace(0, 1, len(categories)))
    custom_cmap = ListedColormap(custom_palette)
    
    # 
    category_codes = data[cluster_column].astype('category').cat.codes
    
    # 
    fig, ax = plt.subplots(figsize=figsize)
    scatter = ax.scatter(data['Longitude'], data['Latitude'], 
                         c=category_codes, cmap=custom_cmap, 
                         s=point_size, alpha=alpha)
    
    # 
    ax.set_xlabel('Longitude (Degrees)')
    ax.set_ylabel('Latitude (Degrees)')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid(False)
    ax.set_title("Equirectangular Projection of Cell Types")
    
    # 
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_ticks(np.linspace(0, len(categories)-1, len(categories)))
    cbar.set_ticklabels(categories)
    cbar.set_label('Cell Types')
    
    plt.show()

def visualize_globe(data, 
                    pcs_columns=['rotated_PC1', 'rotated_PC2', 'rotated_PC3'], 
                    cluster_column='cluster', 
                    colormap='viridis', 
                    point_size=5, 
                    interactive=False, 
                    output_html=None, 
                    figsize=(6, 6), 
                    legend_columns=3):
    """
    Visualizes cells on a globe, either as a static 3D plot or an interactive globe. Allows saving the interactive globe as HTML.

    Args:
        data (pd.DataFrame): A DataFrame containing 3D spherical data with cluster annotations.
        pcs_columns (list of str): Column names for the rotated PC coordinates. Default is ['rotated_PC1', 'rotated_PC2', 'rotated_PC3'].
        cluster_column (str): Column name for the cell type or cluster annotations.
        colormap (str): Colormap to use for visualizing clusters. Default is 'viridis'.
        point_size (int): Size of the points in the scatterplot. Default is 5.
        interactive (bool): If True, creates an interactive Plotly globe. If False, creates a static Matplotlib plot.
        output_html (str): Filepath to save the interactive globe as an HTML file (only applicable if `interactive=True`).
        figsize (tuple): Figure size for the static Matplotlib plot. Default is (6, 6).
        legend_columns (int): Number of columns for the legend in the static plot when too many cell types exist. Default is 3.

    Returns:
        None: Displays the visualization or saves it as an HTML file (interactive=True) if specified.
    """
    # Extract data for plotting
    x = data[pcs_columns[0]].values
    y = data[pcs_columns[1]].values
    z = data[pcs_columns[2]].values
    clusters = data[cluster_column].astype('category')
    unique_celltypes = clusters.cat.categories
    celltypes = clusters.cat.codes

    # Static visualization
    if not interactive:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(x, y, z, c=celltypes, cmap=colormap, s=point_size)

        #
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        #
        colormap_obj = plt.cm.get_cmap(colormap)
        legend_handles = [
            Line2D([0], [0], marker='o', color=colormap_obj(i / len(unique_celltypes)), 
                   label=unique_celltypes[i], linestyle='', markersize=5) 
            for i in range(len(unique_celltypes))
        ]
        ax.legend(handles=legend_handles, title="Cell Types", loc="center left", 
                  bbox_to_anchor=(1.05, 0.5), ncol=legend_columns, fontsize='small', 
                  title_fontsize='small', framealpha=0.3)

        # 
        plt.tight_layout()
        plt.show()

    # Interactive visualization
    else:
        fig = go.Figure()

        # Add scatter points
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(
                size=point_size,
                color=celltypes,
                colorscale=colormap,
                opacity=0.7,
            ),
            text=clusters,  
            hoverinfo='text'
        ))

        # 
        fig.update_layout(
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                xaxis=dict(showgrid=False, zeroline=False),
                yaxis=dict(showgrid=False, zeroline=False),
                zaxis=dict(showgrid=False, zeroline=False)
            ),
            margin=dict(r=0, l=0, b=0, t=0),
            showlegend=False,
            title="Interactive 3D Globe"
        )

        # Save as HTML if specified
        if output_html:
            fig.write_html(output_html)
            print(f"Interactive globe saved to {output_html}")

        #
        fig.show()

def calculate_spherical_angles(x, y, z):
    """
    Calculates spherical coordinates (theta and phi) from Cartesian coordinates.
    Args:
        x, y, z (float): Cartesian coordinates.
    Returns:
        theta, phi (float): Spherical coordinates.
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)  # angle from the z-axis
    phi = np.arctan2(y, x)    # angle in the xy-plane from the x-axis
    return theta, phi

def visualize_lineage_correlation(data, 
                                  cell_types_lineage1, 
                                  cell_types_lineage2=None, 
                                  pcs_columns=['rotated_PC1', 'rotated_PC2', 'rotated_PC3'], 
                                  cluster_column='celltype', 
                                  colormap='viridis', 
                                  figsize=(14, 7), 
                                  annotate=True):
    """
    Visualizes the Pearson correlation of cell types in one or two lineages based on pseudotime inference.

    Args:
        data (pd.DataFrame): Input data containing cell types and rotated PCA coordinates.
        cell_types_lineage1 (list): List of cell types for the first lineage.
        cell_types_lineage2 (list, optional): List of cell types for the second lineage. Default is None.
        pcs_columns (list): Column names for rotated PCA coordinates. Default is ['rotated_PC1', 'rotated_PC2', 'rotated_PC3'].
        cluster_column (str): Column name for the cell type annotations. Default is 'celltype'.
        colormap (str): Colormap for visualizing cell types. Default is 'viridis'.
        figsize (tuple): Size of the plot. Default is (14, 7).
        annotate (bool): Whether to annotate the plot with Pearson correlation coefficients. Default is True.

    Returns:
        None: Displays the plot.
    """
    # 
    filtered_data = data[data[cluster_column].isin(cell_types_lineage1 + (cell_types_lineage2 or []))].copy()
    
    # 
    filtered_data[cluster_column] = pd.Categorical(filtered_data[cluster_column], 
                                                    categories=cell_types_lineage1 + (cell_types_lineage2 or []), 
                                                    ordered=True)

    # 
    filtered_data[['theta', 'phi']] = filtered_data.apply(
        lambda row: calculate_spherical_angles(row[pcs_columns[0]], row[pcs_columns[1]], row[pcs_columns[2]]), 
        axis=1, result_type='expand'
    )
    
    # 
    position_map = {cell: i for i, cell in enumerate(cell_types_lineage1 + (cell_types_lineage2 or []))}
    filtered_data['pseudotime'] = filtered_data[cluster_column].map(position_map)

    # 
    lineage1_data = filtered_data[filtered_data[cluster_column].isin(cell_types_lineage1)]
    lineage1_corr, _ = pearsonr(lineage1_data['pseudotime'], lineage1_data['theta'])

    if cell_types_lineage2:
        lineage2_data = filtered_data[filtered_data[cluster_column].isin(cell_types_lineage2)]
        lineage2_corr, _ = pearsonr(lineage2_data['pseudotime'], lineage2_data['theta'])

    #
    fig, ax = plt.subplots(figsize=figsize)
    
    # 
    scatter = ax.scatter(
        filtered_data['pseudotime'], 
        filtered_data['theta'], 
        c=filtered_data[cluster_column].cat.codes, 
        cmap=colormap, 
        s=50, alpha=0.8, label='Theta vs Pseudotime'
    )
    
    # 
    unique_celltypes = filtered_data[cluster_column].cat.categories
    color_map = plt.cm.get_cmap(colormap)
    handles = [plt.Line2D([0], [0], marker='o', color=color_map(i / len(unique_celltypes)), linestyle='', markersize=10, 
                          label=cell) for i, cell in enumerate(unique_celltypes)]
    ax.legend(handles=handles, title='Cell Types', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 
    medians_lineage1 = lineage1_data.groupby(cluster_column)['theta'].median()
    ax.plot(
        [position_map[cell] for cell in medians_lineage1.index], 
        medians_lineage1.values, 
        marker='o', linestyle='-', color='red', label='Lineage 1 Median'
    )

    if cell_types_lineage2:
        medians_lineage2 = lineage2_data.groupby(cluster_column)['theta'].median()
        ax.plot(
            [position_map[cell] for cell in medians_lineage2.index], 
            medians_lineage2.values, 
            marker='o', linestyle='--', color='blue', label='Lineage 2 Median'
        )
    
    # 
    if annotate:
        ax.annotate(f'Lineage 1 Corr: {lineage1_corr:.2f}', xy=(0.1, 0.9), xycoords='axes fraction', fontsize=10, 
                    bbox=dict(boxstyle="round", alpha=0.1))
        if cell_types_lineage2:
            ax.annotate(f'Lineage 2 Corr: {lineage2_corr:.2f}', xy=(0.1, 0.85), xycoords='axes fraction', fontsize=10, 
                        bbox=dict(boxstyle="round", alpha=0.1))
    
    # 
    ax.set_xlabel('Pseudotime')
    ax.set_ylabel(r'$\theta$')
    ax.set_xticks(list(position_map.values()))
    ax.set_xticklabels(list(position_map.keys()))
    ax.set_title('Lineage Correlation')
    
    plt.tight_layout()
    plt.show()

def statistical_test_on_lineages(data, 
                                  lineage1, 
                                  lineage2, 
                                  pcs_columns=['rotated_PC1', 'rotated_PC2', 'rotated_PC3'], 
                                  cluster_column='celltype', 
                                  figsize=(12, 5)):
    """
    Performs statistical testing (KS Test) on two user-selected lineages and visualizes the distributions.

    Args:
        data (pd.DataFrame): Input data containing PCA coordinates and cell type annotations.
        lineage1 (list): List of cell types in the first lineage.
        lineage2 (list): List of cell types in the second lineage.
        pcs_columns (list): Column names for rotated PCA coordinates. Default is ['rotated_PC1', 'rotated_PC2', 'rotated_PC3'].
        cluster_column (str): Column name for the cell type annotations. Default is 'celltype'.
        figsize (tuple): Size of the plot. Default is (12, 5).

    Returns:
        dict: Test results and types of test performed.
    """
    # 
    filtered_data = data[data[cluster_column].isin(lineage1 + lineage2)].copy()
    
    # 
    filtered_data[['theta', 'phi']] = filtered_data.apply(
        lambda row: calculate_spherical_angles(row[pcs_columns[0]], row[pcs_columns[1]], row[pcs_columns[2]]), 
        axis=1, result_type='expand'
    )
    
    # 
    filtered_data['ϴ'] = np.degrees(filtered_data['theta'])

    #
    lineage1_data = filtered_data[filtered_data[cluster_column].isin(lineage1)]['ϴ']
    lineage2_data = filtered_data[filtered_data[cluster_column].isin(lineage2)]['ϴ']

    # 
    normality_lineage1 = shapiro(lineage1_data)
    normality_lineage2 = shapiro(lineage2_data)

    # 
    if normality_lineage1.pvalue < 0.05 or normality_lineage2.pvalue < 0.05:
        # 
        test_result = ks_2samp(lineage1_data, lineage2_data, alternative='two-sided')
        test_type = 'Kolmogorov-Smirnov Test'
    else:
        # 
        test_result = ttest_ind(lineage1_data, lineage2_data, equal_var=False)
        test_type = 't-test'

    # 
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    sns.histplot(lineage1_data, kde=True, ax=ax[0], color='orchid', bins=20)
    ax[0].set_title(f'Distribution of ϴ for Lineage 1\nShapiro p={normality_lineage1.pvalue:.1e}')
    sns.histplot(lineage2_data, kde=True, ax=ax[1], color='darkseagreen', bins=20)
    ax[1].set_title(f'Distribution of ϴ for Lineage 2\nShapiro p={normality_lineage2.pvalue:.1e}')

    # 
    test_result_text = f"{test_type}:\nStatistic: {test_result.statistic:.3f}\nP-value: {test_result.pvalue:.2e}"
    fig.suptitle(test_result_text, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.show()

    return {
        'test_type': test_type,
        'test_statistic': test_result.statistic,
        'p_value': test_result.pvalue
    }

def find_and_visualize_correlated_genes(data, 
                                        gene_data, 
                                        band_ranges, 
                                        target_band, 
                                        correlation_cutoff=0.4, 
                                        pseudotime_conversion_func=None, 
                                        pseudotime_order=None, 
                                        figsize=(5, 8)):
    """
    Finds and visualizes genes with positive and negative correlations to stripe formation or pseudotime.

    Args:
        data (pd.DataFrame): Data containing cell metadata including longitude, cluster, and pseudotime information.
        gene_data (pd.DataFrame): Gene expression data with cells as rows and genes as columns.
        band_ranges (dict): Dictionary specifying the longitude ranges for different bands.
        target_band (str): The band to analyze (e.g., 'Band1').
        correlation_cutoff (float): Minimum absolute correlation coefficient for filtering genes. Default is 0.4.
        pseudotime_conversion_func (function): Function to convert cluster/pseudotime to numeric values. Default is None.
        pseudotime_order (list): Ordered list of cell types for pseudotime. Used if `pseudotime_conversion_func` is not provided.
        figsize (tuple): Size of the visualization plot. Default is (5, 8).

    Returns:
        pd.DataFrame: A DataFrame of correlated genes with correlation coefficients and p-values.
    """
    # 
    data['band'] = 'Outside'
    for band_name, (lower_bound, upper_bound) in band_ranges.items():
        in_band = (data['Longitude'] >= lower_bound) & (data['Longitude'] <= upper_bound)
        data.loc[in_band, 'band'] = band_name

    # 
    band_data = data[data['band'] == target_band]

    # 
    if pseudotime_conversion_func is None:
        if pseudotime_order is None:
            raise ValueError("You must provide either a `pseudotime_conversion_func` or a `pseudotime_order`.")
        
        def default_pseudotime_conversion_func(cluster):
            return pseudotime_order.index(cluster) if cluster in pseudotime_order else np.nan
        
        pseudotime_conversion_func = default_pseudotime_conversion_func

    # 
    band_data['cluster_numeric'] = band_data['cluster'].apply(pseudotime_conversion_func)

    # 
    band_data.dropna(subset=['cluster_numeric'], inplace=True)

    # 
    aligned_gene_data = gene_data[gene_data['Unnamed: 0'].isin(band_data['Unnamed: 0'])]
    aligned_data = pd.merge(band_data[['Unnamed: 0', 'cluster_numeric']], aligned_gene_data, on='Unnamed: 0')

    # 
    results = []
    for gene in aligned_gene_data.columns[1:]:
        if np.var(aligned_data[gene]) < 1e-8 or np.var(aligned_data['cluster_numeric']) < 1e-8:
            continue  

        correlation, p_value = spearmanr(aligned_data['cluster_numeric'], aligned_data[gene])
        results.append({
            'Gene': gene,
            'Correlation': correlation,
            'P-value': p_value
        })

    # 
    results_df = pd.DataFrame(results)
    results_df.dropna(inplace=True)
    results_df.sort_values(by='Correlation', ascending=False, inplace=True)

    # 
    adjusted_pvals = multipletests(results_df['P-value'], alpha=0.05, method='fdr_bh')[1]
    results_df['Adjusted P-Value'] = adjusted_pvals
    epsilon = 1e-10
    results_df['-log10(Adjusted P-Value)'] = -np.log10(results_df['Adjusted P-Value'] + epsilon)

    # 
    filtered_genes = results_df[
        (results_df['Correlation'] > correlation_cutoff) | 
        (results_df['Correlation'] < -correlation_cutoff)
    ]

    # 
    fig, ax = plt.subplots(figsize=figsize)
    norm = Normalize(vmin=2, vmax=10)
    sm = plt.cm.ScalarMappable(cmap='viridis_r', norm=norm)
    sm.set_array([])

    for index, (_, row) in enumerate(filtered_genes.iterrows()):
        ax.plot([0, row['Correlation']], [index, index], 'gray')
        ax.scatter(row['Correlation'], index, 
                   color=sm.to_rgba(row['-log10(Adjusted P-Value)']), 
                   s=np.abs(row['Correlation']) * 100)

    ax.set_yticks(range(len(filtered_genes)))
    ax.set_yticklabels(filtered_genes['Gene'])
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical')
    cbar.set_label('-log10(Adjusted P-Value') 

    ax.set_xlabel('Correlation Coefficient')
    ax.grid(True)
    ax.set_title(f'{target_band}: Genes with abs(correlation) > {correlation_cutoff}')
    plt.tight_layout()
    plt.show()

    return results_df
