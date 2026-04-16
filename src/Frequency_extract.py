# -*- coding: utf-8 -*-

import os

import anndata
import numpy as np
import pandas as pd
import pywt
import scanpy as sc
from scipy.interpolate import make_interp_spline
from sklearn.metrics import pairwise_distances


TRAJECTORY_META_ROWS = ("time", "path", "sep", "step", "id")
TIME_KEY_CANDIDATES = (
    "latent_time",
    "velocity_pseudotime",
    "pseudotime",
    "pseudo_time",
    "dpt_pseudotime",
    "palantir_pseudotime",
    "veloagent_pseudotime",
    "agent_time",
    "time",
)
PATH_KEY_CANDIDATES = (
    "path",
    "path_id",
    "trajectory",
    "trajectory_id",
    "lineage",
    "lineage_id",
    "branch",
    "branch_id",
    "traj",
    "traj_id",
    "cell_path",
)
CELL_KEY_CANDIDATES = ("cellID", "cell_id", "cell", "barcode", "obs_names")
GENE_KEY_CANDIDATES = ("gene_name", "gene_id", "gene", "var_names")
VALUE_KEY_CANDIDATES = (
    "splice",
    "spliced",
    "Ms",
    "expression",
    "expr",
    "count",
    "counts",
    "value",
)


def _as_path(path):
    return os.fspath(path) if isinstance(path, os.PathLike) else path


def _is_anndata(obj):
    return isinstance(obj, anndata.AnnData)


def _to_dense_array(matrix):
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix)


def _get_adata_matrix(adata, layer=None):
    if layer is None:
        return adata.X
    if layer not in adata.layers:
        raise KeyError("AnnData does not contain layer '%s'." % layer)
    return adata.layers[layer]


def _pick_existing_key(container, preferred, candidates, key_name):
    if preferred is not None:
        if isinstance(preferred, str):
            if preferred in container:
                return preferred
            raise KeyError("%s '%s' was not found." % (key_name, preferred))
        return preferred

    for key in candidates:
        if key in container:
            return key
    return None


def _resolve_vector(frame_or_obs, key, expected_len, key_name, required=True):
    if key is None:
        if required:
            raise ValueError("Could not infer %s; please pass the key explicitly." % key_name)
        return None

    if isinstance(key, str):
        if key not in frame_or_obs:
            if required:
                raise KeyError("%s '%s' was not found." % (key_name, key))
            return None
        values = frame_or_obs[key].to_numpy()
    else:
        values = np.asarray(key)

    if len(values) != expected_len:
        raise ValueError(
            "%s has length %d, but %d values were expected."
            % (key_name, len(values), expected_len)
        )
    return values


def _load_trajectory_frame(trajectory):
    trajectory = _as_path(trajectory)
    if isinstance(trajectory, pd.DataFrame):
        return trajectory.copy()
    if isinstance(trajectory, str):
        return pd.read_csv(trajectory, index_col=0)
    raise TypeError("trajectory must be an AnnData, a pandas DataFrame, or a CSV path.")


def _complete_formatted_trajectory(trajectory_info):
    trajectory_info = trajectory_info.copy()
    missing_required = [row for row in ("time", "path") if row not in trajectory_info.index]
    if missing_required:
        raise ValueError(
            "Formatted trajectory must contain index rows: %s."
            % ", ".join(missing_required)
        )

    n_cols = trajectory_info.shape[1]
    if "sep" not in trajectory_info.index:
        trajectory_info.loc["sep"] = np.ones(n_cols, dtype=int)
    if "step" not in trajectory_info.index:
        trajectory_info.loc["step"] = np.arange(1, n_cols + 1, dtype=int)
    if "id" not in trajectory_info.index:
        trajectory_info.loc["id"] = np.arange(1, n_cols + 1, dtype=int)

    gene_rows = [row for row in trajectory_info.index if row not in TRAJECTORY_META_ROWS]
    return trajectory_info.loc[gene_rows + list(TRAJECTORY_META_ROWS)]


def _save_formatted_trajectory(trajectory_info, dataset=None, output_csv=None, save_csv=False):
    if output_csv is None and save_csv and dataset is not None:
        output_csv = "%s_trajectory.csv" % dataset
    if output_csv is not None:
        trajectory_info.to_csv(output_csv)


def _encode_paths(path_values, n_cells):
    if path_values is None:
        return np.ones(n_cells, dtype=int), {1: 1}

    path_series = pd.Series(path_values)
    numeric = pd.to_numeric(path_series, errors="coerce")
    if numeric.notna().all():
        encoded = numeric.to_numpy()
        if np.all(np.isclose(encoded, np.round(encoded))):
            encoded = np.round(encoded).astype(int)
        return encoded, dict((v, v) for v in pd.unique(encoded))

    categories = pd.Categorical(path_series)
    mapping = dict((cat, i + 1) for i, cat in enumerate(categories.categories))
    encoded = categories.codes.astype(float) + 1
    encoded[categories.codes < 0] = np.nan
    return encoded, mapping


def _mean_expression(matrix, row_indices, agg):
    block = matrix[row_indices, :]
    if agg == "mean":
        values = block.mean(axis=0)
        return np.asarray(values).reshape(-1)
    if agg == "median":
        return np.median(_to_dense_array(block), axis=0)
    raise ValueError("agg must be either 'mean' or 'median'.")


def _build_formatted_trajectory(
    expression,
    gene_names,
    time_values,
    path_values=None,
    n_bins=20,
    min_cells_per_bin=1,
    agg="mean",
):
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2.")
    if min_cells_per_bin < 1:
        raise ValueError("min_cells_per_bin must be at least 1.")

    time_values = pd.to_numeric(pd.Series(time_values), errors="coerce").to_numpy()
    if expression.shape[0] != len(time_values):
        raise ValueError(
            "Expression has %d cells, but %d pseudotime values were provided."
            % (expression.shape[0], len(time_values))
        )

    encoded_paths, _ = _encode_paths(path_values, expression.shape[0])
    valid = np.isfinite(time_values) & pd.notna(encoded_paths)
    if not np.any(valid):
        raise ValueError("No cells with valid pseudotime/path values were found.")

    columns = []
    trajectory_blocks = []
    point_id = 1

    for path_id in sorted(pd.unique(encoded_paths[valid])):
        cell_idx = np.where(valid & (encoded_paths == path_id))[0]
        if len(cell_idx) == 0:
            continue
        cell_idx = cell_idx[np.argsort(time_values[cell_idx])]
        chunks = np.array_split(cell_idx, min(n_bins, len(cell_idx)))
        step = 1

        for chunk in chunks:
            if len(chunk) < min_cells_per_bin:
                continue
            expr_mean = _mean_expression(expression, chunk, agg)
            point = np.concatenate(
                [
                    expr_mean,
                    np.asarray(
                        [
                            float(np.nanmean(time_values[chunk])),
                            path_id,
                            1,
                            step,
                            point_id,
                        ],
                        dtype=float,
                    ),
                ]
            )
            trajectory_blocks.append(point)
            columns.append("T%d" % point_id)
            point_id += 1
            step += 1

    if not trajectory_blocks:
        raise ValueError("No trajectory bins were generated.")

    index = list(map(str, gene_names)) + list(TRAJECTORY_META_ROWS)
    trajectory_info = pd.DataFrame(
        np.column_stack(trajectory_blocks), index=index, columns=columns
    )
    return _complete_formatted_trajectory(trajectory_info)


def _choose_value_key(df, value_key):
    if value_key is not None:
        if value_key not in df.columns:
            raise KeyError("value_key '%s' was not found." % value_key)
        return value_key
    for key in VALUE_KEY_CANDIDATES:
        if key in df.columns:
            return key
    raise ValueError("Could not infer expression value column; pass value_key explicitly.")


def _choose_column(df, preferred, candidates, key_name):
    if preferred is not None:
        if preferred not in df.columns:
            raise KeyError("%s '%s' was not found." % (key_name, preferred))
        return preferred
    for key in candidates:
        if key in df.columns:
            return key
    raise ValueError("Could not infer %s; pass it explicitly." % key_name)


def _cell_metadata_from_frame(df, cell_key):
    if cell_key is None:
        for candidate in CELL_KEY_CANDIDATES:
            if candidate in df.columns:
                cell_key = candidate
                break

    if cell_key is not None:
        if cell_key not in df.columns:
            raise KeyError("cell_key '%s' was not found." % cell_key)
        meta = df.drop_duplicates(cell_key).set_index(cell_key)
        meta.index = meta.index.astype(str)
        return meta

    if df.index.is_unique:
        meta = df.copy()
        meta.index = meta.index.astype(str)
        return meta

    raise ValueError("Could not infer cell_key; pass it explicitly.")


def _frame_with_adata_to_trajectory(
    df,
    adata,
    time_key,
    path_key,
    expression_layer,
    cell_key,
    n_bins,
    min_cells_per_bin,
    agg,
):
    meta = _cell_metadata_from_frame(df, cell_key)
    time_key = _pick_existing_key(meta, time_key, TIME_KEY_CANDIDATES, "time_key")
    if time_key is None:
        raise ValueError("Could not infer pseudotime column; pass time_key explicitly.")

    path_key = _pick_existing_key(meta, path_key, PATH_KEY_CANDIDATES, "path_key")
    cells = adata.obs_names.astype(str)
    missing = [cell for cell in cells if cell not in meta.index]
    if missing:
        raise ValueError(
            "%d AnnData cells were not found in the trajectory table, e.g. %s."
            % (len(missing), ", ".join(missing[:5]))
        )

    aligned = meta.loc[cells]
    time_values = aligned[time_key].to_numpy()
    path_values = aligned[path_key].to_numpy() if path_key is not None else None

    return _build_formatted_trajectory(
        _get_adata_matrix(adata, expression_layer),
        adata.var_names,
        time_values,
        path_values=path_values,
        n_bins=n_bins,
        min_cells_per_bin=min_cells_per_bin,
        agg=agg,
    )


def _long_frame_to_trajectory(
    df,
    time_key,
    path_key,
    cell_key,
    gene_key,
    value_key,
    n_bins,
    min_cells_per_bin,
    agg,
):
    cell_key = _choose_column(df, cell_key, CELL_KEY_CANDIDATES, "cell_key")
    gene_key = _choose_column(df, gene_key, GENE_KEY_CANDIDATES, "gene_key")
    value_key = _choose_value_key(df, value_key)

    time_key = _pick_existing_key(df, time_key, TIME_KEY_CANDIDATES, "time_key")
    if time_key is None:
        raise ValueError("Could not infer pseudotime column; pass time_key explicitly.")
    path_key = _pick_existing_key(df, path_key, PATH_KEY_CANDIDATES, "path_key")

    expression_df = df.pivot_table(
        index=cell_key, columns=gene_key, values=value_key, aggfunc=agg, fill_value=0
    )
    expression_df.index = expression_df.index.astype(str)
    expression_df.columns = expression_df.columns.astype(str)

    meta_agg = {time_key: "mean"}
    if path_key is not None:
        meta_agg[path_key] = lambda x: x.dropna().iloc[0] if len(x.dropna()) else np.nan
    cell_meta = df.groupby(cell_key).agg(meta_agg)
    cell_meta.index = cell_meta.index.astype(str)
    cell_meta = cell_meta.loc[expression_df.index]

    return _build_formatted_trajectory(
        expression_df.to_numpy(),
        expression_df.columns,
        cell_meta[time_key].to_numpy(),
        path_values=cell_meta[path_key].to_numpy() if path_key is not None else None,
        n_bins=n_bins,
        min_cells_per_bin=min_cells_per_bin,
        agg=agg,
    )


def _adata_from_long_frame(df, cell_key=None, gene_key=None, value_key=None, agg="mean"):
    cell_key = _choose_column(df, cell_key, CELL_KEY_CANDIDATES, "cell_key")
    gene_key = _choose_column(df, gene_key, GENE_KEY_CANDIDATES, "gene_key")
    value_key = _choose_value_key(df, value_key)
    expression_df = df.pivot_table(
        index=cell_key, columns=gene_key, values=value_key, aggfunc=agg, fill_value=0
    )
    adata = anndata.AnnData(expression_df.to_numpy())
    adata.obs_names = expression_df.index.astype(str)
    adata.var_names = expression_df.columns.astype(str)
    return adata


def frequency_extract_format(
    trajectory,
    adata=None,
    dataset=None,
    time_key=None,
    path_key=None,
    expression_layer=None,
    cell_key=None,
    gene_key=None,
    value_key=None,
    n_bins=20,
    min_cells_per_bin=1,
    agg="mean",
    save_csv=False,
    output_csv=None,
):
    """
    Format an inferred/imported trajectory for scGeneRhythm frequency extraction.

    The returned DataFrame uses the repository's trajectory format:
    gene rows followed by metadata rows ``time``, ``path``, ``sep``, ``step``
    and ``id``. The input can be an already formatted CSV/DataFrame, AnnData
    with pseudotime in ``obs``, cell metadata plus AnnData expression, or a
    long expression table such as cellDancer output.
    """
    if _is_anndata(trajectory):
        time_key = _pick_existing_key(
            trajectory.obs, time_key, TIME_KEY_CANDIDATES, "time_key"
        )
        if time_key is None:
            raise ValueError("Could not infer pseudotime key from AnnData.obs.")
        path_key = _pick_existing_key(
            trajectory.obs, path_key, PATH_KEY_CANDIDATES, "path_key"
        )
        time_values = _resolve_vector(trajectory.obs, time_key, trajectory.n_obs, "time_key")
        path_values = (
            _resolve_vector(trajectory.obs, path_key, trajectory.n_obs, "path_key", False)
            if path_key is not None
            else None
        )
        trajectory_info = _build_formatted_trajectory(
            _get_adata_matrix(trajectory, expression_layer),
            trajectory.var_names,
            time_values,
            path_values=path_values,
            n_bins=n_bins,
            min_cells_per_bin=min_cells_per_bin,
            agg=agg,
        )
    else:
        df = _load_trajectory_frame(trajectory)
        if "time" in df.index and "path" in df.index:
            trajectory_info = _complete_formatted_trajectory(df)
        elif adata is not None:
            trajectory_info = _frame_with_adata_to_trajectory(
                df,
                adata,
                time_key,
                path_key,
                expression_layer,
                cell_key,
                n_bins,
                min_cells_per_bin,
                agg,
            )
        else:
            trajectory_info = _long_frame_to_trajectory(
                df,
                time_key,
                path_key,
                cell_key,
                gene_key,
                value_key,
                n_bins,
                min_cells_per_bin,
                agg,
            )

    _save_formatted_trajectory(trajectory_info, dataset, output_csv, save_csv)
    return trajectory_info


def _prepare_path_data(trajectory_info, path_id):
    trajectory_info = _complete_formatted_trajectory(trajectory_info)
    gene_rows = [row for row in trajectory_info.index if row not in TRAJECTORY_META_ROWS]

    raw_paths = trajectory_info.loc["path"]
    paths = pd.to_numeric(raw_paths, errors="coerce")
    if path_id is None:
        mask = np.ones(trajectory_info.shape[1], dtype=bool)
    elif paths.notna().any():
        mask = np.asarray(paths == path_id)
    else:
        mask = np.asarray(raw_paths.astype(str) == str(path_id))
    if not np.any(mask):
        raise ValueError("No trajectory points found for path_id=%s." % path_id)

    y = trajectory_info.loc[gene_rows, :].iloc[:, mask]
    y = y.apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(dtype=float)
    x = pd.to_numeric(trajectory_info.loc["time"].iloc[mask], errors="coerce").to_numpy()

    valid = np.isfinite(x)
    x = x[valid]
    y = y[:, valid]
    if len(x) < 2:
        raise ValueError("At least two trajectory points are required.")

    order = np.argsort(x)
    x = x[order]
    y = y[:, order]

    unique_x = []
    unique_y = []
    for value in pd.unique(x):
        same = x == value
        unique_x.append(value)
        unique_y.append(y[:, same].mean(axis=1))
    x = np.asarray(unique_x, dtype=float)
    y = np.column_stack(unique_y)
    if len(x) < 2:
        raise ValueError("At least two unique pseudotime values are required.")
    return x, y, gene_rows


def _frequency_features_from_trajectory(
    trajectory_info,
    path_id=1,
    n_smooth=100,
    smooth_k=3,
    wavelet="mexh",
    scales=None,
    threshold=0.01,
):
    if scales is None:
        scales = np.arange(1, 51)

    x, y, gene_rows = _prepare_path_data(trajectory_info, path_id)
    k = min(int(smooth_k), len(x) - 1)
    x_smooth = np.linspace(start=x.min(), stop=x.max(), num=n_smooth)
    y_smooth = make_interp_spline(x, y.transpose(), k=k)(x_smooth)

    dt = x_smooth[1] - x_smooth[0] if len(x_smooth) > 1 else 1.0
    coeffs, _ = pywt.cwt(y_smooth.transpose(), scales, wavelet, sampling_period=dt)
    coeffs_thresh = pywt.threshold(coeffs, threshold, mode="soft")

    yf_smooth = np.sum(coeffs_thresh, axis=2).T
    yf_info = np.abs(yf_smooth)
    yf_den = yf_info.max(axis=1)
    yf_den[yf_den == 0] = 1.0
    yf_info = yf_info / yf_den[:, None]

    y_info = y_smooth.transpose()
    y_den = y_info.max(axis=1)
    y_den[y_den == 0] = 1.0
    y_info = y_info / y_den[:, None]
    return np.concatenate((y_info, yf_info), axis=1), gene_rows


def _align_adata_to_genes(adata, gene_rows):
    gene_rows = list(map(str, gene_rows))
    adata_var = list(map(str, adata.var_names))
    if all(gene in adata_var for gene in gene_rows):
        return adata[:, gene_rows].copy()
    if len(gene_rows) == adata.n_vars:
        adata_copy = adata.copy()
        adata_copy.var_names = gene_rows
        return adata_copy
    missing = [gene for gene in gene_rows if gene not in adata_var]
    raise ValueError(
        "Trajectory genes do not match AnnData.var_names; missing examples: %s"
        % ", ".join(missing[:5])
    )


def _pca_features_for_genes(adata, gene_rows, n_pca=100):
    adata_gene_aligned = _align_adata_to_genes(adata, gene_rows)
    adata_t = adata_gene_aligned.T.copy()
    if adata_t.n_obs == 0:
        raise ValueError("No genes available for PCA feature calculation.")
    if adata_t.n_obs < 2 or adata_t.n_vars < 2:
        return np.zeros((adata_t.n_obs, n_pca), dtype=float)

    sc.pp.normalize_total(adata_t)
    sc.pp.log1p(adata_t)
    sc.pp.scale(adata_t)
    max_comps = max(1, min(int(n_pca), adata_t.n_obs - 1, adata_t.n_vars - 1))
    sc.tl.pca(adata_t, n_comps=max_comps, svd_solver="auto")
    pca = adata_t.obsm["X_pca"]
    if pca.shape[1] < n_pca:
        pad = np.zeros((pca.shape[0], n_pca - pca.shape[1]), dtype=pca.dtype)
        pca = np.concatenate((pca, pad), axis=1)
    return pca[:, :n_pca]


def frequency_extract(
    trajectory_info,
    adata,
    dataset,
    path_id=1,
    output_npy=None,
    n_smooth=100,
    n_pca=100,
    smooth_k=3,
    wavelet="mexh",
    scales=None,
    threshold=0.01,
):
    freq_features, gene_rows = _frequency_features_from_trajectory(
        trajectory_info,
        path_id=path_id,
        n_smooth=n_smooth,
        smooth_k=smooth_k,
        wavelet=wavelet,
        scales=scales,
        threshold=threshold,
    )
    pca_features = _pca_features_for_genes(adata, gene_rows, n_pca=n_pca)
    res = np.concatenate((freq_features, pca_features), axis=1)

    if output_npy is None:
        output_npy = "%s.npy" % dataset
    np.save(output_npy, np.asarray(res[:, :]))
    return res


def _default_velocity_expression_layer(adata, preferred=None):
    if preferred is not None:
        return preferred
    for layer in ("Ms", "spliced", "X_spliced"):
        if layer in adata.layers:
            return layer
    return None


def frequency_extract_scvelo(
    adata,
    dataset,
    mode="dynamical",
    time_key=None,
    path_key=None,
    path_id=1,
    expression_layer=None,
    n_top_genes=None,
    min_shared_counts=20,
    n_pcs=30,
    n_neighbors=30,
    max_iter=10,
    n_jobs=None,
    n_bins=20,
    min_cells_per_bin=1,
    agg="mean",
    save_trajectory=True,
    trajectory_csv=None,
    output_npy=None,
    copy=True,
    return_adata=False,
    **kwargs,
):
    """
    Infer RNA-velocity trajectory with scVelo and extract frequency features.

    ``adata`` must contain ``spliced`` and ``unspliced`` layers. In dynamical
    mode this wrapper uses ``latent_time`` when available and otherwise falls
    back to ``velocity_pseudotime``.
    """
    try:
        import scvelo as scv
    except ImportError as exc:
        raise ImportError("frequency_extract_scvelo requires scvelo.") from exc

    required_layers = ("spliced", "unspliced")
    missing_layers = [layer for layer in required_layers if layer not in adata.layers]
    if missing_layers:
        raise ValueError(
            "scVelo requires AnnData layers %s; missing %s."
            % (required_layers, missing_layers)
        )

    adata_v = adata.copy() if copy else adata
    scv.pp.filter_and_normalize(
        adata_v, min_shared_counts=min_shared_counts, n_top_genes=n_top_genes
    )
    scv.pp.moments(adata_v, n_pcs=n_pcs, n_neighbors=n_neighbors)

    velocity_kwargs = kwargs.pop("velocity_kwargs", {})
    if mode == "dynamical":
        recover_kwargs = kwargs.pop("recover_dynamics_kwargs", {})
        recover_kwargs.setdefault("max_iter", max_iter)
        if n_jobs is not None:
            recover_kwargs.setdefault("n_jobs", n_jobs)
        scv.tl.recover_dynamics(adata_v, **recover_kwargs)

    scv.tl.velocity(adata_v, mode=mode, **velocity_kwargs)
    graph_kwargs = kwargs.pop("velocity_graph_kwargs", {})
    if n_jobs is not None:
        graph_kwargs.setdefault("n_jobs", n_jobs)
    scv.tl.velocity_graph(adata_v, **graph_kwargs)
    scv.tl.velocity_pseudotime(adata_v)

    if mode == "dynamical":
        try:
            scv.tl.latent_time(adata_v)
        except Exception:
            pass

    if time_key is None:
        time_key = "latent_time" if "latent_time" in adata_v.obs else "velocity_pseudotime"

    trajectory_info = frequency_extract_format(
        adata_v,
        dataset=dataset,
        time_key=time_key,
        path_key=path_key,
        expression_layer=_default_velocity_expression_layer(adata_v, expression_layer),
        n_bins=n_bins,
        min_cells_per_bin=min_cells_per_bin,
        agg=agg,
        save_csv=save_trajectory,
        output_csv=trajectory_csv,
    )
    res = frequency_extract(
        trajectory_info,
        adata_v,
        dataset,
        path_id=path_id,
        output_npy=output_npy,
    )
    if return_adata:
        return res, trajectory_info, adata_v
    return res


def _ensure_umap_for_celldancer(adata, embedding_key):
    if embedding_key in adata.obsm:
        return
    adata_tmp = adata.copy()
    sc.pp.normalize_total(adata_tmp)
    sc.pp.log1p(adata_tmp)
    sc.tl.pca(adata_tmp)
    sc.pp.neighbors(adata_tmp)
    sc.tl.umap(adata_tmp)
    adata.obsm[embedding_key] = adata_tmp.obsm["X_umap"]


def _celldancer_dataframe_from_adata(
    adata,
    cdutil,
    us_para=None,
    cell_type_key=None,
    embedding_key="X_umap",
    save_path=None,
):
    adata_cd = adata.copy()
    if us_para is None:
        if "unspliced" in adata_cd.layers and "spliced" in adata_cd.layers:
            us_para = ["unspliced", "spliced"]
        elif "Mu" in adata_cd.layers and "Ms" in adata_cd.layers:
            us_para = ["Mu", "Ms"]
        else:
            raise ValueError(
                "cellDancer conversion requires unspliced/spliced or Mu/Ms layers."
            )

    if cell_type_key is None or cell_type_key not in adata_cd.obs:
        cell_type_key = "__cell_type"
        adata_cd.obs[cell_type_key] = "cell"

    _ensure_umap_for_celldancer(adata_cd, embedding_key)
    if save_path is None:
        save_path = "cellDancer_input.csv"

    return cdutil.adata_to_df_with_embed(
        adata_cd,
        us_para=us_para,
        cell_type_para=cell_type_key,
        embed_para=embedding_key,
        save_path=save_path,
    )


def frequency_extract_celldancer(
    adata_or_df,
    dataset,
    adata=None,
    gene_list=None,
    time_key="pseudotime",
    path_key=None,
    path_id=1,
    cell_key="cellID",
    gene_key="gene_name",
    value_key="splice",
    us_para=None,
    cell_type_key=None,
    embedding_key="X_umap",
    n_bins=20,
    min_cells_per_bin=1,
    agg="mean",
    velocity_kwargs=None,
    projection_kwargs=None,
    pseudotime_kwargs=None,
    save_trajectory=True,
    trajectory_csv=None,
    output_npy=None,
    return_intermediate=False,
):
    """
    Infer trajectory with cellDancer, format it, and extract frequency features.

    ``adata_or_df`` may be an AnnData object or a cellDancer-style long table.
    For long-table input, pass ``adata`` if PCA features should come from the
    original AnnData; otherwise they are computed from splice expression in the
    table.
    """
    try:
        import celldancer as cd
        import celldancer.utilities as cdutil
    except ImportError as exc:
        raise ImportError("frequency_extract_celldancer requires celldancer.") from exc

    velocity_kwargs = {} if velocity_kwargs is None else dict(velocity_kwargs)
    projection_kwargs = {} if projection_kwargs is None else dict(projection_kwargs)
    pseudotime_kwargs = {} if pseudotime_kwargs is None else dict(pseudotime_kwargs)

    if _is_anndata(adata_or_df):
        adata_for_frequency = adata_or_df
        input_df = _celldancer_dataframe_from_adata(
            adata_or_df,
            cdutil,
            us_para=us_para,
            cell_type_key=cell_type_key,
            embedding_key=embedding_key,
            save_path="%s_celldancer_input.csv" % dataset,
        )
    else:
        input_df = _load_trajectory_frame(adata_or_df)
        adata_for_frequency = adata

    if gene_list is None:
        gene_list = list(pd.unique(input_df[gene_key])) if gene_key in input_df else None

    loss_df, celldancer_df = cd.velocity(input_df, gene_list=gene_list, **velocity_kwargs)
    celldancer_df = cd.compute_cell_velocity(celldancer_df, **projection_kwargs)

    if not pseudotime_kwargs:
        pseudotime_kwargs.update(
            {
                "grid": (30, 30),
                "dt": 0.05,
                "t_total": 200,
                "n_repeats": 10,
                "n_jobs": -1,
                "speed_up": (60, 60),
                "n_paths": 5,
            }
        )
    pseudo_result = cd.pseudo_time(celldancer_df, **pseudotime_kwargs)
    if isinstance(pseudo_result, pd.DataFrame):
        celldancer_df = pseudo_result
    elif isinstance(pseudo_result, (tuple, list)):
        frames = [item for item in pseudo_result if isinstance(item, pd.DataFrame)]
        if frames:
            celldancer_df = frames[-1]

    if adata_for_frequency is None:
        adata_for_frequency = _adata_from_long_frame(
            celldancer_df,
            cell_key=cell_key,
            gene_key=gene_key,
            value_key=value_key,
            agg=agg,
        )

    trajectory_info = frequency_extract_format(
        celldancer_df,
        dataset=dataset,
        time_key=time_key,
        path_key=path_key,
        cell_key=cell_key,
        gene_key=gene_key,
        value_key=value_key,
        n_bins=n_bins,
        min_cells_per_bin=min_cells_per_bin,
        agg=agg,
        save_csv=save_trajectory,
        output_csv=trajectory_csv,
    )
    res = frequency_extract(
        trajectory_info,
        adata_for_frequency,
        dataset,
        path_id=path_id,
        output_npy=output_npy,
    )

    if return_intermediate:
        return res, trajectory_info, celldancer_df, loss_df
    return res


def _run_veloagent(adata, runner=None, runner_kwargs=None, model_kwargs=None):
    runner_kwargs = {} if runner_kwargs is None else dict(runner_kwargs)
    model_kwargs = {} if model_kwargs is None else dict(model_kwargs)

    if runner is not None:
        return runner(adata, **runner_kwargs)

    try:
        import veloagent
    except ImportError as exc:
        raise ImportError(
            "frequency_extract_veloagent requires veloagent or a custom runner."
        ) from exc

    for name in ("run", "fit_transform", "fit"):
        if hasattr(veloagent, name):
            return getattr(veloagent, name)(adata, **runner_kwargs)

    if hasattr(veloagent, "VeloAgent"):
        model = veloagent.VeloAgent(**model_kwargs)
        for name in ("fit_transform", "run", "fit"):
            if hasattr(model, name):
                result = getattr(model, name)(adata, **runner_kwargs)
                return adata if result is None else result

    raise AttributeError(
        "Could not find a supported VeloAgent entry point. Pass runner=callable "
        "that returns an AnnData/DataFrame or mutates the AnnData with pseudotime."
    )


def _extract_veloagent_result(result, fallback_adata):
    if result is None:
        return fallback_adata
    if _is_anndata(result) or isinstance(result, pd.DataFrame) or isinstance(result, str):
        return result
    if isinstance(result, (tuple, list)):
        for item in result:
            if _is_anndata(item):
                return item
        for item in result:
            if isinstance(item, pd.DataFrame):
                return item
    if isinstance(result, dict):
        for key in ("adata", "annData", "trajectory", "trajectory_info"):
            if key in result:
                return result[key]
        if "pseudotime" in result:
            fallback_adata.obs["pseudotime"] = result["pseudotime"]
            if "path" in result:
                fallback_adata.obs["path"] = result["path"]
            return fallback_adata
    return fallback_adata


def frequency_extract_veloagent(
    adata,
    dataset,
    runner=None,
    runner_kwargs=None,
    model_kwargs=None,
    time_key=None,
    path_key=None,
    path_id=1,
    expression_layer=None,
    n_bins=20,
    min_cells_per_bin=1,
    agg="mean",
    save_trajectory=True,
    trajectory_csv=None,
    output_npy=None,
    copy=True,
    return_intermediate=False,
):
    """
    Run VeloAgent trajectory inference and extract frequency features.

    VeloAgent's public Python API is still evolving, so this wrapper supports
    automatic discovery of common entry points and a custom
    ``runner(adata, **runner_kwargs)`` callable. The runner may return an
    AnnData, a trajectory DataFrame, or mutate ``adata.obs`` in place.
    """
    adata_va = adata.copy() if copy else adata
    result = _run_veloagent(
        adata_va,
        runner=runner,
        runner_kwargs=runner_kwargs,
        model_kwargs=model_kwargs,
    )
    trajectory_source = _extract_veloagent_result(result, adata_va)

    if _is_anndata(trajectory_source):
        time_key = _pick_existing_key(
            trajectory_source.obs, time_key, TIME_KEY_CANDIDATES, "time_key"
        )
        if time_key is None:
            raise ValueError(
                "VeloAgent output does not contain a recognized pseudotime column; "
                "pass time_key explicitly."
            )
        trajectory_info = frequency_extract_format(
            trajectory_source,
            dataset=dataset,
            time_key=time_key,
            path_key=path_key,
            expression_layer=_default_velocity_expression_layer(
                trajectory_source, expression_layer
            ),
            n_bins=n_bins,
            min_cells_per_bin=min_cells_per_bin,
            agg=agg,
            save_csv=save_trajectory,
            output_csv=trajectory_csv,
        )
        adata_for_frequency = trajectory_source
    else:
        trajectory_info = frequency_extract_format(
            trajectory_source,
            adata=adata_va,
            dataset=dataset,
            time_key=time_key,
            path_key=path_key,
            expression_layer=expression_layer,
            n_bins=n_bins,
            min_cells_per_bin=min_cells_per_bin,
            agg=agg,
            save_csv=save_trajectory,
            output_csv=trajectory_csv,
        )
        adata_for_frequency = adata_va

    res = frequency_extract(
        trajectory_info,
        adata_for_frequency,
        dataset,
        path_id=path_id,
        output_npy=output_npy,
    )
    if return_intermediate:
        return res, trajectory_info, trajectory_source
    return res


def frequency_extract_spatial(adata, dataset, start, end):
    spatial_coords = adata.obsm["spatial"]

    num_points = 10
    distances = np.linspace(0, 1, num_points)
    center_points = np.array([start + d * (end - start) for d in distances])

    trajectory_means = []
    for point in center_points:
        distances = pairwise_distances([point], spatial_coords)[0]
        nearest_indices = np.argsort(distances)[:100]
        mean_count = np.mean(_to_dense_array(adata.X[nearest_indices, :]), axis=0)
        trajectory_means.append(mean_count)

    trajectory_array = np.array(trajectory_means)
    x = np.arange(trajectory_array.shape[0])
    x_smooth = np.linspace(start=x.min(), stop=x.max(), num=100)
    y_smooth = make_interp_spline(x, trajectory_array, k=3)(x_smooth)

    wavelet = "mexh"
    scales = np.arange(1, 51)
    dt = x_smooth[1] - x_smooth[0]
    coeffs, _ = pywt.cwt(y_smooth.transpose(), scales, wavelet, sampling_period=dt)
    coeffs_thresh = pywt.threshold(coeffs, 0.01, mode="soft")

    yf_smooth = np.sum(coeffs_thresh, axis=2).T
    yf_info = np.abs(yf_smooth)
    yf_den = yf_info.max(axis=1)
    yf_den[yf_den == 0] = 1.0
    yf_info = yf_info / yf_den[:, None]

    y_info = y_smooth.transpose()
    y_den = y_info.max(axis=1)
    y_den[y_den == 0] = 1.0
    y_info = y_info / y_den[:, None]
    res = np.concatenate((y_info, yf_info), axis=1)

    pca_features = _pca_features_for_genes(adata, adata.var_names, n_pca=100)
    res = np.concatenate((res, pca_features), axis=1)

    np.save(dataset + ".npy", np.asarray(res[:, :]))
    return res
