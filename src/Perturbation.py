# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import torch
import torch.nn as nn
from scipy.interpolate import make_interp_spline
import pywt

from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops
from GCN_VAE import GCN_VAE




def Gene_Perturbation(
    control_traj_csv,
    pert_traj_csv,
    control_mtx_dir,
    pert_mtx_dir,
    out_csv="gene_perturbation.csv",
    gene_names=None,
    n_pca=100,
    wavelet="mexh",
    scales=np.arange(1, 51),
    threshold=0.1,
    pval_cutoff=0.05,
    graph=None,
    device=None,
):
    """
    KO at gene expression level (trajectory + mtx).
    Delta formula kept as: delta = 0.5*d1 + d2 + 0.5*d3
    but we keep only d2 by forcing d1=d3=0 (so delta==d2).
    Output: gene_name, pval_empirical (only p<pval_cutoff)
    """

    # ---- model path fixed here ----
    MODEL_PATH = "ALL_GCN_VAE.pth"

    # ---- device ----
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)

    # ---- load model ----
    model = GCN_VAE().to(torch_device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch_device))
    model.eval()

    # ---- helpers ----
    def load_adata(mtx_dir):
        X = sc.read_mtx(os.path.join(mtx_dir, "matrix.mtx")).X.T  # cells x genes
        obs = pd.read_csv(os.path.join(mtx_dir, "barcodes.csv"), index_col=0)
        var = pd.read_csv(os.path.join(mtx_dir, "features.csv"), index_col=0)
        return ad.AnnData(X, obs=obs, var=var)

    def pca_features(adata):
        a = adata.copy()
        sc.pp.normalize_total(a)
        sc.pp.log1p(a)
        sc.tl.pca(a, n_comps=n_pca, svd_solver="auto")
        return a.obsm["X_pca"]

    def traj_to_150(traj_df):
        if "path" not in traj_df.index or "time" not in traj_df.index:
            raise ValueError("trajectory csv must contain index rows: 'path' and 'time'")

        mask = (traj_df.loc["path"].values == 1)
        y = traj_df.iloc[:n_genes_traj, mask].to_numpy()
        x = traj_df.loc["time", mask].to_numpy()

        order = np.argsort(x)
        x = x[order]
        y = y[:, order]

        x_smooth = np.linspace(x.min(), x.max(), 100)
        y_smooth = make_interp_spline(x, y.T, k=3)(x_smooth).T  # (G,100)

        # time normalize per gene
        y_info = y_smooth.copy()
        for gi in range(y_info.shape[0]):
            m = y_info[gi].max()
            if m > 0:
                y_info[gi] = y_info[gi] / m

        # freq block
        base_cwt = pywt.cwt([1], scales, wavelet)[0]
        yf_info = np.zeros((y_info.shape[0], 50), dtype=float)

        for gi in range(y_info.shape[0]):
            coeffs, _ = pywt.cwt(y_info[gi], scales, wavelet)
            coeffs = pywt.threshold(coeffs, threshold, mode="soft")
            rec = np.sum(coeffs * np.conj(base_cwt), axis=0)
            v = np.abs(rec[:50])
            mv = v.max()
            if mv > 0:
                v = v / mv
            yf_info[gi] = v

        return np.concatenate([y_info, yf_info], axis=1)  # (G,150)

    def build_data(feats_np):
        x = torch.tensor(np.float32(feats_np))
        if graph:
            edge_index = torch.tensor(np.load(graph).transpose(), dtype=torch.long)
            data = Data(x=x, edge_index=edge_index)
            data.edge_index = add_self_loops(data.edge_index)[0]
        else:
            data = Data(x=x)
        return data

    def to_latent_z(feats_np):
        data = build_data(feats_np).to(torch_device)
        with torch.no_grad():
            _, _, _, z = model(data)
        return z.detach().cpu().numpy()

    def centroid_dist(A, B):
        return float(np.linalg.norm(A.mean(axis=0) - B.mean(axis=0)))

    def knockout_traj(traj_df, gi, gname):
        df = traj_df.copy()
        if gname in df.index:
            df.loc[gname, :] = 0.0
        elif gi < df.shape[0]:
            df.iloc[gi, :] = 0.0
        return df

    def knockout_adata(adata_in, gi, gname):
        a = adata_in.copy()
        if gname in a.var_names:
            j = int(np.where(a.var_names == gname)[0][0])
        else:
            j = gi

        X = a.X
        if hasattr(X, "tocsc"):
            Xc = X.tocsc(copy=True)
            Xc[:, j] = 0.0
            a.X = Xc.tocsr()
        else:
            a.X[:, j] = 0.0
        return a

    # ---- load data ----
    traj_control = pd.read_csv(control_traj_csv, index_col=0)
    traj_base = pd.read_csv(pert_traj_csv, index_col=0)
    # ===== infer gene list & gene count automatically =====
    all_rows = traj_base.index.tolist()
    gene_names_auto = [g for g in all_rows if g not in ("time", "path")]

    if gene_names is None:
        gene_names = gene_names_auto

    n_genes_traj = len(gene_names)

    adata_control = load_adata(control_mtx_dir)
    adata_base = load_adata(pert_mtx_dir)

    # ---- baseline control latent ----
    feats_control = np.concatenate([traj_to_150(traj_control), pca_features(adata_control)], axis=1)
    z_control = to_latent_z(feats_control)

    # ---- baseline pert latent (unperturbed) ----
    feats_base = np.concatenate([traj_to_150(traj_base), pca_features(adata_base)], axis=1)
    z_base = to_latent_z(feats_base)

    # ---- baseline distances (compute all, but will use only d2) ----
    dist_base_1 = centroid_dist(z_base[:100], z_control[:100])
    dist_base_2 = centroid_dist(z_base[100:150], z_control[100:150])
    dist_base_3 = centroid_dist(z_base[150:], z_control[150:])

    # ---- genes ----
    if gene_names is None:
        # avoid time/path if they are in index; keep first n_genes_traj rows of the matrix block
        gene_names = traj_base.index.tolist()[:n_genes_traj]
    gene_names = list(gene_names)

    # ---- KO loop ----
    rows = []
    for gi, gname in enumerate(gene_names):
        traj_ko = knockout_traj(traj_base, gi, gname)
        adata_ko = knockout_adata(adata_base, gi, gname)

        try:
            feats_ko = np.concatenate([traj_to_150(traj_ko), pca_features(adata_ko)], axis=1)
        except Exception as e:
            print(f"[WARN] skip {gname}: {e}")
            continue

        z_ko = to_latent_z(feats_ko)

        dist_pert_1 = centroid_dist(z_ko[:100], z_control[:100])
        dist_pert_2 = centroid_dist(z_ko[100:150], z_control[100:150])
        dist_pert_3 = centroid_dist(z_ko[150:], z_control[150:])

        # distances deltas
        d1 = dist_pert_1 - dist_base_1
        d2 = dist_pert_2 - dist_base_2
        d3 = dist_pert_3 - dist_base_3


        delta =  d2

        rows.append({"gene_name": gname, "delta_dist": float(delta)})

    df = pd.DataFrame(rows)

    # ---- empirical p-value from |delta| ranking ----
    abs_vals = np.abs(df["delta_dist"].to_numpy())
    N = len(abs_vals)
    if N == 0:
        out = pd.DataFrame(columns=["gene_name", "pval_empirical"])
        out.to_csv(out_csv, index=False)
        print(f"[DONE] saved empty {out_csv}")
        return out

    sorted_abs = np.sort(abs_vals)[::-1]
    idx = np.searchsorted(-sorted_abs, -abs_vals, side="left")
    df["pval_empirical"] = (idx + 1) / N

    # ---- output: only p<cutoff, only two columns ----
    out = (
        df.loc[df["pval_empirical"] < pval_cutoff, ["gene_name", "pval_empirical"]]
        .sort_values("pval_empirical", ascending=True)
        .reset_index(drop=True)
    )

    out.to_csv(out_csv, index=False)
    print(f"[DONE] saved {out_csv} with {len(out)} genes (p<{pval_cutoff})")
    return out


# -*- coding: utf-8 -*-
"""
Unified pathway/drug multi-gene perturbation runner.

This script is condensed into a single public function:
    Pathway_Drug_Perturbation(...)

It supports:
- Trajectory + expression multiplicative perturbation (direction-aware: up/down/default)
- Empirical null distribution with fixed n_med (median term size) and disk cache
- Optional parallel background computation via joblib (disabled by default on GPU)
- Optional graph input for torch_geometric Data construction:
      if graph:
          edge_index = torch.tensor(np.load(graph).transpose(), dtype=torch.long)
          data = Data(x=x, edge_index=edge_index)
          data.edge_index = add_self_loops(data.edge_index)[0]
      else:
          data = Data(x=x)

Requirements:
- A file "GCN_VAE.py" in the same directory exporting class GCN_VAE
- model_path points to the saved weights (e.g., ALL_GCN_VAE.pth)
- mtx directories contain: matrix.mtx, barcodes.csv, features.csv
- trajectory CSVs contain rows named "time" and "path" in index
"""

import os
import re
import json
import hashlib
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import torch
import pywt
from scipy.interpolate import make_interp_spline
from joblib import Parallel, delayed
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops


def Pathway_Drug_Perturbation(
    control_traj_csv,
    pert_traj_csv,
    control_mtx_dir,
    pert_mtx_dir,
    model_path="ALL_GCN_VAE.pth",
    term_profile_path=None,
    cmap_df=None,
    out_csv="term_multigene_diraware_perturb_with_empirical_FDR.csv",
    cache_dir="./.perturb_cache",
    uns_prefix="data",
    from_uns="term",
    keep_sign=True,
    sign_default=+1,
    factor_up=2.0,
    factor_down=0.5,
    factor_default=0.0,
    rng_seed=20251030,
    n_perm=1000,
    n_jobs=5,
    graph=None,
    wavelet="mexh",
    scales=np.arange(1, 128),
    threshold=0.1,
    pca_n_comps=100,
    verbose=True,
):
    """
    Unified multi-gene perturbation framework for both pathways and drug signatures.

    This function evaluates the effect of coordinated perturbation of a gene set
    (e.g. biological pathway or drug-induced signature) by:
      1) multiplicatively perturbing gene expression and trajectory profiles,
      2) projecting perturbed data into a pretrained latent space,
      3) measuring centroid shifts relative to baseline,
      4) estimating empirical p-values using a size-matched random-gene null.

    Parameters
    ----------
    control_traj_csv : str
        Trajectory CSV for control baseline (must include index rows 'time' and 'path').
    pert_traj_csv : str
        Trajectory CSV for perturbed/baseline condition (e.g. all_PDAC) used for perturbation runs.
    control_mtx_dir : str
        Directory containing control matrix.mtx, barcodes.csv, features.csv.
    pert_mtx_dir : str
        Directory containing perturbed/all matrix.mtx, barcodes.csv, features.csv.
    model_path : str
        Path to pretrained model weights (state_dict), e.g. ALL_GCN_VAE.pth.
    term_profile_path : str or None
        Path to a numpy file loadable by np.load(..., allow_pickle=True) that yields
        a dict: {term_name: [ "GENE", "GENE:+", "GENE:-", "GENE:up", "GENE:down", ... ] }.
        This can represent drug signatures or pathway gene sets.
    cmap_df : pandas.DataFrame or None
        Alternative term definition: DataFrame with index=term_name and columns=genes.
        If used, directions default to sign_default.
    out_csv : str
        Output CSV path.
    cache_dir : str
        Directory for caching null distribution (npz) by configuration hash.
    uns_prefix : str
        Prefix used when storing overlap sets in adata.uns (internal).
    from_uns : str
        A label to store term overlap keys in adata.uns (internal).
    keep_sign : bool
        Whether to keep directional annotations when provided (GENE:+/-).
    sign_default : int
        Default sign (+1 or -1) when sign is absent or not parseable.
    factor_up, factor_down, factor_default : float
        Multiplicative perturbation factors for +, -, and unknown direction respectively.
        Note: setting factor_default=0.0 makes unknown direction behave like knockout.
    rng_seed : int
        Random seed for reproducible null draws.
    n_perm : int
        Number of permutations for the empirical null distribution.
    n_jobs : int
        Parallel jobs for background null computation (joblib). On GPU, it will be forced to 1.
    graph : str or None
        If provided, a .npy path storing edge_index with shape (E,2) or (2,E) after transpose.
        Data will be created with edges and self-loops as required.
    wavelet, scales, threshold : wavelet configuration
        Must match your training/preprocessing settings for reproducibility.
    pca_n_comps : int
        Number of PCA components appended to trajectory features.
    verbose : bool
        Print progress logs.

    Returns
    -------
    pandas.DataFrame
        Results table with columns:
        pathway (term), k_genes, genes_used, delta_dist, pval_empirical, qval_bh, neglog10_q
    """

    # ---------------------------------------------------------
    # Basic validations
    # ---------------------------------------------------------
    if (term_profile_path is None) and (cmap_df is None):
        raise ValueError("Provide either term_profile_path (np file) or cmap_df (DataFrame).")

    # ---------------------------------------------------------
    # Utilities (local)
    # ---------------------------------------------------------
    def _log(msg):
        if verbose:
            print(msg)

    def _parse_sign_token(tok: str):
        t = tok.strip().lower()
        if t in {"+", "up", "1", "pos", "positive"}:
            return +1
        if t in {"-", "down", "-1", "neg", "negative"}:
            return -1
        return None

    def _bh_fdr(p):
        p = np.asarray(p, dtype=float)
        m = p.size
        order = np.argsort(p)
        ranked = p[order]
        q = ranked * m / (np.arange(m) + 1)
        q = np.minimum.accumulate(q[::-1])[::-1]
        q = np.clip(q, 0.0, 1.0)
        out = np.empty_like(q)
        out[order] = q
        return out

    def _stable_hash(obj) -> str:
        s = json.dumps(obj, sort_keys=True, default=str)
        return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]

    def _save_npz(path, **arrays):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez_compressed(path, **arrays)

    def _load_npz(path):
        with np.load(path, allow_pickle=False) as data:
            return {k: data[k] for k in data.files}

    def _load_adata_from_dir(mtx_dir: str) -> ad.AnnData:
        X = sc.read_mtx(os.path.join(mtx_dir, "matrix.mtx")).X.T  # cells x genes
        obs = pd.read_csv(os.path.join(mtx_dir, "barcodes.csv"), sep=",", index_col=0)
        var = pd.read_csv(os.path.join(mtx_dir, "features.csv"), header=0, index_col=0)
        return ad.AnnData(X, obs=obs, var=var)

    def _recompute_pca_block(adata_in_T: ad.AnnData, n_comps: int) -> np.ndarray:
        a = adata_in_T.copy()
        sc.pp.normalize_total(a)
        sc.pp.log1p(a)
        sc.tl.pca(a, n_comps=n_comps, svd_solver="auto")
        return a.obsm["X_pca"]

    def _traj_features(traj_df: pd.DataFrame):
        if "path" not in traj_df.index or "time" not in traj_df.index:
            raise ValueError("Trajectory CSV must contain index rows: 'time' and 'path'.")

        gene_rows = [g for g in traj_df.index.tolist() if g not in ("time", "path")]

        a = pd.Index(traj_df.loc["path"] == 1)
        y = traj_df.loc[gene_rows, :].iloc[:, a].to_numpy()  # (G, M)
        x = traj_df.loc["time", a].to_numpy()                # (M,)

        order = np.argsort(x)
        x = x[order]
        y = y[:, order]

        x_smooth = np.linspace(x.min(), x.max(), 100)
        y_smooth = make_interp_spline(x, y.T, k=3)(x_smooth)  # (100, G)

        coeffs, _ = pywt.cwt(y_smooth, scales, wavelet)       # (S, 100, G)
        coeffs = pywt.threshold(coeffs, threshold, mode="soft")
        base_cwt = pywt.cwt([1], scales, wavelet)[0]          # (S, 1)

        yf_smooth = np.zeros_like(y_smooth)                   # (100, G)
        for i in range(y_smooth.shape[0]):
            yf_smooth[i] = np.sum(coeffs[:, i] * np.conj(base_cwt), axis=0)

        yf_info = np.abs(yf_smooth[:50, :]).T                 # (G, 50)
        denom = yf_info.max(axis=1)
        denom[denom == 0] = 1.0
        yf_info = yf_info / denom[:, None]

        y_info = y_smooth.T                                   # (G, 100)
        denom2 = y_info.max(axis=1)
        denom2[denom2 == 0] = 1.0
        y_info = y_info / denom2[:, None]

        res150 = np.concatenate([y_info, yf_info], axis=1)     # (G, 150)
        return res150, gene_rows

    def _features_concat(res150, pca100):
        return np.concatenate([res150, pca100], axis=1)

    def _make_data(x):
        # Keep the exact logic requested by the user.
        if graph:
            edge_index = torch.tensor(np.load(graph).transpose(), dtype=torch.long)
            data = Data(x=x, edge_index=edge_index)
            data.edge_index = add_self_loops(data.edge_index)[0]
        else:
            data = Data(x=x)
        return data

    def _to_latent_z(feats_np):
        x = torch.tensor(np.float32(feats_np))
        data = _make_data(x).to(device)
        with torch.no_grad():
            _, _, _, z = gcn_vae(data)
        return z.detach().cpu().numpy()

    def _centroid_dist(A, B):
        return float(np.linalg.norm(A.mean(axis=0) - B.mean(axis=0)))

    def _pick_factor_from_sign(s: int) -> float:
        if s > 0:
            return float(factor_up)
        if s < 0:
            return float(factor_down)
        return float(factor_default)

    def _perturb_adata_columns(adata_cells_genes: ad.AnnData, col_idx, signs):
        adx = adata_cells_genes.copy()
        X = adx.X
        if hasattr(X, "tocsc"):
            Xc = X.tocsc(copy=True)
            for j, s in zip(col_idx, signs):
                fac = _pick_factor_from_sign(int(s))
                start, end = Xc.indptr[j], Xc.indptr[j + 1]
                if end > start:
                    Xc.data[start:end] *= fac
            adx.X = Xc.tocsr()
        else:
            for j, s in zip(col_idx, signs):
                fac = _pick_factor_from_sign(int(s))
                adx.X[:, j] = adx.X[:, j] * fac
        return adx

    def _perturb_traj_multi(traj_df: pd.DataFrame, genes, signs):
        df = traj_df.copy()
        for g, s in zip(genes, signs):
            fac = _pick_factor_from_sign(int(s))
            hit = False
            if g in df.index:
                df.loc[g, :] = df.loc[g, :] * fac
                hit = True
            if (not hit) and (g in df.columns):
                df.loc[:, g] = df.loc[:, g] * fac
        return df

    def _forward_once(traj_df_pert, adata_T_pert):
        res150, _ = _traj_features(traj_df_pert)
        pca100 = _recompute_pca_block(adata_T_pert, n_comps=pca_n_comps)
        feats = _features_concat(res150, pca100)
        return _to_latent_z(feats)

    # ---------------------------------------------------------
    # Load model
    # ---------------------------------------------------------
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device_obj = torch.device(device_str)
    if device_obj != device:
        device = device_obj  # ensure consistent
    _log(f"[INFO] device={device}")

    from GCN_VAE import GCN_VAE  # same folder
    gcn_vae = GCN_VAE().to(device)
    gcn_vae.load_state_dict(torch.load(model_path, map_location=device))
    gcn_vae.eval()

    # ---------------------------------------------------------
    # Load baseline inputs
    # ---------------------------------------------------------
    traj_ctrl = pd.read_csv(control_traj_csv, index_col=0)
    traj_base = pd.read_csv(pert_traj_csv, index_col=0)

    adata_ctrl_T = _load_adata_from_dir(control_mtx_dir).T
    adata_base_T = _load_adata_from_dir(pert_mtx_dir).T

    # cells x genes version (for sparse column updates by var index)
    adata_base_cells_genes = _load_adata_from_dir(pert_mtx_dir)

    # ---------------------------------------------------------
    # Compute z_control and z_base (baseline)
    # ---------------------------------------------------------
    res_ctrl_150, ctrl_gene_rows = _traj_features(traj_ctrl)
    res_base_150, base_gene_rows = _traj_features(traj_base)

    pca_ctrl = _recompute_pca_block(adata_ctrl_T, n_comps=pca_n_comps)
    pca_base = _recompute_pca_block(adata_base_T, n_comps=pca_n_comps)

    feats_ctrl = _features_concat(res_ctrl_150, pca_ctrl)
    feats_base = _features_concat(res_base_150, pca_base)

    z_control = _to_latent_z(feats_ctrl)
    z_base = _to_latent_z(feats_base)

    dist_base_1 = _centroid_dist(z_base[:100], z_control[:100])
    dist_base_2 = _centroid_dist(z_base[100:150], z_control[100:150])
    dist_base_3 = _centroid_dist(z_base[150:], z_control[150:])

    _log(f"[INFO] baseline dists: {dist_base_1:.4f}, {dist_base_2:.4f}, {dist_base_3:.4f}")

    # ---------------------------------------------------------
    # Build term->genes and term->signs from input (profile file or cmap_df)
    # ---------------------------------------------------------
    def _calculate_term_sets(adata_in, cmap_df_in=None, profile_path=None):
        genenames = set(map(str, adata_in.var.index.tolist()))

        if profile_path is not None:
            term_dict = dict(np.load(profile_path, allow_pickle=True).tolist())

            per_term_genes = {}
            per_term_signs = {}
            for term in list(term_dict.keys()):
                genes = []
                signs = {}
                for item in term_dict[term]:
                    if not isinstance(item, str):
                        item = str(item)
                    gene, sep, tail = item.partition(":")
                    gene = gene.strip()
                    if gene in genenames:
                        genes.append(gene)
                        if keep_sign and sep == ":" and tail.strip():
                            s = _parse_sign_token(tail)
                            if s is not None:
                                signs[gene] = s
                genes = sorted(set(genes))
                if genes:
                    per_term_genes[term] = genes
                    if keep_sign:
                        per_term_signs[term] = {g: signs.get(g, sign_default) for g in genes}
                    else:
                        per_term_signs[term] = {g: sign_default for g in genes}

            # Merge identical gene sets to match your original behavior
            tmp = {}
            tmp_signs = {}
            for term, glist in per_term_genes.items():
                key_genes = "!".join(glist)
                tmp.setdefault(key_genes, []).append(term)
                tmp_signs.setdefault(key_genes, []).append(per_term_signs[term])

            out_genes = {}
            out_signs = {}
            for gene_key, tlist in tmp.items():
                genes = gene_key.split("!")
                if keep_sign:
                    votes = {g: 0 for g in genes}
                    for sd in tmp_signs[gene_key]:
                        for g in genes:
                            votes[g] += int(sd.get(g, sign_default))
                    merged_signs = {g: (1 if votes[g] >= 0 else -1) for g in genes}
                else:
                    merged_signs = {g: sign_default for g in genes}

                out_key = ",".join(tlist)
                out_genes[out_key] = genes
                out_signs[out_key] = merged_signs

            adata_in.uns[f"{uns_prefix}_{from_uns}_overlap_genes"] = out_genes
            adata_in.uns[f"{uns_prefix}_{from_uns}_overlap_signs"] = out_signs
            adata_in.uns["term-gene_len_dict"] = {k: len(v) for k, v in out_genes.items()}
            return adata_in

        if cmap_df_in is not None:
            cmap_df_in = cmap_df_in[cmap_df_in.columns.intersection(list(genenames))]
            cols = cmap_df_in.columns.tolist()
            rows = cmap_df_in.index.tolist()
            adata_in.uns[f"{uns_prefix}_{from_uns}_overlap_genes"] = {row: cols for row in rows}
            adata_in.uns[f"{uns_prefix}_{from_uns}_overlap_signs"] = {row: {g: sign_default for g in cols} for row in rows}
            adata_in.uns["term-gene_len_dict"] = {row: len(cols) for row in rows}
            return adata_in

        raise ValueError("Provide either term_profile_path or cmap_df.")

    def _build_sets_from_uns(adata_in):
        key_g = f"{uns_prefix}_{from_uns}_overlap_genes"
        key_s = f"{uns_prefix}_{from_uns}_overlap_signs"
        if key_g not in adata_in.uns:
            raise KeyError(f"Missing adata.uns['{key_g}']")
        if key_s not in adata_in.uns:
            raise KeyError(f"Missing adata.uns['{key_s}']")
        return adata_in.uns[key_g], adata_in.uns[key_s]

    _calculate_term_sets(adata_base_cells_genes, cmap_df_in=cmap_df, profile_path=term_profile_path)
    term2genes, term2signs = _build_sets_from_uns(adata_base_cells_genes)

    # ---------------------------------------------------------
    # Universe construction (intersection of trajectory genes and adata vars)
    # ---------------------------------------------------------
    traj_gene_list = [g for g in traj_base.index.tolist() if g not in ("time", "path")]
    traj_gene_list = list(map(str, traj_gene_list))

    gene_to_pos = {g: i for i, g in enumerate(traj_gene_list)}
    varname_set = set(map(str, adata_base_cells_genes.var_names))

    universe = [g for g in traj_gene_list if g in varname_set]
    universe_idx = np.array([gene_to_pos[g] for g in universe], dtype=int)

    # median term size for background
    sizes = [len(v) for v in term2genes.values() if v]
    if len(sizes) == 0:
        raise ValueError("No terms with genes after overlap/intersection.")
    n_med = int(np.median(sizes))
    if n_med < 1:
        raise RuntimeError("Median term size (n_med) < 1.")

    # ---------------------------------------------------------
    # Background null distribution with cache
    # ---------------------------------------------------------
    cfg = dict(
        rng=int(rng_seed),
        n_perm=int(n_perm),
        n_med=int(n_med),
        universe=int(len(universe_idx)),
        factor_up=float(factor_up),
        factor_down=float(factor_down),
        factor_default=float(factor_default),
        base=[float(dist_base_1), float(dist_base_2), float(dist_base_3)],
        graph=str(graph) if graph else None,
        wavelet=str(wavelet),
        threshold=float(threshold),
        pca_n_comps=int(pca_n_comps),
    )
    cfg_hash = _stable_hash(cfg)
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"bg_{cfg_hash}.npz")

    rng = np.random.default_rng(int(rng_seed))

    def _bg_single(draw_idx, draw_signs):
        draw_genes = [traj_gene_list[i] for i in draw_idx]
        traj_b = _perturb_traj_multi(traj_base, draw_genes, draw_signs)

        # map to var indices (safe: universe genes are in var)
        var_idx = [int(np.where(adata_base_cells_genes.var_names == g)[0][0]) for g in draw_genes]
        ad_b_T = _perturb_adata_columns(adata_base_cells_genes, var_idx, draw_signs).T

        try:
            z_b = _forward_once(traj_b, ad_b_T)
            d1 = _centroid_dist(z_b[:100], z_control[:100])
            d2 = _centroid_dist(z_b[100:150], z_control[100:150])
            d3 = _centroid_dist(z_b[150:], z_control[150:])
            delta_b = 0.5 * (d1 - dist_base_1) + (d2 - dist_base_2) + 0.5 * (d3 - dist_base_3)
        except Exception:
            delta_b = 0.0
        return float(delta_b)

    if os.path.exists(cache_path):
        delta_null = np.asarray(_load_npz(cache_path)["delta_null"], dtype=float)
        _log(f"[INFO] Loaded background from cache: {cache_path} (n={len(delta_null)})")
    else:
        if universe_idx.size < n_med:
            raise RuntimeError(f"Universe smaller than n_med ({universe_idx.size} < {n_med}).")

        draws = [rng.choice(universe_idx, size=n_med, replace=False) for _ in range(int(n_perm))]
        signs_list = [rng.choice([-1, 1], size=n_med) for _ in range(int(n_perm))]

        use_jobs = int(n_jobs)
        if torch.cuda.is_available():
            use_jobs = 1  # avoid multi-process GPU contention

        _log(f"[INFO] Computing background: n_perm={n_perm}, n_med={n_med}, n_jobs={use_jobs}")
        if use_jobs <= 1:
            delta_null = np.array([_bg_single(d, s) for d, s in zip(draws, signs_list)], dtype=float)
        else:
            delta_null = np.array(
                Parallel(n_jobs=use_jobs, backend="loky")(
                    delayed(_bg_single)(d, s) for d, s in zip(draws, signs_list)
                ),
                dtype=float,
            )
        _save_npz(cache_path, delta_null=delta_null)
        _log(f"[INFO] Saved background cache: {cache_path}")

    # ---------------------------------------------------------
    # Run each term (pathway/drug)
    # ---------------------------------------------------------
    records = []

    def _term_run(term, genes, sign_map):
        genes = [str(g) for g in genes]
        genes = [g for g in genes if (g in gene_to_pos) and (g in varname_set)]
        if not genes:
            return None

        signs = []
        for g in genes:
            if (sign_map is None) or (not keep_sign):
                s = 0
            else:
                s = int(np.sign(sign_map.get(g, 0)))
            signs.append(s)
        signs = np.asarray(signs, dtype=int)

        try:
            traj_pert = _perturb_traj_multi(traj_base, genes, signs)
            var_idx = [int(np.where(adata_base_cells_genes.var_names == g)[0][0]) for g in genes]
            ad_pert_T = _perturb_adata_columns(adata_base_cells_genes, var_idx, signs).T

            z_pert = _forward_once(traj_pert, ad_pert_T)

            d1 = _centroid_dist(z_pert[:100], z_control[:100])
            d2 = _centroid_dist(z_pert[100:150], z_control[100:150])
            d3 = _centroid_dist(z_pert[150:], z_control[150:])
            delta = d2 - dist_base_2
        except Exception:
            return None

        abs_obs = abs(delta)
        ge = np.count_nonzero(np.abs(delta_null) >= abs_obs)
        p_emp = (ge + 1) / (len(delta_null) + 1)

        return dict(
            pathway=str(term),
            k_genes=int(len(genes)),
            genes_used=",".join(genes),
            delta_dist=float(delta),
            pval_empirical=float(p_emp),
        )

    for term, genes in term2genes.items():
        sign_map = term2signs.get(term, None)
        rec = _term_run(term, genes, sign_map)
        if rec is not None:
            records.append(rec)

    res_df = pd.DataFrame.from_records(records)
    if res_df.empty:
        _log("[WARN] No results produced. Saving empty CSV.")
        res_df.to_csv(out_csv, index=False)
        return res_df

    res_df["qval_bh"] = _bh_fdr(res_df["pval_empirical"].values)
    eps = 1e-300
    res_df["neglog10_q"] = -np.log10(np.clip(res_df["qval_bh"].values, eps, 1.0))
    res_df = res_df.sort_values(["qval_bh", "pval_empirical", "delta_dist"],
                                ascending=[True, True, True]).reset_index(drop=True)

    res_df.to_csv(out_csv, index=False)
    _log(f"[SAVE] {out_csv}")
    return res_df


# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from typing import Sequence, Tuple, Optional, Union, Dict, Any


def Survival(
    bulk_csv: str,
    gene_list: Union[str, Sequence[str]],
    duration_col: str = "duration",
    event_col: str = "event",
    event_mapping: Optional[Dict[Any, int]] = None,
    data_name: str = "DATA",
    high_risk_label: str = "High risk",
    low_risk_label: str = "Low risk",
    colors: Tuple[str, str] = ("tab:red", "tab:gray"),
    fontsize: int = 12,
    quantiles: Tuple[float, float] = (0.25, 0.75),
    save_pdf_path: Optional[str] = None,
    export_classified_csv_path: Optional[str] = None,
    export_significant_csv_path: Optional[str] = None,
    show: bool = True,
) -> Tuple[plt.Figure, plt.Axes, Dict[str, Any]]:
    """
    Kaplan-Meier survival analysis using a multi-gene signature score from bulk RNA expression.

    Workflow
    --------
    1) Load bulk CSV and intersect with provided gene list.
    2) Z-score normalize each gene across samples.
    3) Compute signature score per sample as the mean of normalized gene expression.
    4) Stratify samples using quantile cutoffs:
          score <= q_low  -> low_risk_label
          score >= q_high -> high_risk_label
          otherwise       -> "Middle" (excluded from KM/log-rank)
    5) Run log-rank test and plot KM curves for High vs Low.

    Parameters
    ----------
    bulk_csv : str
        Bulk expression CSV with genes as columns and clinical columns (duration/event).
    gene_list : str or sequence of str
        Either a path to a text file listing genes (one per line) or a list/tuple/set of genes.
    duration_col : str
        Column name for survival time.
    event_col : str
        Column name for event indicator (0/1).
    event_mapping : dict or None
        Optional mapping to convert event labels (e.g. {"dead":1,"alive":0}).
    data_name : str
        Display name for legend title.
    high_risk_label : str
        Label for high group in plot.
    low_risk_label : str
        Label for low group in plot.
    colors : (str, str)
        Line colors for (high, low).
    fontsize : int
        Font size.
    quantiles : (float, float)
        (q_low, q_high), default (0.25, 0.75).
    save_pdf_path : str or None
        If provided, save figure as PDF.
    export_classified_csv_path : str or None
        If provided, export per-sample classification table (includes Middle).
    export_significant_csv_path : str or None
        If provided, export only samples in High/Low groups (excludes Middle).
    show : bool
        Whether to show the plot.

    Returns
    -------
    fig, ax, results : (Figure, Axes, dict)
        results contains p_value, cutoffs, group_counts, risk_scores, classified_df.
    """

    def _load_gene_list(x: Union[str, Sequence[str]]) -> list:
        if isinstance(x, str) and os.path.exists(x):
            with open(x, "r", encoding="utf-8") as f:
                return [ln.strip() for ln in f if ln.strip()]
        if isinstance(x, (list, tuple, set)):
            return list(x)
        raise ValueError("gene_list must be a list/tuple/set of gene names or a text file path.")

    # --- Load data ---
    df = pd.read_csv(bulk_csv)
    genes = _load_gene_list(gene_list)

    missing_cols = [c for c in (duration_col, event_col) if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns in CSV: {missing_cols}")

    genes_in = [g for g in genes if g in df.columns]
    if not genes_in:
        raise ValueError("No overlap between gene_list and CSV columns.")

    use_df = df.loc[:, genes_in + [duration_col, event_col]].copy()

    # --- Event mapping ---
    if event_mapping is not None:
        mapping_norm = {str(k).strip().lower(): int(v) for k, v in event_mapping.items()}
        use_df[event_col] = use_df[event_col].apply(lambda x: mapping_norm.get(str(x).strip().lower(), x))

    # --- Coerce to numeric ---
    for col in genes_in + [duration_col, event_col]:
        use_df[col] = pd.to_numeric(use_df[col], errors="coerce")
    use_df = use_df.dropna(subset=genes_in + [duration_col, event_col]).reset_index(drop=True)

    # --- Validate event values ---
    unique_events = set(use_df[event_col].unique().tolist())
    if not unique_events.issubset({0, 1}):
        raise ValueError(f"{event_col} must be 0/1. Found: {sorted(unique_events)}")

    # --- Z-score normalize genes across samples ---
    gene_mat = use_df.loc[:, genes_in]
    gene_mean = gene_mat.mean(axis=0)
    gene_std = gene_mat.std(axis=0, ddof=0).replace(0, 1.0)
    use_df.loc[:, genes_in] = (gene_mat - gene_mean) / gene_std

    # --- Risk score: mean of z-scored genes ---
    risk_scores = use_df.loc[:, genes_in].mean(axis=1).to_numpy()

    q_low, q_high = float(quantiles[0]), float(quantiles[1])
    if not (0.0 < q_low < q_high < 1.0):
        raise ValueError("quantiles must satisfy 0 < q_low < q_high < 1.")

    cutoff_low = float(np.quantile(risk_scores, q_low))
    cutoff_high = float(np.quantile(risk_scores, q_high))

    risk_group = np.array(["Middle"] * len(risk_scores), dtype=object)
    risk_group[risk_scores <= cutoff_low] = low_risk_label
    risk_group[risk_scores >= cutoff_high] = high_risk_label

    classified = use_df.copy()
    classified["risk"] = risk_group
    classified["score"] = risk_scores

    low = classified[classified["risk"] == low_risk_label]
    high = classified[classified["risk"] == high_risk_label]

    if len(low) == 0 or len(high) == 0:
        raise ValueError(
            f"One group is empty after quantile split: "
            f"{low_risk_label}={len(low)}, {high_risk_label}={len(high)}. "
            f"Try different quantiles, e.g. (0.30, 0.70)."
        )

    # --- Log-rank test ---
    lr = logrank_test(
        durations_A=low[duration_col],
        durations_B=high[duration_col],
        event_observed_A=low[event_col],
        event_observed_B=high[event_col],
    )
    p_value = float(lr.p_value)

    # --- KM plot ---
    fig, ax = plt.subplots(figsize=(8, 6))
    kmf = KaplanMeierFitter()

    kmf.fit(high[duration_col], event_observed=high[event_col], label=high_risk_label)
    kmf.plot_survival_function(ax=ax, ci_show=False, linewidth=2.5, color=colors[0])

    kmf.fit(low[duration_col], event_observed=low[event_col], label=low_risk_label)
    kmf.plot_survival_function(ax=ax, ci_show=False, linewidth=2.5, color=colors[1])

    ax.set_xlabel("Time", fontsize=fontsize)
    ax.set_ylabel("Survival Probability", fontsize=fontsize)
    ax.tick_params(axis="both", labelsize=fontsize)
    ax.set_ylim(0, 1)

    legend_labels = [data_name, high_risk_label, low_risk_label]
    legend_handles = [plt.Line2D([0], [0], color="w", label=legend_labels[0])] + [
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=c, markersize=8, label=lab)
        for c, lab in zip(colors, legend_labels[1:])
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=3,
        frameon=False,
        fontsize=fontsize,
    )

    ax.text(
        0.5,
        1.02,
        f"P = {p_value:.2e}",
        transform=ax.transAxes,
        ha="center",
        fontsize=fontsize,
        fontstyle="italic",
        weight="bold",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    # --- Optional exports ---
    if save_pdf_path is not None:
        out_pdf = save_pdf_path if save_pdf_path.lower().endswith(".pdf") else f"{save_pdf_path}.pdf"
        fig.savefig(out_pdf, bbox_inches="tight")

    if export_classified_csv_path is not None:
        classified.to_csv(export_classified_csv_path, index=False)

    if export_significant_csv_path is not None:
        classified[classified["risk"].isin([low_risk_label, high_risk_label])].to_csv(
            export_significant_csv_path, index=False
        )

    if show:
        plt.show()
    else:
        plt.close(fig)

    results = {
        "p_value": p_value,
        "cutoffs": {f"q{int(q_low*100)}": cutoff_low, f"q{int(q_high*100)}": cutoff_high},
        "group_counts": {
            high_risk_label: int((risk_group == high_risk_label).sum()),
            low_risk_label: int((risk_group == low_risk_label).sum()),
            "Middle": int((risk_group == "Middle").sum()),
        },
        "risk_scores": risk_scores,
        "classified_df": classified,
        "genes_used": genes_in,
    }
    return fig, ax, results





