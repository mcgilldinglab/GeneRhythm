# -*- coding: utf-8 -*-

from __future__ import print_function

import mygene
from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline
from sklearn.cluster import KMeans
from scipy.fftpack import fft

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy import stats
import seaborn as sns
import pywt

# ====== NEW: MusicXML deps ======
from music21 import stream, note as m21note, tempo, meter, instrument, dynamics
import os


# ============================================================
# NEW: Power -> note events (3 voices) and MusicXML writer
# ============================================================
def _energy_to_dynamic_mark(energy):
    if energy < 0.15:
        return "p"
    if energy < 0.30:
        return "mp"
    if energy < 0.55:
        return "mf"
    if energy < 0.80:
        return "f"
    return "ff"


def power_to_note_events(
    power,
    energy_threshold=0.60,
    low_midi=(45, 48, 50, 52, 55),
    mid_midi=(57, 59, 60, 62, 64, 67),
    high_midi=(69, 71, 72, 74, 76),
):
    """
    Convert wavelet power (n_scales, n_time) into note events for 3 voices.
    Returns: note_events dict + total_frames
    note_events[voice] = list of (start_frame, dur_frames, midi_pitch, mean_energy_norm)
    """
    power = np.asarray(power, dtype=float)
    power[power < 0] = 0.0
    n_scales, n_time = power.shape

    low_m = np.array(low_midi, dtype=int)
    mid_m = np.array(mid_midi, dtype=int)
    high_m = np.array(high_midi, dtype=int)

    # Split scales into 3 bands by index (compatible with your original code)
    n = n_scales
    idx_high = np.arange(0, max(1, n // 3))
    idx_mid = np.arange(max(1, n // 3), max(2, 2 * n // 3))
    idx_low = np.arange(max(2, 2 * n // 3), n)

    bands = {
        "high": (idx_high, high_m),
        "mid": (idx_mid, mid_m),
        "low": (idx_low, low_m),
    }

    # Normalize energy within each band across time
    band_energy = {}
    band_max = {}
    for name, (idxs, _) in bands.items():
        if len(idxs) == 0:
            band_energy[name] = np.zeros(n_time, dtype=float)
            band_max[name] = 1.0
        else:
            e = np.sum(power[idxs, :], axis=0)
            band_energy[name] = e
            mx = float(np.max(e))
            band_max[name] = mx if mx > 0 else 1.0

    band_energy_norm = {k: band_energy[k] / band_max[k] for k in bands.keys()}

    # Frame-wise chosen pitch index per band
    pitch_idx_seq = {k: [None] * n_time for k in bands.keys()}

    for t in range(n_time):
        amps = power[:, t]
        for name, (idxs, midi_list) in bands.items():
            if len(idxs) == 0:
                continue
            e_norm = float(band_energy_norm[name][t])
            if e_norm < float(energy_threshold):
                pitch_idx_seq[name][t] = None
                continue

            band = amps[idxs]
            if float(band.sum()) <= 0:
                pitch_idx_seq[name][t] = None
                continue

            # choose scale with max power, map to midi palette position
            loc = int(np.argmax(band))
            scale_idx = int(idxs[loc])
            if len(idxs) > 1:
                pos = (scale_idx - int(idxs[0])) / float(int(idxs[-1]) - int(idxs[0]))
            else:
                pos = 0.0
            pi = int(round(pos * (len(midi_list) - 1)))
            pi = max(0, min(len(midi_list) - 1, pi))
            pitch_idx_seq[name][t] = pi

    def _compress(seq, voice):
        _, midi_list = bands[voice]
        e_series = band_energy_norm[voice]
        events = []
        cur = None
        start = 0
        for i, val in enumerate(seq + [None]):
            if val != cur:
                if cur is not None:
                    dur = i - start
                    midi_pitch = int(midi_list[cur])
                    e_mean = float(np.mean(e_series[start:i])) if dur > 0 else 0.0
                    events.append((start, dur, midi_pitch, e_mean))
                cur = val
                start = i
        return events

    note_events = {
        "high": _compress(pitch_idx_seq["high"], "high"),
        "mid": _compress(pitch_idx_seq["mid"], "mid"),
        "low": _compress(pitch_idx_seq["low"], "low"),
    }
    return note_events, int(n_time)


def note_events_to_musicxml(
    note_events,
    out_path,
    tempo_bpm=60.0,
    base_ql=1 / 8.0,
    total_frames=None,
):
    """
    Write note events into a MusicXML file.
    base_ql: quarterLength per frame
    """
    sc_score = stream.Score()
    sc_score.insert(0, tempo.MetronomeMark(number=float(tempo_bpm)))
    sc_score.insert(0, meter.TimeSignature("4/4"))

    for v_name in ["high", "mid", "low"]:
        part = stream.Part()
        part.id = v_name
        inst = instrument.Instrument()
        inst.partName = v_name
        part.insert(0, inst)

        events = note_events.get(v_name, [])
        cur_frame = 0
        cur_time_ql = 0.0
        last_dyn = None

        for start_f, dur_f, midi_pitch, e_mean in events:
            # rests before the note
            if start_f > cur_frame:
                rest_frames = start_f - cur_frame
                rest_ql = float(rest_frames) * float(base_ql)
                if rest_ql > 0:
                    r = m21note.Rest()
                    r.quarterLength = rest_ql
                    part.append(r)
                    cur_time_ql += rest_ql

            ql = float(dur_f) * float(base_ql)
            if ql <= 0:
                cur_frame = start_f + dur_f
                continue

            dyn = _energy_to_dynamic_mark(float(e_mean))
            if dyn != last_dyn:
                part.insert(float(cur_time_ql), dynamics.Dynamic(dyn))
                last_dyn = dyn

            n = m21note.Note(int(midi_pitch))
            n.quarterLength = ql
            part.append(n)

            cur_time_ql += ql
            cur_frame = start_f + dur_f

        # optional tail rest to reach total length
        if total_frames is not None and cur_frame < int(total_frames):
            tail_frames = int(total_frames) - cur_frame
            tail_ql = float(tail_frames) * float(base_ql)
            if tail_ql > 0:
                r = m21note.Rest()
                r.quarterLength = tail_ql
                part.append(r)

        sc_score.append(part)

    if not str(out_path).lower().endswith(".musicxml"):
        out_path = str(out_path) + ".musicxml"

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    sc_score.write("musicxml", fp=out_path)
    return out_path


def export_musicxml_from_wavelet(
    power,
    cluster_num,
    m,
    out_dir="musicxml",
    tempo_bpm=60.0,
    base_ql=1 / 8.0,
    energy_threshold=0.60,
    low_midi=(45, 48, 50, 52, 55),
    mid_midi=(57, 59, 60, 62, 64, 67),
    high_midi=(69, 71, 72, 74, 76),
):
    """
    Convenience wrapper: power -> note events -> MusicXML
    """
    note_events, total_frames = power_to_note_events(
        power=power,
        energy_threshold=energy_threshold,
        low_midi=low_midi,
        mid_midi=mid_midi,
        high_midi=high_midi,
    )
    out_path = os.path.join(out_dir, f"cluster_{cluster_num}_path_{m}.musicxml")
    return note_events_to_musicxml(
        note_events=note_events,
        out_path=out_path,
        tempo_bpm=tempo_bpm,
        base_ql=base_ql,
        total_frames=total_frames,
    )


# ============================================================
# ORIGINAL plot_wavelet (MODIFIED: adds MusicXML export)
# ============================================================
def plot_wavelet(
    time,
    signal,
    scales,
    cluster_num,
    m,
    label=0,
    waveletname="mexh",
    cmap=plt.cm.seismic,
    title="Wavelet Transform (Power Spectrum) of signal",
    ylabel="1/Frequency",
    xlabel="Time",
    # ====== NEW knobs ======
    export_musicxml=True,
    musicxml_out_dir="musicxml",
    musicxml_tempo_bpm=60.0,
    musicxml_base_ql=1 / 8.0,
    musicxml_energy_threshold=0.60,
    musicxml_low_midi=(45, 48, 50, 52, 55),
    musicxml_mid_midi=(57, 59, 60, 62, 64, 67),
    musicxml_high_midi=(69, 71, 72, 74, 76),
):
    dt = time[1] - time[0]
    [coefficients, frequencies] = pywt.cwt(signal, scales, waveletname, dt)
    power = (abs(coefficients)) ** 2

    # ====== NEW: export MusicXML directly from this power ======
    if export_musicxml:
        export_musicxml_from_wavelet(
            power=power,
            cluster_num=cluster_num,
            m=m,
            out_dir=musicxml_out_dir,
            tempo_bpm=musicxml_tempo_bpm,
            base_ql=musicxml_base_ql,
            energy_threshold=musicxml_energy_threshold,
            low_midi=musicxml_low_midi,
            mid_midi=musicxml_mid_midi,
            high_midi=musicxml_high_midi,
        )

    period = 1.0 / frequencies
    levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8]
    contourlevels = np.log2(levels)

    fig, ax = plt.subplots(figsize=(15, 10))
    im = ax.contourf(range(50), np.log2(period), np.log2(power), contourlevels, extend="both", cmap=cmap)

    ax.set_title(title, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_xlabel(xlabel, fontsize=18)

    yticks = 2 ** np.arange(np.ceil(np.log2(period.min())), np.ceil(np.log2(period.max())))
    ax.set_yticks(np.log2(yticks))
    ax.set_yticklabels(yticks)
    ax.invert_yaxis()
    ylim = ax.get_ylim()
    ax.set_ylim(ylim[0], -1)

    cbar_ax = fig.add_axes([0.95, 0.5, 0.03, 0.25])
    fig.colorbar(im, cax=cbar_ax, orientation="vertical")
    filename = "figure_of_time_period_" + cluster_num + "_path_" + str(m) + ".pdf"
    if label == 1:
        plt.show()

    plt.savefig(filename, bbox_inches="tight")
    plt.close()


# ============================================================
# YOUR show_result (UNCHANGED except plot_wavelet now exports musicxml)
# ============================================================
def show_result(gene_info, trajectory_info, latent="ALL_mu.npy"):

    data = np.load(latent)
    adata = ad.AnnData(data)
    adata.obs_names = gene_info.loc[:, "gene_id"]
    adata.var_names = ["C" + str(i) for i in range(20)]
    adata.obs["indexa"] = range(len(adata.obs_names))

    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=10)
    sc.tl.umap(adata)

    sc.tl.leiden(adata, resolution=1)

    for i in range(len(gene_info)):
        gene_info.loc[i, "gene_id"] = gene_info.loc[i, "gene_id"][:18]

    adata.obs_names = gene_info.loc[:, "gene_id"]

    a = pd.Index(trajectory_info.loc["path"] == 1)  # dataset1
    y = np.array(trajectory_info.iloc[:-5, a])
    x = np.array(trajectory_info.loc["time", a])

    x.sort()
    x_smooth = np.linspace(start=x.min(), stop=x.max(), num=100)
    y_smooth = make_interp_spline(x, y.transpose())(x_smooth)

    wavelet = "mexh"
    scales = np.arange(1, 51)
    dt = x_smooth[1] - x_smooth[0]

    coeffs, freqs = pywt.cwt(y_smooth.transpose(), scales, wavelet, sampling_period=dt)

    threshold = 0.01
    coeffs_thresh = pywt.threshold(coeffs, threshold, mode="soft")

    yf_smooth = np.sum(coeffs_thresh, axis=2).T
    yf_info = np.array([abs(yf_smooth[i, :]) for i in range(yf_smooth.shape[0])])

    for i in range(len(yf_info)):
        if yf_info[i].max() == 0:
            continue
        yf_info[i] = yf_info[i] / yf_info[i].max()

    y_k = y_smooth.transpose()

    for i in range(len(y)):
        y[i, :] = y[i, :] - y[i, 0]

    def movingaverage(data, window_size):
        window = np.ones(int(window_size)) / float(window_size)
        a = np.convolve(data[window_size:], window, "valid")
        return np.concatenate([np.zeros(window_size), a])

    print_first = [1, 1, 1]

    for j in range(200):
        cluster_num = str(j)
        cluster = adata.obs.loc[adata.obs["leiden"] == cluster_num, "indexa"].to_numpy()

        if len(cluster) == 0:
            continue

        error_t = np.max(y[cluster, :], axis=1) - np.min(y[cluster, :], axis=1)
        error_index_t = np.argsort(error_t)

        cluster_filter, y_info_filter, yf_info_filter = [], [], []
        cluater_index, name = [], []
        for idx in error_index_t:
            a_sm = movingaverage(y[cluster[idx], :], 5)
            if abs(np.max(np.abs(a_sm))) < 0.01 or abs(np.max(np.abs(a_sm))) > 5:
                continue
            cluster_filter.append(a_sm)
            y_info_filter.append(y_k[cluster[idx], :])
            yf_info_filter.append(yf_info[cluster[idx], :])
            cluater_index.append(cluster[idx])
            name.append(adata.obs_names[cluster[idx]])

        cluster_filter = np.asarray(cluster_filter)
        y_info_filter = np.asarray(y_info_filter)
        yf_info_filter = np.asarray(yf_info_filter)
        cluater_index = np.asarray(cluater_index)

        color_map = ["b", "r", "y", "c", "g"]
        n_cluster = 5
        index = [[] for _ in range(n_cluster)]
        namei = [[] for _ in range(n_cluster)]

        if len(cluster_filter) >= 5:
            y_pred = KMeans(n_clusters=n_cluster, random_state=9).fit_predict(cluster_filter)
            number_of_genes = np.zeros(n_cluster, dtype=int)
            for i, lab in enumerate(y_pred):
                index[lab].append(i)
                namei[lab].append(name[i])
                number_of_genes[lab] += 1

            avg = np.zeros((n_cluster, cluster_filter.shape[1]))
            std = np.zeros_like(avg)
            yf_avg = np.zeros((n_cluster, yf_info_filter.shape[1]))
            for i in range(n_cluster):
                if len(index[i]) == 0:
                    continue
                avg[i] = np.mean(cluster_filter[index[i]], axis=0)
                std[i] = np.std(cluster_filter[index[i]], axis=0)
                yf_avg[i] = np.mean(yf_info_filter[index[i]], axis=0)

            m = 0
            for i in range(n_cluster):
                if number_of_genes[i] < 5:
                    continue
                plt.errorbar(x[:-4], avg[i], yerr=std[i], c=color_map[i], label="path" + str(m))
                m += 1
            plt.xlabel("Pseudotime", fontdict={"family": "Arial", "size": 18})
            plt.ylabel("Expression log2 fold change", fontdict={"family": "Arial", "size": 18})
            plt.title("cluster" + str(j), fontdict={"family": "Arial", "size": 18})
            plt.legend()
            filename = "figure_of_ds1_avg_time_data_" + cluster_num + ".pdf"
            plt.savefig(filename, bbox_inches="tight")
            if print_first[0] == 1:
                plt.show()
                print_first[0] = 0
            plt.close()

            m = 0
            for i in range(n_cluster):
                if number_of_genes[i] < 5:
                    continue
                plt.plot(range(len(yf_avg[i])), yf_avg[i][::-1], c=color_map[i], label="path" + str(m))
                m += 1
            plt.xlabel("Frequency", fontdict={"family": "Arial", "size": 18})
            plt.ylabel("Amplitude", fontdict={"family": "Arial", "size": 18})
            plt.title("cluster" + str(j), fontdict={"family": "Arial", "size": 18})
            plt.legend()
            filename = "figure_of_ds1_frequency_" + cluster_num + ".pdf"
            plt.savefig(filename, bbox_inches="tight")
            if print_first[1] == 1:
                plt.show()
                print_first[1] = 0
            plt.close()
            plt.gcf().clear()

            scales = np.arange(1, 51)
            m = 0
            for i in range(n_cluster):
                if number_of_genes[i] < 5:
                    continue

                # plot_wavelet now auto-exports MusicXML under ./musicxml/
                plot_wavelet(
                    x_smooth,
                    y_info_filter[index[i]],
                    scales,
                    cluster_num,
                    m,
                    print_first[2],
                    export_musicxml=True,
                    musicxml_out_dir="musicxml",
                    musicxml_tempo_bpm=60.0,
                    musicxml_base_ql=1 / 8.0,
                    musicxml_energy_threshold=0.60,
                )
                print_first[2] = 0
                m += 1

            mg = mygene.MyGeneInfo()
            m = 0
            for i in range(n_cluster):
                if number_of_genes[i] < 5 or len(namei[i]) == 0:
                    continue
                gene_ids = mg.getgenes(namei[i], "name, symbol, entrezgene", as_dataframe=True)
                gene_ids.index.name = "UNIPROT"
                gene_ids.reset_index(inplace=True)
                gene_symbols = gene_ids["symbol"]
                gene_symbols.to_csv(f"Gene_{cluster_num}_path_{m}.csv", index=False)
                m += 1








def differential_frequency(dataset1,dataset2,dataset1_name,dataset2_name,features):

    trajectory_info = pd.read_csv(dataset1)  # dataset1

    a = pd.Index(trajectory_info.loc["path"] == 1)  # dataset1
    y = np.array(trajectory_info.iloc[:-5, a])
    x = np.array(trajectory_info.loc["time", a])

    x.sort()
    x_smooth = np.linspace(start=x.min(), stop=x.max(), num=100)
    y_smooth = make_interp_spline(x, y.transpose())(x_smooth)
    wavelet = 'mexh'
    scales = np.arange(1, 51)  # Define the range of scales
    dt = x_smooth[1] - x_smooth[0]

    coeffs, freqs = pywt.cwt(y_smooth.transpose(), scales, wavelet, sampling_period=dt)

    # Thresholding
    threshold = 0.1  # Set your threshold value here
    coeffs_thresh = pywt.threshold(coeffs, threshold, mode='soft')

    yf_smooth = np.sum(coeffs_thresh, axis=2).T

    yf_info = np.array([abs(yf_smooth[i, :]) for i in range(yf_smooth.shape[0])])

    for i in range(len(yf_info)):
        if yf_info[i].max() == 0:
            continue
        yf_info[i] = yf_info[i] / yf_info[i].max()


    result_df_c = pd.DataFrame(yf_info, index=trajectory_info.index[:len(yf_info)])
    result_df_c.columns = [f'c{i + 1}' for i in range(yf_info.shape[1])]

    trajectory_info_a = pd.read_csv(dataset2)  # dataset1

    a = pd.Index(trajectory_info_a.loc["path"] == 1)  # dataset1
    y = np.array(trajectory_info_a.iloc[:-5, a])
    x = np.array(trajectory_info_a.loc["time", a])

    x.sort()
    x_smooth = np.linspace(start=x.min(), stop=x.max(), num=100)
    y_smooth = make_interp_spline(x, y.transpose())(x_smooth)
    # Perform wavelet transform
    wavelet = 'mexh'
    scales = np.arange(1, 51)  # Define the range of scales
    dt = x_smooth[1] - x_smooth[0]

    coeffs, freqs = pywt.cwt(y_smooth.transpose(), scales, wavelet, sampling_period=dt)

    # Thresholding
    threshold = 0.1  # Set your threshold value here
    coeffs_thresh = pywt.threshold(coeffs, threshold, mode='soft')


    yf_smooth = np.sum(coeffs_thresh, axis=2).T

    yf_info = np.array([abs(yf_smooth[i, :]) for i in range(yf_smooth.shape[0])])

    for i in range(len(yf_info)):
        if yf_info[i].max() == 0:
            continue
        yf_info[i] = yf_info[i] / yf_info[i].max()


    result_df = pd.DataFrame(yf_info, index=trajectory_info_a.index[:len(yf_info)])
    result_df.columns = [f'a{i + 1}' for i in range(yf_info.shape[1])]

    merged_df = result_df.merge(result_df_c, left_index=True, right_index=True, how='inner', suffixes=('_df1', '_df2'))

    import anndata as ad

    adata = ad.AnnData(merged_df.T)
    import scanpy as sc
    adata.obs['cluster'] = 0
    adata.obs['cluster'][0:50] = dataset1_name
    adata.obs['cluster'][50:100] = dataset2_name

    import pandas as pd

    # Load the CSV file

    df = features # Adjust separator if needed
    # Ensure the data has 'gene_id' and 'gene_name' columns
    # If you already have the mapping in the file, you can directly use it
    # Otherwise, you may need to create a dictionary or use mygene for mapping

    # Create a dictionary from 'gene_id' to 'gene_name'
    gene_mapping = dict(zip(df['gene_id'], df['gene_name']))

    # Load the dataset where you want to replace gene IDs
    # Replace `adata.var_names` with gene names using the dictionary
    adata.var_names = [gene_mapping.get(gene_id, gene_id) for gene_id in adata.var_names]
    adata.var_names_make_unique()
    sc.tl.rank_genes_groups(adata, 'cluster', groups=[dataset1_name], reference=dataset2_name, method='t-test')

    save_name = 'rank_genes_groups_'+dataset1_name+'_vs_'+dataset2_name+'.pdf'
    sc.pl.rank_genes_groups_heatmap(adata, n_genes=20, show_gene_labels=True,
                                    save=save_name)


    ranked_genes = adata.uns['rank_genes_groups']

    filtered_genes = ranked_genes['pvals_adj'][dataset1_name] < 0.05

    filtered_genes_info = pd.DataFrame({
        'gene': ranked_genes['names'][dataset1_name][filtered_genes],
        'logfoldchange': ranked_genes['logfoldchanges'][dataset1_name][filtered_genes],
        'pval': ranked_genes['pvals'][dataset1_name][filtered_genes],
        'pval_adj': ranked_genes['pvals_adj'][dataset1_name][filtered_genes]
    })

    save_name_csv = 'filtered_genes_pval_adj_0.05_'+dataset1_name+'_vs_'+dataset2_name+'.csv'





def show_result_spatial(
    adata_s,
    start,
    end,
    latent="ALL_mu.npy",
    n_points=10,
    k_neighbors=100,
    leiden_resolution=1,
    n_leiden_clusters=200,
    n_cluster_sub=5,
    min_genes_per_path=5,
    smooth_k=3,
    moving_avg_window=5,
    waveletname="mexh",
    wavelet_scales=np.arange(1, 51),
    wavelet_threshold=0.1,
    export_musicxml=True,
    musicxml_out_dir="musicxml_spatial",
    musicxml_tempo_bpm=60.0,
    musicxml_base_ql=1 / 8.0,
    musicxml_energy_threshold=0.60,
):
    """
    Spatial version of show_result() with MusicXML integrated via plot_wavelet().

    Inputs match your style:
      - adata_s: AnnData with .X and .obsm["spatial"]
      - start/end: np.array-like (x,y) coordinates defining a line
      - latent: path to latent embedding .npy (genes x 20)

    Outputs:
      - writes PDFs (avg curves / frequency / wavelet) as before
      - writes MusicXML into musicxml_out_dir if export_musicxml=True
      - writes gene list CSVs per (cluster,path) as before
    """
    gene_info = adata_s
    data = np.load(latent)
    adata = ad.AnnData(data)

    # assume latent rows correspond to genes in adata_s.var_names
    adata.obs_names = gene_info.var_names
    adata.var_names = ["C" + str(i) for i in range(adata.shape[1])]
    adata.obs["indexa"] = range(len(adata.obs_names))

    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=min(10, adata.shape[1]))
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=leiden_resolution)

    adata.obs_names = gene_info.var_names
    adata1 = adata_s

    # spatial coords
    if "spatial" not in adata1.obsm:
        raise KeyError("adata_s.obsm['spatial'] not found.")
    spatial_coords = adata1.obsm["spatial"]

    # build trajectory means along the line
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    distances = np.linspace(0, 1, int(n_points))
    center_points = np.array([start + d * (end - start) for d in distances])

    trajectory_means = []
    for point in center_points:
        dists = pairwise_distances([point], spatial_coords)[0]
        nearest_indices = np.argsort(dists)[: int(k_neighbors)]
        Xsub = adata1.X[nearest_indices, :]
        if hasattr(Xsub, "toarray"):
            Xsub = Xsub.toarray()
        mean_count = np.mean(Xsub, axis=0)
        trajectory_means.append(mean_count)

    trajectory_array = np.array(trajectory_means)  # (n_points, n_genes)
    y = trajectory_array.T  # (n_genes, n_points)

    # smooth
    x = np.arange(trajectory_array.shape[0])
    x_smooth = np.linspace(start=x.min(), stop=x.max(), num=100)
    y_smooth = make_interp_spline(x, trajectory_array, k=int(smooth_k))(x_smooth)

    # wavelet per gene
    dt = float(x_smooth[1] - x_smooth[0])
    coeffs, freqs = pywt.cwt(y_smooth.transpose(), wavelet_scales, waveletname, sampling_period=dt)
    coeffs_thresh = pywt.threshold(coeffs, float(wavelet_threshold), mode="soft")
    yf_smooth = np.sum(coeffs_thresh, axis=2).T
    yf_info = np.array([np.abs(yf_smooth[i, :]) for i in range(yf_smooth.shape[0])])

    for i in range(len(yf_info)):
        mx = float(np.max(yf_info[i]))
        if mx > 0:
            yf_info[i] = yf_info[i] / mx

    y_k = y_smooth.transpose()

    # baseline subtract as your original
    for i in range(y.shape[0]):
        y[i, :] = y[i, :] - y[i, 0]

    def movingaverage(arr, window_size):
        window_size = int(window_size)
        window = np.ones(window_size) / float(window_size)
        a = np.convolve(arr[window_size:], window, "valid")
        return np.concatenate([np.zeros(window_size), a])

    print_first = [0, 0, 0]

    for j in range(int(n_leiden_clusters)):
        cluster_num = str(j)
        cluster = adata.obs.loc[adata.obs["leiden"] == cluster_num, "indexa"].to_numpy()
        if cluster.size == 0:
            continue

        error_t = np.max(y[cluster, :], axis=1) - np.min(y[cluster, :], axis=1)
        error_index_t = np.argsort(error_t)

        cluster_filter, y_info_filter, yf_info_filter = [], [], []
        name = []

        for idx in error_index_t:
            a_ma = movingaverage(y[cluster[idx], :], moving_avg_window)
            if abs(np.max(np.abs(a_ma))) < 0.01 or abs(np.max(np.abs(a_ma))) > 5:
                continue
            cluster_filter.append(a_ma)
            y_info_filter.append(y_k[cluster[idx], :])
            yf_info_filter.append(yf_info[cluster[idx], :])
            name.append(adata.obs_names[cluster[idx]])

        cluster_filter = np.asarray(cluster_filter)
        y_info_filter = np.asarray(y_info_filter)
        yf_info_filter = np.asarray(yf_info_filter)

        if len(cluster_filter) < int(n_cluster_sub):
            continue

        color_map = ["b", "r", "y", "c", "g"]
        n_cluster = int(n_cluster_sub)
        index = [[] for _ in range(n_cluster)]
        namei = [[] for _ in range(n_cluster)]

        y_pred = KMeans(n_clusters=n_cluster, random_state=9).fit_predict(cluster_filter)
        number_of_genes = np.zeros(n_cluster, dtype=int)
        for ii, lab in enumerate(y_pred):
            index[lab].append(ii)
            namei[lab].append(name[ii])
            number_of_genes[lab] += 1

        avg = np.zeros((n_cluster, cluster_filter.shape[1]))
        std = np.zeros_like(avg)
        yf_avg = np.zeros((n_cluster, yf_info_filter.shape[1]))
        for ii in range(n_cluster):
            if len(index[ii]) == 0:
                continue
            avg[ii] = np.mean(cluster_filter[index[ii]], axis=0)
            std[ii] = np.std(cluster_filter[index[ii]], axis=0)
            yf_avg[ii] = np.mean(yf_info_filter[index[ii]], axis=0)

        # avg time plot
        m = 0
        for ii in range(n_cluster):
            if number_of_genes[ii] < int(min_genes_per_path):
                continue
            plt.errorbar(x[:-4], avg[ii], yerr=std[ii], c=color_map[ii % len(color_map)], label="path" + str(m))
            m += 1
        plt.xlabel("Pseudotime", fontdict={"family": "Arial", "size": 18})
        plt.ylabel("Expression log2 fold change", fontdict={"family": "Arial", "size": 18})
        plt.title("cluster" + str(j), fontdict={"family": "Arial", "size": 18})
        plt.legend()
        filename = "figure_of_ds1_avg_time_data_" + cluster_num + ".pdf"
        plt.savefig(filename, bbox_inches="tight")
        if print_first[0] == 1:
            plt.show()
            print_first[0] = 0
        plt.close()

        # frequency plot
        m = 0
        for ii in range(n_cluster):
            if number_of_genes[ii] < int(min_genes_per_path):
                continue
            plt.plot(range(len(yf_avg[ii])), yf_avg[ii][::-1], c=color_map[ii % len(color_map)], label="path" + str(m))
            m += 1
        plt.xlabel("Frequency", fontdict={"family": "Arial", "size": 18})
        plt.ylabel("Amplitude", fontdict={"family": "Arial", "size": 18})
        plt.title("cluster" + str(j), fontdict={"family": "Arial", "size": 18})
        plt.legend()
        filename = "figure_of_ds1_frequency_" + cluster_num + ".pdf"
        plt.savefig(filename, bbox_inches="tight")
        if print_first[1] == 1:
            plt.show()
            print_first[1] = 0
        plt.close()
        plt.gcf().clear()

        # wavelet plot + MusicXML
        scales = np.arange(1, 51)
        m = 0
        for ii in range(n_cluster):
            if number_of_genes[ii] < int(min_genes_per_path):
                continue

            plot_wavelet(
                x_smooth,
                y_info_filter[index[ii]],
                scales,
                cluster_num,
                m,
                print_first[2],
                waveletname=waveletname,
                export_musicxml=bool(export_musicxml),
                musicxml_out_dir=str(musicxml_out_dir),
                musicxml_tempo_bpm=float(musicxml_tempo_bpm),
                musicxml_base_ql=float(musicxml_base_ql),
                musicxml_energy_threshold=float(musicxml_energy_threshold),
            )
            print_first[2] = 0
            m += 1

        # gene list export
        mg = mygene.MyGeneInfo()
        m = 0
        for ii in range(n_cluster):
            if number_of_genes[ii] < int(min_genes_per_path) or len(namei[ii]) == 0:
                continue
            gene_ids = mg.getgenes(namei[ii], "name, symbol, entrezgene", as_dataframe=True)
            gene_ids.index.name = "UNIPROT"
            gene_ids.reset_index(inplace=True)
            gene_symbols = gene_ids["symbol"]
            gene_symbols.to_csv(f"Gene_{cluster_num}_path_{m}.csv", index=False)
            m += 1






def differential_frequency_spatial(adata, start1, end1, start2, end2, direction1, direction2):
    # Extract spatial coordinates
    spatial_coords = adata.obsm["spatial"]


    start = start1
    end = end1

    num_points = 20  # Number of center points along the line
    distances = np.linspace(0, 1, num_points)  # Fractional distances along the line
    center_points = np.array([start + d * (end - start) for d in distances])

    # Initialize a list to store mean counts for each center point
    trajectory_means = []

    # For each center point, find the 100 nearest cells and compute mean counts
    ind_2 = []
    for point in center_points:
        distances = pairwise_distances([point], spatial_coords)[0]
        nearest_indices = np.argsort(distances)[:100]  # Get indices of 100 nearest cells
        ind_2 = ind_2 + list(nearest_indices)
        mean_count = np.mean(adata.X[nearest_indices, :].toarray(), axis=0)  # Compute mean counts
        trajectory_means.append(mean_count)  # Flatten to ensure 1D array

    # Convert trajectory_means to a numpy array for further processing
    trajectory_array = np.array(trajectory_means)

    # Smooth the data for frequency analysis
    x = np.arange(trajectory_array.shape[0])
    x_smooth = np.linspace(start=x.min(), stop=x.max(), num=100)
    y_smooth = make_interp_spline(x, trajectory_array, k=3)(x_smooth)

    # Perform continuous wavelet transform
    wavelet = 'mexh'
    scales = np.arange(1, 51)  # Define the range of scales
    dt = x_smooth[1] - x_smooth[0]

    coeffs, freqs = pywt.cwt(y_smooth.transpose(), scales, wavelet, sampling_period=dt)

    # Thresholding
    threshold = 0.1  # Set your threshold value here
    coeffs_thresh = pywt.threshold(coeffs, threshold, mode='soft')

    yf_smooth = np.sum(coeffs_thresh, axis=2).T

    yf_info = np.array([abs(yf_smooth[i, :]) for i in range(yf_smooth.shape[0])])

    for i in range(len(yf_info)):
        if yf_info[i].max() == 0:
            continue
        yf_info[i] = yf_info[i] / yf_info[i].max()


    result_df_c = pd.DataFrame(yf_info, index=np.arange(0, len(yf_info)))
    result_df_c.columns = [f'c{i + 1}' for i in range(yf_info.shape[1])]


    # Extract spatial coordinates
    spatial_coords = adata.obsm["spatial"]

    start = start2
    end = end2

    num_points = 20  # Number of center points along the line
    distances = np.linspace(0, 1, num_points)  # Fractional distances along the line
    center_points = np.array([start + d * (end - start) for d in distances])

    # Initialize a list to store mean counts for each center point
    trajectory_means = []
    ind_1 = []

    # For each center point, find the 100 nearest cells and compute mean counts
    for point in center_points:
        distances = pairwise_distances([point], spatial_coords)[0]
        nearest_indices = np.argsort(distances)[:100]  # Get indices of 100 nearest cells
        ind_1 = ind_1 + list(nearest_indices)
        mean_count = np.mean(adata.X[nearest_indices, :].toarray(), axis=0)  # Compute mean counts
        trajectory_means.append(mean_count)  # Flatten to ensure 1D array

    # Convert trajectory_means to a numpy array for further processing
    trajectory_array = np.array(trajectory_means)

    # Smooth the data for frequency analysis
    x = np.arange(trajectory_array.shape[0])
    x_smooth = np.linspace(start=x.min(), stop=x.max(), num=100)
    y_smooth = make_interp_spline(x, trajectory_array, k=3)(x_smooth)

    # Perform continuous wavelet transform
    wavelet = 'mexh'
    scales = np.arange(1, 51)  # Define the range of scales
    dt = x_smooth[1] - x_smooth[0]

    coeffs, freqs = pywt.cwt(y_smooth.transpose(), scales, wavelet, sampling_period=dt)

    # Thresholding
    threshold = 0.1  # Set your threshold value here
    coeffs_thresh = pywt.threshold(coeffs, threshold, mode='soft')

    yf_smooth = np.sum(coeffs_thresh, axis=2).T

    yf_info = np.array([abs(yf_smooth[i, :]) for i in range(yf_smooth.shape[0])])

    for i in range(len(yf_info)):
        if yf_info[i].max() == 0:
            continue
        yf_info[i] = yf_info[i] / yf_info[i].max()



    result_df = pd.DataFrame(yf_info, index=np.arange(0, len(yf_info)))
    result_df.columns = [f'a{i + 1}' for i in range(yf_info.shape[1])]

    merged_df = result_df.merge(result_df_c, left_index=True, right_index=True, how='inner', suffixes=('_df1', '_df2'))

    import anndata as ad

    adata = ad.AnnData(merged_df.T)
    import scanpy as sc
    adata.obs['cluster'] = 0
    adata.obs['cluster'][0:50] = dataset1_name
    adata.obs['cluster'][50:100] = dataset2_name

    import pandas as pd

    # Load the CSV file

    df = features # Adjust separator if needed
    # Ensure the data has 'gene_id' and 'gene_name' columns
    # If you already have the mapping in the file, you can directly use it
    # Otherwise, you may need to create a dictionary or use mygene for mapping

    # Create a dictionary from 'gene_id' to 'gene_name'
    gene_mapping = dict(zip(df['gene_id'], df['gene_name']))

    # Load the dataset where you want to replace gene IDs
    # Replace `adata.var_names` with gene names using the dictionary
    adata.var_names = [gene_mapping.get(gene_id, gene_id) for gene_id in adata.var_names]
    adata.var_names_make_unique()
    sc.tl.rank_genes_groups(adata, 'cluster', groups=[dataset1_name], reference=dataset2_name, method='t-test')

    save_name = 'rank_genes_groups_'+dataset1_name+'_vs_'+dataset2_name+'.pdf'
    sc.pl.rank_genes_groups_heatmap(adata, n_genes=20, show_gene_labels=True,
                                    save=save_name)


    ranked_genes = adata.uns['rank_genes_groups']

    filtered_genes = ranked_genes['pvals_adj'][dataset1_name] < 0.05

    filtered_genes_info = pd.DataFrame({
        'gene': ranked_genes['names'][dataset1_name][filtered_genes],
        'logfoldchange': ranked_genes['logfoldchanges'][dataset1_name][filtered_genes],
        'pval': ranked_genes['pvals'][dataset1_name][filtered_genes],
        'pval_adj': ranked_genes['pvals_adj'][dataset1_name][filtered_genes]
    })

    save_name_csv = 'filtered_genes_pval_adj_0.05_'+dataset1_name+'_vs_'+dataset2_name+'.csv'




import os
import re
import math
from fractions import Fraction
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import pywt
import anndata as ad
import scanpy as sc

from music21 import stream, note as m21note, tempo, meter, instrument, dynamics
from music21 import key as m21key
from music21 import metadata as m21metadata


def run_diff_wavelet_musicxml_pipeline(
    control_traj_csv,
    disease_traj_csv,
    gene_id_csv,
    disease_label="COVID",
    control_label="Control",
    outdir_pdf="out_pdf",
    outdir_mxml_piano="out_musicxml_piano",
    outdir_mxml_flute="out_musicxml_flute",
    outdir_mxml_violin="out_musicxml_violin",
    wavelet_name="mexh",
    threshold=0.1,
    scales=np.arange(1, 128),
    padj_cutoff=0.05,
    abs_logfc_cutoff=1.0,
    keep_only_positive_diff=True,
    test_n_genes=None,                 # None = all genes
    topn_markers_to_export=None,        # None = all markers
    topn_nonmarkers_to_export=None,     # None = all non-markers
    base_ql=Fraction(1, 32),            # smallest unit
    bpm_min=30.0,
    bpm_max=130.0,
    total_beats_min=8.0,
    total_beats_max=16.0,
    # sigmoid anchors (reviewer-resistant knobs)
    anchor_ahf_low=150.0,
    anchor_rp_low=0.20,
    anchor_ahf_high=3000.0,
    anchor_rp_high=0.80,
    anchor_tilt_low=-1.05,
    anchor_rt_low=0.20,
    anchor_tilt_high=-0.75,
    anchor_rt_high=0.80,
    # visualization scaling (keep yours)
    pdf_power_scale_mul=5.0,
    pdf_power_scale_add=-4.5,
    diff_power_floor=0.1,
    gain_clip=(0.5, 2.0),
    verbose=True,
):
    # ----------------------------
    # XML post-processing helpers
    # ----------------------------
    def hide_tempo_marks_in_musicxml(xml_path: str):
        """
        Hide tempo in printed score, keep playback tempo via <sound tempo="...">.
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()
        ns = ""
        if root.tag.startswith("{"):
            ns = root.tag.split("}")[0].strip("{")

        def q(tag):
            return f"{{{ns}}}{tag}" if ns else tag

        for direction in root.iter(q("direction")):
            sound = None
            for s in direction.findall(q("sound")):
                if "tempo" in s.attrib:
                    sound = s
                    break
            if sound is None:
                continue

            direction.set("print-object", "no")
            for dt in list(direction.findall(q("direction-type"))):
                for met in list(dt.findall(q("metronome"))):
                    dt.remove(met)
                for w in list(dt.findall(q("words"))):
                    dt.remove(w)
                if len(list(dt)) == 0:
                    direction.remove(dt)

        tree.write(xml_path, encoding="utf-8", xml_declaration=True)

    def set_first_system_names_hide_later(xml_path: str, voice_names=("high", "mid", "low")):
        """
        First system shows part-name; later systems hide by setting part-abbreviation="".
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()
        ns = ""
        if root.tag.startswith("{"):
            ns = root.tag.split("}")[0].strip("{")

        def q(tag):
            return f"{{{ns}}}{tag}" if ns else tag

        part_elems = list(root.findall(q("part")))
        part_ids_in_order = [p.get("id") for p in part_elems]

        id_to_name = {}
        for i, pid in enumerate(part_ids_in_order):
            if pid is None:
                continue
            if i < len(voice_names):
                id_to_name[pid] = str(voice_names[i])

        part_list = root.find(q("part-list"))
        if part_list is None:
            tree.write(xml_path, encoding="utf-8", xml_declaration=True)
            return

        score_parts = list(part_list.findall(q("score-part")))
        for i, sp in enumerate(score_parts):
            spid = sp.get("id")
            if spid in id_to_name:
                name = id_to_name[spid]
            else:
                name = str(voice_names[i]) if i < len(voice_names) else (spid or f"part{i+1}")

            pn = sp.find(q("part-name"))
            if pn is None:
                pn = ET.SubElement(sp, q("part-name"))
            pn.text = name

            pa = sp.find(q("part-abbreviation"))
            if pa is None:
                pa = ET.SubElement(sp, q("part-abbreviation"))
            pa.text = ""

            for tag in ["part-name-display", "part-abbreviation-display"]:
                elem = sp.find(q(tag))
                if elem is not None:
                    sp.remove(elem)

        tree.write(xml_path, encoding="utf-8", xml_declaration=True)

    # ----------------------------
    # math helpers
    # ----------------------------
    EPS_STYLE = 1e-9

    def clamp(x, lo, hi):
        return max(lo, min(hi, x))

    def sigmoid(z):
        z = float(z)
        if z >= 0:
            ez = math.exp(-z)
            return 1.0 / (1.0 + ez)
        else:
            ez = math.exp(z)
            return ez / (1.0 + ez)

    def _logit(p):
        p = float(min(1.0 - 1e-9, max(1e-9, p)))
        return math.log(p / (1.0 - p))

    def _solve_sigmoid_mu_scale(x_low, r_low, x_high, r_high):
        y1 = _logit(r_low)
        y2 = _logit(r_high)
        scale = (float(x_high) - float(x_low)) / (y2 - y1 + 1e-12)
        mu = float(x_low) - scale * y1
        return float(mu), float(scale)

    SIGMOID_MU_B, SIGMOID_SCALE_B = _solve_sigmoid_mu_scale(
        math.log(anchor_ahf_low + EPS_STYLE), anchor_rp_low,
        math.log(anchor_ahf_high + EPS_STYLE), anchor_rp_high
    )
    SIGMOID_MU_T, SIGMOID_SCALE_T = _solve_sigmoid_mu_scale(
        anchor_tilt_low, anchor_rt_low,
        anchor_tilt_high, anchor_rt_high
    )

    def sanitize_gene_name(g):
        g = str(g)
        g = re.sub(r"[^\w\-.]+", "_", g)
        return g[:120]

    def energy_to_dynamic_mark(energy):
        if energy < 0.15:
            return "p"
        if energy < 0.30:
            return "mp"
        if energy < 0.55:
            return "mf"
        if energy < 0.80:
            return "f"
        return "ff"

    # ----------------------------
    # I/O folders & instrument variants
    # ----------------------------
    os.makedirs(outdir_pdf, exist_ok=True)
    os.makedirs(outdir_mxml_piano, exist_ok=True)
    os.makedirs(outdir_mxml_flute, exist_ok=True)
    os.makedirs(outdir_mxml_violin, exist_ok=True)

    MUSICXML_INSTRUMENT_VARIANTS = [
        ("piano", "Piano", outdir_mxml_piano),
        ("flute", "Flute", outdir_mxml_flute),
        ("violin", "Violin", outdir_mxml_violin),
    ]

    # ----------------------------
    # trajectory loader (same as yours)
    # ----------------------------
    def load_and_smooth_trajectory(csv_path):
        traj = pd.read_csv(csv_path)
        a = pd.Index(traj.loc["path"] == 1)
        y = np.array(traj.iloc[:-5, a])
        x = np.array(traj.loc["time", a])
        x.sort()

        x_smooth = np.linspace(start=x.min(), stop=x.max(), num=100)
        y_smooth = make_interp_spline(x, y.transpose())(x_smooth)  # (100, n_genes)
        y_k = y_smooth.transpose()                                  # (n_genes, 100)

        gene_index = traj.index[: y_k.shape[0]]
        df_y = pd.DataFrame(y_k, index=gene_index)
        df_y.columns = [f"t{i+1}" for i in range(y_k.shape[1])]
        return x_smooth, df_y

    x_smooth_ctrl, result_df_y_c = load_and_smooth_trajectory(control_traj_csv)
    x_smooth_dis,  result_df_y   = load_and_smooth_trajectory(disease_traj_csv)

    common_genes = result_df_y.index.intersection(result_df_y_c.index)
    if test_n_genes is not None:
        common_genes = common_genes[: int(test_n_genes)]

    result_df_y   = result_df_y.loc[common_genes]
    result_df_y_c = result_df_y_c.loc[common_genes]

    N_TIME = result_df_y.shape[1]  # 100

    # reference frequencies for log-period axis & band split
    dt_ref = 0.1
    frequencies_ref = pywt.scale2frequency(wavelet_name, scales) / dt_ref
    period_ref = 1.0 / (frequencies_ref + 1e-12)
    log_period_ref = np.log2(period_ref + 1e-12)

    def _band_indices_by_logp(frequencies):
        freq = np.asarray(frequencies, dtype=float)
        period = 1.0 / (freq + 1e-12)
        logp = np.log2(period + 1e-12)
        b1 = np.quantile(logp, 1/3)
        b2 = np.quantile(logp, 2/3)
        high_idx = np.where(logp <= b1)[0]
        mid_idx  = np.where((logp > b1) & (logp <= b2))[0]
        low_idx  = np.where(logp > b2)[0]
        return low_idx, mid_idx, high_idx

    LOW_IDX, MID_IDX, HIGH_IDX = _band_indices_by_logp(frequencies_ref)

    # ----------------------------
    # 3) marker selection (same logic)
    # ----------------------------
    def wavelet_feature_matrix(df_y, scales_used=scales):
        dt = float(x_smooth_dis[1] - x_smooth_dis[0])
        feats = []
        for g in df_y.index:
            sig = df_y.loc[g].values.astype(float)
            coeffs, _ = pywt.cwt(sig, scales_used, wavelet_name, sampling_period=dt)
            power = (np.abs(coeffs)) ** 2
            power = pywt.threshold(power, threshold, mode="soft")
            v = np.sum(power, axis=1)
            vmax = float(v.max())
            if vmax > 0:
                v = v / vmax
            feats.append(v)
        X = np.vstack(feats)
        return pd.DataFrame(X, index=df_y.index, columns=[f"s{i}" for i in range(X.shape[1])])

    feat_dis  = wavelet_feature_matrix(result_df_y,  scales_used=scales)
    feat_ctrl = wavelet_feature_matrix(result_df_y_c, scales_used=scales)

    merged_df = feat_dis.merge(feat_ctrl, left_index=True, right_index=True, how="inner", suffixes=("_dis", "_ctrl"))
    adata_mark = ad.AnnData(merged_df.T)

    adata_mark.obs["cluster"] = "NA"
    adata_mark.obs.loc[adata_mark.obs.index[:len(scales)], "cluster"] = disease_label
    adata_mark.obs.loc[adata_mark.obs.index[len(scales):2 * len(scales)], "cluster"] = control_label

    df_gid = pd.read_csv(gene_id_csv)
    # keep your behavior: map var_names from gene_id list
    adata_mark.var_names = list(df_gid["gene_id"][: adata_mark.shape[1]])
    adata_mark.var_names_make_unique()

    sc.pp.filter_genes(adata_mark, min_cells=3)
    sc.pp.normalize_total(adata_mark)

    sc.tl.rank_genes_groups(
        adata_mark, "cluster", groups=[disease_label], reference=control_label, method="t-test"
    )
    ranked = adata_mark.uns["rank_genes_groups"]

    p_adj = pd.Series(ranked["pvals_adj"][disease_label], index=ranked["names"][disease_label]).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    logfc = pd.Series(ranked["logfoldchanges"][disease_label], index=ranked["names"][disease_label]).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    marker_mask = (p_adj < padj_cutoff) & (logfc.abs() > abs_logfc_cutoff)
    marker_genes = [g for g in p_adj.index[marker_mask] if g in common_genes]
    marker_set = set(marker_genes)

    if verbose:
        print(f"[INFO] common_genes = {len(common_genes)}")
        print(f"[INFO] markers(p_adj<{padj_cutoff},|logFC|>{abs_logfc_cutoff}) = {len(marker_genes)}")
        print(f"[INFO] non-markers = {len(common_genes) - len(marker_genes)}")

    # ----------------------------
    # 4) DIFF power (two versions as yours)
    # ----------------------------
    def compute_diff_power_for_gene(gene):
        sig_dis  = result_df_y.loc[gene].values.astype(float)
        sig_ctrl = result_df_y_c.loc[gene].values.astype(float)

        dt_dis  = float(x_smooth_dis[1] - x_smooth_dis[0])
        dt_ctrl = float(x_smooth_ctrl[1] - x_smooth_ctrl[0])

        coeffs_dis, _  = pywt.cwt(sig_dis,  scales, wavelet_name, sampling_period=dt_dis)
        coeffs_ctrl, _ = pywt.cwt(sig_ctrl, scales, wavelet_name, sampling_period=dt_ctrl)

        power_dis  = (np.abs(coeffs_dis)) ** 2
        power_ctrl = (np.abs(coeffs_ctrl)) ** 2

        diff = power_dis - power_ctrl
        if keep_only_positive_diff:
            diff[diff < 0] = 0.0
        return diff

    def compute_diff_power_for_gene_y(gene):
        sig_dis  = result_df_y.loc[gene].values.astype(float)
        sig_ctrl = result_df_y_c.loc[gene].values.astype(float)

        dt_dis  = float(x_smooth_dis[1] - x_smooth_dis[0])
        dt_ctrl = float(x_smooth_ctrl[1] - x_smooth_ctrl[0])

        coeffs_dis, _  = pywt.cwt(sig_dis,  scales, wavelet_name, sampling_period=dt_dis)
        coeffs_ctrl, _ = pywt.cwt(sig_ctrl, scales, wavelet_name, sampling_period=dt_ctrl)

        power_dis  = (np.abs(coeffs_dis)) ** 2
        power_ctrl = (np.abs(coeffs_ctrl)) ** 2

        mx_dis = float(np.max(power_dis))
        mx_ctrl = float(np.max(power_ctrl))
        if mx_dis > 0:
            power_dis = power_dis / mx_dis
        if mx_ctrl > 0:
            power_ctrl = power_ctrl / mx_ctrl

        diff = power_dis - power_ctrl
        if keep_only_positive_diff:
            diff[diff < 0] = 0.0
        return diff

    # ----------------------------
    # 5) Global style (weighted by |logFC|)
    # ----------------------------
    def build_global_diff_weighted():
        genes = list(common_genes)
        acc = None
        wsum = 0.0
        n_used = 0
        for g in genes:
            w = float(abs(logfc.get(g, 1.0)))
            if (not np.isfinite(w)) or (w <= 0):
                continue
            try:
                mat = compute_diff_power_for_gene(g)
            except Exception:
                continue
            if acc is None:
                acc = np.zeros_like(mat, dtype=float)
            acc += w * mat
            wsum += w
            n_used += 1
        if acc is None or wsum <= 0:
            raise RuntimeError("Failed to build global DIFF: no genes contributed.")
        return acc / wsum, n_used

    def infer_style_from_BT(global_diff):
        P = np.asarray(global_diff, dtype=float)
        P[P < 0] = 0.0

        A_HF = float(np.sum(P[HIGH_IDX, :])) if len(HIGH_IDX) else 0.0
        A_LF = float(np.sum(P[LOW_IDX,  :])) if len(LOW_IDX)  else 0.0

        B = float(math.log(A_HF + EPS_STYLE))
        T = float(math.log((A_HF + EPS_STYLE) / (A_LF + EPS_STYLE)))

        zB = (B - SIGMOID_MU_B) / (SIGMOID_SCALE_B + 1e-12)
        r_pitch = float(sigmoid(zB))

        zT = (T - SIGMOID_MU_T) / (SIGMOID_SCALE_T + 1e-12)
        r_rhythm = float(sigmoid(zT))

        pitch_shift = int(round(clamp(-10 + 22 * r_pitch, -12, +20)))

        low_midi_base  = np.array([36, 40, 43, 45, 48], dtype=int)
        mid_midi_base  = np.array([60, 62, 64, 65, 67, 69, 71], dtype=int)
        high_midi_base = np.array([72, 74, 76, 77, 79], dtype=int)

        if r_pitch < 0.35:
            mid_midi_base  = np.array([60, 62, 64, 67, 69], dtype=int)
            high_midi_base = np.array([72, 74, 76], dtype=int)
        elif r_pitch < 0.65:
            mid_midi_base  = np.array([60, 62, 64, 65, 67, 69], dtype=int)
            high_midi_base = np.array([72, 74, 76, 79], dtype=int)
        else:
            mid_midi_base  = np.array([60, 61, 62, 64, 65, 67, 69, 71, 72], dtype=int)
            high_midi_base = np.array([72, 74, 76, 77, 79, 81], dtype=int)

        low_midi  = (low_midi_base  + pitch_shift).tolist()
        mid_midi  = (mid_midi_base  + pitch_shift).tolist()
        high_midi = (high_midi_base + pitch_shift).tolist()

        bpm = float(clamp(bpm_min + (bpm_max - bpm_min) * r_rhythm, bpm_min, bpm_max))

        energy_thr = float(clamp(0.32 - 0.28 * r_rhythm, 0.04, 0.36))
        min_run = int(round(clamp(4 - 3 * r_rhythm, 1, 4)))

        target_beats = float(clamp(
            total_beats_max - (total_beats_max - total_beats_min) * r_rhythm,
            total_beats_min, total_beats_max
        ))

        time_signature = "4/4"
        key_sig = int(round(clamp(-3 + 6 * r_rhythm, -5, +5)))

        energy_t = np.sum(P, axis=0)
        if float(np.max(energy_t)) > 0:
            e = energy_t / float(np.max(energy_t))
        else:
            e = np.zeros_like(energy_t, dtype=float)

        if r_rhythm < 0.35:
            lo_u, hi_u = 3, 6
        elif r_rhythm < 0.65:
            lo_u, hi_u = 2, 5
        else:
            lo_u, hi_u = 1, 3

        units = []
        span = (hi_u - lo_u)
        for i in range(len(e)):
            u = (lo_u + hi_u) / 2.0 + span * (0.55 * e[i] + 0.10 * math.sqrt(float(e[i]) + 1e-12)) - (0.35 * span * r_rhythm)
            u = int(round(clamp(u, lo_u, hi_u)))
            units.append(u)

        target_units = int(round(target_beats * 32))
        total_units = int(sum(units))
        if total_units > 0:
            scale_factor = target_units / total_units
            units_scaled = []
            for u in units:
                u2 = int(round(u * scale_factor))
                u2 = int(clamp(u2, lo_u, hi_u))
                units_scaled.append(u2)
            units = units_scaled

        frame_ql = [Fraction(u, 32) for u in units]
        total_ql = sum(frame_ql, start=Fraction(0, 1))
        target_seconds = float(total_ql) * 60.0 / bpm
        frame_sec = float(target_seconds / N_TIME)

        return {
            "A_HF": A_HF,
            "A_LF": A_LF,
            "B": B,
            "T": T,
            "r_pitch": r_pitch,
            "r_rhythm": r_rhythm,
            "pitch_shift": pitch_shift,
            "energy_threshold": energy_thr,
            "bpm": bpm,
            "time_signature": time_signature,
            "key_sig": key_sig,
            "low_midi": low_midi,
            "mid_midi": mid_midi,
            "high_midi": high_midi,
            "frame_ql": frame_ql,
            "total_ql": total_ql,
            "target_seconds": target_seconds,
            "frame_sec": frame_sec,
            "min_run_frames": min_run,
        }

    GLOBAL_DIFF, N_USED = build_global_diff_weighted()
    STYLE = infer_style_from_BT(GLOBAL_DIFF)

    BPM_TAG = f"BPM{int(round(STYLE['bpm']))}"
    if verbose:
        print(f"[GLOBAL_DIFF] built from n={N_USED} genes (weighted by |logFC| when available)")
        print(f"[CALIB_B] MU_B={SIGMOID_MU_B:.4f} SCALE_B={SIGMOID_SCALE_B:.4f}")
        print(f"[CALIB_T] MU_T={SIGMOID_MU_T:.4f} SCALE_T={SIGMOID_SCALE_T:.4f}")
        print(
            f"[STYLE] A_HF={STYLE['A_HF']:.4g} | A_LF={STYLE['A_LF']:.4g} | "
            f"B=log(A_HF)={STYLE['B']:.3f} -> r_pitch={STYLE['r_pitch']:.3f} | "
            f"T=log(HF/LF)={STYLE['T']:.3f} -> r_rhythm={STYLE['r_rhythm']:.3f} | "
            f"pitch_shift={STYLE['pitch_shift']:+d} | energy_thr={STYLE['energy_threshold']:.3f} | "
            f"BPM={STYLE['bpm']:.2f} | TS={STYLE['time_signature']} | KS={STYLE['key_sig']} | "
            f"min_run={STYLE['min_run_frames']} | TOTAL_BEATS={float(STYLE['total_ql']):.3f} | "
            f"len≈{STYLE['target_seconds']:.2f}s (frame_sec={STYLE['frame_sec']:.4f})"
        )

    # ----------------------------
    # 6) Wavelet power -> note events (NO AUDIO)
    # ----------------------------
    def wavelet_power_to_events(power_mat, style):
        coeffs = np.asarray(power_mat, dtype=float)
        coeffs[coeffs < 0] = 0.0
        _, n_time = coeffs.shape
        assert n_time == N_TIME, "Expected n_time == 100"

        energy_thr = float(style["energy_threshold"])
        min_run = int(style.get("min_run_frames", 2))

        low_midi  = np.array(style["low_midi"], dtype=int)
        mid_midi  = np.array(style["mid_midi"], dtype=int)
        high_midi = np.array(style["high_midi"], dtype=int)

        globals_max = {}
        band_energy_all = {}
        for name, idxs in {"low": LOW_IDX, "mid": MID_IDX, "high": HIGH_IDX}.items():
            if len(idxs) == 0:
                globals_max[name] = 1.0
                band_energy_all[name] = np.zeros(n_time)
            else:
                e = np.sum(coeffs[idxs, :], axis=0)
                band_energy_all[name] = e
                gm = np.quantile(e, 0.95)
                globals_max[name] = gm if gm > 0 else 1.0

        band_energy_norm = {k: band_energy_all[k] / globals_max[k] for k in band_energy_all.keys()}
        note_idx_seq = {"low": [None] * n_time, "mid": [None] * n_time, "high": [None] * n_time}

        for ti in range(n_time):
            amps = coeffs[:, ti]
            for vname, idxs, midi_list in [
                ("low",  LOW_IDX,  low_midi),
                ("mid",  MID_IDX,  mid_midi),
                ("high", HIGH_IDX, high_midi),
            ]:
                if len(idxs) == 0:
                    continue
                band = amps[idxs]
                energy = float(band.sum())
                if energy <= 0:
                    note_idx_seq[vname][ti] = None
                    continue
                e_norm = float(band_energy_norm[vname][ti])
                if e_norm < energy_thr:
                    note_idx_seq[vname][ti] = None
                    continue

                loc = int(np.argmax(band))
                scale_idx = int(idxs[loc])
                if len(idxs) > 1:
                    pos = (scale_idx - idxs[0]) / (idxs[-1] - idxs[0])
                else:
                    pos = 0.0

                if len(midi_list) > 1:
                    note_idx = int(round(pos * (len(midi_list) - 1)))
                else:
                    note_idx = 0
                note_idx = max(0, min(len(midi_list) - 1, note_idx))
                note_idx_seq[vname][ti] = note_idx

        def remove_short_runs(seq, min_len):
            if min_len <= 1:
                return seq
            out = list(seq)
            i = 0
            while i < len(out):
                val = out[i]
                j = i + 1
                while j < len(out) and out[j] == val:
                    j += 1
                run_len = j - i
                if val is not None and run_len < min_len:
                    for k in range(i, j):
                        out[k] = None
                i = j
            return out

        note_idx_seq["low"]  = remove_short_runs(note_idx_seq["low"],  min_run)
        note_idx_seq["mid"]  = remove_short_runs(note_idx_seq["mid"],  min_run)
        note_idx_seq["high"] = remove_short_runs(note_idx_seq["high"], min_run)

        def compress(seq, vname):
            events = []
            current = None
            start = 0
            midi_list = {"low": low_midi, "mid": mid_midi, "high": high_midi}[vname]
            energy_series = band_energy_norm[vname]

            for i, idx in enumerate(seq + [None]):
                if idx != current:
                    if current is not None:
                        dur = i - start
                        midi_pitch = int(midi_list[current])
                        e_mean = float(np.mean(energy_series[start:i])) if dur > 0 else 0.0
                        events.append((start, dur, midi_pitch, e_mean))
                    current = idx
                    start = i
            return events

        return {
            "low":  compress(note_idx_seq["low"],  "low"),
            "mid":  compress(note_idx_seq["mid"],  "mid"),
            "high": compress(note_idx_seq["high"], "high"),
        }

    # ----------------------------
    # 7) Events -> MusicXML
    # ----------------------------
    def frames_to_ql(start_f, dur_f, frame_ql):
        if dur_f <= 0:
            return Fraction(0, 1)
        return sum(frame_ql[start_f:start_f + dur_f], start=Fraction(0, 1))

    def pad_to_measure_multiple(part, measure_len_ql):
        total = Fraction(part.highestTime).limit_denominator(1024)
        if total <= 0:
            return
        rem = total % measure_len_ql
        if rem != 0:
            need = measure_len_ql - rem
            r = m21note.Rest()
            r.quarterLength = float(need)
            part.append(r)

    def note_events_to_score_musicxml(note_events, out_base_noext, style, instrument_mode="piano", title_text=None):
        bpm = float(style["bpm"])
        ts = str(style["time_signature"])
        key_sig = int(style["key_sig"])
        frame_ql = style["frame_ql"]

        num, den = ts.split("/")
        num = int(num)
        den = int(den)
        measure_len = Fraction(num * 4, den)

        def _make_inst(mode: str):
            mode = str(mode).lower()
            if mode == "piano":
                inst = instrument.Piano()
                inst.partName = "Piano"
            elif mode == "flute":
                inst = instrument.Flute()
                inst.partName = "Flute"
            elif mode == "violin":
                inst = instrument.Violin()
                inst.partName = "Violin"
            else:
                inst = instrument.Instrument()
                inst.partName = mode
            return inst

        sc_score = stream.Score()

        # title + remove composer
        if title_text is not None:
            md = m21metadata.Metadata()
            md.title = str(title_text)
            md.composer = ""  # remove "music21" credit
            sc_score.insert(0, md)

        sc_score.insert(0, tempo.MetronomeMark(number=bpm))
        sc_score.insert(0, meter.TimeSignature(ts))
        sc_score.insert(0, m21key.KeySignature(key_sig))

        voice_order = ["high", "mid", "low"]
        for v_name in voice_order:
            part = stream.Part()
            part.id = v_name

            part.insert(0, _make_inst(instrument_mode))
            part.insert(0, meter.TimeSignature(ts))
            part.insert(0, m21key.KeySignature(key_sig))
            part.insert(0, tempo.MetronomeMark(number=bpm))  # ensure per-part tempo

            events = note_events[v_name]
            current_frame = 0
            current_time = Fraction(0, 1)
            last_dyn = None

            for start_f, dur_f, midi_pitch, energy in events:
                if start_f > current_frame:
                    rest_ql = frames_to_ql(current_frame, start_f - current_frame, frame_ql)
                    if rest_ql > 0:
                        r = m21note.Rest()
                        r.quarterLength = float(rest_ql)
                        part.append(r)
                        current_time += rest_ql

                note_ql = frames_to_ql(start_f, dur_f, frame_ql)
                if note_ql <= 0:
                    current_frame = start_f + dur_f
                    continue

                dyn = energy_to_dynamic_mark(energy)
                if dyn != last_dyn:
                    part.insert(float(current_time), dynamics.Dynamic(dyn))
                    last_dyn = dyn

                n = m21note.Note(int(midi_pitch))
                n.quarterLength = float(note_ql)
                part.append(n)

                current_time += note_ql
                current_frame = start_f + dur_f

            if current_frame < N_TIME:
                tail_ql = frames_to_ql(current_frame, N_TIME - current_frame, frame_ql)
                if tail_ql > 0:
                    r_tail = m21note.Rest()
                    r_tail.quarterLength = float(tail_ql)
                    part.append(r_tail)
                    current_time += tail_ql

            pad_to_measure_multiple(part, measure_len)
            part.makeMeasures(inPlace=True)
            sc_score.append(part)

        out_xml = out_base_noext + ".musicxml"
        sc_score.write("musicxml", fp=out_xml)

        # hide printed tempo; keep playback
        hide_tempo_marks_in_musicxml(out_xml)
        # first system show names, later hide
        set_first_system_names_hide_later(out_xml, voice_names=("high", "mid", "low"))

        return out_xml

    # ----------------------------
    # 8) Export one gene (PDF + 3x MusicXML)
    # ----------------------------
    def export_diff_for_gene(gene, rank=None, prefix="DIFF", category_tag="NA"):
        gene_s = sanitize_gene_name(gene)
        ts_tag = STYLE["time_signature"].replace("/", "-")
        ks = int(STYLE["key_sig"])
        cat = str(category_tag).upper()

        if rank is not None:
            base = f"{prefix}_{cat}_{disease_label}-minus-{control_label}_{BPM_TAG}_TS{ts_tag}_KS{ks}_top{rank:03d}_{gene_s}"
        else:
            base = f"{prefix}_{cat}_{disease_label}-minus-{control_label}_{BPM_TAG}_TS{ts_tag}_KS{ks}_{gene_s}"

        diff_power = compute_diff_power_for_gene_y(gene)

        lfc = float(logfc.get(gene, 0.0))
        gain = abs(lfc)
        gain = min(gain, float(gain_clip[1]))
        gain = max(gain, float(gain_clip[0]))

        diff_power = diff_power * gain
        diff_power[diff_power < float(diff_power_floor)] = 0.0

        note_events = wavelet_power_to_events(diff_power, STYLE)

        cat_human = "Marker" if cat.startswith("MARKER") else ("Non-Marker" if cat.startswith("NON") else str(category_tag))
        title_text = f"{disease_label} {cat_human} {gene}"

        # MusicXML variants
        for tag, _human, outdir in MUSICXML_INSTRUMENT_VARIANTS:
            out_base_noext = os.path.join(outdir, f"{base}_{tag}")
            note_events_to_score_musicxml(
                note_events=note_events,
                out_base_noext=out_base_noext,
                style=STYLE,
                instrument_mode=tag,
                title_text=title_text,
            )

        # PDF
        power_vis = diff_power.copy()
        power_vis = power_vis * float(pdf_power_scale_mul) + float(pdf_power_scale_add)

        levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8]
        contourlevels = np.log2(levels)

        fig, ax = plt.subplots(figsize=(15, 10))
        im = ax.contourf(
            x_smooth_dis,
            log_period_ref,
            power_vis,
            levels=contourlevels,
            cmap=plt.cm.seismic,
            extend="both",
        )
        ax.set_title(f"Wavelet Power DIFF ({disease_label} - {control_label})\n{gene}", fontsize=22)
        ax.set_ylabel("1/Frequency (Period)", fontsize=18)
        ax.set_xlabel("Time", fontsize=18)

        yticks = 2 ** np.arange(np.floor(log_period_ref.min()), np.ceil(log_period_ref.max()))
        ax.set_yticks(np.log2(yticks))
        ax.set_yticklabels(yticks)
        ax.invert_yaxis()

        cbar_ax = fig.add_axes([0.95, 0.5, 0.03, 0.25])
        fig.colorbar(im, cax=cbar_ax, orientation="vertical")

        out_pdf = os.path.join(outdir_pdf, base + ".pdf")
        plt.savefig(out_pdf, bbox_inches="tight")
        plt.close()

        if verbose:
            print(f"[OK] {base}  (PDF + 3xMusicXML)")

    # ----------------------------
    # 9) Run export (ALL markers + ALL non-markers)
    # ----------------------------
    markers_sorted = marker_genes.copy()
    markers_sorted.sort(key=lambda g: float(abs(logfc.get(g, 0.0))), reverse=True)
    if topn_markers_to_export is not None:
        markers_sorted = markers_sorted[: int(topn_markers_to_export)]

    if verbose:
        print(f"[EXPORT] DIFF-only marker genes: n={len(markers_sorted)}")
    for rk, g in enumerate(markers_sorted, start=1):
        export_diff_for_gene(gene=g, rank=rk, prefix="DIFF", category_tag="MARKER")

    nonmarker_candidates = [g for g in list(common_genes) if g not in marker_set]
    nonmarker_candidates.sort(key=lambda g: float(abs(logfc.get(g, 0.0))))
    if topn_nonmarkers_to_export is not None:
        nonmarkers_pick = nonmarker_candidates[: int(topn_nonmarkers_to_export)]
    else:
        nonmarkers_pick = nonmarker_candidates

    if verbose:
        print(f"[EXPORT] DIFF-only non-marker genes: n={len(nonmarkers_pick)}")
    for rk, g in enumerate(nonmarkers_pick, start=1):
        export_diff_for_gene(gene=g, rank=rk, prefix="DIFF", category_tag="NONMARKER")

    if verbose:
        print("Done (ALL markers + ALL non-markers).")

    # return useful stuff for debugging / downstream
    return {
        "STYLE": STYLE,
        "marker_genes": marker_genes,
        "common_genes": list(common_genes),
        "logfc": logfc,
        "p_adj": p_adj,
        "outdirs": {
            "pdf": outdir_pdf,
            "piano": outdir_mxml_piano,
            "flute": outdir_mxml_flute,
            "violin": outdir_mxml_violin,
        },
    }



