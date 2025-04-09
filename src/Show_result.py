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


def plot_wavelet(time, signal, scales, cluster_num, m,label=0,
                 waveletname='mexh',
                 cmap=plt.cm.seismic,
                 title='Wavelet Transform (Power Spectrum) of signal',
                 ylabel='1/Frequency',
                 xlabel='Time'):
    dt = time[1] - time[0]
    [coefficients, frequencies] = pywt.cwt(signal, scales, waveletname, dt)
    power = (abs(coefficients)) ** 2
    period = 1. / frequencies
    levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8]
    contourlevels = np.log2(levels)

    fig, ax = plt.subplots(figsize=(15, 10))
    im = ax.contourf(range(50), np.log2(period), np.log2(power), contourlevels, extend='both', cmap=cmap)

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
    filename = 'figure_of_time_period_' + cluster_num+'_path_'+str(m) + '.pdf'
    if label == 1:
        plt.show()

    plt.savefig(filename, bbox_inches="tight")
    plt.close()


def show_result(gene_info,trajectory_info,latent='ALL_mu.npy'):

    data = np.load(latent)
    adata = ad.AnnData(data)
    adata.obs_names = gene_info.loc[:, "gene_id"]
    adata.var_names = ["C" + str(i) for i in range(20)]
    adata.obs['indexa'] = range(len(adata.obs_names))

    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=10)
    sc.tl.umap(adata)

    sc.tl.leiden(adata,resolution=1)


    for i in range(len(gene_info)):
        gene_info.loc[i, "gene_id"] = gene_info.loc[i, "gene_id"][:18]

    adata.obs_names = gene_info.loc[:, "gene_id"]


    a = pd.Index(trajectory_info.loc["path"] == 1)  # dataset1
    y = np.array(trajectory_info.iloc[:-5, a])
    x = np.array(trajectory_info.loc["time", a])

    x.sort()
    x_smooth = np.linspace(start=x.min(), stop=x.max(), num=100)
    y_smooth = make_interp_spline(x, y.transpose())(x_smooth)
    yf_smooth = fft(y_smooth.transpose())
    # Perform wavelet transform
    wavelet = 'db4'
    coeffs = pywt.wavedec(y_smooth.transpose(), wavelet)

    # Thresholding
    threshold = 0.01  # Set your threshold value here
    coeffs_thresh = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]

    # Reconstructing the signal
    yf_smooth = pywt.waverec(coeffs_thresh, wavelet)
    yf_info = np.array([abs(yf_smooth[i, :50]) for i in range(yf_smooth.shape[0])])

    for i in range(len(yf_info)):
        if yf_info[i].max() == 0:
            continue
        yf_info[i] = yf_info[i] / yf_info[i].max()

    for i in range(len(y)):
        y[i, :] = y[i, :] - y[i, 0]


    cluster = adata.obs[adata.obs['leiden'] == '0']['indexa']


    def movingaverage(data, window_size):
        window = np.ones(int(window_size)) / float(window_size)
        a = np.convolve(data[window_size:], window, 'valid')
        return np.concatenate([np.zeros(window_size), a])

    print_first = [1,1,1]

    for j in range(200):
        cluster_num = str(j)
        cluster = adata.obs[adata.obs['leiden'] == cluster_num]['indexa']

        error_t = []
        for i in range(len(cluster)):
            error_t.append(np.max(y[cluster[i], :] - np.min(y[cluster[i], :])))

        error_index_t = np.argsort(error_t)
        error_t = np.array(error_t)
        num = 0
        cluster_filter = []
        yf_info_filter = []
        cluater_index = []
        name = []
        for i in error_index_t[:]:
            a = movingaverage(y[cluster[i], :], 5)
            if abs(np.max(abs(a))) < 0.01 or abs(np.max(abs(a))) > 5:
                continue
            num = num + 1
            cluster_filter.append(a)
            cluater_index.append(cluster[i])
            yf_info_filter.append(yf_info[cluster[i], :])
            name.append(adata.obs_names[cluster[i]])
        cluster_filter = np.array(cluster_filter)
        cluater_index = np.array(cluater_index)
        yf_info_filter = np.array(yf_info_filter)


        color_map = ['b', 'r', 'y', 'c', 'g']
        n_cluster = 5
        index = [[] for i in range(n_cluster)]
        namei = [[] for i in range(n_cluster)]

        if len(cluster_filter) >= 5:
            y_pred = KMeans(n_clusters=n_cluster, random_state=9).fit_predict(cluster_filter)
            color = []
            number_of_genes = np.zeros([n_cluster])
            for i in range(len(y_pred)):
                index[y_pred[i]].append(i)
                namei[y_pred[i]].append(name[i])
            for i in y_pred:
                number_of_genes[i] = number_of_genes[i] + 1
                color.append(color_map[i])
            for i in range(len(cluster_filter)):
                if number_of_genes[y_pred[i]] < 5:
                    continue

            cluster_num = cluster_num

            avg = np.zeros([n_cluster, cluster_filter.shape[1]])
            std = np.zeros([n_cluster, cluster_filter.shape[1]])
            yf_avg = np.zeros([n_cluster, yf_info_filter.shape[1]])
            for i in range(n_cluster):
                avg[i] = np.mean(cluster_filter[index[i]], axis=0)
                std[i] = np.std(cluster_filter[index[i]], axis=0)
                yf_avg[i] = np.mean(yf_info_filter[index[i]], axis=0)
            m = 0
            for i in range(n_cluster):
                if number_of_genes[i] < 5:
                    continue
                plt.errorbar(x[:-4], avg[i], yerr=std[i], c=color_map[i], label = 'path'+ str(m))
                m = m+1
            plt.xlabel("Pseudotime", fontdict={'family': 'Arial', 'size': 18})
            plt.ylabel("Expression log2 fold change", fontdict={'family': 'Arial', 'size': 18})
            plt.title('cluster' + str(j), fontdict={'family': 'Arial', 'size': 18})
            plt.legend()
            filename = 'figure_of_ds1_avg_time_data_' + cluster_num + '.pdf'
            plt.savefig(filename, bbox_inches="tight")
            if print_first[0] == 1:
                plt.show()
                print_first[0] = 0

            plt.close()
            avgf_smooth = fft(avg.transpose())





            m = 0
            for i in range(n_cluster):
                if number_of_genes[i] < 5:
                    continue
                plt.plot(range(len(yf_avg[i])), yf_avg[i], c=color_map[i], label = 'path'+str(m))
                m = m+1

            plt.xlabel("Frequency", fontdict={'family': 'Arial', 'size': 18})
            plt.ylabel("Amplitude", fontdict={'family': 'Arial', 'size': 18})
            plt.title('cluster' + str(j), fontdict={'family': 'Arial', 'size': 18})
            plt.legend()
            filename = 'figure_of_ds1_frequency_' + cluster_num + '.pdf'
            plt.savefig(filename, bbox_inches="tight")
            if print_first[1] == 1:
                plt.show()
                print_first[1] = 0
            plt.close()
            plt.gcf().clear()

            scales = np.arange(1, 128)
            m = 0
            for i in range(n_cluster):
                if number_of_genes[i] < 5:
                    continue

                plot_wavelet(x[:-4], yf_avg[i], scales, cluster_num, m, print_first[2])
                print_first[2] = 0

                m = m + 1

            m = 0
            for i in range(n_cluster):
                if number_of_genes[i] < 5:
                    continue
                plt.plot(range(cluster_filter.shape[1]), avg[i], c = color_map[i])
                mg = mygene.MyGeneInfo()
                gene_ids = mg.getgenes(namei[i], 'name, symbol, entrezgene', as_dataframe=True)
                gene_ids.index.name = "UNIPROT"
                gene_ids.reset_index(inplace=True)

                gene_symbols = gene_ids['symbol']
                gene_symbols.to_csv('Gene_' + cluster_num + '_path_' + str(m) + '.csv')
                m = m+1






def differential_frequency(dataset1,dataset2,dataset1_name,dataset2_name,features):

    trajectory_info = pd.read_csv(dataset1)  # dataset1

    a = pd.Index(trajectory_info.loc["path"] == 1)  # dataset1
    y = np.array(trajectory_info.iloc[:-5, a])
    x = np.array(trajectory_info.loc["time", a])

    x.sort()
    x_smooth = np.linspace(start=x.min(), stop=x.max(), num=100)
    y_smooth = make_interp_spline(x, y.transpose())(x_smooth)
    yf_smooth = fft(y_smooth.transpose())
    # Perform wavelet transform
    wavelet = 'db4'
    coeffs = pywt.wavedec(y_smooth.transpose(), wavelet)

    # Thresholding
    threshold = 0.01  # Set your threshold value here
    coeffs_thresh = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]

    # Reconstructing the signal
    yf_smooth = pywt.waverec(coeffs_thresh, wavelet)
    yf_info = np.array([abs(yf_smooth[i, :50]) for i in range(yf_smooth.shape[0])])

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
    wavelet = 'db4'
    coeffs = pywt.wavedec(y_smooth.transpose(), wavelet)

    # Thresholding
    threshold = 0.01  # Set your threshold value here
    coeffs_thresh = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]

    # Reconstructing the signal
    yf_smooth = pywt.waverec(coeffs_thresh, wavelet)
    yf_info = np.array([abs(yf_smooth[i, :50]) for i in range(yf_smooth.shape[0])])

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





def show_result_spatial(adata_s, start, end,latent='ALL_mu.npy'):
    gene_info = adata_s
    data = np.load(latent)
    adata = ad.AnnData(data)
    adata.obs_names = gene_info.var_names
    adata.var_names = ["C" + str(i) for i in range(20)]
    adata.obs['indexa'] = range(len(adata.obs_names))

    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=10)
    sc.tl.umap(adata)

    sc.tl.leiden(adata, resolution=1)

    adata.obs_names = gene_info.var_names

    adata1 = adata_s

    # Extract spatial coordinates
    spatial_coords = adata1.obsm["spatial"]

    # Define a single line from bottom-left to top-right
    num_points = 10  # Number of center points along the line
    distances = np.linspace(0, 1, num_points)  # Fractional distances along the line
    center_points = np.array([start + d * (end - start) for d in distances])

    # Initialize a list to store mean counts for each center point
    trajectory_means = []

    # For each center point, find the 100 nearest cells and compute mean counts
    for point in center_points:
        distances = pairwise_distances([point], spatial_coords)[0]
        nearest_indices = np.argsort(distances)[:100]  # Get indices of 100 nearest cells
        mean_count = np.mean(adata1.X[nearest_indices, :].toarray(), axis=0)  # Compute mean counts
        trajectory_means.append(mean_count)  # Flatten to ensure 1D array

    # Convert trajectory_means to a numpy array for further processing
    trajectory_array = np.array(trajectory_means)
    y = trajectory_array.T
    # Smooth the data for frequency analysis
    x = np.arange(trajectory_array.shape[0])
    x_smooth = np.linspace(start=x.min(), stop=x.max(), num=100)
    y_smooth = make_interp_spline(x, trajectory_array, k=3)(x_smooth)

    yf_smooth = fft(y_smooth.transpose())
    # Perform wavelet transform
    wavelet = 'db4'
    coeffs = pywt.wavedec(y_smooth.transpose(), wavelet)

    # Thresholding
    threshold = 0.01  # Set your threshold value here
    coeffs_thresh = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]

    # Reconstructing the signal
    yf_smooth = pywt.waverec(coeffs_thresh, wavelet)
    yf_info = np.array([abs(yf_smooth[i, :50]) for i in range(yf_smooth.shape[0])])

    for i in range(len(yf_info)):
        if yf_info[i].max() == 0:
            continue
        yf_info[i] = yf_info[i] / yf_info[i].max()

    for i in range(len(y)):
        y[i, :] = y[i, :] - y[i, 0]

    cluster = adata.obs[adata.obs['leiden'] == '0']['indexa']


    def movingaverage(data, window_size):
        window = np.ones(int(window_size)) / float(window_size)
        a = np.convolve(data[window_size:], window, 'valid')
        return np.concatenate([np.zeros(window_size), a])

    print_first = [0,0,0]
    for j in range(200):
        cluster_num = str(j)
        cluster = adata.obs[adata.obs['leiden'] == cluster_num]['indexa']

        error_t = []
        for i in range(len(cluster)):
            error_t.append(np.max(y[cluster[i], :] - np.min(y[cluster[i], :])))

        error_index_t = np.argsort(error_t)
        error_t = np.array(error_t)
        num = 0
        cluster_filter = []
        yf_info_filter = []
        cluater_index = []
        name = []
        for i in error_index_t[:]:
            a = movingaverage(y[cluster[i], :], 5)
            if abs(np.max(abs(a))) < 0.01 or abs(np.max(abs(a))) > 5:
                continue
            num = num + 1
            cluster_filter.append(a)
            cluater_index.append(cluster[i])
            yf_info_filter.append(yf_info[cluster[i], :])
            name.append(adata.obs_names[cluster[i]])

        cluster_filter = np.array(cluster_filter)
        cluater_index = np.array(cluater_index)
        yf_info_filter = np.array(yf_info_filter)

        color_map = ['b', 'r', 'y', 'c', 'g']
        n_cluster = 5
        index = [[] for i in range(n_cluster)]
        namei = [[] for i in range(n_cluster)]

        if len(cluster_filter) >= 5:
            y_pred = KMeans(n_clusters=n_cluster, random_state=9).fit_predict(cluster_filter)
            color = []
            number_of_genes = np.zeros([n_cluster])
            for i in range(len(y_pred)):
                index[y_pred[i]].append(i)
                namei[y_pred[i]].append(name[i])
            for i in y_pred:
                number_of_genes[i] = number_of_genes[i] + 1
                color.append(color_map[i])
            for i in range(len(cluster_filter)):
                if number_of_genes[y_pred[i]] < 5:
                    continue

            cluster_num = cluster_num

            avg = np.zeros([n_cluster, cluster_filter.shape[1]])
            std = np.zeros([n_cluster, cluster_filter.shape[1]])
            yf_avg = np.zeros([n_cluster, yf_info_filter.shape[1]])
            for i in range(n_cluster):
                avg[i] = np.mean(cluster_filter[index[i]], axis=0)
                std[i] = np.std(cluster_filter[index[i]], axis=0)
                yf_avg[i] = np.mean(yf_info_filter[index[i]], axis=0)
            m = 0
            for i in range(n_cluster):
                if number_of_genes[i] < 5:
                    continue
                plt.errorbar(x[:-4], avg[i], yerr=std[i], c=color_map[i], label='path' + str(m))
                m = m + 1
            plt.xlabel("Pseudotime", fontdict={'family': 'Arial', 'size': 18})
            plt.ylabel("Expression log2 fold change", fontdict={'family': 'Arial', 'size': 18})
            plt.title('cluster' + str(j), fontdict={'family': 'Arial', 'size': 18})
            plt.legend()
            filename = 'figure_of_ds1_avg_time_data_' + cluster_num + '.pdf'
            plt.savefig(filename, bbox_inches="tight")
            if print_first[0] == 1:
                plt.show()
                print_first[0] = 0
            plt.close()
            avgf_smooth = fft(avg.transpose())




            m = 0
            for i in range(n_cluster):
                if number_of_genes[i] < 5:
                    continue
                plt.plot(range(len(yf_avg[i])), yf_avg[i], c=color_map[i], label='path' + str(m))
                m = m + 1

            plt.xlabel("Frequency", fontdict={'family': 'Arial', 'size': 18})
            plt.ylabel("Amplitude", fontdict={'family': 'Arial', 'size': 18})
            plt.title('cluster' + str(j), fontdict={'family': 'Arial', 'size': 18})
            plt.legend()
            filename = 'figure_of_ds1_frequency_' + cluster_num + '.pdf'
            plt.savefig(filename, bbox_inches="tight")
            if print_first[1] == 1:
                plt.show()
                print_first[1] = 0
            plt.close()
            plt.gcf().clear()

            scales = np.arange(1, 128)
            m = 0
            for i in range(n_cluster):
                if number_of_genes[i] < 5:
                    continue
                plot_wavelet(x[:-4], yf_avg[i], scales, cluster_num, m, print_first[2])
                print_first[2] = 0
                m = m + 1







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
    wavelet = 'mexh'  # Morlet wavelet for continuous wavelet transform
    scales = np.arange(1, 128)  # Define the range of scales
    coeffs, freqs = pywt.cwt(y_smooth.transpose(), scales, wavelet)

    # Thresholding
    threshold = 0.01  # Set your threshold value here
    coeffs_thresh = pywt.threshold(coeffs, threshold, mode='soft')

    # Reconstructing the signal
    yf_smooth = np.zeros_like(y_smooth.transpose())
    yf_smooth = np.sum(coeffs_thresh, axis=0)

    yf_info = np.array([abs(yf_smooth[i, :50]) for i in range(yf_smooth.shape[0])])

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
    wavelet = 'mexh'  # Morlet wavelet for continuous wavelet transform
    scales = np.arange(1, 128)  # Define the range of scales
    coeffs, freqs = pywt.cwt(y_smooth.transpose(), scales, wavelet)

    # Thresholding
    threshold = 0.01  # Set your threshold value here
    coeffs_thresh = pywt.threshold(coeffs, threshold, mode='soft')

    # Reconstructing the signal
    yf_smooth = np.zeros_like(y_smooth.transpose())
    yf_smooth = np.sum(coeffs_thresh, axis=0)

    yf_info = np.array([abs(yf_smooth[i, :50]) for i in range(yf_smooth.shape[0])])

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



