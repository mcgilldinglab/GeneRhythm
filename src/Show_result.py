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


def plot_wavelet(time, signal, scales, cluster_num, m,
                 waveletname='mexh',
                 cmap=plt.cm.seismic,
                 title='Wavelet Transform (Power Spectrum) of signal',
                 ylabel='1/Frequency',
                 xlabel='Time'):
    dt = time[1] - time[0]
    [coefficients, frequencies] = pywt.cwt(signal, scales, waveletname, dt)
    power = (abs(coefficients)) ** 2
    period = 1. / frequencies
    #period =  frequencies
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
    sc.pl.umap(adata, color=['leiden'], legend_loc='on data', legend_fontsize='x-large', save='test.pdf')

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


    # np.save('cluster1_8.npy', cluster)
    # cluster = np.array(cluster)
    # np.save('cluster_index.npy',cluster)


    def movingaverage(data, window_size):
        window = np.ones(int(window_size)) / float(window_size)
        a = np.convolve(data[window_size:], window, 'valid')
        return np.concatenate([np.zeros(window_size), a])


    for j in range(200):
        cluster_num = str(j)
        cluster = adata.obs[adata.obs['leiden'] == cluster_num]['indexa']

        # for s in adata.obs_names[adata.obs['leiden'] == cluster_num]:
        #     print(s)
        # print("********************")
        error_t = []
        for i in range(len(cluster)):
            # print(np.max(y[cluster[i],:] - np.min(y[cluster[i],:])))
            error_t.append(np.max(y[cluster[i], :] - np.min(y[cluster[i], :])))

        error_index_t = np.argsort(error_t)
        error_t = np.array(error_t)
        num = 0
        cluster_filter = []
        yf_info_filter = []
        cluater_index = []
        name = []
        for i in error_index_t[:]:
            # a = movingaverage(y[cluster[i],:],5)
            a = movingaverage(y[cluster[i], :], 5)
            if abs(np.max(abs(a))) < 0.01 or abs(np.max(abs(a))) > 5:
                continue
            num = num + 1
            cluster_filter.append(a)
            cluater_index.append(cluster[i])
            yf_info_filter.append(yf_info[cluster[i], :])
            name.append(adata.obs_names[cluster[i]])
        print(len(name))
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
                # plt.plot(x[:-4], cluster_filter[i], c=color[i])

            cluster_num = cluster_num

            plt.xlabel("Pseudotime", fontdict={'family': 'Arial', 'size': 18})
            plt.ylabel("Expression log2 fold change", fontdict={'family': 'Arial', 'size': 18})
            plt.title('cluster' + str(j), fontdict={'family': 'Arial', 'size': 18})
            filename = 'figure_of_ds1_time_data_' + cluster_num + '.pdf'
            plt.savefig(filename, bbox_inches="tight")
            # plt.show()
            plt.close()
            print(y_pred)

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
                # plt.plot(range(cluster_filter.shape[1]), avg[i], c = color_map[i])
                plt.errorbar(x[:-4], avg[i], yerr=std[i], c=color_map[i], label = 'path'+ str(m))
                m = m+1
            plt.xlabel("Pseudotime", fontdict={'family': 'Arial', 'size': 18})
            plt.ylabel("Expression log2 fold change", fontdict={'family': 'Arial', 'size': 18})
            plt.title('cluster' + str(j), fontdict={'family': 'Arial', 'size': 18})
            plt.legend()
            filename = 'figure_of_ds1_avg_time_data_' + cluster_num + '.pdf'
            plt.savefig(filename, bbox_inches="tight")
            #plt.show()
            plt.close()
            avgf_smooth = fft(avg.transpose())
            print(avgf_smooth.shape)

            print('cluster' + str(j))

            scales = np.arange(1, 128)
            m = 0
            for i in range(n_cluster):
                if number_of_genes[i] < 5:
                    continue
                # plt.plot(range(cluster_filter.shape[1]), avg[i], c = color_map[i])
                plot_wavelet(x[:-4], yf_avg[i], scales, cluster_num, m)
                m = m + 1


            m = 0
            for i in range(n_cluster):
                if number_of_genes[i] < 5:
                    continue
                # plt.plot(range(cluster_filter.shape[1]), avg[i], c = color_map[i])
                plt.plot(range(len(yf_avg[i])), yf_avg[i], c=color_map[i], label = 'path'+str(m))
                m = m+1

            plt.xlabel("Frequency", fontdict={'family': 'Arial', 'size': 18})
            plt.ylabel("Amplitude", fontdict={'family': 'Arial', 'size': 18})
            plt.title('cluster' + str(j), fontdict={'family': 'Arial', 'size': 18})
            plt.legend()
            filename = 'figure_of_ds1_frequency_' + cluster_num + '.pdf'
            plt.savefig(filename, bbox_inches="tight")
            #plt.show()
            plt.close()
            plt.gcf().clear()
            print(yf_info_filter.shape)

            m = 0
            for i in range(n_cluster):
                if number_of_genes[i] < 5:
                    continue
                plt.plot(range(cluster_filter.shape[1]), avg[i], c = color_map[i])
                mg = mygene.MyGeneInfo()
                gene_ids = mg.getgenes(namei[i], 'name, symbol, entrezgene', as_dataframe=True)
                gene_ids.index.name = "UNIPROT"
                gene_ids.reset_index(inplace=True)
                for index1, row in gene_ids.iterrows():
                    print(row['symbol'])
                    # if pd.isna(row['symbol']):
                    #     gene_ids.at[index1, 'symbol'] = namei[i][index1]
                    #     print(row['entrezgene'])

                gene_symbols = gene_ids['symbol']
                gene_symbols.to_csv('Gene_' + cluster_num + '_path_' + str(m) + '.csv')
                yff = pd.DataFrame(data=yf_info_filter[index[i]], index=gene_symbols).iloc[:, :15]
                #yff.to_csv('Gene_' + cluster_num +'_path_'+str(m)+ '.csv')
                sns.heatmap(data=yff, cmap='Reds')
                filename = 'figure_of_ds1_heatmap_' + cluster_num +'_path_'+str(m)+ '.pdf'
                plt.savefig(filename, bbox_inches="tight")
                m = m+1


            mg = mygene.MyGeneInfo()
            gene_ids = mg.getgenes(name, 'name, symbol, entrezgene', as_dataframe=True)
            gene_ids.index.name = "UNIPROT"
            gene_ids.reset_index(inplace=True)
            for index, row in gene_ids.iterrows():
                print(row['symbol'])
                # if pd.isna(row['symbol']):
                #     gene_ids.at[index, 'symbol'] = name[index]
                #     print(row['entrezgene'])


            yff = pd.DataFrame(data = yf_info_filter, index=name).iloc[:, :15]
            print(yff)
            sns.heatmap(data = yff, cmap = 'Reds')
            filename = 'figure_of_ds1_heatmap_' + cluster_num + '.pdf'
            plt.savefig(filename, bbox_inches="tight")
            #plt.show()






