# -*- coding: utf-8 -*-

import pandas as pd
from scipy.fftpack import fft
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import pywt
import anndata
import scanpy as sc


def frequency_extract(trajectory_info, adata, dataset):
    a = pd.Index(trajectory_info.loc["path"] ==1)

    y = np.array(trajectory_info.iloc[:-5,a])

    x = np.array(trajectory_info.loc["time",a])

    x.sort()
    x_smooth = np.linspace(start=x.min(), stop=x.max(), num=100)
    y_smooth = make_interp_spline(x, y.transpose(), k =3)(x_smooth)

    # Perform continuous wavelet transform
    wavelet = 'mexh'  # Morlet wavelet for continuous wavelet transform
    scales = np.arange(1, 128)  # Define the range of scales
    coeffs, freqs = pywt.cwt(y_smooth.transpose(), scales, wavelet)

    # Thresholding
    threshold = 0.01  # Set your threshold value here
    coeffs_thresh = pywt.threshold(coeffs, threshold, mode='soft')

    yf_smooth = np.zeros_like(y_smooth.transpose())
    for i in range(len(y_smooth.transpose())):
        yf_smooth[i] = np.sum(coeffs_thresh[:, i] * np.conj(pywt.cwt([1], scales, wavelet)[0]))

    yf_info = np.array([abs(yf_smooth[i,:50]) for i in range(yf_smooth.shape[0])])

    for i in range(len(yf_info)):
        if yf_info[i].max() == 0:
            continue
        yf_info[i] = yf_info[i]/yf_info[i].max()

    y_info = y_smooth.transpose()
    for i in range(len(y_info)):
        if y_info[i].max() == 0:
            continue
        y_info[i] = y_info[i]/y_info[i].max()
    res = np.array([np.append(y_info[i],yf_info[i]) for i in range(yf_smooth.shape[0])])



    adata = adata.T
    sc.pp.normalize_total(adata1)
    sc.pp.log1p(adata1)
    sc.pp.scale(adata1)
    sc.tl.pca(adata1, n_comps=100, svd_solver="auto")
    res = np.concatenate((res,adata1.obsm['X_pca']),axis=1)
    name = dataset+'.npy'
    np.save(name, np.array(res[:,:]))


def frequency_extract_spatial( adata, dataset,  start, end):
    # Extract spatial coordinates
    spatial_coords = adata.obsm["spatial"]

    num_points = 10  # Number of center points along the line
    distances = np.linspace(0, 1, num_points)  # Fractional distances along the line
    center_points = np.array([start + d * (end - start) for d in distances])

    # Initialize a list to store mean counts for each center point
    trajectory_means = []

    # For each center point, find the 100 nearest cells and compute mean counts
    for point in center_points:
        distances = pairwise_distances([point], spatial_coords)[0]
        nearest_indices = np.argsort(distances)[:100]  # Get indices of 100 nearest cells
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
    for i in range(len(y_smooth.transpose())):
        yf_smooth[i] = np.sum(coeffs_thresh[:, i] * np.conj(pywt.cwt([1], scales, wavelet)[0]))

    yf_info = np.array([abs(yf_smooth[i, :50]) for i in range(yf_smooth.shape[0])])

    for i in range(len(yf_info)):
        if yf_info[i].max() == 0:
            continue
        yf_info[i] = yf_info[i] / yf_info[i].max()

    y_info = y_smooth.transpose()
    for i in range(len(y_info)):
        if y_info[i].max() == 0:
            continue
        y_info[i] = y_info[i] / y_info[i].max()
    res = np.array([np.append(y_info[i], yf_info[i]) for i in range(yf_smooth.shape[0])])

    adata1 = adata.T
    sc.pp.normalize_total(adata1)
    sc.pp.log1p(adata1)
    sc.pp.scale(adata1)
    sc.tl.pca(adata1, n_comps=100, svd_solver="auto")
    res = np.concatenate((res, adata1.obsm['X_pca']), axis=1)

    name = dataset + '.npy'
    np.save(name, np.array(res[:, :]))