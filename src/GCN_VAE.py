
import argparse

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops
from tqdm import tqdm
import torch.nn as nn
import time
import scanpy as sc
import anndata as ad
import pandas as pd
from sklearn.metrics import calinski_harabaz_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import pdist, squareform


def dunn_index(X, labels):
    unique_labels = np.unique(labels)
    num_clusters = len(unique_labels)

    # Compute pairwise distances between all points
    pairwise_dist = squareform(pdist(X))

    intra_cluster_distances = []
    for label in unique_labels:
        # Extract points belonging to the current cluster
        cluster_points = X[labels == label]

        # Compute pairwise distances within the cluster
        if len(cluster_points) > 1:
            cluster_dist = pdist(cluster_points)
            intra_cluster_distances.extend(cluster_dist)

    # Compute minimum inter-cluster distance
    inter_cluster_distances = []
    for i in range(num_clusters):
        for j in range(i + 1, num_clusters):
            inter_cluster_dist = pairwise_dist[labels == unique_labels[i]][:, labels == unique_labels[j]].min()
            inter_cluster_distances.append(inter_cluster_dist)

    min_inter_cluster_distance = np.min(inter_cluster_distances)

    # Avoid division by zero if there's only one cluster
    if len(intra_cluster_distances) > 0:
        max_intra_cluster_distance = np.max(intra_cluster_distances)
        dunn_index_value = min_inter_cluster_distance / max_intra_cluster_distance
    else:
        dunn_index_value = np.inf  # Return infinity if there's only one cluster

    return dunn_index_value


class GCN_VAE(torch.nn.Module):
    def __init__(self, latent_size = 20):
        super().__init__()
        self.latent_size = latent_size
        self.encoder_forward1 = nn.Sequential(
            nn.Linear(100 ,70),
            nn.LeakyReLU(),
            nn.Linear(70, 40),
            nn.LeakyReLU(),
            nn.Linear(40, 20)
        )
        self.encoder_forward2 = nn.Sequential(
            nn.Linear(50, 40),
            nn.LeakyReLU(),
            nn.Linear(40, 30),
            nn.LeakyReLU(),
            nn.Linear(30, 20)
        )
        self.encoder_forward3 = nn.Sequential(
            nn.Linear(100, 70),
            nn.LeakyReLU(),
            nn.Linear(70, 40),
            nn.LeakyReLU(),
            nn.Linear(40, 20)
        )
        self.FC = nn.Sequential(
            nn.Linear(30, 20),
            nn.ReLU()
        )
        self.decoder_forward1 = nn.Sequential(
            nn.Linear(20, 40),
            nn.LeakyReLU(),
            nn.Linear(40, 70),
            nn.LeakyReLU(),
            nn.Linear(70, 100),
            nn.Sigmoid()
        )
        self.decoder_forward2 = nn.Sequential(
            nn.Linear(20, 30),
            nn.LeakyReLU(),
            nn.Linear(30, 40),
            nn.LeakyReLU(),
            nn.Linear(40, 50),
            nn.Sigmoid()
        )
        self.decoder_forward3 = nn.Sequential(
            nn.Linear(20, 40),
            nn.LeakyReLU(),
            nn.Linear(40, 70),
            nn.LeakyReLU(),
            nn.Linear(70, 100),
            nn.Sigmoid()
        )

    def encoder(self, X, encoder_forward , latent_size):
        out = encoder_forward(X)
        mu = out[:, :latent_size]
        log_var = out[:, latent_size:]
        return mu, log_var

    def decoder(self, z, decoder_forward):
        mu_prime = decoder_forward(z)
        return mu_prime


    def reparameterization(self, mu, log_var):
        epsilon = torch.randn_like(log_var)
        z = mu + epsilon * torch.sqrt(log_var.exp())
        return z

    def loss(self, X, mu_prime, mu, log_var):
        reconstruction_loss = torch.mean(torch.square(X - mu_prime).sum(dim=1))

        latent_loss = torch.mean(0.5 * (log_var.exp() + torch.square(mu) - log_var).sum(dim=1))

        loss = reconstruction_loss + 0.05 * latent_loss

        return reconstruction_loss


    def forward(self, data):
        x = data.x
        print(x.shape)
        x1 = x[:, :100]
        x2 = x[:, 100:150]
        x3 = x[:, 150:]
        mu1, log_var1 = self.encoder(x1, self.encoder_forward1, 10)
        mu2, log_var2 = self.encoder(x2, self.encoder_forward2, 10)
        mu3, log_var3 = self.encoder(x3, self.encoder_forward3, 10)
        z1 = self.reparameterization(mu1, log_var1)
        z2 = self.reparameterization(mu2, log_var2)
        z3 = self.reparameterization(mu3, log_var3)
        z = torch.cat((z1, z2, z3), dim=1)
        z = self.FC(z)
        mu_prime1 = self.decoder(z, self.decoder_forward1)
        mu_prime2 = self.decoder(z, self.decoder_forward2)
        mu_prime3 = self.decoder(z, self.decoder_forward3)
        r1 = [mu_prime1, mu1, log_var1]
        r2 = [mu_prime2, mu2, log_var2]
        r3 = [mu_prime3, mu3, log_var3]

        return r1, r2, r3, z


class GCN_VAE_Graph(torch.nn.Module):
    def __init__(self, latent_size = 20):
        super().__init__()
        self.latent_size = latent_size
        self.encoder_forward1 = nn.Sequential(
            nn.Linear(100 ,70),
            nn.LeakyReLU(),
            nn.Linear(70, 40),
            nn.LeakyReLU(),
            nn.Linear(40, 20)
        )
        self.encoder_forward2 = nn.Sequential(
            nn.Linear(50, 40),
            nn.LeakyReLU(),
            nn.Linear(40, 30),
            nn.LeakyReLU(),
            nn.Linear(30, 20)
        )
        self.encoder_forward3 = nn.Sequential(
            nn.Linear(100, 70),
            nn.LeakyReLU(),
            nn.Linear(70, 40),
            nn.LeakyReLU(),
            nn.Linear(40, 20)
        )
        self.FC = nn.Sequential(
            nn.Linear(30, 20),
            nn.ReLU()
        )
        self.decoder_forward1 = nn.Sequential(
            nn.Linear(20, 40),
            nn.LeakyReLU(),
            nn.Linear(40, 70),
            nn.LeakyReLU(),
            nn.Linear(70, 100),
            nn.Sigmoid()
        )
        self.decoder_forward2 = nn.Sequential(
            nn.Linear(20, 30),
            nn.LeakyReLU(),
            nn.Linear(30, 40),
            nn.LeakyReLU(),
            nn.Linear(40, 50),
            nn.Sigmoid()
        )
        self.decoder_forward3 = nn.Sequential(
            nn.Linear(20, 40),
            nn.LeakyReLU(),
            nn.Linear(40, 70),
            nn.LeakyReLU(),
            nn.Linear(70, 100),
            nn.Sigmoid()
        )

    def encoder(self, X, encoder_forward , latent_size):
        out = encoder_forward(X)
        mu = out[:, :latent_size]
        log_var = out[:, latent_size:]
        return mu, log_var

    def decoder(self, z, decoder_forward):
        mu_prime = decoder_forward(z)
        return mu_prime


    def reparameterization(self, mu, log_var):
        epsilon = torch.randn_like(log_var)
        z = mu + epsilon * torch.sqrt(log_var.exp())
        return z

    def loss(self, X, mu_prime, mu, log_var):

        reconstruction_loss = torch.mean(torch.square(X - mu_prime).sum(dim=1))

        latent_loss = torch.mean(0.5 * (log_var.exp() + torch.square(mu) - log_var).sum(dim=1))

        loss = reconstruction_loss + 0.05 * latent_loss

        return reconstruction_loss


    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        #x = data.x
        print(x.shape)
        x1 = x[:, :100]
        x2 = x[:, 100:150]
        x3 = x[:, 150:]
        x1 = self.conv1(x1, edge_index)
        x2 = self.conv2(x2, edge_index)
        x3 = self.conv3(x3, edge_index)
        mu1, log_var1 = self.encoder(x1, self.encoder_forward1, 10)
        mu2, log_var2 = self.encoder(x2, self.encoder_forward2, 10)
        mu3, log_var3 = self.encoder(x3, self.encoder_forward3, 10)
        z1 = self.reparameterization(mu1, log_var1)
        z2 = self.reparameterization(mu2, log_var2)
        z3 = self.reparameterization(mu3, log_var3)
        z = torch.cat((z1, z2, z3), dim=1)
        z = self.FC(z)
        mu_prime1 = self.decoder(z, self.decoder_forward1)
        mu_prime2 = self.decoder(z, self.decoder_forward2)
        mu_prime3 = self.decoder(z, self.decoder_forward3)
        r1 = [mu_prime1, mu1, log_var1]
        r2 = [mu_prime2, mu2, log_var2]
        r3 = [mu_prime3, mu3, log_var3]

        return r1, r2, r3, z


def train(model, optimizer, data_loader, device, name='GCN_VAE'):
    model.train()

    total_loss = 0
    for X in data_loader:
        X = X.to(device)
        model.zero_grad()
        r1, r2, r3, z = model(X)

        loss = model.loss(X.x[:, :100], r1[0], r1[1], r1[2])
        loss += model.loss(X.x[:, 100:150], r2[0], r2[1], r2[2])
        loss += 0.001*model.loss(X.x[:, 150:], r3[0], r3[1], r3[2])

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss




def benchmark(adata):
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=10)
    sc.tl.umap(adata)

    sc.tl.leiden(adata)
    sc.pl.umap(adata, color=['leiden'], legend_loc='on data', legend_fontsize='x-large', save='ALL.pdf')
    CHscore = calinski_harabaz_score(adata.X, adata.obs['leiden'])
    Silscore = silhouette_score(adata.X, adata.obs['leiden'])
    DBscore = davies_bouldin_score(adata.X, adata.obs['leiden'])
    Dunnscore = dunn_index(adata.X, adata.obs['leiden'])
    print("ALL: CH: " + str(CHscore) + " Silhoutte: " + str(Silscore) + " DB: " + str(DBscore) + " Dunn: " + str(Dunnscore))


def GeneRhythm_Model(input_data,graph=None, model_output='ALL_mu.npy', latent_output='gcn_vae.pth', sc_data, lr=0.00005, n_epoch=1000, batch_size=32):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)


    batch_size = batch_size
    epochs = n_epoch
    lr = lr


    x = torch.tensor(np.float32(np.load(input_data)))

    if graph:
        edge_index = torch.tensor(np.load(graph).transpose(), dtype=torch.long)
        data = Data(x=x, edge_index=edge_index)
        data.edge_index = add_self_loops(data.edge_index)[0]
    else:
        data = Data(x=x)

    data_loader = DataLoader(dataset=[data], batch_size=1, shuffle=False)
    # train VAE
    if graph:
        gcn_vae = GCN_VAE_Graph().to(device)
    else:
        gcn_vae = GCN_VAE().to(device)

    optimizer = torch.optim.AdamW(gcn_vae.parameters(), lr=lr)
    all_loss = []
    all_epoch = np.array(range(epochs))
    print('Start Training VAE...')
    time1 = time.perf_counter()
    for epoch in range(1, 1 + epochs):
        loss = train(gcn_vae, optimizer, data_loader, device, name='GCN_VAE')
        all_loss.append(loss)
        print("Epochs: {epoch}, AvgLoss: {loss:.4f}".format(epoch=epoch, loss=loss))

    time2 = time.perf_counter()
    print('Training for VAE has been done.')

    print(time2 - time1)

    np.array(all_loss)
    plt.plot(all_epoch, all_loss)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()

    PATH = model_output
    torch.save(gcn_vae.state_dict(), PATH)
    gcn_vae.eval()
    if torch.cuda.is_available():
        _, _,_, z = gcn_vae(data.cuda())
    else:
        _, _,_, z = gcn_vae(data)

    z = z.cpu().detach().numpy()
    np.save(latent_output, z)







