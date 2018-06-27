#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""See Unsupervised Deep Embedding for Clustering Analysis"""

__author__ = "Wang Gang"
__email__ = "wanggang15@jd.com"
__status__ = "Production"

import torch
import numpy as np


class AutoEncoder(torch.nn.Module):
    """Fully connected auto-encoder model, symmetric"""

    def __init__(self, dims, act='relu', init='glorot_uniform'):
        """
        dims[0] -> dims[1] -> ... -> dims[-1] -> dims[-2] -> ... -> dims[0]

        Input:
          dims: dims[0] is input dim, dims[-1] is units in hidden layer. the decoder is symmetric with the encode.

        # FIXME: act and init NOT working now
        """

        super().__init__()

        self.encoder = torch.nn.ModuleList()
        n_stacks = len(dims) - 1
        for idx in np.arange(n_stacks - 1):
            _in = dims[idx]
            _out = dims[idx + 1]
            self.encoder.append(torch.nn.Linear(in_features=_in, out_features=_out))
            self.encoder.append(torch.nn.ReLU())
        self.encoder.append(torch.nn.Linear(in_features=dims[-2], out_features=dims[-1]))

        self.decoder = torch.nn.ModuleList()
        for idx in np.arange(n_stacks, 1, -1):
            _in = dims[idx]
            _out = dims[idx - 1]
            self.decoder.append(torch.nn.Linear(in_features=_in, out_features=_out))
            self.decoder.append(torch.nn.ReLU())
        self.decoder.append(torch.nn.Linear(in_features=dims[1], out_features=dims[0]))

    def forward(self, x):
        """pass"""
        for l in self.encoder:
            x = l(x)

        encoded = x

        for l in self.decoder:
            x = l(x)
        decoded = x

        del x
        return encoded, decoded


class DEC(torch.nn.Module):
    def __init__(self, ae_dims, ae_weights_path=None, n_cluster=10, alpha=1.0):
        """
        Input:
          ae_dims: auto encoder dims
          ae_weights_path_path: pretrained ae encoder state
        """

        super().__init__()

        self.n_cluster = n_cluster
        self.alpha = alpha

        self.ae_dims = ae_dims
        self.auto_encoder = AutoEncoder(dims=self.ae_dims)
        self.ae_weights_path = ae_weights_path

        self.cluster_centroid = torch.nn.Parameter(data=torch.randn(
            self.n_cluster, self.ae_dims[-1]), requires_grad=True)

        #     cluster centroids
        from sklearn.cluster import KMeans
        self.kmeans = KMeans(n_clusters=self.n_cluster, n_init=20)

    def init_ae(self):
        """Init the auto encoder (theta)
        """
        if self.ae_weights_path is None:
            raise ValueError("pre_training NOT implemented yet")
        else:
            print("  load ae weights from {} ...".format(self.ae_weights_path))
            self.auto_encoder.load_state_dict(torch.load(self.ae_weights_path))

    def init_centroid(self, x_all):
        """
        Init centorid using KMeans

        Input:
          x_all: all data, NOT a mini batch
        """
        print("  initial centorid using kmeans ...")
        encoded, _ = self.auto_encoder(x_all)
        self.kmeans.fit(encoded.detach().numpy())
        self.cluster_centroid.data = torch.from_numpy(self.kmeans.cluster_centers_)

    def forward(self, x):
        """
        Input:
          x: batch_size, d_feature

        Output:
          q: soft assignment of x
        """
        x, _ = self.auto_encoder(x)
        # print("encoder output", x[:2])

        # soft assignment:
        #   q: batch_size, n_cluster,
        #   c: n_cluster,  d_feature -> 1, n_cluster, d_feature -> batch_size, n_cluster, d_feature
        #   x: batch_size, d_feature -> batch_size, 1, d_feature -> batch_size, n_cluster, d_feature
        batch_size = x.shape[0]
        n_cluster = self.n_cluster
        q_unnormalized = (torch.unsqueeze(self.cluster_centroid, 0).expand(batch_size, -1, -1) -
                          torch.unsqueeze(x, 1).expand(-1, n_cluster, -1)) \
            .pow(2) \
            .sum(dim=2, keepdim=False) \
            .div(self.alpha) \
            .add(1) \
            .pow(-1 * (self.alpha + 1) * 0.5)

        q = q_unnormalized / q_unnormalized.sum(dim=1, keepdim=True).expand(-1, n_cluster)  # batch_size, n_cluster

        return q

    def kl_divergence(self, x):
        """
        Input:
          x: batch_size, d_feature

        Output:
          loss
        """
        batch_size = x.shape[0]
        n_cluster = self.n_cluster

        q = self.forward(x)

        # calculate the auxiliary distribution p
        f = q.sum(dim=0, keepdim=True)  # soft cluster frequencies
        p_unmormalized = q.pow(2) / f.expand(batch_size, -1)
        p = p_unmormalized / p_unmormalized.sum(dim=1, keepdim=True).expand(-1, n_cluster)  # batch_size, n_cluster
        p_no_grad = p.detach()

        # calculate KL divengence loss: clustering_loss
        # print("x", x[:2])
        # print("p", p_no_grad[:2])
        # print("q", q[:2])
        print(q.shape, p.shape)
        loss = torch.nn.KLDivLoss(size_average=True, reduce=True)(input=torch.log(q), target=p_no_grad)

        return loss


def train():

    from torch.utils.data import DataLoader

    dec = DEC(ae_dims=[784, 500, 500, 2000, 10], ae_weights_path="../IDEC/ae_weights/mnist_ae_weights.pth")

    from datasets import load_mnist
    x, y = load_mnist()

    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float()

    # x = x[:500, :]

    optimizer = torch.optim.Adam(dec.parameters(), lr=0.01)

    dataloader = DataLoader(x, batch_size=256, shuffle=False, num_workers=4)

    print("before init", dec.cluster_centroid)
    # Phase 1: Parameter Initialization
    dec.init_ae()
    dec.init_centroid(x)
    print("after init", dec.cluster_centroid)

    # Test
    q = dec(x)
    y_pred = q.argmax(1)
    from DEC import cluster_acc
    acc = np.round(cluster_acc(y.numpy(), y_pred.numpy()), 5)
    print("kmeans, acc={}".format(acc))

    # Phase 2: Paramtger optimization
    for i_epoch in np.arange(10):
        for i_batch, sample_batched in enumerate(dataloader):
            # sample_batched.requires_grad = False  # ?
            loss = dec.kl_divergence(sample_batched)

            y_pred_batch = dec(sample_batched)
            # print(y_pred_batch[:2, :])

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if i_batch % 10 == 0:
                q = dec(x)
                y_pred = q.argmax(1)
                from DEC import cluster_acc
                acc = np.round(cluster_acc(y.numpy(), y_pred.numpy()), 5)
                print("i_epoch={}, i_batch={}, acc={}, loss={}".format(i_epoch, i_batch, acc, loss.item()))


def main():
    pass


if __name__ == '__main__':
    # dec = DEC(ae_dims=[5, 10, 2], ae_weights_path=None)

    # ae = AutoEncoder([5, 10, 2])
    # x = torch.randn(3, 5)
    # y_enc, y_dec = ae(x)
    # print(y_enc.shape, y_dec.shape)

    train()
