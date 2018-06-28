#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""See Unsupervised Deep Embedding for Clustering Analysis"""

__author__ = "Wang Gang"
__email__ = "wanggang15@jd.com"
__status__ = "Production"

import torch
import numpy as np
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader


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
        """
        Input:
          x: batch_size, dim_feature

        Output:
          encoded: batch_size, dims[-1]
          doceded: batch_size, dim_feature
        """
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
        self.kmeans = KMeans(n_clusters=self.n_cluster, n_init=20)

    def init_ae(self):
        """Init the auto encoder (theta)
             If self.ae_weights_path is None, train the ae encoder
             else, load the pre-trained weights
        """
        if self.ae_weights_path is None:
            raise ValueError("pre_training NOT implemented, only load pretrained model supported")
        else:
            print("  load ae weights from {} ...".format(self.ae_weights_path))
            self.auto_encoder.load_state_dict(torch.load(self.ae_weights_path))

    def init_centroid(self, x_all):
        """
        Init centorid using KMeans, make sure the auto encoder is trained (call init_ae() => then init_centroid())

        Input:
          x_all: all data, NOT a mini batch
        """
        print("  initial centorid using kmeans ...")
        encoded, _ = self.auto_encoder(x_all)
        self.kmeans.fit(encoded.detach().cpu().numpy())
        self.cluster_centroid.data = torch.from_numpy(
            self.kmeans.cluster_centers_).to(device=self.cluster_centroid.device)

    def forward(self, x):
        """
        Input:
          x: batch_size, d_feature

        Output:
          q: (batch_size, n_cluster) soft assignment of x
        """
        x, _ = self.auto_encoder(x)

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

    def kl_loss(self, x):
        """ Get the KL loss
        Input:
          x: batch_size, d_feature

        Output:
          loss
        """
        batch_size = x.shape[0]
        n_cluster = self.n_cluster

        q = self.forward(x)

        with torch.no_grad():
            # calculate the auxiliary distribution p
            f = q.sum(dim=0, keepdim=True)  # soft cluster frequencies
            p_unmormalized = q.pow(2) / f.expand(batch_size, -1)
            p = p_unmormalized / p_unmormalized.sum(dim=1, keepdim=True).expand(-1, n_cluster)  # batch_size, n_cluster

        # calculate KL divengence loss: clustering_loss
        loss = torch.nn.KLDivLoss(size_average=True, reduce=True)(input=torch.log(q), target=p)

        return loss


def train(max_epoch=20, print_interval=10, tol=0.001, use_gpu=True, data='mnist'):

    def load_data(data='mnist'):
        from datasets import load_mnist, load_reuters, load_usps

        if data == 'mnist':
            x, y = load_mnist()
            n_cluster = 10
        elif data == "usps":
            x, y = load_usps()
            n_cluster = 10
        elif data == "reutersidf10k":
            x, y = load_reuters()
            n_cluster = 4

        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        return x, y, n_cluster

    x, y, n_cluster = load_data(data=data)
    input_dim = x.shape[1]
    dec = DEC(ae_dims=[input_dim, 500, 500, 2000, 10],
              n_cluster=n_cluster,
              ae_weights_path="../IDEC/ae_weights/{}_ae_weights.pth".format(data))

    if use_gpu:
        dec = dec.cuda()

    optimizer = torch.optim.SGD(dec.parameters(), lr=0.1, momentum=0.9)
    optimizer = torch.optim.Adam(dec.parameters(), lr=0.0001)

    dataloader = DataLoader(x, batch_size=256, shuffle=True, num_workers=1,
                            drop_last=False)  # parallel for gpu tensorf, raise error

    # Phase 1: Parameter Initialization
    dec.init_ae()
    if use_gpu:
        dec.init_centroid(x.cuda())
    else:
        dec.init_centroid(x)

    # Test
    result = eval_model(dec, x, y, use_gpu=use_gpu)
    print("after init(pre-train ae, kmeans=>centroid):\n  acc={:7.5f}, nmi={:7.5f}, ari={:7.5f}".format(
        result["acc"], result["nmi"], result["ari"]))
    y_pred_hard_last = result["y_pred_hard"]

    # Phase 2: Paramtger optimization
    for i_epoch in np.arange(max_epoch):
        for i_batch, sample_batched in enumerate(dataloader):
            if use_gpu:
                sample_batched = sample_batched.cuda()

            optimizer.zero_grad()
            loss = dec.kl_loss(sample_batched)
            loss.backward()
            optimizer.step()

            if i_batch % print_interval == 0:
                result = eval_model(dec, x, y, use_gpu=use_gpu)
                y_pred_hard = result["y_pred_hard"]
                delta_label_ratio = np.sum(y_pred_hard != y_pred_hard_last).astype(np.float32) / y_pred_hard.shape[0]
                y_pred_hard_last = y_pred_hard

                print("i_epoch={:3d}, i_batch={:8d}, acc={:7.5f}, nmi={:7.5f}, ari={:7.5f}, delta_label_ratio={:7.5f}, kl_loss(@batch)={:8.5f}".format(
                    i_epoch, i_batch, result["acc"], result["nmi"], result["ari"], delta_label_ratio, loss.item()))

                if delta_label_ratio < tol:
                    break


def eval_model(model, x, y, use_gpu=True):
    """
    eval the model
    """
    from metrics import nmi, acc, ari

    if use_gpu:
        x = x.cuda()

    y_pred_soft = model(x)
    y_pred_hard = y_pred_soft.argmax(1)

    # make sure numpy array
    y_pred_hard = y_pred_hard.cpu().numpy()
    y = y.cpu().numpy()

    acc_value = acc(y, y_pred_hard)
    nmi_value = nmi(y, y_pred_hard)
    ari_value = ari(y, y_pred_hard)
    return {"acc": acc_value, "nmi": nmi_value, "ari": ari_value, "y_pred_hard": y_pred_hard}


def main():
    pass


if __name__ == '__main__':

    train(data="reutersidf10k")
    # train(data="usps")
