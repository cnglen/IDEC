#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""See Unsupervised Deep Embedding for Clustering Analysis

DEC  done
IDEC done (pretrain)

caec todo?

"""

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

        # x = torch.nn.functional.normalize(x, p=2, dim=1)  # normalize
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

    def pretrain(self, x_all, n_epoch=40, batch_size=256):
        """pre-training autoencoder"""

        print("pretrain ...")
        dataloader = DataLoader(x_all, batch_size=batch_size, shuffle=True, num_workers=1,
                                drop_last=False)  # parallel for gpu tensorf, raise error

        optimizer = torch.optim.Adam(self.auto_encoder.parameters())

        for i_epoch in np.arange(n_epoch):
            for i_batch, sample_batched in enumerate(dataloader):
                sample_batched = sample_batched.cuda()

                optimizer.zero_grad()
                _, sample_batched_est = self.auto_encoder(sample_batched)
                mse_loss = torch.nn.MSELoss(size_average=True, reduce=True)(
                    input=sample_batched_est, target=sample_batched)
                mse_loss.backward()
                optimizer.step()

                if i_batch % 10 == 0:
                    print("i_epoch={:4d}, i_batch={:8d}, mse={:10.5f}".format(i_epoch, i_batch, mse_loss.item()))

        torch.save(self.auto_encoder.state_dict(), "./ae_weights/pretrained_mnist_ae_weights.pth")
        print("pretrain done")

    def init_ae(self, x_all=None):
        """Init the auto encoder (theta)
             If self.ae_weights_path is None, train the ae encoder
             else, load the pre-trained weights
        """
        if self.ae_weights_path is None:
            #raise ValueError("pre_training NOT implemented, only load pretrained model supported")
            if x_all is None:
                raise ValueError("please specify x_all")

            self.pretrain(x_all)
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
          x_est: (batch_size, d_feature) reconstructed x
        """
        x, x_est = self.auto_encoder(x)

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

        return q, x_est

    def kl_ae_loss(self, x, gamma=0.1):
        """Get the weighted average KL loss and auto-encoder reconstrution MSE loss (IDEC)
        """
        batch_size = x.shape[0]
        n_cluster = self.n_cluster

        q, x_est = self.forward(x)

        with torch.no_grad():
            # calculate the auxiliary distribution p
            f = q.sum(dim=0, keepdim=True)  # soft cluster frequencies
            p_unmormalized = q.pow(2) / f.expand(batch_size, -1)
            p = p_unmormalized / p_unmormalized.sum(dim=1, keepdim=True).expand(-1, n_cluster)  # batch_size, n_cluster

        # calculate KL divengence loss: clustering_loss
        Lc = torch.nn.KLDivLoss(size_average=True, reduce=True)(input=torch.log(q), target=p)  # * n_cluster

        # Lr
        Lr = torch.nn.MSELoss(size_average=True, reduce=True)(input=x_est, target=x)  # * x.shape[1]

        # print("Lr={:.5f}, Lc={:.5f}, gamma={}".format(Lr, Lc, gamma))
        loss = Lr + gamma * Lc
        return loss

    def kl_loss(self, x):
        """ Get the KL loss(used in DEC)
        Input:
          x: batch_size, d_feature

        Output:
          loss
        """
        batch_size = x.shape[0]
        n_cluster = self.n_cluster

        q, _ = self.forward(x)

        with torch.no_grad():
            # calculate the auxiliary distribution p
            f = q.sum(dim=0, keepdim=True)  # soft cluster frequencies
            p_unmormalized = q.pow(2) / f.expand(batch_size, -1)
            p = p_unmormalized / p_unmormalized.sum(dim=1, keepdim=True).expand(-1, n_cluster)  # batch_size, n_cluster

        # calculate KL divengence loss: clustering_loss
        loss = torch.nn.KLDivLoss(size_average=True, reduce=True)(input=torch.log(q), target=p)

        return loss


def train(max_epoch=20, watch_interval=10, tol=0.001, use_gpu=True, data='mnist', mode='dec', gamma=1.0, lr=0.001):
    """
    Input:
      mode: dec using Lc; idec using Lr+gamma*Lc

    Note:
    watch_interval: not only show the eval results, but also check delta_diff, which determines when to stop iteration
    """

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
    # dec = DEC(ae_dims=[input_dim, 500, 500, 2000, 10],
    #           n_cluster=n_cluster,
    #           ae_weights_path="../IDEC/ae_weights/{}_ae_weights.pth".format(data))

    # dec = DEC(ae_dims=[input_dim, 500, 500, 2000, 10],
    #           n_cluster=n_cluster,
    #           ae_weights_path="./ae_weights/pretrained_{}_ae_weights.pth".format(data))

    dec = DEC(ae_dims=[input_dim, 500, 500, 2000, 10],
              n_cluster=n_cluster,
              ae_weights_path=None)

    if use_gpu:
        dec = dec.cuda()

    # optimizer = torch.optim.SGD(dec.parameters(), lr=0.1, momentum=0.9)

    # optimizer = torch.optim.SGD([
    #     {"params": dec.auto_encoder.parameters(), 'lr': 0.5 * 1e-4},
    #     {"params": dec.cluster_centroid, 'lr': 0.1}
    # ],
    #     lr=0.1, momentum=0.9)

    # optimizer = torch.optim.Adam([
    #     {"params": dec.auto_encoder.parameters(), 'lr': lr * 1.0},
    #     {"params": dec.cluster_centroid, 'lr': lr},
    # ], lr=lr)
    optimizer = torch.optim.Adam(dec.parameters(), lr=lr)

    dataloader = DataLoader(x, batch_size=256, shuffle=True, num_workers=1,
                            drop_last=False)  # parallel for gpu tensorf, raise error

    # Phase 1: Parameter Initialization
    # dec.init_ae()    # using pretrained model
    dec.init_ae(x)           # pretraing model
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
    # FIXME: raise error if using break
    for i_epoch in np.arange(max_epoch):
        for i_batch, sample_batched in enumerate(dataloader):
            if use_gpu:
                sample_batched = sample_batched.cuda()

            optimizer.zero_grad()
            if mode == "dec":
                loss = dec.kl_loss(sample_batched)
            else:
                loss = dec.kl_ae_loss(sample_batched, gamma=gamma)  # 1.2 ~ .886; 1~

            loss.backward()
            optimizer.step()

            if i_batch % watch_interval == 0:
                result = eval_model(dec, x, y, use_gpu=use_gpu)
                y_pred_hard = result["y_pred_hard"]
                delta_label_ratio = np.sum(y_pred_hard != y_pred_hard_last).astype(np.float32) / y_pred_hard.shape[0]
                y_pred_hard_last = y_pred_hard

                print("i_epoch={:3d}, i_batch={:8d}, acc={:7.5f}, nmi={:7.5f}, ari={:7.5f}, delta_label_ratio={:7.5f}, kl_loss(@batch)={:8.5f}".format(
                    i_epoch, i_batch, result["acc"], result["nmi"], result["ari"], delta_label_ratio, loss.item()))

                if delta_label_ratio < tol:  # or result["acc"] < 0.50:
                    break

    return result


def eval_model(model, x, y, use_gpu=True):
    """
    eval the model
    """
    from metrics import nmi, acc, ari

    if use_gpu:
        x = x.cuda()

    y_pred_soft, _ = model(x)
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

    # DEC
    # train(data="mnist", mode="dec", tol=0.001, lr=0.0001, watch_interval=10)    # wi=5, 0.865; wi=10, 有机会达到0.872

    # train(data="reutersidf10k")
    # train(data="usps")

    # # IDEC 0.885: gamma=1.0, lr=0.001
    # with open("./result.csv", "w") as f:
    #     f.write("lr,gamma,acc,nmi\n")

    #     for lr in [0.0001, 0.001, 0.01, 0.1]:
    #         for gamma in [0.01, 0.1, 1.0, 10, 100.0, 1000]:
    #             result = train(data="mnist", mode="idec", tol=0.001, max_epoch=50,
    #                            watch_interval=100, lr=lr, gamma=gamma)  # 0.886
    #             f.write("{},{},{},{}\n".format(lr, gamma, result["acc"], result["nmi"]))

    # result = train(data="mnist", mode="idec", tol=0.001, max_epoch=100,
    #                watch_interval=5,
    #                lr=0.001,
    #                gamma=1.0)    # 0.886

    result = train(data="mnist", mode="idec", tol=0.001, max_epoch=20,
                   watch_interval=5,
                   lr=0.001,
                   gamma=1.0)    # ae: normailized
