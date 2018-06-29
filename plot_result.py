#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""desc"""


import pandas as pd
import matplotlib.pyplot as plt


def main():
    raw_data = pd.read_csv("./result_v1.csv")

    for lr in set(raw_data.lr):
        condition = raw_data["lr"] == lr
        gamma = raw_data.loc[condition, "gamma"]
        acc = raw_data.loc[condition, "acc"]
        plt.semilogx(gamma, acc, label="lr={}".format(lr))

    plt.legend()
    plt.grid()
    plt.xlabel("gamma")
    plt.ylabel("acc")
    plt.show()


if __name__ == '__main__':
    main()
