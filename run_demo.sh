python DEC.py mnist --ae_weights ./ae_weights/mnist_ae_weights.h5
python DEC.py usps --update_interval 30 --ae_weights ./ae_weights/usps_ae_weights.h5
python DEC.py reutersidf10k --n_clusters 4 --update_interval 20 --ae_weights ./ae_weights/reutersidf10k_ae_weights.h5
