import os
import imp


def make_network(cfg, training=False):
    module = cfg.network_module
    path = cfg.network_path

    network = imp.load_source(module, path).Network()
    return network
