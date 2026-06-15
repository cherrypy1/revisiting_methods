"""Cosmos-Predict2-2B, cfgpp (factory)."""
from ._patch import load_pipeline


def pipeline(device):
    return load_pipeline("cfgpp", device)
