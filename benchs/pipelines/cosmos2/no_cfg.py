"""Cosmos-Predict2-2B, no_cfg (factory)."""
from ._patch import load_pipeline


def pipeline(device):
    return load_pipeline("no_cfg", device)
