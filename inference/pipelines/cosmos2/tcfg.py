"""Cosmos-Predict2-2B, tcfg (factory)."""
from ._patch import load_pipeline


def pipeline(device):
    return load_pipeline("tcfg", device)
