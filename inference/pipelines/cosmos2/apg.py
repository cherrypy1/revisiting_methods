"""Cosmos-Predict2-2B, apg (factory)."""
from ._patch import load_pipeline


def pipeline(device):
    return load_pipeline("apg", device)
