INF_SIGMA = 9_999_999.0


BASE_PARAMS = {
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 30,
}

SWEEPS = [
    {
        "name": "sigma_x_seg",
        "row": ("guidance_scale", [4.0, 5.0, 6.0]),
        "col": ("seg_scale", [0.0, 2.0]),
        "fixed": {
            "seg_blur_sigma": 10.0,
            "seg_applied_layers": ["d4", "s1"],
        },
    },
  
]

FINALS = [
    {
        "label": "d4+s1 / sigma 5",
        "guidance_scale": 4.5,
        "seg_applied_layers": ["d4", "s1"],
        "seg_scale": 3.0,
        "seg_blur_sigma": 5.0,
    },
    {
        "label": "d4+s2 / sigma 5",
        "guidance_scale": 4.5,
        "seg_applied_layers": ["d4", "s2"],
        "seg_scale": 3.0,
        "seg_blur_sigma": 5.0,
    },
    {
        "label": "d3+d4 / sigma 10",
        "guidance_scale": 4.5,
        "seg_applied_layers": ["d3", "d4"],
        "seg_scale": 3.0,
        "seg_blur_sigma": 10.0,
    },
]


def get_params():
    return BASE_PARAMS, SWEEPS


def get_finals():
    return FINALS
