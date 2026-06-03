INF_SIGMA = 9_999_999.0


BASE_PARAMS = {
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 30,
}

SWEEPS = [
    {
        "name": "sigma_x_oseg",
        "row": ("seg_blur_sigma", [1.0, 5.0, 10.0, INF_SIGMA]),
        "col": ("oseg_scale", [0.0, 0.5, 1.0, 1.5]),
        "fixed": {
            "guidance_scale": 4.5,
            "seg_scale": 0.0,
            "seg_applied_layers": ["d4", "s1"],
        },
    },
    {
        "name": "layers_x_oseg",
        "row": (
            "seg_applied_layers",
            [
                ["d3", "s1"],
                ["d3", "s2"],
                ["d4", "s1"],
                ["d4", "s2"],
            ],
        ),
        "col": ("oseg_scale", [0.0, 0.5, 1.0, 1.5]),
        "fixed": {
            "guidance_scale": 4.5,
            "seg_scale": 0.0,
            "seg_blur_sigma": 5.0,
        },
    },
]

FINALS = [
    {
        "label": "d4+s1 / 1.0",
        "guidance_scale": 4.5,
        "seg_applied_layers": ["d4", "s1"],
        "seg_scale": 0.0,
        "oseg_scale": 1.0,
        "seg_blur_sigma": 5.0,
    },
    {
        "label": "d4+s2 / 1.0",
        "guidance_scale": 4.5,
        "seg_applied_layers": ["d4", "s2"],
        "seg_scale": 0.0,
        "oseg_scale": 1.0,
        "seg_blur_sigma": 5.0,
    },
    {
        "label": "d3+s1 / 0.5",
        "guidance_scale": 4.5,
        "seg_applied_layers": ["d3", "s1"],
        "seg_scale": 0.0,
        "oseg_scale": 0.5,
        "seg_blur_sigma": 5.0,
    },
]


def get_params():
    return BASE_PARAMS, SWEEPS


def get_finals():
    return FINALS
