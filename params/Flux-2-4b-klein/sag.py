BASE_PARAMS = {
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 30,
}

SWEEPS = [
    {
        "name": "layers_x_sag",
        "row": (
            "sag_applied_layers",
            [
                ["d4", "s1"],
                ["d4", "s2"],
            ],
        ),
        "col": ("sag_scale", [0.0, 0.75, 1.0, 1.5]),
        "fixed": {
            "guidance_scale": 4.5,
            "sag_blur_sigma": 1.0,
        },
    },
    {
        "name": "sigma_x_sag",
        "row": ("sag_blur_sigma", [1.0, 2.0, 3.0]),
        "col": ("sag_scale", [0.0, 1.0, 1.5]),
        "fixed": {
            "guidance_scale": 4.5,
            "sag_applied_layers": ["d4", "s1"],
        },
    },
]

FINALS = [
    {
        "label": "d4+s1 / 1.5",
        "guidance_scale": 4.5,
        "sag_applied_layers": ["d4", "s1"],
        "sag_scale": 1.5,
        "sag_blur_sigma": 2.0,
    },
    {
        "label": "d4+s2 / 1.5",
        "guidance_scale": 4.5,
        "sag_applied_layers": ["d4", "s2"],
        "sag_scale": 1.5,
        "sag_blur_sigma": 2.0,
    },
    {
        "label": "d3+d4+s1+s2 / 1.0",
        "guidance_scale": 4.5,
        "sag_applied_layers": ["d3", "d4", "s1", "s2"],
        "sag_scale": 1.0,
        "sag_blur_sigma": 2.0,
    },
]


def get_params():
    return BASE_PARAMS, SWEEPS


def get_finals():
    return FINALS
