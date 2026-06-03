BASE_PARAMS = {
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 30,
}

SWEEPS = [
    {
        "name": "layers_x_pag",
        "row": (
            "pag_applied_layers",
            [
                ["d2", "d3", "d4"],
                ["d4", "s1"],
                ["s0", "s1", "s2"],
                ["s4", "s5", "s6", "s7"],
            ],
        ),
        "col": ("pag_scale", [0.0, 1.0, 1.5, 2.0, 2.5]),
        "fixed": {
            "guidance_scale": 4.5,
        },
    },
    
]

FINALS = [
    {
        "label": "d4+s0 / 2.5",
        "guidance_scale": 4.5,
        "pag_applied_layers": ["d4", "s0"],
        "pag_scale": 2.5,
    },
    {
        "label": "d4+s1 / 2.5",
        "guidance_scale": 4.5,
        "pag_applied_layers": ["d4", "s1"],
        "pag_scale": 2.5,
    },
    {
        "label": "d4+s2 / 2.5",
        "guidance_scale": 4.5,
        "pag_applied_layers": ["d4", "s2"],
        "pag_scale": 2.5,
    },
]


def get_params():
    return BASE_PARAMS, SWEEPS


def get_finals():
    return FINALS
