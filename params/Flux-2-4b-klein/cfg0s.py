BASE_PARAMS = {
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 30,
    "guidance_scale": 6.5,
    "use_cfg_zero_star": True,
    "use_zero_init": True,
    "zero_steps": 0,
}

SWEEPS = [
    {
        "name": "scale_x_zero_steps",
        "row": ("guidance_scale", [5.0, 6.0, 7.0]),
        "col": ("zero_steps", [0, 1, 2]),
        "fixed": {
            "use_cfg_zero_star": True,
            "use_zero_init": True,
        },
    },
    {
        "name": "components",
        "row": ("use_cfg_zero_star", [False, True]),
        "col": ("use_zero_init", [False, True]),
        "fixed": {
            "guidance_scale": 6.5,
            "zero_steps": 0,
        },
    },
]

FINALS = [
    {
        "label": "zero star + init 0",
        "guidance_scale": 6.5,
        "use_cfg_zero_star": True,
        "use_zero_init": True,
        "zero_steps": 0,
    },
    {
        "label": "zero star only",
        "guidance_scale": 6.5,
        "use_cfg_zero_star": True,
        "use_zero_init": False,
        "zero_steps": 0,
    },
    {
        "label": "zero init 0 only",
        "guidance_scale": 6.5,
        "use_cfg_zero_star": False,
        "use_zero_init": True,
        "zero_steps": 0,
    },
]


def get_params():
    return BASE_PARAMS, SWEEPS


def get_finals():
    return FINALS
