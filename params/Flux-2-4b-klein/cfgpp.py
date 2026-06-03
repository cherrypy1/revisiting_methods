BASE_PARAMS = {
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 30,
    "guidance_scale": 1.2,
    "use_cfgpp": True,
}

SWEEPS = [
    {
        "name": "scale_x_steps",
        "row": ("guidance_scale", [0.7, 1.0, 1.2, 1.4, 1.6]),
        "col": ("num_inference_steps", [25, 30]),
        "fixed": {
            "use_cfgpp": True,
        },
    },
]

FINALS = [
    {
        "label": "scale 1.0 / 30",
        "guidance_scale": 1.0,
        "num_inference_steps": 30,
        "use_cfgpp": True,
    },
    {
        "label": "scale 1.2 / 30",
        "guidance_scale": 1.2,
        "num_inference_steps": 30,
        "use_cfgpp": True,
    },
    {
        "label": "scale 1.4 / 30",
        "guidance_scale": 1.4,
        "num_inference_steps": 30,
        "use_cfgpp": True,
    },
]


def get_params():
    return BASE_PARAMS, SWEEPS


def get_finals():
    return FINALS
