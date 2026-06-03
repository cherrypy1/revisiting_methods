BASE_PARAMS = {
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 30,
    "guidance_scale": 5.5,
}

SWEEPS = [
    {
        "name": "scale_x_steps",
        "row": ("guidance_scale", [4.0, 5.0, 5.5, 6.0, 7.0]),
        "col": ("num_inference_steps", [25, 30, 40]),
        "fixed": {},
    },
    {
        "name": "fine_scale",
        "row": ("guidance_scale", [5.0, 5.5, 6.0, 6.5]),
        "col": ("num_inference_steps", [28, 30, 40]),
        "fixed": {},
    },
]

FINALS = [
    {
        "label": "cfg 5",
        "guidance_scale": 5.0,
        "num_inference_steps": 30,
    },
    {
        "label": "cfg 5.5",
        "guidance_scale": 5.5,
        "num_inference_steps": 30,
    },
    {
        "label": "cfg 6",
        "guidance_scale": 6.0,
        "num_inference_steps": 30,
    },
]


def get_params():
    return BASE_PARAMS, SWEEPS


def get_finals():
    return FINALS
