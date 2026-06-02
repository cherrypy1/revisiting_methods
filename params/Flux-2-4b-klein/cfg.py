BASE_PARAMS = {
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 30,
    "guidance_scale": 4.0,
}

SWEEPS = [
    {
        "name": "scale_x_steps",
        "row": ("guidance_scale", [1.0, 4.0, 5.0, 7.0, 15.0]),
        "col": ("num_inference_steps", [30]),
        "fixed": {},
    },
]


def get_params():
    return BASE_PARAMS, SWEEPS
