BASE_PARAMS = {
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 30,
    "guidance_scale": 6.0,
    "use_tcfg": True,
    "tcfg_rank": 2,
}

SWEEPS = [
    {
        "name": "fine_scale",
        "row": ("guidance_scale", [5.0, 6.0, 7.0]),
        "col": ("tcfg_rank", [1, 2]),
        "fixed": {
            "num_inference_steps": 30,
            "use_tcfg": True,
        },
    },
]

FINALS = [
    {
        "label": "rank 2 / scale 6",
        "guidance_scale": 6.0,
        "num_inference_steps": 30,
        "use_tcfg": True,
        "tcfg_rank": 2,
    },
    {
        "label": "rank 1 / scale 6",
        "guidance_scale": 6.0,
        "num_inference_steps": 30,
        "use_tcfg": True,
        "tcfg_rank": 1,
    },
    {
        "label": "rank 2 / scale 6.5",
        "guidance_scale": 6.5,
        "num_inference_steps": 30,
        "use_tcfg": True,
        "tcfg_rank": 2,
    },
]


def get_params():
    return BASE_PARAMS, SWEEPS


def get_finals():
    return FINALS
