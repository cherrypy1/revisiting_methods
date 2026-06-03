BASE_PARAMS = {
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 30,
    "guidance_scale": 7.0,
    "use_apg": True,
    "apg_eta": 0.0,
    "apg_step_radius": 15.0,
    "apg_momentum": -0.5,
}

SWEEPS = [
    {
        "name": "scale_x_radius",
        "row": ("guidance_scale", [5.5, 6.5, 7.0]),
        "col": ("apg_step_radius", [10.0, 15.0, 22.5, 30.0]),
        "fixed": {
            "use_apg": True,
            "apg_eta": 0.0,
            "apg_momentum": -0.5,
        },
    },
    {
        "name": "momentum_x_eta",
        "row": ("apg_momentum", [-0.75, -0.5, -0.25, 0.0]),
        "col": ("apg_eta", [0.0, 0.15, 0.25]),
        "fixed": {
            "use_apg": True,
            "guidance_scale": 7.0,
            "apg_step_radius": 15.0,
        },
    },
]

FINALS = [
    {
        "label": "r15 beta -0.5",
        "guidance_scale": 7.0,
        "use_apg": True,
        "apg_eta": 0.0,
        "apg_step_radius": 15.0,
        "apg_momentum": -0.5,
    },
    {
        "label": "r22.5 beta -0.5",
        "guidance_scale": 7.0,
        "use_apg": True,
        "apg_eta": 0.0,
        "apg_step_radius": 22.5,
        "apg_momentum": -0.5,
    },
    {
        "label": "scale 6.5 r15",
        "guidance_scale": 6.5,
        "use_apg": True,
        "apg_eta": 0.0,
        "apg_step_radius": 15.0,
        "apg_momentum": -0.5,
    },
]


def get_params():
    return BASE_PARAMS, SWEEPS


def get_finals():
    return FINALS
