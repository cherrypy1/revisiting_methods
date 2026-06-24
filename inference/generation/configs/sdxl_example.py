# Example config for SDXL model
# Usage: python generation/diffusers_generate.py prompts/evaluation_metadata.jsonl --config generation/configs/sdxl_example.py

pipeline = {
    "class": "StableDiffusionXLPipeline",
    "pretrained": "stabilityai/stable-diffusion-xl-base-1.0",
    "params": {
        "torch_dtype": "float16",  # Will be converted to torch.float16
        "use_safetensors": True,
        "variant": "fp16",
    }
}

# Override generation parameters
generation_params = {
    "steps": 30,
    "scale": 7.5,
    "H": 1024,
    "W": 1024,
    "n_samples": 4,
}
