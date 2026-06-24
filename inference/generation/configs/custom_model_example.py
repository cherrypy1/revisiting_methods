# Example: Loading a custom/local model with custom pipeline
# This shows how to use a callable for maximum flexibility

import torch

def pipeline(device):
    """
    Custom pipeline loader function.
    
    Args:
        device: torch device to load the model to
    
    Returns:
        A pipeline object with __call__ method that accepts:
        - prompt
        - height, width
        - num_inference_steps
        - guidance_scale
        - num_images_per_prompt
        - negative_prompt
        And returns object with .images attribute
    """
    from diffusers import StableDiffusionPipeline
    
    # Load your custom model
    model = StableDiffusionPipeline.from_pretrained(
        "/path/to/your/local/model",  # Local path or HuggingFace ID
        torch_dtype=torch.float16,
        # Add any custom params here
    )
    model = model.to(device)
    model.enable_attention_slicing()
    
    # You can also load LoRA weights here:
    # model.load_lora_weights("/path/to/lora")
    
    return model


# Override generation parameters
generation_params = {
    "steps": 25,
    "scale": 7.0,
    "seed": 12345,
}
