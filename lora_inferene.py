"""
conda activate simpletuner
export CUDA_VISIBLE_DEVICES=3
python lora_inferene.py
"""

import torch
import os
import json
import time
import random
import hashlib
import gradio as gr
from PIL import Image
from helpers.models.hidream.pipeline import HiDreamImagePipeline
from helpers.models.hidream.transformer import HiDreamImageTransformer2DModel
from transformers import PreTrainedTokenizerFast, LlamaForCausalLM
from lycoris import create_lycoris_from_weights

# Load base model components
llama_repo = "/local/yada/apps/HiDream-I1-a/Meta-Llama-3.1-8B-Instruct"
model_id = "HiDream-ai/HiDream-I1-Full"

# --- LoRA setup ---
lora_options = {
    "anime 400 run4": "/local/yada/apps/SimpleTuner-a/output/models-hidream-run04/pytorch_lora_weights.safetensors",
}

save_dir = "output/gradio_3".rstrip('/')

tokenizer_4 = PreTrainedTokenizerFast.from_pretrained(llama_repo)
text_encoder_4 = LlamaForCausalLM.from_pretrained(llama_repo, output_hidden_states=True, torch_dtype=torch.bfloat16)
transformer = HiDreamImageTransformer2DModel.from_pretrained(model_id, torch_dtype=torch.bfloat16, subfolder="transformer")

pipeline = HiDreamImagePipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    tokenizer_4=tokenizer_4,
    text_encoder_4=text_encoder_4,
    transformer=transformer,
)

pipeline.to('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')


current_lora_path = None
curr_lora_scale = None

def apply_lora_if_needed(new_lora_path, lora_scale=1.0):
    global current_lora_path
    global curr_lora_scale
    global pipeline

    if current_lora_path == new_lora_path and curr_lora_scale == lora_scale:
        return  # No need to reapply

    # Re-initialize transformer to discard previous merged LoRA
    transformer = HiDreamImageTransformer2DModel.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, subfolder="transformer"
    )

    wrapper, _ = create_lycoris_from_weights(lora_scale, new_lora_path, transformer)
    wrapper.merge_to()

    # Attach updated transformer to pipeline
    pipeline.transformer = transformer

    current_lora_path = new_lora_path
    curr_lora_scale = lora_scale


# --- Image Generation ---
def generate_image(prompt, negative_prompt, width, height, seed, guidance_scale, num_inference_steps, lora_choice, lora_scale=1.0):
    lora_path = lora_options[lora_choice]
    apply_lora_if_needed(lora_path, lora_scale)

    if seed == -1:
        seed = random.randint(0, 2**32 - 1)

    # Encode prompts
    t5_embeds, llama_embeds, negative_t5_embeds, negative_llama_embeds, pooled_embeds, negative_pooled_embeds = pipeline.encode_prompt(
        prompt=prompt, prompt_2=prompt, prompt_3=prompt, prompt_4=prompt, num_images_per_prompt=1
    )

    # Generate
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    image = pipeline(
        t5_prompt_embeds=t5_embeds,
        llama_prompt_embeds=llama_embeds,
        pooled_prompt_embeds=pooled_embeds,
        negative_t5_prompt_embeds=negative_t5_embeds,
        negative_llama_prompt_embeds=negative_llama_embeds,
        negative_pooled_prompt_embeds=negative_pooled_embeds,
        num_inference_steps=num_inference_steps,
        generator=torch.Generator(device=device).manual_seed(seed),
        width=width,
        height=height,
        guidance_scale=guidance_scale,
    ).images[0]

    # Save with metadata
    timestamp = int(time.time())
    image_hash = hashlib.sha256(f"{prompt}{negative_prompt}{timestamp}".encode()).hexdigest()[:10]
    base_path = f"{save_dir}/{timestamp}_{image_hash}"
    os.makedirs(save_dir, exist_ok=True)
    image.save(f"{base_path}.png")

    metadata = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "timestamp": timestamp,
        # "image_path": f"{base_path}.png",
        "seed": seed,
        "lora": lora_path,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
    }
    with open(f"{base_path}.json", "w") as f:
        json.dump(metadata, f, indent=4)

    return image

# --- Gradio Interface ---
demo = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Textbox(label="Prompt", lines=4, value="an anime illustration of hatsune miku, 1girl, white gloves, dress, cat ears, maid outfit, indoors, cat ears"),
        gr.Textbox(label="Negative Prompt", lines=2, value="ugly, cropped, blurry, low-quality, mediocre average"),
        gr.Slider(minimum=256, maximum=1536, step=64, value=768, label="Width"),
        gr.Slider(minimum=256, maximum=1536, step=64, value=1280, label="Height"),
        gr.Slider(minimum=-1, maximum=2**32 - 1, step=1, value=-1, label="Seed"),
        gr.Slider(minimum=0.1, maximum=20.0, step=0.1, value=4, label="Guidance Scale"),
        gr.Slider(minimum=1, maximum=200, step=1, value=50, label="Inference Steps"),
        gr.Dropdown(choices=list(lora_options.keys()), label="LoRA Weights", value=list(lora_options.keys())[0]),
        gr.Slider(minimum=0, maximum=5.0, step=0.1, value=1.0, label="LoRA Scale"),
    ],
    outputs=gr.Image(label="Generated Image"),
    title="HiDream Demo: LoRA Swap Edition",
    description=f"Now with dynamic LoRA switching. Currently using base model: `{model_id}`.",
)

if __name__ == "__main__":
    demo.launch(share=True)