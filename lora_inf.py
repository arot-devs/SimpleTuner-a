"""
conda activate simpletuner
export CUDA_VISIBLE_DEVICES=3
python lora_inferene.py
"""

import gc
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
llama_repo = "/local/yada/models/Meta-Llama-3.1-8B-Instruct"
model_id = "HiDream-ai/HiDream-I1-Full"

# --- LoRA setup ---
lora_options = {
    # "anime 400 run4": "/local/yada/apps/SimpleTuner-a/output/models-hidream-run04/pytorch_lora_weights.safetensors",
    # "nai artist run5": "/local/yada/apps/SimpleTuner-a/output/models-hidream-run05-nai4/pytorch_lora_weights.safetensors",
    # "run06-nai4": "/local/yada/apps/SimpleTuner-a/output/models-hidream-run06-nai4/pytorch_lora_weights.safetensors",
    "run07-higherlr-n4": "/local/yada/apps/SimpleTuner-a/output/models-hidream-run07-nai4-longer-higherlr/checkpoint-1200/ema/ema_model.safetensors",
    "run08-continue-s25": "/local/yada/apps/SimpleTuner-a/output/models-hidream-run08-continue-from-07/checkpoint-2500/ema/ema_model.safetensors",
    }

save_dir = "output/gradio_3".rstrip('/')

tokenizer_4 = PreTrainedTokenizerFast.from_pretrained(llama_repo)
text_encoder_4 = LlamaForCausalLM.from_pretrained(
    llama_repo, output_hidden_states=True, torch_dtype=torch.bfloat16)
transformer = HiDreamImageTransformer2DModel.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, subfolder="transformer")





def unload_pipeline_model():
    global pipeline
    if pipeline is not None:
        # try:
        #     pipeline.to("meta")  # aggressively clear GPU memory
        # except Exception:
        pipeline.to("cpu")  # fallback in case of meta issues
        del pipeline
        pipeline = None

    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(1)


current_lora_path = None


def apply_lora_if_needed(new_lora_path, lora_scale=1.0):
    global current_lora_path
    global pipeline

    # Only reload if LoRA path or scale actually changed
    if current_lora_path == new_lora_path and getattr(pipeline, "lora_scale", None) == lora_scale:
        return

    # Fully unload the old pipeline from GPU
    unload_pipeline_model()

    # Now load or merge the new LoRA
    transformer = HiDreamImageTransformer2DModel.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, subfolder="transformer"
    )
    wrapper, _ = create_lycoris_from_weights(lora_scale, new_lora_path, transformer)
    wrapper.merge_to()

    # Rebuild pipeline on CPU or GPU
    pipeline = HiDreamImagePipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        tokenizer_4=tokenizer_4,
        text_encoder_4=text_encoder_4,
        transformer=transformer,
    )

    pipeline.lora_scale = lora_scale
    pipeline.to("cuda")  # or whichever device
    current_lora_path = new_lora_path

pipeline = None
# Preload the default LoRA at startup
default_lora_key = list(lora_options.keys())[-1]
apply_lora_if_needed(lora_options[default_lora_key])
current_lora_path = lora_options[default_lora_key]


# --- Image Generation ---
def generate_image(prompt, negative_prompt, width, height, seed, guidance_scale, num_inference_steps, lora_choice, lora_scale=1.0):
    lora_path = lora_options[lora_choice]
    apply_lora_if_needed(lora_path)

    if seed == -1:
        seed = random.randint(0, 2**32 - 1)

    # Encode prompts
    t5_embeds, llama_embeds, negative_t5_embeds, negative_llama_embeds, pooled_embeds, negative_pooled_embeds = pipeline.encode_prompt(
        prompt=prompt, prompt_2=prompt, prompt_3=prompt, prompt_4=prompt, num_images_per_prompt=1
    )

    # Generate
    device = 'cuda' if torch.cuda.is_available(
    ) else 'mps' if torch.backends.mps.is_available() else 'cpu'
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
    image_hash = hashlib.sha256(
        f"{prompt}{negative_prompt}{timestamp}".encode()).hexdigest()[:10]
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
        gr.Textbox(label="Prompt", lines=4,
                   value="an anime illustration of hatsune miku, 1girl, white gloves, dress, cat ears, maid outfit, indoors, holding a sign that writes 'HiDream', trending on pxiv"),
        gr.Textbox(label="Negative Prompt", lines=2,
                   value="ugly, cropped, blurry, low-quality, mediocre average"),
        gr.Slider(minimum=256, maximum=2048,
                  step=64, value=832, label="Width"),
        gr.Slider(minimum=256, maximum=2048, step=64,
                  value=1216, label="Height"),
        gr.Slider(minimum=-1, maximum=2**32 - 1,
                  step=1, value=1337, label="Seed"),
        gr.Slider(minimum=0, maximum=20.0, step=0.1,
                  value=4, label="Guidance Scale"),
        gr.Slider(minimum=1, maximum=200, step=1,
                  value=30, label="Inference Steps"),
        gr.Dropdown(choices=list(lora_options.keys()),
                    label="LoRA Weights", value=list(lora_options.keys())[-1]),
        # gr.Slider(minimum=0.0, maximum=2.0, step=0.1,
        #           value=1.0, label="LoRA Scale"),
    ],
    outputs=gr.Image(label="Generated Image"),
    title="HiDream Demo: LoRA Swap Edition",
    description=f"Now with dynamic LoRA switching. Currently using base model: `{model_id}`.",
)

if __name__ == "__main__":
    demo.launch(share=False)
