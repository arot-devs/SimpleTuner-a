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
llama_repo = "/local/yada/apps/HiDream-I1-a/Meta-Llama-3.1-8B-Instruct"
model_id = "HiDream-ai/HiDream-I1-Full"

# --- LoRA setup ---
lora_options = {
    "anime 400 run4": "/local/yada/apps/SimpleTuner-a/output/models-hidream-run04/pytorch_lora_weights.safetensors",
    # "nai artist run5": "/local/yada/apps/SimpleTuner-a/output/models-hidream-run05-nai4/pytorch_lora_weights.safetensors",
    "run06-nai4": "/local/yada/apps/SimpleTuner-a/output/models-hidream-run06-nai4/pytorch_lora_weights.safetensors",}

save_dir = "output/gradio_3".rstrip('/')

tokenizer_4 = PreTrainedTokenizerFast.from_pretrained(llama_repo)
# tokenizer_4.model_max_length = 512

text_encoder_4 = LlamaForCausalLM.from_pretrained(
    llama_repo, output_hidden_states=True, torch_dtype=torch.bfloat16)


transformer = HiDreamImageTransformer2DModel.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, subfolder="transformer")

pipeline = HiDreamImagePipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    tokenizer_4=tokenizer_4,
    text_encoder_4=text_encoder_4,
    transformer=transformer,
)

pipeline.to('cuda' if torch.cuda.is_available()
            else 'mps' if torch.backends.mps.is_available() else 'cpu')

print(f"Llama Tokenizer model_max_length: {pipeline.tokenizer_4.model_max_length}")
if hasattr(pipeline.text_encoder_4, 'config') and hasattr(pipeline.text_encoder_4.config, 'max_position_embeddings'):
    print(f"Llama Model max_position_embeddings: {pipeline.text_encoder_4.config.max_position_embeddings}")
else:
    print("Could not determine Llama model's max_position_embeddings from config.")

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


# --- Image Generation ---
def generate_image(prompt, negative_prompt, width, height, seed, guidance_scale, num_inference_steps, lora_choice, lora_scale=1.0):
    lora_path = lora_options[lora_choice]
    # Pass lora_scale to apply_lora_if_needed
    apply_lora_if_needed(lora_path, lora_scale)

    # --- Define desired max length ---
    # Ensure it's within the model's capabilities (8192 based on your logs)
    # And practical limits (e.g., VRAM)
    desired_max_len = 256
    actual_model_limit = 8192 # From your log: pipeline.text_encoder_4.config.max_position_embeddings
    effective_max_len = min(desired_max_len, actual_model_limit)

    print(f"--- Using effective max_sequence_length for encoding: {effective_max_len} ---")

    if seed == -1:
        seed = random.randint(0, 2**32 - 1)

    device = 'cuda' if torch.cuda.is_available(
    ) else 'mps' if torch.backends.mps.is_available() else 'cpu'

    # Encode prompts - PASS max_sequence_length HERE
    (
        t5_prompt_embeds, llama_prompt_embeds,
        negative_t5_prompt_embeds, negative_llama_prompt_embeds,
        pooled_prompt_embeds, negative_pooled_prompt_embeds
    ) = pipeline.encode_prompt(
        prompt=prompt, prompt_2=prompt, prompt_3=prompt, prompt_4=prompt,
        negative_prompt=negative_prompt, negative_prompt_2=negative_prompt,
        negative_prompt_3=negative_prompt, negative_prompt_4=negative_prompt,
        num_images_per_prompt=1,
        max_sequence_length=effective_max_len, # <<< ADD THIS ARGUMENT >>>
        device=device,
        dtype=pipeline.text_encoder_4.dtype # Use appropriate dtype for safety
    )

    # --- Verify Embeddings Shape (Optional Debugging) ---
    if llama_prompt_embeds is not None:
        # Shape can be [num_layers, batch, seq, dim] or [batch, num_layers, 1, seq, dim]
        llama_seq_len = llama_prompt_embeds.shape[-2]
        print(f"Llama embedding sequence length after encoding: {llama_seq_len}")
        if llama_seq_len < effective_max_len and len(prompt.split()) > llama_seq_len // 4: # Rough check
             print(f"WARNING: Llama embedding length ({llama_seq_len}) is less than requested ({effective_max_len}), might still be truncated elsewhere or prompt is short.")
        elif llama_seq_len > 128:
             print(f"Llama embedding length ({llama_seq_len}) successfully exceeds 128.")

    # Generate using pre-computed embeddings
    image = pipeline(
        t5_prompt_embeds=t5_prompt_embeds,
        llama_prompt_embeds=llama_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_t5_prompt_embeds=negative_t5_prompt_embeds,
        negative_llama_prompt_embeds=negative_llama_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        num_inference_steps=num_inference_steps,
        generator=torch.Generator(device=device).manual_seed(seed),
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        # This is less critical now, but doesn't hurt
        max_sequence_length=effective_max_len,
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
        "lora_scale": lora_scale, # Add LoRA scale to metadata
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
        "max_sequence_length_used": effective_max_len, # Log the length used
    }
    with open(f"{base_path}.json", "w") as f:
        json.dump(metadata, f, indent=4)

    return image

# --- Gradio Interface ---
demo = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Textbox(label="Prompt", lines=4,
                   value="an anime illustration of hatsune miku, 1girl, white gloves, dress, cat ears, maid outfit, indoors, cat ears"),
        gr.Textbox(label="Negative Prompt", lines=2,
                   value="ugly, cropped, blurry, low-quality, mediocre average"),
        gr.Slider(minimum=256, maximum=1536,
                  step=64, value=768, label="Width"),
        gr.Slider(minimum=256, maximum=1536, step=64,
                  value=1280, label="Height"),
        gr.Slider(minimum=-1, maximum=2**32 - 1,
                  step=1, value=-1, label="Seed"),
        gr.Slider(minimum=0.1, maximum=20.0, step=0.1,
                  value=4, label="Guidance Scale"),
        gr.Slider(minimum=1, maximum=200, step=1,
                  value=30, label="Inference Steps"),
        gr.Dropdown(choices=list(lora_options.keys()),
                    label="LoRA Weights", value=list(lora_options.keys())[-1]),
        gr.Slider(minimum=0.0, maximum=2.0, step=0.1,
                  value=1.0, label="LoRA Scale"),
    ],
    outputs=gr.Image(label="Generated Image"),
    title="HiDream Demo: LoRA Swap Edition",
    description=f"Now with dynamic LoRA switching. Currently using base model: `{model_id}`.",
)

if __name__ == "__main__":
    demo.launch(share=True, server_port=7998)
