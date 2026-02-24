import time
import argparse
from typing import Any

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info


from vllm import LLM, SamplingParams


# Dummy prompt to simulate inference
DUMMY_MESSAGES = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Describe this image in detail."},
        ],
    }
]

def load_model_and_processor(model_id: str, device_map: Any):
    # Load processor
    processor = AutoProcessor.from_pretrained(model_id)
    
    # Load model
    # We use bfloat16 for optimal performance on modern GPUs (Ampere and later)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, 
        dtype=torch.bfloat16, 
        device_map=device_map
    )
    return model, processor

def run_inference(model, processor, messages, device):
    # Prepare input
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )
    
    first_device = next(model.parameters()).device
    inputs = inputs.to(first_device)
        

    # Generate output
    # Limit max_new_tokens for predictable benchmark duration
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]

def benchmark_pipeline_parallel(model_id: str, num_samples: int):
    print(f"\n--- Starting Pipeline Parallelism Benchmark (device_map='auto') ---")
    start_load = time.time()
    model, processor = load_model_and_processor(model_id, device_map="auto")
    load_time = time.time() - start_load
    print(f"Model loaded in {load_time:.2f} seconds.")

    # Warmup
    print("Running warmup...")
    run_inference(model, processor, DUMMY_MESSAGES, "auto")

    # Benchmark
    print(f"Running inference on {num_samples} samples...")
    start_infer = time.time()
    for i in range(num_samples):
        run_inference(model, processor, DUMMY_MESSAGES, "auto")
        if (i + 1) % 5 == 0:
            print(f"Processed {i + 1}/{num_samples} samples")
            
    infer_time = time.time() - start_infer
    throughput = num_samples / infer_time
    
    print(f"[Results] Pipeline Parallelism: {num_samples} samples take {infer_time:.2f}s (Throughput: {throughput:.2f} samples/sec)")
    
    # Clean up to free VRAM for the next test
    del model
    del processor
    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-GPU Inference Benchmark for Qwen-VL")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-VL-30B-A3B-Instruct", help="HuggingFace model ID")
    parser.add_argument("--samples", type=int, default=20, help="Total number of samples to process in benchmark")
    args = parser.parse_args()
    benchmark_pipeline_parallel(args.model, args.samples)
