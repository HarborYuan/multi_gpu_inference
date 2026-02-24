import time
import argparse
from vllm import LLM, SamplingParams

# Dummy prompt to simulate inference
DUMMY_MESSAGES = [
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"},
            },
            {"type": "text", "text": "Describe this image in detail."},
        ],
    }
]

def benchmark_vllm(model_id: str, num_samples: int, tp_size: int):
    print(f"\n--- Starting vLLM Benchmark (tensor_parallel_size={tp_size}) ---")
    start_load = time.time()

    llm = LLM(
        model=model_id,
        tensor_parallel_size=tp_size,
        dtype="bfloat16",
        limit_mm_per_prompt={"image": 1, "video": 0},
        trust_remote_code=True
    )
    load_time = time.time() - start_load
    print(f"Model loaded in {load_time:.2f} seconds.")
    sampling_params = SamplingParams(
        max_tokens=128, 
        temperature=0.0 # Greedy decoding
    )

    # Warmup
    print("Running warmup...")
    llm.chat(messages=DUMMY_MESSAGES, sampling_params=sampling_params)
    print(f"Running batch inference on {num_samples} samples...")
    start_infer = time.time()
    
    requests = [DUMMY_MESSAGES for _ in range(num_samples)]
    outputs = llm.chat(messages=requests, sampling_params=sampling_params, use_tqdm=True)

    infer_time = time.time() - start_infer
    throughput = num_samples / infer_time

    print(f"\n[Results] vLLM Output Sample: {outputs[0].outputs[0].text.strip()}")
    print(f"[Results] vLLM: {num_samples} samples take {infer_time:.2f}s (Throughput: {throughput:.2f} samples/sec)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-GPU Inference Benchmark for Qwen-VL using vLLM")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-VL-30B-A3B-Instruct", help="HuggingFace model ID")
    parser.add_argument("--samples", type=int, default=20, help="Total number of samples to process in benchmark")
    parser.add_argument("--tp", type=int, default=8, help="Tensor Parallel size (Number of GPUs)")
    args = parser.parse_args()
    benchmark_vllm(args.model, args.samples, args.tp)
