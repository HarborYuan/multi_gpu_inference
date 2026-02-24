# Multi-GPU Inference Benchmark

This project provides a benchmark tool to compare different methods for running multiple GPUs for inference of a Large Language Model using Hugging Face Transformers. 

The test defaults to `Qwen/Qwen2.5-VL-7B-Instruct` as the inference target.

## Methods Tested

1. **Pipeline Parallelism (`device_map="auto"`)**:
   - The model is split across all available GPUs.
   - It processes queries one by one (or in small batches), sequentially passing through GPU 0, then GPU 1, etc.
   - **Pros:** Can fit models that do not fit on a single GPU (e.g., 70B models).
   - **Cons:** Does not improve throughput (in fact, can be slower due to inter-GPU communication).

2. **Data Parallelism via Multiprocessing**:
   - Spawns N independent processes (one for each GPU).
   - Each GPU loads a full replica of the model.
   - The total number of requests is divided evenly among the GPUs and processed in parallel.
   - **Pros:** Maximum throughput scaling (nearly linear scaling as you add more GPUs).
   - **Cons:** Requires the model to completely fit into a single GPU's VRAM.

## How to use

Ensure you have your environment set up and the packages installed via `uv`.

To install dependencies:
```bash
uv sync
```

To run the benchmark comparing both methods:
```bash
uv run python benchmark.py --samples 20 --model Qwen/Qwen2.5-VL-7B-Instruct --method both
```

You can test a specific method by using `--method pipeline` or `--method multiprocessing`.

### Notes
- Ensure you have authenticated with Hugging Face (`huggingface-cli login`) if you use gated models (e.g. Llama-3).
- The dummy inference uses a sample image URL. In actual high-throughput scenarios, downloading images could be a bottleneck, so consider replacing the image input with local images.
