from vllm import LLM, SamplingParams
messages = [
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
print("Running test...")
llm = LLM(model="Qwen/Qwen3-VL-30B-A3B-Instruct", tensor_parallel_size=8, dtype="bfloat16", limit_mm_per_prompt={"image": 1, "video": 0}, trust_remote_code=True)
sampling_params = SamplingParams(max_tokens=128, temperature=0.0)
outputs = llm.chat(messages=messages, sampling_params=sampling_params)
print(outputs[0].outputs[0].text)
