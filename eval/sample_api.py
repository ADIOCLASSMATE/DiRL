import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import random
from typing import List
from dataclasses import dataclass
import json
from transformers import AutoTokenizer
from openai import OpenAI
import socket
from tqdm import tqdm
def get_host_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip


@dataclass
class GenerationOutput:
    """模拟lmdeploy的输出格式"""
    text: str
    prompt_ids: List[int]
    token_ids: List[int]
    step_map: List[int]
    generation_time: float = 0.0
    prompt_length: int = 0
    token_length: int = 0


def sample(client, model_name, gen_config, prompts, k_sample=1, config=None, accelerator=None, use_tqdm=True):
    """
    使用同步API调用来生成响应，保持与原sample函数相同的输入输出接口
    
    Args:
        model_path: 模型路径（这里可以用来标识模型名称）
        backend_config: 后端配置（API模式下不使用）
        gen_config: 生成配置
        prompts: 输入提示列表
        k_sample: 每个提示生成k个响应
        config: 全局配置对象
        accelerator: Accelerator对象
        use_tqdm: 是否使用进度条
    
    Returns:
        List[GenerationOutput]: 生成的输出列表
    """

    def call_api(prompt, idx):
        """同步API调用，为每个请求生成唯一的session_id"""
        prompts_sub = [prompt] * k_sample
        # 为每个请求生成唯一的整数session_id（使用时间戳纳秒+索引+随机数）
        session_id = int(time.time() * 1e9) + idx * 1000000 + random.randint(0, 999999)

        response = client.completions.create(
            model=model_name,
            prompt=prompts_sub,
            temperature=gen_config.temperature,
            top_p=gen_config.top_p,
            max_tokens=gen_config.max_new_tokens,
            stop=gen_config.stop_words,
            extra_body={
                "top_k": gen_config.top_k,
                "session_id": session_id
            },
        )
        struct_output = [GenerationOutput(
            text=g.text, 
            prompt_ids=g.input_ids, 
            token_ids=g.token_ids, 
            step_map=g.step_map, 
            generation_time=g.generation_time,
            prompt_length=len(g.input_ids) if g.input_ids else 0,
            token_length=len(g.token_ids) if g.token_ids else 0
        ) for g in response.choices]
        return idx, struct_output
    
    # 顺序执行所有请求
    outputs = []
    tasks = list(enumerate(prompts))
    
    # 使用tqdm包装循环
    iterator = tqdm(tasks, desc="Generating responses") if use_tqdm else tasks
    results = []
    for i, prompt in iterator:
        try:
            result = call_api(prompt, i)
            results.append(result)
        except Exception as e:
            print(f"[API Error] {e}")
            if accelerator and accelerator.is_main_process:
                print(f"[API Error] {e}")
    
    # 收集结果并按原始顺序排序
    indexed_results = results
    
    # 按索引排序确保顺序正确
    indexed_results.sort(key=lambda x: x[0])
    
    # 提取输出
    for idx, struct_output in indexed_results:
        outputs.extend(struct_output)

    print(f"[API Mode] [Rank {accelerator.process_index}] Successfully generated {len(outputs)}/{len(prompts)*k_sample} responses")
    return outputs

def main():
    model_name = "test"
    model_path = "/inspire/hdd/global_user/liuxiaoran-240108120089/public/SDAR-8B-Chat"
    local_ip = get_host_ip()
    rank = 0
    port = 12349
    base_url = f'http://{local_ip}:{port}/v1'
    # print(base_url)
    api_key = "sk-"
    print("开始测试")
    client = OpenAI(api_key="sk", base_url=f"http://0.0.0.0:{port}/v1")

    prompts = ["<|im_start|>user\nFind the greatest integer less than $(\\sqrt{7} + \\sqrt{5})^6.$  (Do not use a calculator!)\nPlease reason step by step, and put your final answer within $\\boxed{}$.<|im_end|>\n<|im_start|>assistant\n"]*10
    k_sample = 16
    config = None
    accelerator = None
    use_tqdm = True
    outputs = sample(client, model_name, gen_config=type('obj', (object,), {
        'temperature': 1.0,
        'top_p': 1.0,
        'top_k': 50,
        'max_new_tokens': 2048,
        'stop_words': None
    })(), prompts=prompts, k_sample=k_sample, config=config, accelerator=accelerator, use_tqdm=use_tqdm)

if __name__ == "__main__":
    main()
