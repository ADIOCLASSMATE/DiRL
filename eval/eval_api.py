import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import torch
import torch.distributed as dist
from accelerate import Accelerator
from accelerate.utils import gather_object
from transformers import AutoTokenizer
from datasets import load_dataset
from omegaconf import OmegaConf
import argparse
from openai import OpenAI
import socket
# 假设这些模块存在

from utils.data_chunk import get_data_chunk
from utils.math_utils import equation
from sample_api import sample
from train.utils import get_config

from lmdeploy import pipeline, PytorchEngineConfig, GenerationConfig, ChatTemplateConfig, serve
from lmdeploy.pytorch.tools.utils import Timer, visualize_pipe_out
from accelerate import Accelerator, utils
import accelerate
from transformers import AutoTokenizer
import re
import time

from train.orm import orms
math_judge = orms['accuracy']()

def get_host_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip


def extract_final_boxed_answer(s: str):
    """从输出中提取\\boxed{}中的答案"""
    tag = r'\boxed{'
    start = s.rfind(tag)          # last \boxed{
    if start == -1:
        return "Can not extract the answer!"

    i = start + len(tag)
    depth = 1                    # we are already inside one '{'
    buf = []

    while i < len(s) and depth:
        ch = s[i]
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:       # matching '}' for the opening \boxed{
                break
        buf.append(ch)
        i += 1

    return ''.join(buf) if depth == 0 else "Can not extract the answer!"


def get_token_lengths(strings, tokenizer):
    """计算字符串列表的token长度"""
    pad_token = tokenizer.pad_token

    escaped = re.escape(pad_token)
    pattern = rf"(?:{escaped})+"
    remove_pattern = escaped

    collapse_re = re.compile(pattern)

    lengths = []
    for s in strings:
        s_clean = collapse_re.sub(lambda _: pad_token if isinstance(pad_token, str) else '', s)
        s_clean = re.sub(remove_pattern, '', s_clean)
        lengths.append(len(tokenizer.encode(s_clean, add_special_tokens=False)))
    return lengths



def main():
    config = get_config()
    accelerator = Accelerator()
    # 获取项目名称
    project_name = config.experiment.project
    reward = config.dataset.data_type
    model_path = os.path.expanduser(config.model)
    # 从model_path中提取model_name
    model_name = os.path.basename(model_path)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    stop_token_list = config.rollout.stop_token_list

    backend_config = PytorchEngineConfig(
        dtype="bfloat16",
        max_prefill_token_num=config.rollout.max_token*32,
        cache_max_entry_count=0.95,
        dllm_block_length=config.rollout.block_size,
        dllm_denoising_steps=config.rollout.denoising_steps_per_block,
        dllm_unmasking_strategy=config.rollout.remasking_strategy,
        dllm_confidence_threshold=config.rollout.dynamic_threshold,
    )
    ## serve
    # master_addr = os.environ.get('MASTER_ADDR', 'localhost')
    local_ip = get_host_ip()
    rank = accelerator.process_index
    port = 12446 + 10*int(rank)
    base_url = f'http://{local_ip}:{port}/v1'
    # print(base_url)
    api_key = "sk-"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(rank%8)
    server = serve(
        model_path = model_path,
        model_name = model_name,
        backend = 'pytorch',
        backend_config = backend_config,
        server_name = local_ip,
        server_port = port,
        # cache_max_entry_count = 0.95,
    )
    # time.sleep(30)
    accelerate.utils.wait_for_everyone()
    print("Start Evaluation~")
    client = OpenAI(api_key="sk", base_url=f"http://{local_ip}:{port}/v1")

    
    do_sample=config.rollout.do_sample
    gen_config = GenerationConfig(
        n=1,
        top_p=config.rollout.top_p,
        top_k=config.rollout.top_k,
        temperature=config.rollout.temperature,
        do_sample=do_sample, # greedy decoding
        min_new_tokens=128,
        max_new_tokens=config.rollout.max_token,
        stop_words=["<|im_end|>"],
        skip_special_tokens=False,
        bad_words=["<|endoftext|>"],
        # random_seed=10086 if "Qwen" not in model_path else None,
    )

    print("expect:", gen_config)
    start_with_think = config.rollout.start_with_think
    dataset_name = config.dataset.eval_dataset
    
    # 构建输出文件名（类似ref.py）
    pretrained_model = config.model
    outputs_name = "eval-" + pretrained_model.replace("/", ".") + "-" + dataset_name
    # Load data,
    dataset = load_dataset("json", data_files=f"data/{dataset_name}.json")
    ds = dataset['train']
    prompts = ds['question']
    gts = ds['ground_truth_answer']
    
    reason_prompt = "<|im_start|>user\n{problem}\nPlease reason step by step, and put your final answer within $\\boxed{{}}$.<|im_end|>\n<|im_start|>assistant\n"

    # reason_prompt = "<|im_start|>user\n{problem}\nPlease reason step by step in <think>...</think>, and put your final answer within $\\boxed{{}}$ in <answer>...</answer>.<|im_end|>\n<|im_start|>assistant\n"
    # start_with_think=True
    if start_with_think:
        reason_prompt = reason_prompt + "<think>"
    # if "Qwen3" in model_path:
        # reason_prompt = reason_prompt + "<think></think>"
    # 构建带原始索引的数据
    all_data = list(enumerate(zip(prompts, gts))) # [(idx, (prompt, gt)), ...]

    # Shard data WITH indices
    num_processes = accelerator.num_processes
    rank = accelerator.process_index
    local_data = get_data_chunk(all_data, num_processes, rank)

    if len(local_data) == 0:
        accelerator.print(f"[Rank {rank}] No prompts assigned.")
        # 仍需返回空结果以便 gather
        data = []
    else:
        indices, prompt_gt_pairs = zip(*local_data)
        local_prompts, local_gts = zip(*prompt_gt_pairs)

        # Apply chat template
        prompts_list = [reason_prompt.format(problem=p) for p in local_prompts]
        
        print("参考prompt: ", repr(prompts_list[0]))

        # sample函数现在返回完整的data列表，支持k次采样
        k_sample = getattr(config.rollout, 'num_response_per_task', 1)  # 从配置中获取k值
        
        outputs = sample(client, model_name, gen_config, prompts_list, k_sample, config, accelerator)

        data = []
        for original_idx in range(0, len(outputs) // k_sample):
            question = local_prompts[original_idx]
            prompt = prompts_list[original_idx]
            gt = local_gts[original_idx]

            combined_texts = []
            combined_step_maps = []
            extracted_answers = []
            response_lengths = []
            rewards = []
            truncs = []
            speed_rewards = []
            combined_prompt_ids = []
            combined_token_ids = []
            generation_times = []
            prompt_lengths = []
            token_lengths = []
            for i in range(k_sample):
                output_idx = original_idx * k_sample + i
                o = outputs[output_idx]
                text = o.text
                token_ids = o.token_ids
                step_map = o.step_map
                prompt_ids = o.prompt_ids
                generation_time = o.generation_time
                prompt_length = o.prompt_length
                token_length = o.token_length
                # assert "<|im_end|>" in text, text
                combined_texts.append(text)
                combined_step_maps.append(step_map)
                combined_prompt_ids.append(prompt_ids)
                combined_token_ids.append(token_ids)
                generation_times.append(generation_time)
                prompt_lengths.append(prompt_length)
                token_lengths.append(token_length)
                extracted_answers.append(extract_final_boxed_answer(text))
                response_lengths.append(len(token_ids))
                if len(token_ids) >= config.rollout.max_token:
                    truncs.append(1)
                else:
                    truncs.append(0)

                if reward == "math":
                    # rewards.append(float(equation(text,gt)))
                    correctness = math_judge([text], [gt])[0]
                    rewards.append(correctness)
                    # 计算 speed_reward
                    speed = getattr(config.rollout, 'speed', False)
                    if correctness == 1.0 and speed:
                        speed_reward = len(step_map)/len(set(step_map))/config.rollout.block_size
                    else:
                        speed_reward = 0.0
                    speed_rewards.append(speed_reward)
                else:
                    # print(text,gt)
                    rewards.append(0.0)
                    speed_rewards.append(0.0)


            data.append({
                "question": question,
                "prompt": prompt,
                "ground_truth_answer": gt,
                "full_output": combined_texts,
                "extracted_output": extracted_answers,
                "response_length": response_lengths,
                "rewards": rewards,
                "step_map": combined_step_maps,
                "trunc": truncs,
                "speed_rewards": speed_rewards,
                "prompt_ids": combined_prompt_ids,
                "token_ids": combined_token_ids,
                "generation_time": generation_times,
                "prompt_length": prompt_lengths,
                "token_length": token_lengths,
            })

        
    accelerator.wait_for_everyone()

    # 直接gather data
    all_data = gather_object(data)

    if accelerator.is_main_process:
        import json
        
        # 计算平均reward
        all_rewards = []
        all_truncs = []
        all_speed_rewards = []
        for d in all_data:
            rewards = d.get("rewards", [])
            if rewards:
                all_rewards.extend(rewards)  # Flatten
            truncs = d.get("trunc", [])
            if truncs:
                all_truncs.extend(truncs)
            speed_rewards = d.get("speed_rewards", [])
            if speed_rewards:
                # 只收集正确答案的speed_reward（>0的）
                for sr, r in zip(speed_rewards, rewards):
                    if r == 1.0 and sr > 0:
                        all_speed_rewards.append(sr)
        
        print(all_rewards)
        avg_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0
        avg_speed_reward = sum(all_speed_rewards) / len(all_speed_rewards) if all_speed_rewards else 0
        print(f"{config.dataset.eval_dataset}, length: {len(all_rewards)}, reward: {avg_reward}, trunc: {sum(all_truncs) / len(all_truncs) if all_truncs else 0}, speed_reward: {avg_speed_reward} (n={len(all_speed_rewards)})")
        
        # 保存到文件（使用ref.py的路径规则）
        output_file_name = project_name + "/temp_data-official/outputs-" + outputs_name + ".json"
        
        os.makedirs(os.path.dirname(output_file_name), exist_ok=True)
        
        # 构建包含total信息的完整数据结构
        output_data = {
            "total": {
                "reward": avg_reward,
                "num_samples": len(all_rewards),
                "trunc_rate": sum(all_truncs) / len(all_truncs) if all_truncs else 0,
                "speed_reward": avg_speed_reward,
                "speed_reward_count": len(all_speed_rewards),
                "dataset": config.dataset.eval_dataset
            },
            "data": all_data
        }
        
        with open(output_file_name, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"{output_file_name}保存成功，一共{len(all_data)}条数据···")
    
    accelerator.wait_for_everyone()

    # 显式销毁进程组
    if dist.is_initialized():
        dist.destroy_process_group()

    os._exit(0)

if __name__ == "__main__":
    main()