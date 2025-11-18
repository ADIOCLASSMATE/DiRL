import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import torch
import torch.distributed as dist
from accelerate import Accelerator
from accelerate.utils import gather_object
from transformers import AutoTokenizer, AutoConfig
from datasets import load_dataset
from omegaconf import OmegaConf
import argparse

# 假设这些模块存在

from utils.data_chunk import get_data_chunk
from utils.sample import to_single_token_stop_ids
from utils.dataset import NormalDataset
from utils.math_utils import equation
from train.utils import get_config

from lmdeploy import pipeline, PytorchEngineConfig, GenerationConfig, ChatTemplateConfig
from lmdeploy.pytorch.tools.utils import Timer, visualize_pipe_out
from accelerate import Accelerator, utils
import accelerate
from transformers import AutoTokenizer
import gc
import shutil
import time
import importlib.util
import types
import re
import torch.distributed as dist


import os, shutil, glob

def force_inject(checkpoint_dir):
    """
    从 checkpoint_dir 自动识别 cache_dir，
    并将源码文件覆盖注入到 cache 目录
    """
    checkpoint_dir = os.path.abspath(checkpoint_dir)
    base_name = os.path.basename(checkpoint_dir.rstrip("/"))
    cache_dir = os.path.expanduser(
        os.path.join("~/.cache/huggingface/modules/transformers_modules", base_name)
    )
    os.makedirs(cache_dir, exist_ok=True)

    patterns = ["configuration_*.py", "modeling_*.py", "tokenization_*.py", "processing_*.py", "config.json"]
    copied = []
    for pat in patterns:
        for src in glob.glob(os.path.join(checkpoint_dir, pat)):
            dst = os.path.join(cache_dir, os.path.basename(src))
            shutil.copy2(src, dst)  # 强行覆盖
            copied.append((src, dst))

    print(f"[Inject] Copied {len(copied)} files from {checkpoint_dir} to {cache_dir}")
    for s, d in copied:
        print(f"  {s} -> {d}")
    return cache_dir

def inject_debug_print_all(model_path: str, target_files=None):
    """
    在模型目录下的多个文件顶部插入调试打印语句。
    默认包含：configuration_*.py, tokenization_*.py, modeling_*.py
    """
    if target_files is None:
        target_files = [
            "configuration_sdar.py",
            "tokenization_qwen2.py",
            "modeling_sdar.py"
        ]

    for filename in target_files:
        file_path = os.path.join(model_path, filename)
        if not os.path.exists(file_path):
            print(f"[跳过] {file_path} 不存在")
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            original = f.read()

        injection = f'print(f"--- I am EXECUTING {filename} from location: {{__file__}} ---")\n'

        if injection.strip() not in original:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(injection + original)
            print(f"[完成] 已在 {file_path} 顶部插入调试打印。")
        else:
            print(f"[跳过] {file_path} 已经有调试打印了。")


def sample(model_path, backend_config, gen_config, prompts, k_sample=1, config=None, accelerator=None, use_tqdm=True):
    expanded_prompts = []
    for prompt in prompts:
        expanded_prompts.extend([prompt] * k_sample)

    from pathlib import Path

    # ===== 强制 offline 模式，禁止走 cache =====
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"

    if accelerator.is_main_process:
        force_inject(model_path)

    # 等 rank0 写完，其他 rank 再继续
    if dist.is_initialized():
        dist.barrier()

    retry = 3
    while retry > 0:
        try:  
            if dist.get_rank() == 0:
                _ = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                _ = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

            dist.barrier()

            with pipeline(model_path, backend_config=backend_config) as pipe:
                outputs = pipe(expanded_prompts, gen_config=gen_config, use_tqdm=use_tqdm)   
            # 检查outputs里面的step_map是否为None
            for output in outputs:
                if output.step_map is None:
                    print(f"Step map is None for output: {output}")
                    raise ValueError("Step map is None")
            gc.collect()
            torch.cuda.empty_cache()
            return outputs
        except Exception as e:
            print(f"Retry {3-retry} Error: {e}")
            retry -= 1
            time.sleep(1)
    outputs = []
    gc.collect()
    torch.cuda.empty_cache()
    return outputs