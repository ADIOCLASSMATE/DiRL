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


def sample(model_path, backend_config, gen_config, prompts, k_sample=1, use_tqdm=True):
    expanded_prompts = []
    for prompt in prompts:
        expanded_prompts.extend([prompt] * k_sample)

    with pipeline(model_path, backend_config=backend_config) as pipe:
        outputs = pipe(expanded_prompts, gen_config=gen_config, use_tqdm=use_tqdm)
    
    gc.collect()
    torch.cuda.empty_cache()

    return outputs