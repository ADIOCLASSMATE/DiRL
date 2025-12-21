"""
修改说明：统一使用右padding，返回 valid_indices 标识（prompt=1, response=2, padding=0）
"""
from accelerate.logging import get_logger
from transformers import AutoTokenizer
logger = get_logger(__name__, log_level="INFO")

import torch

class UniversalPrompting():
    def __init__(self, text_tokenizer, max_prompt_len=1024, max_gen_length=8192, ignore_id=-100, block_size=4):
        self.tokenizer = text_tokenizer
        self.ignore_id = ignore_id
        self.max_prompt_len = max_prompt_len
        self.max_gen_length = max_gen_length
        self.block_size = block_size  # 保留参数以保持接口兼容

    def __call__(self, input, step_map_list=None):
        prompt_ids_list, response_ids_list = input
        
        # 拼接ids
        texts_ids = [p + r for p, r in zip(prompt_ids_list, response_ids_list)]
        prompt_lens = [len(p) for p in prompt_ids_list]
        
        # 固定长度：max_prompt_len + max_gen_length，截断超长序列
        max_len = self.max_prompt_len + self.max_gen_length
        batch_size = len(texts_ids)
        
        # 初始化所有tensor
        input_ids = torch.full((batch_size, max_len), self.tokenizer.pad_token_id, dtype=torch.long)
        labels = torch.full((batch_size, max_len), self.ignore_id, dtype=torch.long)
        valid_indices = torch.zeros((batch_size, max_len), dtype=torch.long)
        step_map_tensor = torch.full((batch_size, max_len), -1, dtype=torch.long) if step_map_list else None
        
        # 统一填充
        for i in range(batch_size):
            seq_len = min(len(texts_ids[i]), max_len)
            prompt_len = min(prompt_lens[i], max_len)
            response_len = seq_len - prompt_len
            
            # 断言：检查序列长度是block_size的倍数
            # assert seq_len % self.block_size == 0, \
                # f"序列长度 {seq_len} 不是 block_size {self.block_size} 的倍数"
            
            # input_ids: 填充原始数据
            input_ids[i, :seq_len] = torch.tensor(texts_ids[i][:seq_len], dtype=torch.long)
            
            # valid_indices: prompt=1, response=2, padding=0
            valid_indices[i, :prompt_len] = 1
            valid_indices[i, prompt_len:seq_len] = 2
            
            # labels: 仅response部分
            labels[i, prompt_len:seq_len] = input_ids[i, prompt_len:seq_len]
            
            # step_map: 仅response部分，截断对齐
            if step_map_tensor is not None and response_len > 0:
                step_map_tensor[i, prompt_len:seq_len] = torch.tensor(
                    step_map_list[i][:response_len], dtype=torch.long
                )
        
        return (input_ids, labels, valid_indices, step_map_tensor) if step_map_tensor is not None else (input_ids, labels, valid_indices)


if __name__ == '__main__':
    # 测试基本功能
    prompts = [
        "Hello, how are you?",
        "I am fine, thank you."
    ]
    responses = [
        "I am a student.",
        "Sure! Let me help you with that question."
    ]
    tokenizer = AutoTokenizer.from_pretrained("/inspire/hdd/global_user/liuxiaoran-240108120089/zhuying/DiRL-8B-Instruct")
    block_size = 4
    prompting = UniversalPrompting(tokenizer, max_prompt_len=8, max_gen_length=12, block_size=block_size)
    
    print(f"=== 配置信息 ===")
    print(f"block_size: {block_size} (保留参数但不使用)")
    print(f"pad_token_id: {tokenizer.pad_token_id}")
    
    # 先tokenize成id列表
    prompt_ids_list = [tokenizer(p, add_special_tokens=False)['input_ids'] for p in prompts]
    response_ids_list = [tokenizer(r, add_special_tokens=False)['input_ids'] for r in responses]
    print(f"prompt_ids_list: {prompt_ids_list}")
    print(f"response_ids_list: {response_ids_list}")
    # exit(0)
    # 构造step_map
    step_map_list = [
        list(range(1, len(response_ids_list[0]) + 1)),
        list(range(1, len(response_ids_list[1]) + 1))
    ]
    
    # 调用prompting
    result = prompting((prompt_ids_list, response_ids_list), step_map_list=step_map_list)
    input_ids, labels, valid_indices, step_map_tensor = result
    
    print(f"\n=== 输出tensor shapes ===")
    print(f"input_ids: {input_ids.shape}")
    print(f"labels: {labels.shape}")
    print(f"valid_indices: {valid_indices.shape}")
    print(f"step_map_tensor: {step_map_tensor.shape}")
    print(f"{tokenizer.batch_decode(input_ids)}")
    
    # 详细检查每个样本
    for i in range(len(prompts)):
        print(f"\n=== Sample {i} 详细信息 ===")
        prompt_len = len(prompt_ids_list[i])
        response_len = len(response_ids_list[i])
        original_seq_len = prompt_len + response_len
        
        print(f"原始长度: prompt={prompt_len}, response={response_len}, total={original_seq_len}")
        
        # 统计各部分长度
        prompt_count = (valid_indices[i] == 1).sum().item()
        response_count = (valid_indices[i] == 2).sum().item()
        padding_count = (valid_indices[i] == 0).sum().item()
        
        print(f"valid_indices统计: prompt(1)={prompt_count}, response(2)={response_count}, padding(0)={padding_count}")
        
        # 显示前面部分的token IDs
        display_len = min(original_seq_len + 5, 30)
        print(f"input_ids前{display_len}个: {input_ids[i, :display_len].tolist()}")
        print(f"valid_indices前{display_len}个: {valid_indices[i, :display_len].tolist()}")
        print(f"step_map前{display_len}个: {step_map_tensor[i, :display_len].tolist()}")
        
        # 验证labels只在response部分有值
        label_count = (labels[i] != prompting.ignore_id).sum().item()
        print(f"labels有效位置数: {label_count} (应该等于response长度 {response_len})")
        
        # 验证step_map
        step_count = (step_map_tensor[i] > 0).sum().item()
        print(f"step_map有效位置数: {step_count} (应该等于response长度 {response_len})")
        
        # 显示response部分的step_map内容
        response_mask = valid_indices[i] == 2
        if response_mask.sum() > 0:
            response_steps = step_map_tensor[i][response_mask].tolist()
            print(f"response部分的step_map: {response_steps}")