import json
import re

# 1. 检查模板定义
print("=" * 60)
print("1. 检查 qwen3_nothink 模板定义")
print("=" * 60)

template_file = '/inspire/hdd/global_user/liuxiaoran-240108120089/zhuying/SDAR/training/llama_factory_sdar/src/llamafactory/data/template.py'
with open(template_file) as f:
    content = f.read()

# 找到qwen3_nothink的定义
pattern = r'register_template\(\s*name="qwen3_nothink".*?\)'
match = re.search(pattern, content, re.DOTALL)
if match:
    template_def = match.group(0)
    print("找到 qwen3_nothink 模板定义:")
    print(template_def[:500])
    
    # 检查是否有thought_words参数
    if 'thought_words' in template_def:
        print("\n✓ 模板中有 thought_words 参数")
        thought_match = re.search(r'thought_words\s*=\s*\((.*?)\)', template_def, re.DOTALL)
        if thought_match:
            print(f"  值: thought_words=({thought_match.group(1)})")
    else:
        print("\n✗ 模板中没有 thought_words 参数 (将使用默认值)")
        # 查找默认值
        default_match = re.search(r'thought_words\s*=\s*thought_words\s+or\s+\((.*?)\)', content)
        if default_match:
            print(f"  默认值: {default_match.group(1)}")
    
    # 检查是否有template_class参数
    if 'template_class' in template_def:
        print("\n✓ 模板中有 template_class 参数")
        class_match = re.search(r'template_class\s*=\s*(\w+)', template_def)
        if class_match:
            print(f"  值: template_class={class_match.group(1)}")
    else:
        print("\n✗ 模板中没有 template_class 参数 (将使用 Template 基类)")
else:
    print("未找到 qwen3_nothink 模板定义!")

# 2. 检查训练输出的tokenizer
print("\n" + "=" * 60)
print("2. 检查训练后的 tokenizer 词表")
print("=" * 60)

added_tokens_file = '/inspire/hdd/global_user/liuxiaoran-240108120089/zhuying/SDAR/training/llama_factory_sdar/sdar_ckpt/sdar_8b_math_r1_cot2/full/sft/added_tokens.json'
with open(added_tokens_file) as f:
    added_tokens = json.load(f)

print(f"Added tokens 数量: {len(added_tokens)}")
print(f"\nAdded tokens 列表:")
for token, idx in sorted(added_tokens.items(), key=lambda x: x[1]):
    print(f"  {token}: {idx}")

if '<think>' in added_tokens:
    print(f"\n!!! 发现: <think> 被添加到词表, token id = {added_tokens['<think>']}")
    print(f"!!! 发现: </think> 被添加到词表, token id = {added_tokens['</think>']}")
else:
    print("\n✓ 词表中没有 <think> token")

# 3. 检查配置文件
print("\n" + "=" * 60)
print("3. 检查训练配置")
print("=" * 60)

config_file = '/inspire/hdd/global_user/liuxiaoran-240108120089/zhuying/SDAR/training/llama_factory_sdar/examples/train_full_sdar/sdar_8b/sdar_8b_math_cot_full.yaml'
with open(config_file) as f:
    config_content = f.read()

print("相关配置:")
for line in config_content.split('\n'):
    if any(key in line.lower() for key in ['template', 'thinking', 'dataset']):
        print(f"  {line}")

# 4. 检查数据集
print("\n" + "=" * 60)
print("4. 检查数据集内容")
print("=" * 60)

data_file = '/inspire/hdd/global_user/liuxiaoran-240108120089/zhuying/data/glm_openr1math-sharegpt.json'
with open(data_file) as f:
    data = json.load(f)

print(f"数据集样本数: {len(data)}")
print(f"\n检查前5个样本的assistant内容是否包含 <think> 标签:")
for i in range(min(5, len(data))):
    sample = data[i]
    if len(sample['messages']) > 1:
        assistant_content = sample['messages'][1]['content']
        has_think = '<think>' in assistant_content or '</think>' in assistant_content
        print(f"  样本 {i}: {'有' if has_think else '无'} <think> 标签, 长度={len(assistant_content)}")
        if has_think:
            print(f"    内容预览: {assistant_content[:100]}")

print("\n" + "=" * 60)
print("总结")
print("=" * 60)
print("问题分析:")
print("1. qwen3_nothink 模板定义中没有 thought_words 参数")
print("2. register_template 函数会使用默认值: ('<think>\\n', '\\n</think>\\n\\n')")
print("3. 这些默认的 thought_words 会被添加到 tokenizer 的词表中")
print("4. 即使 enable_thinking=false, token 已经在词表里，模型可以生成")
print("\n解决方法:")
print("在 qwen3_nothink 模板定义中添加: thought_words=('', '')")
print("这样就不会往词表里添加 <think> 和 </think> token")

