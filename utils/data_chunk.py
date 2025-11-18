from typing import Any, Union, Sequence
from datasets import Dataset as HFDataset


def get_data_chunk(data: Any, num_processes: int, rank: int):
    """
    通用数据分块函数，支持：
    - list, tuple
    - Hugging Face Dataset (datasets.Dataset)
    - 任何支持 len() 和 slicing 的序列
    
    不支持 IterableDataset（无长度），会报错。
    
    Args:
        data: 输入数据
        num_processes: 总进程数
        rank: 当前进程的 rank
        
    Returns:
        分配给当前 rank 的子数据（类型与输入一致）
    """
    # 检查是否为 Hugging Face Dataset
    if isinstance(data, HFDataset):
        total = len(data)
        if total == 0:
            return data.select([])

        base_size = total // num_processes
        remainder = total % num_processes

        if rank < remainder:
            start = rank * (base_size + 1)
            end = start + base_size + 1
        else:
            start = rank * base_size + remainder
            end = start + base_size

        return data.select(range(start, end))

    # 检查是否为普通序列（list, tuple, etc.）
    elif hasattr(data, '__len__') and hasattr(data, '__getitem__'):
        total = len(data)
        if total == 0:
            return data

        base_size = total // num_processes
        remainder = total % num_processes

        if rank < remainder:
            start = rank * (base_size + 1)
            end = start + base_size + 1
        else:
            start = rank * base_size + remainder
            end = start + base_size

        chunk = data[start:end]

        # 尽量保持原始类型（对 list/tuple 有效）
        if isinstance(data, list):
            return list(chunk)
        elif isinstance(data, tuple):
            return tuple(chunk)
        else:
            # 对其他序列类型（如自定义类），直接返回切片
            return chunk

    else:
        raise TypeError(
            f"Unsupported data type: {type(data)}. "
            "Supported types: list, tuple, datasets.Dataset, or any sequence with __len__ and __getitem__."
        )