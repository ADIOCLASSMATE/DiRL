import socket
import os
import time
from lmdeploy import pipeline, PytorchEngineConfig, GenerationConfig, ChatTemplateConfig, serve
from lmdeploy.pytorch.tools.utils import Timer, visualize_pipe_out
import requests
from openai import OpenAI
import accelerate
import torch
from typing import List, Dict
from lmdeploy.utils import serialize_state_dict, FlattenedTensorBucket


def get_host_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip


def start_server(local_ip, rank, port, model_path, model_name, backend_config):
    base_url = f'http://{local_ip}:{port}/v1'
    # print(base_url)
    api_key = "sk-"
    origin = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ else str(rank%8)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(rank%8)
    server = serve(
        model_path = model_path,
        model_name = model_name,
        backend = 'pytorch',
        backend_config = backend_config,
        server_name = local_ip,
        server_port = port,
        cache_max_entry_count = 0.8,
        log_level = 'CRITICAL',
    )
    # 配置更大的连接池和超时，避免高并发时连接耗尽或请求卡死
    client = OpenAI(api_key="sk", base_url=f"http://{local_ip}:{port}/v1")
    os.environ['CUDA_VISIBLE_DEVICES'] = origin
    return client

def server_sleep(base_url, level=2):
    api_key = 'sk-xxx'
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    start_time = time.time()
    response = requests.post(f"{base_url}/sleep", headers=headers, params=dict(tags=['weights', 'kv_cache'], level=level))
    assert response.status_code == 200, response.text
    print(f"server_sleep: {time.time() - start_time:.2f}s")

def server_update_weights(base_url, new_model):
    api_key = 'sk-xxx'
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    start_total = time.time()
    
    # Prepare state dict
    state_dict = new_model.state_dict()
    segmented_state_dict: List[Dict[str, torch.Tensor]] = [state_dict]
    
    num_segment = len(segmented_state_dict)
    for seg_idx in range(num_segment):
        seg_start = time.time()
        
        named_tensors = list(segmented_state_dict[seg_idx].items())
        bucket = FlattenedTensorBucket(named_tensors=named_tensors)
        metadata = bucket.get_metadata()
        flattened_tensor_data = dict(flattened_tensor=bucket.get_flattened_tensor(), metadata=metadata)
        serialized_data = serialize_state_dict(flattened_tensor_data)
        
        # Free memory immediately
        del bucket
        del flattened_tensor_data
        torch.cuda.empty_cache()
        
        # Send data
        data = dict(
            serialized_named_tensors=serialized_data, 
            finished=seg_idx == num_segment-1, 
            load_format='flattened_bucket'
        )
        response = requests.post(f"{base_url}/update_weights", headers=headers, json=data)
        
        if response.status_code != 200:
            print(f"Error: {response.status_code}, {response.text}")
            raise RuntimeError(f"Failed to update weights")
        
        del serialized_data
        del data
        print(f"segment {seg_idx+1}/{num_segment}: {time.time() - seg_start:.2f}s")
    
    # Clean up
    del state_dict
    del segmented_state_dict
    del named_tensors
    torch.cuda.empty_cache()
    
    print(f"server_update_weights: {time.time() - start_total:.2f}s")

def server_wakeup(base_url, tags=None):
    if tags is None:
        tags = ['weights', 'kv_cache']
    api_key = 'sk-xxx'
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    start_time = time.time()
    response = requests.post(f"{base_url}/wakeup", headers=headers, params=dict(tags=tags))
    assert response.status_code == 200, response.status_code
    print(f"server_wakeup({tags}): {time.time() - start_time:.2f}s")


