import time
import torch
from torch import Tensor
from typing import Dict, List, Tuple
import numpy as np

from ..helper import MessageType
from ..communicator import BITS_SET
from ..communicator import Communicator as comm
from ..manager import GraphEngine as engine

'''
*************************************************
********** cost model related functions *********
*************************************************
'''

def build_dummy(feat_dim: int, hidden_dim: int, num_data: int, rank: int):
    '''
    build dumpy data for profiling communication cost of each worker (device) pair
    '''
    send_dummpy: Dict[int,List[Tensor]] = {}
    recv_dummpy: Dict[int,List[Tensor]] = {}
    # generate sending dummy data
    send_idx = engine.ctx.send_idx
    for pid, idx in send_idx.items():
        num_nodes = idx[1] - idx[0]
        low = round( num_nodes * min(feat_dim, hidden_dim) * BITS_SET[0] / 8)  # lowest total bytes of sending
        high = round(num_nodes * max(feat_dim, hidden_dim) * BITS_SET[-1] / 8)  # highest total bytes of sending
        tolerance = round(low / 2) # tolerance for better sampling
        data_size = torch.linspace(start=low - tolerance, end=high + tolerance, steps=num_data, dtype=torch.int64)
        data = [torch.zeros(size=(b_size,), dtype=torch.uint8, device=f'cuda') for b_size in data_size] # generated byte stream
        send_dummpy[pid] = data
    # generate receiving dummy data (serve as the buffer for receiving)
    recv_idx = engine.ctx.recv_idx
    for pid, idx in recv_idx.items():
        num_nodes = len(idx)
        low = round( num_nodes * min(feat_dim, hidden_dim) * BITS_SET[0] / 8)
        high = round(num_nodes * max(feat_dim, hidden_dim) * BITS_SET[-1] / 8)
        tolerance = round(low / 2)
        data_size = torch.linspace(start=low - tolerance, end=high + tolerance, steps=num_data, dtype=torch.int64)
        data = [torch.zeros(size=(b_size,), dtype=torch.uint8, device=f'cuda') for b_size in data_size]
        recv_dummpy[pid] = data
    return send_dummpy, recv_dummpy

def generate_sender(send_dumpy: Dict[int,List[Tensor]], warmup:int) -> Tuple[Dict[str, List[Tensor]], Dict[str, List[float]]]:
    '''
    generate the datatset for cost model of each worker pair. dataset: (data_size, time)
    '''
    rank, world_size = comm.get_rank(), comm.get_world_size()
    data_size_buffer: Dict[str, List[Tensor]] = {}
    time_buffer: Dict[str, List[float]] = {}
    # generate sending dataset for device pairs whose src device is rank
    for i in range(world_size):
        if i != rank:
            data_size_buffer[f'{rank}_{i}'] = []
            time_buffer[f'{rank}_{i}'] = []
            dummy_data = send_dumpy[i]
            for n in range(len(dummy_data)):
                avg_time = []
                for epoch in range(1, 3 * warmup):
                    start = time.time()
                    comm.sync_send(dummy_data[n], i, MessageType.DATA)
                    end = time.time()
                    if epoch > warmup:
                        avg_time.append(end - start)
                data_size_buffer[f'{rank}_{i}'].append(dummy_data[n].shape[0] / (1024 ** 2)) # MB
                time_buffer[f'{rank}_{i}'].append(sum(1000 * avg_time) / len(avg_time)) # ms
            data_size_buffer[f'{rank}_{i}'] = torch.tensor(data_size_buffer[f'{rank}_{i}'])
            time_buffer[f'{rank}_{i}'] = torch.tensor(time_buffer[f'{rank}_{i}'])
    # sync
    comm.barrier()
    return data_size_buffer, time_buffer

def generate_receiver(dumpy_data: List[Tensor], sender_rank: int, warmup: int):
    '''
    waiting for receiving dummy data from the sender.
    '''
    for n in range(len(dumpy_data)):
        for _ in range(1, 3 * warmup):
            comm.sync_recv(dumpy_data[n], sender_rank, MessageType.DATA)
    # sync
    comm.barrier()

def generate_cost_model_dataset(feat_dim: int, hidden_dim: int, num_data: int, warmup: int) -> Tuple[Dict[str, List[Tensor]], Dict[str, List[float]]]:
    rank, world_size = comm.get_rank(), comm.get_world_size()
    # get send & recv dummy data
    send_dummpy, recv_dummpy = build_dummy(feat_dim, hidden_dim, num_data, rank)
    # get dataset for each worker pair
    for sender in range(world_size):
        if sender == rank:
            dataset = generate_sender(send_dummpy, warmup)
        else:
            generate_receiver(recv_dummpy[sender], sender, warmup)
    return dataset

def fit_cost_model(dataset: Tuple[Dict[str, List[Tensor]], Dict[str, List[float]]]) -> Dict[str, np.ndarray]:
    '''
    fit the cost model for each worker pair
    '''
    cost_model: Dict[str, np.ndarray] = {}
    data_size_buffer, time_buffer = dataset
    for pair_key in data_size_buffer.keys():
        model = np.polyfit(data_size_buffer[pair_key], time_buffer[pair_key], 1) # alpha-beta model (y = alpha * x + beta)
        cost_model[pair_key] = model
    return cost_model
