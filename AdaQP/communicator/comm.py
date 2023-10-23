import os
import logging
import torch
import torch.distributed as dist
from torch import Tensor
from typing import Dict, List, Any, Tuple
from queue import Queue

from .buffer import CommBuffer, Basic_Buffer_Type
from ..helper import MessageType

logger = logging.getLogger('trainer')

class Communicator(object):
    '''
    the communicator class for distributed training. Communicator is a wrapper of torch.distributed, and managers all the communication buffers and operations.
    '''
    def __init__(self, backend: str ='gloo', init_method: str ='env://'):
        self._init(backend, init_method)
        self.comm_buffer: CommBuffer = None
        # set ctx (do not share across processes)
        Communicator.ctx = self 

    def _init(self, backend: str, init_method: str):
        '''
        initialize the communicator.
        '''
        if backend != 'gloo':
            raise NotImplementedError('only gloo is supported now')
        dist.init_process_group(backend, init_method="env://")
        self._backend = backend
        self._init_method = init_method
        self._local_rank = int(os.environ['LOCAL_RANK'])
        self._device = torch.device(f'cuda:{self.local_rank}')
        torch.cuda.set_device(self.device)

    def __repr__(self):
        return f'<Communicator(rank: {self.get_rank()}, backend: {self.backend}, world_size: {self.get_world_size()}, local_rank: {self.local_rank}, device: {self.device})>'

    '''
    *************************************************
    ***************** getter methods ****************
    *************************************************
    '''
    
    @property
    def local_rank(self):
        return self._local_rank
    
    @property
    def device(self):
        return self._device
    
    @property
    def init_method(self):
        return self._init_method
    
    @property
    def backend(self):
        return self._backend

    @staticmethod
    def get_rank():
        return dist.get_rank()

    @staticmethod
    def get_world_size():
        return dist.get_world_size()

    @staticmethod
    def get_backend():
        return dist.get_backend()

    @staticmethod
    def _destroy():
        dist.destroy_process_group()

    def __del__(self):
        self._destroy()

    @staticmethod
    def barrier():
        dist.barrier()

    '''
    *************************************************
    ************* collective primitives *************
    *************************************************
    '''

    @staticmethod
    def all_reduce_max(tensor: Tensor):
        '''
        all reduce the tensor with max operation.
        '''
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    
    @staticmethod
    def all_reduce_sum(tensor: Tensor):
        '''
        all reduce the tensor with sum operation.
        '''
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    @staticmethod
    def all_gather_any(obj_list: List[Any], obj: Any):
        '''
        all gather any objects across all the workers. (only support by gloo backend)
        '''
        dist.all_gather_object(obj_list, obj)
    
    @staticmethod
    def broadcast_any(obj_list: List[Any], src: int = 0):
        '''
        broadcast objects to all the workers
        '''
        dist.broadcast_object_list(obj_list, src)
        
    
    @staticmethod
    def scatter_any(output_list: List[Any], input_list: List[Any], src: int = 0):
        '''
        scatter objects from input_list to output_list. (only support by gloo backend)
        '''
        dist.scatter_object_list(output_list, input_list, src)
    
    @staticmethod
    def gather_any(obj, obj_list: List[Any], dst: int = 0):
        dist.gather_object(obj, obj_list, dst)

    # p2p primitives
    @staticmethod
    def sync_send(tensor: Tensor, dst: int, tag: MessageType):
        '''
        send tensor to dst synchronously.
        '''
        return dist.send(tensor, dst, tag=tag.value)

    @staticmethod
    def sync_recv(tensor: Tensor, src: int, tag: MessageType):
        '''
        receive tensor from src synchronously.
        '''
        return dist.recv(tensor, src, tag=tag.value)
    
    @staticmethod
    def async_send(tensor: Tensor, dst: int, tag: MessageType):
        '''
        send tensor to dst asynchronously.
        '''
        return dist.isend(tensor, dst, tag=tag.value)
    
    @staticmethod
    def async_recv(tensor: Tensor, src: int, tag: MessageType):
        '''
        receive tensor from src asynchronously.
        '''
        return dist.irecv(tensor, src, tag=tag.value)

    '''
    *************************************************
    *********** messages exchange methods ***********
    *************************************************
    '''

    def fp_msg_exchange(self, recv_buffer_cpu: Basic_Buffer_Type, recv_buffer_gpu: Basic_Buffer_Type, send_buffer_cpu: Basic_Buffer_Type, send_idx: Dict[int, Tuple[int, int]], send_messages: Tensor):
        '''
        all-to-all full-precision message exchange across all the worker
        '''
        rank, world_size = self.get_rank(), self.get_world_size()
        req_send, req_recv = [], Queue()
        # world_size - 1 round communication
        for i in range(1, world_size):
            left = (rank - i + world_size) % world_size
            right = (rank + i) % world_size
            retrieve_idx = send_idx[right]
            # async send
            send_buffer_cpu[right].copy_(send_messages[retrieve_idx[0]:retrieve_idx[1]])
            logger.debug(f'<worker {dist.get_rank()} send {send_buffer_cpu[right].shape} to {right}, recv {recv_buffer_cpu[left].shape} from {left}>')
            r1 = self.async_send(send_buffer_cpu[right], right, MessageType.DATA)
            req_send.append(r1)
            # async recv
            r2 = self.async_recv(recv_buffer_cpu[left], left, MessageType.DATA)
            req_recv.put((r2, left))
        # wait for completion
        while not req_recv.empty():
            r, left = req_recv.get()
            r.wait()
            recv_buffer_gpu[left].copy_(recv_buffer_cpu[left], non_blocking=True)
        for r in req_send:
            r.wait()
    
    def qt_msg_exchange(self, recv_buffer_cpu: Basic_Buffer_Type, recv_buffer_gpu: Basic_Buffer_Type, send_buffer_cpu: Basic_Buffer_Type):
        '''
        all-to-all quantized message exchange across all the worker
        '''
        rank, world_size = self.get_rank(), self.get_world_size()
        req_send, req_recv = [], Queue()
        # world_size - 1 round communication
        for i in range(1, world_size):
            left = (rank - i + world_size) % world_size
            right = (rank + i) % world_size
            # async send
            send_data = send_buffer_cpu[right]
            q_data, q_params = send_data
            r1_0 = self.async_send(q_data, right, MessageType.DATA)
            r1_1 = self.async_send(q_params, right, MessageType.PARAMs)
            req_send.append(r1_0)
            req_send.append(r1_1)
            # async recv
            r2_0 = self.async_recv(recv_buffer_cpu[left][0], left, MessageType.DATA)
            r2_1 = self.async_recv(recv_buffer_cpu[left][1], left, MessageType.PARAMs)
            req_recv.put((r2_0, r2_1, left))
        # wait for completion
        while not req_recv.empty():
            r_0, r_1, left = req_recv.get()
            r_0.wait()
            r_1.wait()
            recv_buffer_gpu[left][0].copy_(recv_buffer_cpu[left][0], non_blocking=True)
            recv_buffer_gpu[left][1].copy_(recv_buffer_cpu[left][1], non_blocking=True)
        for r in req_send:
            r.wait()
    
    '''
    *************************************************
    *********** buffer management methods ***********
    *************************************************
    '''

    def init_buffer(self, *args, **kwargs):
        '''
        wrapper to initialize the communication buffer
        '''
        self.comm_buffer = CommBuffer(*args, **kwargs, device=self.device)
    
    def update_buffer(self, *args, **kwargs):
        '''
        wrapper to update the communication buffer
        '''
        assert self.comm_buffer is not None, 'please initialize the communication buffer first'
        self.comm_buffer._update(*args, **kwargs)
    
    def delete_buffer(self, *args, **kwargs):
        '''
        wrapper to delete the communication buffer
        '''
        assert self.comm_buffer is not None, 'please initialize the communication buffer first'
        self.comm_buffer._delete(*args, **kwargs)
    
    
    
    
