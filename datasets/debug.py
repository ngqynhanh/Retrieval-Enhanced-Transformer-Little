import os
import json
import pickle
import math
import random
from collections import OrderedDict
from pathlib import Path
import argparse
import yaml # Thêm import yaml

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset as PyTorchDataset, DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm # Sử dụng tqdm cho tiến độ
from torchsummary import summary

# Các thư viện custom cần được định nghĩa hoặc import nếu bạn có chúng.
# Đối với ví dụ này, tôi sẽ định nghĩa các lớp Attention và FeedForward cơ bản.
# Bạn cần đảm bảo các lớp này khớp với định nghĩa của bạn.

# ===============================================
# Các lớp custom (ví dụ) - Bạn cần thay thế bằng định nghĩa chính xác của mình
# ===============================================
model_dir = "/kaggle/working/logs/some_model"
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps: float = 1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        norm = x.norm(2, dim=-1, keepdim=True)
        rms = torch.rsqrt(norm.pow(2) + self.eps)
        return x * rms * self.scale

class SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k, is_causal=False, dropout_rate=0):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        self.q_linear = nn.Linear(d_model, n_heads * d_k)
        self.v_linear = nn.Linear(d_model, n_heads * d_k)
        self.k_linear = nn.Linear(d_model, n_heads * d_k)
        self.dropout = nn.Dropout(dropout_rate)
        self.out = nn.Linear(n_heads * d_k, d_model)
        self.is_causal = is_causal

    def forward(self, q, k=None, v=None, mask=None):
        if k is None: k = q
        if v is None: v = q

        batch_size = q.size(0)

        q = self.q_linear(q).view(batch_size, -1, self.n_heads, self.d_k)
        k = self.k_linear(k).view(batch_size, -1, self.n_heads, self.d_k)
        v = self.v_linear(v).view(batch_size, -1, self.n_heads, self.d_k)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        if self.is_causal:
            seq_len = q.size(1)
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), diagonal=1).bool()
            scores = scores.masked_fill(causal_mask, -1e9)

        scores = self.dropout(torch.softmax(scores, dim=-1))
        output = torch.matmul(scores, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)
        return self.out(output)

class CrossAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k, dropout_rate=0):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        self.q_linear = nn.Linear(d_model, n_heads * d_k)
        self.v_linear = nn.Linear(d_model, n_heads * d_k)
        self.k_linear = nn.Linear(d_model, n_heads * d_k)
        self.dropout = nn.Dropout(dropout_rate)
        self.out = nn.Linear(n_heads * d_k, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        q = self.q_linear(query).view(batch_size, -1, self.n_heads, self.d_k)
        k = self.k_linear(key).view(batch_size, -1, self.n_heads, self.d_k)
        v = self.v_linear(value).view(batch_size, -1, self.n_heads, self.d_k)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        scores = self.dropout(torch.softmax(scores, dim=-1))
        output = torch.matmul(scores, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)
        return self.out(output)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout_rate=0):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU() # Thường dùng GELU trong Transformers

    def forward(self, x):
        x = self.dropout(self.activation(self.linear_1(x)))
        return self.linear_2(x)

class ChunkedCrossAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k, chunk_len, dropout_rate=0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k
        self.chunk_len = chunk_len
        self.q_linear = nn.Linear(d_model, n_heads * d_k)
        self.k_linear = nn.Linear(d_model, n_heads * d_k)
        self.v_linear = nn.Linear(d_model, n_heads * d_k)
        self.dropout = nn.Dropout(dropout_rate)
        self.out = nn.Linear(n_heads * d_k, d_model)

    def forward(self, query, key):
        # query shape: (batch_size, seq_len, d_model)
        # key shape: (batch_size, num_chunks, num_neighbors, neighbor_len, d_model)
        batch_size, seq_len, d_model = query.shape
        _, num_chunks, num_neighbors, neighbor_len, _ = key.shape

        # Mở rộng query để khớp với kích thước chunk của key
        # Ví dụ: query (batch, 256, d_model), key (batch, 16, 8, 32, d_model)
        # Cần biến đổi query thành (batch, num_chunks, chunk_len, d_model)
        # và key thành (batch, num_chunks, num_neighbors * neighbor_len, d_model)
        
        # Reshape query thành (batch_size, num_chunks, chunk_len, d_model)
        q_reshaped = query.view(batch_size, num_chunks, self.chunk_len, d_model)

        # Reshape key thành (batch_size, num_chunks, num_neighbors * neighbor_len, d_model)
        k_reshaped = key.view(batch_size, num_chunks, num_neighbors * neighbor_len, d_model)
        v_reshaped = key.view(batch_size, num_chunks, num_neighbors * neighbor_len, d_model)


        # Áp dụng tuyến tính cho Q, K, V
        q = self.q_linear(q_reshaped).view(batch_size, num_chunks, self.chunk_len, self.n_heads, self.d_k)
        k = self.k_linear(k_reshaped).view(batch_size, num_chunks, num_neighbors * neighbor_len, self.n_heads, self.d_k)
        v = self.v_linear(v_reshaped).view(batch_size, num_chunks, num_neighbors * neighbor_len, self.n_heads, self.d_k)

        # Chuyển đổi và tính toán scores
        q = q.permute(0, 1, 3, 2, 4).contiguous() # (batch, chunks, heads, chunk_len, d_k)
        k = k.permute(0, 1, 3, 2, 4).contiguous() # (batch, chunks, heads, num_neighbors*neighbor_len, d_k)
        v = v.permute(0, 1, 3, 2, 4).contiguous() # (batch, chunks, heads, num_neighbors*neighbor_len, d_k)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = self.dropout(torch.softmax(scores, dim=-1))

        output = torch.matmul(scores, v)
        output = output.permute(0, 1, 3, 2, 4).contiguous().view(batch_size, num_chunks, self.chunk_len, self.n_heads * self.d_k)
        output = self.out(output)

        return output.view(batch_size, seq_len, d_model) # Trở lại hình dạng ban đầu (batch_size, seq_len, d_model)

# ===============================================

import torch.optim as optim # Thêm import này

# ===============================================
# Optimizers (AdamW, RAdam, PlainRAdam) của bạn
# ===============================================

class RAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, degenerated_to_sgd=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        self.degenerated_to_sgd = degenerated_to_sgd
        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            for param in params:
                if 'betas' in param and (param['betas'][0] != betas[0] or param['betas'][1] != betas[1]):
                    param['buffer'] = [[None, None, None] for _ in range(10)]
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, buffer=[[None, None, None] for _ in range(10)])
        super(RAdam, self).__init__(params, defaults)
    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')
                p_data_fp32 = p.data.float()
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                state['step'] += 1
                buffered = group['buffer'][int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma
                    if N_sma >= 5:
                        step_size = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    elif self.degenerated_to_sgd:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    else:
                        step_size = -1
                    buffered[2] = step_size
                if N_sma >= 5:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size * group['lr'], exp_avg, denom)
                    p.data.copy_(p_data_fp32)
                elif step_size > 0:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    p_data_fp32.add_(-step_size * group['lr'], exp_avg)
                    p.data.copy_(p_data_fp32)
        return loss
class PlainRAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, degenerated_to_sgd=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        self.degenerated_to_sgd = degenerated_to_sgd
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(PlainRAdam, self).__init__(params, defaults)
    def __setstate__(self, state):
        super(PlainRAdam, self).__setstate__(state)
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')
                p_data_fp32 = p.data.float()
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                state['step'] += 1
                beta2_t = beta2 ** state['step']
                N_sma_max = 2 / (1 - beta2) - 1
                N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                if N_sma >= 5:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    step_size = group['lr'] * math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size, exp_avg, denom)
                    p.data.copy_(p_data_fp32)
                elif self.degenerated_to_sgd:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    step_size = group['lr'] / (1 - beta1 ** state['step'])
                    p_data_fp32.add_(-step_size, exp_avg)
                    p.data.copy_(p_data_fp32)
        return loss
class AdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, warmup = 0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, warmup = warmup)
        super(AdamW, self).__init__(params, defaults)
    def __setstate__(self, state):
        super(AdamW, self).__setstate__(state)
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                p_data_fp32 = p.data.float()
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                if group['warmup'] > state['step']:
                    scheduled_lr = 1e-8 + state['step'] * group['lr'] / group['warmup']
                else:
                    scheduled_lr = group['lr']
                step_size = scheduled_lr * math.sqrt(bias_correction2) / bias_correction1
                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * scheduled_lr, p_data_fp32)
                p_data_fp32.addcdiv_(-step_size, exp_avg, denom)
                p.data.copy_(p_data_fp32)
        return loss

# Thêm Noam và AdamWarmupCosineDecay nếu bạn có định nghĩa của chúng
class Noam(Optimizer): # Định nghĩa tạm thời
    def __init__(self, params, lr, betas, eps, d_model, warmup):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, d_model=d_model, warmup=warmup))
        # Logic của Noam optimizer

    def step(self, closure=None):
        # Implement Noam step
        pass # Placeholder

class AdamWarmupCosineDecay(Optimizer): # Định nghĩa tạm thời
    def __init__(self, params):
        super().__init__(params)
        # Logic của AdamWarmupCosineDecay

    def step(self, closure=None):
        # Implement AdamWarmupCosineDecay step
        pass # Placeholder

# ===============================================


from transformers import GPT2Model, GPT2Config

# ------------------- Models -------------------
class NearestNeighborEncoder(nn.Module):
    def __init__(self, chunk_len: int, n_layers: int, ca_layers: set[int],d_model: int, n_heads: int, d_k: int, d_ff: int, retro_flag):
        super().__init__()
        self.ca_layers = ca_layers
        self.chunk_len = chunk_len
        self.retro_flag = retro_flag
        # Số lượng lớp cross-attention có thể ít hơn tổng số lớp
        self.ca = nn.ModuleList([CrossAttention(d_model, n_heads, d_k, 0) for _ in range(len(ca_layers))])
        self.attn = nn.ModuleList([SelfAttention(d_model, n_heads, d_k, is_causal=False, dropout_rate=0) for _ in range(n_layers)])
        self.ffw = nn.ModuleList([FeedForward(d_model, d_ff) for _ in range(n_layers)])
        self.norm_h = RMSNorm(d_model)

    def forward(self, e: torch.Tensor, h: torch.Tensor):
        batch_size, chunks, neighbors, neighbor_len, d_model = e.shape
        # Chia h thành các chunk để khớp với e
        h_split = h[:, :self.chunk_len * chunks, :].reshape(batch_size, chunks, self.chunk_len, d_model)
        h_split = self.norm_h(h_split)
        
        p_ca = 0
        for p in range(len(self.attn)):
            # Áp dụng self-attention cho từng neighbor (kết hợp các kích thước batch, chunks, neighbors)
            e_flat = e.view(-1, neighbor_len, d_model) # (batch*chunks*neighbors, neighbor_len, d_model)
            e_flat = self.attn[p](e_flat)
            e = e_flat.view(e.shape) # Trở lại hình dạng ban đầu
            
            if self.retro_flag:
                if p in self.ca_layers:
                    h_expanded = h_split.unsqueeze(2).expand(-1, -1, neighbors, -1, -1) # (batch, chunks, neighbors, chunk_len, d_model)
                    e = self.ca[p_ca](query=e.view(-1, neighbor_len, d_model), # Query: Neighbor features
                                      key=h_expanded.view(-1, self.chunk_len, d_model), # Key: Chunk features
                                      value=h_expanded.view(-1, self.chunk_len, d_model)) # Value: Chunk features
                    e = e.view(batch_size, chunks, neighbors, neighbor_len, d_model) # Reshape lại
                    p_ca += 1
            e = self.ffw[p](e.view(-1, neighbor_len, d_model)).view(e.shape) # Áp dụng FFW
        return e

# Định nghĩa một lớp RetroModel đơn giản để làm placeholder cho type hint
class RetroModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Định nghĩa tối thiểu cho RetroModel
        self.dummy_param = nn.Parameter(torch.empty(0)) # Tham số dummy để tránh lỗi khi không có tham số

    def forward(self, src, neighbors, validate=False):
        raise NotImplementedError("RetroModel needs to be implemented or replaced with RetroFittedGPT2")


class RetroFittedGPT2(nn.Module):
    def __init__(self, ca_layers: Set[int], chunk_len: int,d_k: int, d_ff: int, encoder: NearestNeighborEncoder, retro_flag, gpt2_config = None, gpt2_finetuned_path=None, noisy_embed_sequence=False, noisy_embed_neighbors=False, noise_alpha=0, noise_coeff=None,is_aihwkit=0):
        super().__init__()
        if gpt2_finetuned_path:
            self.transformer = GPT2Model.from_pretrained(gpt2_finetuned_path) 
        else:
            self.transformer = GPT2Model.from_pretrained('gpt2')
        if gpt2_config == None:
            raise Error("Please specify the GPT2 configurations file.")
        else:
            self.config = gpt2_config
        self.transformer.init_weights()
        self.noisy_embed_sequence = noisy_embed_sequence
        self.noisy_embed_neighbors = noisy_embed_neighbors
        self.noise_alpha = noise_alpha
        self.noise_coeff = noise_coeff
        if noise_coeff:
            self.gauss = True
        else:
            self.gauss = False
        self.ca_layers = ca_layers
        self.encoder = encoder
        self.retro_flag = retro_flag
        self.chunk_len = chunk_len
        self.gpt2_finetuned_path = gpt2_finetuned_path
        self.cca = nn.ModuleList(
            [ChunkedCrossAttention(gpt2_config.n_embd, gpt2_config.n_head, d_k, chunk_len, 0.0) for _ in range(len(ca_layers))])
        self.ffw = nn.ModuleList([FeedForward(gpt2_config.n_embd, d_ff) for _ in range(len(self.transformer.h))])
        self.read = nn.Linear(gpt2_config.n_embd, gpt2_config.vocab_size)
        self.is_aihwkit = is_aihwkit
        if self.is_aihwkit:
            self.rpu_config = InferenceRPUConfig()
            self.rpu_config.forward.out_res = -1.0  # Turn off (output) ADC discretization.
            self.rpu_config.forward.w_noise_type = WeightNoiseType.ADDITIVE_CONSTANT
            self.rpu_config.forward.w_noise = 0.02  # Short-term w-noise.

            self.rpu_config.clip.type = WeightClipType.FIXED_VALUE
            self.rpu_config.clip.fixed_value = 1.0
            self.rpu_config.modifier.pdrop = 0.03  # Drop connect.
            self.rpu_config.modifier.type = WeightModifierType.ADD_NORMAL  # Fwd/bwd weight noise.
            self.rpu_config.modifier.std_dev = 0.1
            self.rpu_config.modifier.rel_to_actual_wmax = True
            #retro specifc setting to avoid split into multi tile
            self.rpu_config.mapping.max_output_size = 768
            self.rpu_config.mapping.max_input_size = 768

            # Inference noise model.
            self.rpu_config.noise_model = PCMLikeNoiseModel(g_max=25.0)

            # drift compensation
            self.rpu_config.drift_compensation = GlobalDriftCompensation()

    def noised_embed_aihwkit(self, x, validate=False):
        torch.autograd.set_detect_anomaly(True)
        h = self.transformer.wte(x)
        noisy_h = torch.zeros([0, h.shape[-1]], dtype=h.dtype, device=h.device)
        for index in np.ndindex(tuple(h.shape[:-2])):
            retr_weights = torch.transpose(h[index].view(-1,h.shape[-1]),0,1)
            retrive_model = nn.Linear(retr_weights.shape[1],retr_weights.shape[0])
            retrive_model.weight.data = retr_weights.to("cpu")
            retrive_model.to(h.device)
            inputs_vecs = torch.eye(retr_weights.shape[1]).to(h.device)
            with torch.no_grad():
                retrive_model.bias.fill_(0.)
                model = convert_to_analog(retrive_model, self.rpu_config)
                model.eval()
                model.drift_analog_weights(0)
                dist = model(inputs_vecs)
                noisy_h = torch.cat((noisy_h,dist),0)
        return noisy_h.view(h.shape)
        
            
    def noised_embed_gauss(self, x, coeff, validate=False):
        h = self.transformer.wte(x)
        if not validate:
            mu = torch.mean(torch.abs(h).to(h.device),dtype=torch.float64).to(h.device)
            sigma = coeff*mu
            gaussian_noise = torch.empty(h.shape, device=h.device).normal_(mean=0,std=(sigma.to(h.dtype)))
            return h + gaussian_noise
        else:
            return h
    def noised_embed(self, x, noise_alpha, validate=False): #alpha is a hyperparameter that can / should be tuned
        h = self.transformer.wte(x)
        if not validate:
            dims = torch.tensor(h.size(-2) * h.size(-1)) #L x d
            mag_norm = noise_alpha/torch.sqrt(dims)
            return h + torch.zeros_like(h).uniform_(-mag_norm, mag_norm)
        else:
            return h
    def forward(self, x: torch.Tensor, ret: torch.Tensor, past=None, validate=False, x_attention_mask = None, ret_attention_mask = None,is_aihwkit=0):
        batch_size = x.shape[0]
        if self.noisy_embed_sequence:
            if self.gauss:
                h = self.noised_embed_gauss(x, coeff=self.noise_coeff, validate=validate) # Đã thêm coeff
            else:
                h = self.noised_embed(x, noise_alpha=self.noise_alpha, validate=validate)
        else:
            h = self.transformer.wte(x)

        if self.retro_flag:
            if self.noisy_embed_neighbors:
                if self.is_aihwkit:
                    ret_emb = self.noised_embed_aihwkit(ret, validate=validate)
                else:
                    if self.gauss:
                        ret_emb = self.noised_embed_gauss(ret, coeff=self.noise_coeff, validate=validate)
                    else:
                        ret_emb = self.noised_embed(ret, noise_alpha=self.noise_alpha, validate=validate)
            else:
                ret_emb = self.transformer.wte(ret)
        else:
            ret_emb = None

        p_ca = 0
        if past is None:
            past_length = 0
            past = [None] * len(self.transformer.h)
        self.config.use_cache = True 

        presents = [] 

        for p, (block, layer_past) in enumerate(zip(self.transformer.h, past)):
            outputs = block(
                h,
                layer_past=layer_past,
                attention_mask=x_attention_mask,
                use_cache=self.config.use_cache,
                output_attentions=self.config.output_attentions
            )
            
            # Sửa đổi logic giải nén tại đây
            if self.config.use_cache:
                h, present = outputs[0], outputs[1] # Lấy hidden states và present tuple
                presents.append(present) # Lưu trữ present cho vòng lặp tiếp theo nếu cần
            else:
                h = outputs[0] 
                present = None 
            
            if self.retro_flag:
                if self.ca_layers and p == min(self.ca_layers):
                    e_encoded = self.encoder(ret_emb, h)
                    e = e_encoded 
                if p in self.ca_layers:
                    h = self.cca[p_ca](h, e) 
                    p_ca += 1
            h = self.ffw[p](h)
        return self.read(h)
    @staticmethod
    def from_pretrained(pretrained_model_name_or_path, is_aihwkit=0):
        config_file_path = "/".join(pretrained_model_name_or_path.split("/")[:-1])
        with open(config_file_path+'/run.yaml', 'r') as file:
            config_dict = yaml.safe_load(file)
        retrieval_on = True if (config_dict["config_json"]["retro"] == "On") else False
        if "noisy_embed" in config_dict["config_json"]:#this was the old way we added noise, no longer valid as it does not specify where the noise is added
            raise Exception("This way of adding noise is no longer valid.")
        noisy_embed_sequence = True if ("noisy_embed_sequence" in config_dict["config_json"] and config_dict["config_json"]["noisy_embed_sequence"]== "True") else False
        noisy_embed_neighbors = True if ("noisy_embed_neighbors" in config_dict["config_json"] and config_dict["config_json"]["noisy_embed_neighbors"]== "True") else False
        noise_alpha = int(config_dict["config_json"]["noise_alpha"]) if ("noise_alpha" in config_dict["config_json"]) else None
        noise_coeff = float(config_dict["config_json"]["noise_coeff"]) if ("noise_coeff" in config_dict["config_json"]) else None
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        gpt2_config = GPT2Config()
        nearest_neighbor_encoder = NearestNeighborEncoder(config_dict["config_json"]["chunk_len"], config_dict["config_json"]["n_encoder_layers"], config_dict["config_json"]["encoder_ca_layers"], gpt2_config.n_embd, config_dict["config_json"]["n_heads"], config_dict["config_json"]["d_k"], config_dict["config_json"]["d_ff"], retro_flag=retrieval_on)
        model = RetroFittedGPT2(ca_layers=config_dict["config_json"]["decoder_ca_layers"], chunk_len=config_dict["config_json"]["chunk_len"],d_k=config_dict["config_json"]["d_k"], d_ff=config_dict["config_json"]["d_ff"], encoder=nearest_neighbor_encoder,retro_flag=retrieval_on, gpt2_config=gpt2_config, noisy_embed_sequence=noisy_embed_sequence,noisy_embed_neighbors=noisy_embed_neighbors, noise_alpha=noise_alpha, noise_coeff=noise_coeff,is_aihwkit=is_aihwkit)
        f = open(pretrained_model_name_or_path, 'rb')
        state_dict = torch.load(f, map_location=torch.device(device))["model_state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k[:7] == "module.":
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=False)# load params
        return model
class Trainer:
    def __init__(self, device: torch.device, model: retro.RetroModel, dataloader: DataLoader, val_dataloader, optimizer: torch.optim.Optimizer, model_dir):
        self.optimizer = optimizer
        self.device = device
        self.dataloader = dataloader
        self.model = model
        self.loss_func = nn.CrossEntropyLoss()
        self.model_dir = model_dir
        self.val_dataloader = val_dataloader
        os.makedirs(model_dir, exist_ok=True)
        with open(model_dir+'/run.yaml', 'r') as file:
            config_dict = yaml.safe_load(file)
        self.minifit = True if ("minifit" in config_dict["config_json"] and config_dict["config_json"]["minifit"] == "True") else False
        self.distance_metric = config_dict["config_json"]["quantizer"]
        self.n_neighbors = config_dict["config_json"]["n_neighbors"]
        if "noisy_distance_metric" in config_dict["config_json"]:
            self.noisy_distance_metric = True if (config_dict["config_json"]["noisy_distance_metric"]=="True") else False
        else:
            self.noisy_distance_metric = False
    def __call__(self, epoch, tb_writer):
        train_loss_sum = 0
        n = 0
        if self.minifit: #for minifit experiments only save one ckp per epoch
            checkpoints_modulo = len(self.dataloader)+1
        else:
            checkpoints_modulo = len(self.dataloader)//5 #save five checkpoints per epoch
        for i, data in monit.enum('Train', self.dataloader):
            if len(data) == 4:
                src, tgt, neighbors, _ = data  #the fourth argument would be the distance matrix D
            else:
                src, tgt, neighbors = data
            src, tgt, neighbors = src.to(self.device), tgt.to(self.device), neighbors.to(self.device)
            neighbors = neighbors[:, :, :self.n_neighbors]
            res = self.model(src, neighbors)
            curr_train_loss = self.loss_func(res.view(-1, res.shape[-1]), tgt.view(-1))
            if i == 0 and epoch==0:
                #Track performance soon after loading the checkpoint
                temp_val_loss, temp_val_ppl = validate(self.val_dataloader, self.device, self.model, self.loss_func, i, self.model_dir, epoch, self.n_neighbors)
                tb_writer.add_scalar("Loss/train", curr_train_loss, 0)
                tb_writer.add_scalar("ppl/train", math.exp(curr_train_loss), 0)
                tb_writer.add_scalar("Loss/val", temp_val_loss, 0)
                tb_writer.add_scalar("ppl/val", temp_val_ppl, 0)

            self.optimizer.zero_grad()
            curr_train_loss.backward()
            self.optimizer.step()
            #remove duplicate tensorboard saving
            #tracker.save({'loss.train': curr_train_loss.item()})
            #tracker.add_global_step(len(src))
            train_loss_sum += curr_train_loss.item()
            n += 1
            if i%checkpoints_modulo == 0 and i != 0:
                temp_val_loss, temp_val_ppl = validate(self.val_dataloader, self.device, self.model, self.loss_func, i, self.model_dir, epoch, self.n_neighbors)
                if os.path.exists(self.model_dir+'/best-checkpoint-model.pt'): #only save the best checkpoint
                    fb = open(model_dir+'/best-checkpoint-model.pt', 'rb')
                    best_curr_model = torch.load(fb)
                    best_curr_val_loss = best_curr_model["val_loss"]
                    if temp_val_loss < best_curr_val_loss:
                        torch.save({'epoch': epoch,'i': i,'model_state_dict': self.model.state_dict(),'optimizer_state_dict': self.optimizer.state_dict(),'train_loss': train_loss_sum/i,'val_loss': temp_val_loss,}, self.model_dir+'/best-checkpoint-model.pt')
                else:
                    torch.save({'epoch': epoch,'i': i,'model_state_dict': self.model.state_dict(),'optimizer_state_dict': self.optimizer.state_dict(),'train_loss': train_loss_sum/i,'val_loss': temp_val_loss,}, self.model_dir+'/best-checkpoint-model.pt')
                tb_writer.add_scalar("Loss/train", train_loss_sum/i, (epoch)*len(self.dataloader)+i)
                tb_writer.add_scalar("ppl/train", math.exp(train_loss_sum/i), (epoch)*len(self.dataloader)+i)
                tb_writer.add_scalar("Loss/val", temp_val_loss, (epoch)*len(self.dataloader)+i)
                tb_writer.add_scalar("ppl/val", temp_val_ppl, (epoch)*len(self.dataloader)+i)
        train_loss = train_loss_sum/n
        with torch.no_grad():
            train_perplexity  = torch.exp(torch.tensor(train_loss))
        tb_writer.add_scalar("Loss/train", train_loss_sum/i, (epoch)*len(self.dataloader)+n)
        tb_writer.add_scalar("ppl/train", math.exp(train_loss_sum/i), (epoch)*len(self.dataloader)+n)
        temp_val_loss, temp_val_ppl = validate(self.val_dataloader, self.device, self.model, self.loss_func, i, self.model_dir, epoch, self.n_neighbors)
        tb_writer.add_scalar("Loss/val", temp_val_loss, (epoch)*len(self.dataloader)+n)
        tb_writer.add_scalar("ppl/val", temp_val_ppl, (epoch)*len(self.dataloader)+n)
        if os.path.exists(self.model_dir+'/best-checkpoint-model.pt'):
            fb = open(model_dir+'/best-checkpoint-model.pt', 'rb')
            best_curr_model = torch.load(fb)
            best_curr_val_loss = best_curr_model["val_loss"]
            if temp_val_loss < best_curr_val_loss:
                #save checkpoint
                torch.save({'epoch': epoch,'i': i,'model_state_dict': self.model.state_dict(),'optimizer_state_dict': self.optimizer.state_dict(),'train_loss': train_loss_sum/i,'val_loss': temp_val_loss,}, self.model_dir+'/best-checkpoint-model.pt')
        else:
            torch.save({'epoch': epoch,'i': i,'model_state_dict': self.model.state_dict(),'optimizer_state_dict': self.optimizer.state_dict(),'train_loss': train_loss_sum/i,'val_loss': temp_val_loss,}, self.model_dir+'/best-checkpoint-model.pt')
    def validate(val_dl, device, model, loss_func, training_step, model_dir, epoch, n_neighbors):
        with torch.no_grad():
            val_loss_sum = 0
            m = 0
            for i, data in monit.enum('Val', val_dl):
                if len(data) == 4:
                    src, tgt, neighbors, _ = data #the fourth argument would be the distance matrix D
                else:
                    src, tgt, neighbors = data
                src, tgt, neighbors = src.to(device), tgt.to(device), neighbors.to(device)
                neighbors = neighbors[:, :, :n_neighbors]
                res = model(src, neighbors, validate=True)
                curr_val_loss = loss_func(res.view(-1, res.shape[-1]), tgt.view(-1))
                val_loss_sum += curr_val_loss.item()
                m +=1
            val_loss = val_loss_sum/m
            val_perplexity  = torch.exp(torch.tensor(val_loss))
        return val_loss, val_perplexity.item()
    def load_optimizer(model_parameters, config_json):
        if config_json["optimizer"] == "Noam":
            optimizer = Noam(model_parameters, lr=config_json["optimizer_lr"], betas=(0.9,0.95), eps=1e-8, d_model=config_json["d_model"], warmup=config_json["optimizer_warmup"])
        elif config_json["optimizer"] == "RAdam":
            optimizer = RAdam(model_parameters)
        elif config_json["optimizer"] == "AdamWarmupCosineDecay":
            optimizer = AdamWarmupCosineDecay(model_parameters)
        else:
            optimizer = torch.optim.AdamW(model_parameters)
        return optimizer 
    def train(random_seed, train_dataset_filepath, val_dataset_filepath, device, config_json):
        model_configurations = {"random_seed": random_seed, "train_dataset_filepath": train_dataset_filepath, "val_dataset_filepath": val_dataset_filepath, "device":device, "config_json":config_json}
        lab.configure({})
        print(args.val_dataset_filepath)
        if "minifit" in config_json and config_json["minifit"] == "True" and train_dataset_filepath=="":
            #infer dataset name from validate filepath
            experiment_name = 'minifit_'+config_json['minifit']+'_retro_'+config_json['retro']+'_data_'+args.val_dataset_filepath.split('/')[4]
            val_dataset = Dataset(val_dataset_filepath)
            train_indices = random.sample(range(0, len(val_dataset)-1), int(len(val_dataset)/10))
            train_indices.sort(reverse=True)
            train_dl = DataLoader(val_dataset,batch_size=2,sampler=RandomSampler(train_indices, replacement=False))
            for i in train_indices:
                del val_dataset.samples[i]
            val_dl = DataLoader(val_dataset,batch_size=config_json["dl_batch_size"],sampler=RandomSampler(val_dataset, replacement=False))
        else:
            val_name = os.path.basename(args.val_dataset_filepath)  # chỉ lấy tên file
            experiment_name = f"minifit_off_retro_{config_json['retro']}_data_{val_name}"   
            train_dataset, val_dataset = Dataset(train_dataset_filepath), Dataset(val_dataset_filepath)
            train_dl, val_dl = DataLoader(train_dataset,batch_size=config_json["dl_batch_size"],sampler=RandomSampler(train_dataset, replacement=False)), DataLoader(val_dataset,batch_size=config_json["dl_batch_size"],sampler=RandomSampler(val_dataset, replacement=False))
    
        experiment.create(name="retro_test_v2", comment="new")
        device = torch.device(device)
        model_dir = str(lab.get_experiments_path())+'/'+experiment_name+'/'+str(experiment.get_uuid())
        chunk_len, d_model, d_ff, n_heads, d_k, n_encoder_layers, encoder_ca_layers, n_decoder_layers, decoder_ca_layers = config_json["chunk_len"], config_json["d_model"], config_json["d_ff"], config_json["n_heads"], config_json["d_k"], config_json["n_encoder_layers"], set(config_json["encoder_ca_layers"]), config_json["n_decoder_layers"], set(config_json["decoder_ca_layers"])
        retro_flag = True if (config_json["retro"] == "On") else False
        # where to put the noise and what kind
        if "noisy_embed_neighbors" in config_json.keys() and "noisy_embed_sequence" in config_json.keys() and "noise_alpha" in config_json.keys():
            noisy_embed_neighbors = True if (config_json["noisy_embed_neighbors"] == "True") else False
            noisy_embed_sequence = True if (config_json["noisy_embed_sequence"] == "True") else False
            noise_alpha, noise_coeff = config_json["noise_alpha"], None
        elif "noisy_embed_neighbors" in config_json.keys():
            noisy_embed_sequence = False
            noisy_embed_neighbors = True if (config_json["noisy_embed_neighbors"] == "True") else False
            if "noise_alpha" in config_json.keys():
                noise_alpha, noise_coeff = config_json["noise_alpha"], None
            if "noise_coeff" in config_json.keys():
                noise_alpha, noise_coeff = None, config_json["noise_coeff"]
        elif "noisy_embed_sequence" in config_json.keys() and "noise_alpha" in config_json.keys():
            noisy_embed_neighbors = False
            noisy_embed_sequence = True if (config_json["noisy_embed_sequence"] == "True") else False
            noise_alpha, noise_coeff = config_json["noise_alpha"], None
        else:
            noisy_embed_neighbors, noisy_embed_sequence = False, False
            noise_alpha, noise_coeff = None, None
        # load from checkpoint or no
        if config_json["gpt2"] != "True":
            raise NotImplementedError("Only retro-fitted GPT-2 is implemented.")
        gpt2_config = GPT2Config()
        if config_json["load_from_checkpoint"] != "":
            model = RetroFittedGPT2.from_pretrained(config_json["load_from_checkpoint"])
            if retro_flag != model.retro_flag and "true_retrofit" in config_json and config_json["true_retrofit"] == "True": #we are retro-fitting a retro-off checkpoint into a retro-on model
                    nearest_neighbor_encoder = NearestNeighborEncoder(chunk_len, n_encoder_layers, encoder_ca_layers, gpt2_config.n_embd, n_heads, d_k, d_ff, retro_flag=retro_flag)
                    model = RetroFittedGPT2(ca_layers=decoder_ca_layers, chunk_len=chunk_len,d_k=d_k, d_ff=d_ff, encoder=nearest_neighbor_encoder,retro_flag=retro_flag, gpt2_config=gpt2_config,noisy_embed_sequence=noisy_embed_sequence, noisy_embed_neighbors=noisy_embed_neighbors, noise_alpha=noise_alpha, noise_coeff=noise_coeff)
                    f = open(config_json["load_from_checkpoint"], 'rb')
                    state_dict = torch.load(f, map_location=device)["model_state_dict"]
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        if k[:7] == "module.":
                            name = k[7:] # remove `module.`
                            new_state_dict[name] = v
                        else:
                            new_state_dict[k] = v
                    model.load_state_dict(new_state_dict, strict=False)# load params
                    optimizer = load_optimizer(model.parameters(), config_json)
            else:
                optimizer = load_optimizer(model.parameters(), config_json)
                with open(config_json["load_from_checkpoint"], 'rb') as f:
                    state_dict = torch.load(f)
                    optimizer.load_state_dict(state_dict["optimizer_state_dict"])
        else:
            nearest_neighbor_encoder = NearestNeighborEncoder(chunk_len, n_encoder_layers, encoder_ca_layers, gpt2_config.n_embd, n_heads, d_k, d_ff, retro_flag=retro_flag)
            model = RetroFittedGPT2(ca_layers=decoder_ca_layers, chunk_len=chunk_len,d_k=d_k, d_ff=d_ff, encoder=nearest_neighbor_encoder,retro_flag=retro_flag, gpt2_config=gpt2_config,noisy_embed_sequence=noisy_embed_sequence, noisy_embed_neighbors=noisy_embed_neighbors, noise_alpha=noise_alpha, noise_coeff=noise_coeff)
            optimizer = load_optimizer(model.parameters(), config_json)
            
        # freeze or unfreeze layers
        if "unfreeze_gpt2" not in config_json: #keep it frozen.
            for param in model.transformer.parameters():
                param.requires_grad = False 
        else:
            for param in model.transformer.parameters():
                param.requires_grad = True 
        if retro_flag and "true_retrofit" in config_json and config_json["true_retrofit"] == "True":
                for param in model.ffw.parameters():
                    param.requires_grad = False
                for param in model.read.parameters():
                    param.requires_grad = False
    
        #over ride param updates for minfit
        if config_json["minifit"] == "True":
            for param in model.ffw.parameters():
                param.requires_grad = True
            for param in model.read.parameters():
                param.requires_grad = True
            if config_json["retro"] == "On":
                for param in model.encoder.ca.parameters():
                    param.requires_grad = False
                for param in model.cca.parameters():
                    param.requires_grad = False
    
        model = nn.DataParallel(model)
        model = model.to(device)
        #experiment.add_models(model=model)

    with experiment.start():
        os.makedirs(model_dir, exist_ok=True)
        with open(model_dir + '/model_summary.txt', 'w', encoding='utf-8') as f:
            f.write(str(summary(self.model)))
        with open(model_dir + '/run.yaml', 'a') as fp:
            yaml.dump(model_configurations, fp)
        if "minifit" in config_json and config_json["minifit"]=="True" and train_dataset_filepath=="":
            #save what samples we validated on, what samples we trained on
            with open(model_dir+"/val_train_split.csv", "w") as f:
                f.write("val_len, train_ids")
                f.write("\n")
                f.write(str(len(val_dataset)) +","+str(train_indices))
        trainer = Trainer(device, model, train_dl, val_dl, optimizer, model_dir, )
        loss_func = nn.CrossEntropyLoss()
        tb_writer = SummaryWriter(log_dir=model_dir+'/runs/')
        for epoch in monit.loop(config_json["epochs"]):
            trainer(epoch, tb_writer)
            tracker.new_line()
            torch.cuda.empty_cache()
            tb_writer.flush()
        torch.save({'epoch': epoch,'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict(),}, model_dir+'/last-checkpoint-model.pt')
    tb_writer.close()
    f.close()
    fp.close()
    
  
class Args:
    def __init__(self, random_seed, train_dataset_filepath, val_dataset_filepath, config_filepath):
        self.random_seed = random_seed
        self.train_dataset_filepath = train_dataset_filepath
        self.val_dataset_filepath = val_dataset_filepath
        self.config_filepath = config_filepath

train_data_path = "/kaggle/input/retroli/retroli_train.json"
val_data_path = "/kaggle/input/retroli/retroli_val-custom_phukhoa.json"
config_path = "/kaggle/input/retroli/retro_small_model_wikitext103-gpt2-coeff0196-neigh.json"

# Tạo một đối tượng args giả
args = Args(random_seed=42,
            train_dataset_filepath=train_data_path,
            val_dataset_filepath=val_data_path,
            config_filepath=config_path)

# Tiếp tục với phần còn lại của main
config_json = json.load(open(args.config_filepath))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
seed = int(args.random_seed)
torch.random.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
os.environ["PYTHONHASHSEED"] = str(seed)

Trainer.train(random_seed=seed, train_dataset_filepath=args.train_dataset_filepath, val_dataset_filepath=args.val_dataset_filepath, device=device, config_json=config_json)