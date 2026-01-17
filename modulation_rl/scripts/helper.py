import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal


__REDUCE__ = lambda b: 'mean' if b else 'none'


def l1(pred, target, reduce=False):
	"""Computes the L1-loss between predictions and targets."""
	return F.l1_loss(pred, target, reduction=__REDUCE__(reduce))


def mse(pred, target, reduce=False):
	"""Computes the MSE loss between predictions and targets."""
	return F.mse_loss(pred, target, reduction=__REDUCE__(reduce))


def _get_out_shape(in_shape, layers):
	"""Utility function. Returns the output shape of a network for a given input shape."""
	x = torch.randn(*in_shape).unsqueeze(0)
	return (nn.Sequential(*layers) if isinstance(layers, list) else layers)(x).squeeze(0).shape


def orthogonal_init(m):
	"""Orthogonal layer initialization."""
	if isinstance(m, nn.Linear):
		nn.init.orthogonal_(m.weight.data)
		if m.bias is not None:
			nn.init.zeros_(m.bias)
	elif isinstance(m, nn.Conv2d):
		gain = nn.init.calculate_gain('relu')
		nn.init.orthogonal_(m.weight.data, gain)
		if m.bias is not None:
			nn.init.zeros_(m.bias)


def ema(m, m_target, tau):
	"""Update slow-moving average of online network (target network) at rate tau."""
	with torch.no_grad():
		for p, p_target in zip(m.parameters(), m_target.parameters()):
			p_target.data.lerp_(p.data, tau)


def set_requires_grad(net, value):
	"""Enable/disable gradients for a given (sub)network."""
	for param in net.parameters():
		param.requires_grad_(value)


class TruncatedNormal(pyd.Normal):
	"""Utility class implementing the truncated normal distribution."""
	def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
		super().__init__(loc, scale, validate_args=False)
		self.low = low
		self.high = high
		self.eps = eps

	def _clamp(self, x):
		clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
		x = x - x.detach() + clamped_x.detach()
		return x

	def sample(self, clip=None, sample_shape=torch.Size()):
		shape = self._extended_shape(sample_shape)
		eps = _standard_normal(shape,
							   dtype=self.loc.dtype,
							   device=self.loc.device)
		eps *= self.scale
		if clip is not None:
			eps = torch.clamp(eps, -clip, clip)
		x = self.loc + eps
		return self._clamp(x)


class NormalizeImg(nn.Module):
	"""Normalizes pixel observations to [0,1) range."""
	def __init__(self):
		super().__init__()

	def forward(self, x):
		return x.div(255.)


class Flatten(nn.Module):
	"""Flattens its input to a (batched) vector."""
	def __init__(self):
		super().__init__()
		
	def forward(self, x):
		return x.view(x.size(0), -1)


def enc(cfg):
	"""Returns a TOLD encoder."""
	if cfg.modality == 'pixels':
		C = int(3*cfg.frame_stack)
		layers = [NormalizeImg(),
				  nn.Conv2d(C, cfg.num_channels, 7, stride=2), nn.ReLU(),
				  nn.Conv2d(cfg.num_channels, cfg.num_channels, 5, stride=2), nn.ReLU(),
				  nn.Conv2d(cfg.num_channels, cfg.num_channels, 3, stride=2), nn.ReLU(),
				  nn.Conv2d(cfg.num_channels, cfg.num_channels, 3, stride=2), nn.ReLU()]
		out_shape = _get_out_shape((C, cfg.img_size, cfg.img_size), layers)
		layers.extend([Flatten(), nn.Linear(np.prod(out_shape), cfg.latent_dim)])
	else:
		layers = [nn.Linear(cfg.obs_shape[0][0], cfg.enc_dim), nn.ELU(),
				  nn.Linear(cfg.enc_dim, cfg.latent_dim)]
	return nn.Sequential(*layers)


def mlp(in_dim, mlp_dim, out_dim, act_fn=nn.ELU()):
	"""Returns an MLP."""
	if isinstance(mlp_dim, int):
		mlp_dim = [mlp_dim, mlp_dim]
	return nn.Sequential(
		nn.Linear(in_dim, mlp_dim[0]), act_fn,
		nn.Linear(mlp_dim[0], mlp_dim[1]), act_fn,
		nn.Linear(mlp_dim[1], out_dim))

def q(cfg, act_fn=nn.ELU()):
	"""Returns a Q-function that uses Layer Normalization."""
	return nn.Sequential(nn.Linear(cfg.latent_dim+cfg.action_dim, cfg.mlp_dim), nn.LayerNorm(cfg.mlp_dim), nn.Tanh(),
						 nn.Linear(cfg.mlp_dim, cfg.mlp_dim), nn.ELU(),
						 nn.Linear(cfg.mlp_dim, 1))


class RandomShiftsAug(nn.Module):
	"""
	Random shift image augmentation.
	Adapted from https://github.com/facebookresearch/drqv2
	"""
	def __init__(self, cfg):
		super().__init__()
		self.pad = int(cfg.img_size/21) if cfg.modality == 'pixels' else None

	def forward(self, x):
		if not self.pad:
			return x
		n, c, h, w = x.size()
		assert h == w
		padding = tuple([self.pad] * 4)
		x = F.pad(x, padding, 'replicate')
		eps = 1.0 / (h + 2 * self.pad)
		arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
		arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
		base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
		base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
		shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
		shift *= 2.0 / (h + 2 * self.pad)
		grid = base_grid + shift
		return F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)


class Episode(object):
	"""Storage object for a single episode."""
	def __init__(self, cfg, init_obs, episode_length):
		self.cfg = cfg
		state_shape, map_shape = cfg.obs_shape
		state_shape = state_shape[0]
		img_h, img_w, img_c = map_shape
		# 数据类型
		self.dtype_state = torch.float32
		self.dtype_map = torch.bool
		self.device = torch.device(cfg.device)
		dtype = torch.float32 if cfg.modality == 'state' else torch.uint8
		self.obs_state = torch.empty(
            (episode_length+1, state_shape),
            dtype=self.dtype_state, device=self.device
        )
		self.obs_map = torch.empty(
            (episode_length+1, img_c, img_h, img_w),
            dtype=self.dtype_map, device=self.device
        )		
		# 初始化第一个观测
		init_state, init_map = init_obs  # 解包环境返回的 tuple
		self.obs_state[0] = torch.tensor(init_state, dtype=self.dtype_state, device=self.device)
        # 注意 map 通道次序 (H, W, C) -> (C, H, W)
		self.obs_map[0] = torch.tensor(init_map, dtype=self.dtype_map, device=self.device).permute(2, 0, 1)

		self.action = torch.empty((episode_length, cfg.action_dim), dtype=torch.float32, device=self.device)
		self.reward = torch.empty((episode_length,), dtype=torch.float32, device=self.device)
		self.manipulability_reward = torch.empty((episode_length,), dtype=torch.float32, device=self.device)
		self.cumulative_reward = 0
		self.done = False
		self._idx = 0
	
	def __len__(self):
		return self._idx

	@property
	def first(self):
		return len(self) == 0
	
	def __add__(self, transition):
		self.add(*transition)
		return self

	def add(self, obs, action, reward, done, manipulability_reward):
		step_obs_state, step_obs_map = obs
		self.obs_state[self._idx+1] = torch.tensor(step_obs_state, dtype=self.obs_state.dtype, device=self.obs_state.device)
		self.obs_map[self._idx+1] = torch.tensor(step_obs_map, dtype=self.obs_map.dtype, device=self.obs_map.device).permute(2, 0, 1)
		self.action[self._idx] = action
		self.reward[self._idx] = reward
		self.manipulability_reward[self._idx] = manipulability_reward
		self.cumulative_reward += reward
		self.done = done
		self._idx += 1


class ReplayBuffer():
	"""
	Storage and sampling functionality for training TD-MPC / TOLD.
	The replay buffer is stored in GPU memory when training from state.
	Uses prioritized experience replay by default."""
	def __init__(self, cfg):
		self.cfg = cfg
		self.device = torch.device(cfg.device)
		self.capacity = 20000
		state_shape, map_shape = cfg.obs_shape 
		state_shape = state_shape[0]
		img_h, img_w, img_c = map_shape
		# 分别创建 buffer, 这里为了对齐obs，action,reward,丢弃了最后一个观测，并且最后一步的奖励和动作也不会被采样
		self._obs_state = torch.empty((self.capacity, state_shape),
                                      dtype=torch.float32, device=self.device)
		self._obs_map = torch.empty((self.capacity, img_c, img_h, img_w),
                                      dtype=torch.bool, device=self.device)
		self._action = torch.empty((self.capacity, cfg.action_dim), dtype=torch.float32, device=self.device)
		self._reward = torch.empty((self.capacity,), dtype=torch.float32, device=self.device)
		self._manipulability_reward = torch.empty((self.capacity,), dtype=torch.float32, device=self.device)
		self._priorities = torch.ones((self.capacity,), dtype=torch.float32, device=self.device)
		self.buffer_end_marker = 0
		self.last_obs_marker = 0
		self._eps = 1e-6
		self._full = False
		self.idx = 0

	def __add__(self, episode: Episode):
		self.add(episode)
		return self

	def add(self, episode: Episode):
		ep_len = len(episode)
		start_idx = self.idx
		end_idx = start_idx + ep_len
		if end_idx > self.capacity:
			self.buffer_end_marker = self.idx
			self._full = True
			start_idx = 0
			end_idx = start_idx + ep_len
		# 保存数据，为了和action，reward对其，我们丢掉最后一个观测
		self._obs_state[start_idx:end_idx] = episode.obs_state[:-1]
		self._obs_map[start_idx:end_idx] = episode.obs_map[:-1]
		self._action[start_idx:end_idx] = episode.action
		self._reward[start_idx:end_idx] = episode.reward
		self._manipulability_reward[start_idx:end_idx] = episode.manipulability_reward
		if self._full:
			max_priority = self._priorities[:self.buffer_end_marker].max().to(self.device).item()
		else:
			max_priority = 1. if self.idx == 0 else self._priorities[:self.idx].max().to(self.device).item()
		mask = torch.arange(ep_len) >= ep_len-(self.cfg.horizon+1)
		new_priorities = torch.full((ep_len,), max_priority, device=self.device)
		new_priorities[mask] = 0
		self._priorities[start_idx:end_idx] = new_priorities
		self.idx = end_idx

	def update_priorities(self, idxs, priorities):
		self._priorities[idxs] = priorities.squeeze(1).to(self.device) + self._eps

	def _get_obs(self, idxs):
		obs_state = self._obs_state[idxs].float()
		obs_map = self._obs_map[idxs].float()
		return obs_state, obs_map

	def sample(self):
		probs = (self._priorities[:self.buffer_end_marker] if self._full else self._priorities[:self.idx]) ** self.cfg.per_alpha
		probs /= probs.sum()
		total = len(probs)
		idxs = torch.from_numpy(np.random.choice(total, self.cfg.batch_size, p=probs.cpu().numpy(), replace=not self._full)).to(self.device)
		weights = (total * probs[idxs]) ** (-self.cfg.per_beta)
		weights /= weights.max()

		obs_state, obs_map = self._get_obs(idxs)
		next_obs_state = torch.empty(
			(self.cfg.horizon+1, self.cfg.batch_size, *self._obs_state.shape[1:]),
			dtype=torch.float32, device=self.device
		)
		next_obs_map = torch.empty(
			(self.cfg.horizon+1, self.cfg.batch_size, *self._obs_map.shape[1:]),
			dtype=torch.bool, device=self.device
		)
		action = torch.empty((self.cfg.horizon+1, self.cfg.batch_size, *self._action.shape[1:]), dtype=torch.float32, device=self.device)
		reward = torch.empty((self.cfg.horizon+1, self.cfg.batch_size), dtype=torch.float32, device=self.device)
		manipulability_reward = torch.empty((self.cfg.horizon+1, self.cfg.batch_size), dtype=torch.float32, device=self.device)
		for t in range(self.cfg.horizon+1):
			_idxs = idxs + t
			s, m = self._get_obs(_idxs + 1)
			next_obs_state[t] = s
			next_obs_map[t] = m
			action[t] = self._action[_idxs]
			reward[t] = self._reward[_idxs]
			manipulability_reward[t] = self._manipulability_reward[_idxs]

		if not action.is_cuda:
			action, reward, idxs, weights = (
				action.cuda(), reward.cuda(), idxs.cuda(), weights.cuda(), manipulability_reward.cuda()
			)

		return (obs_state, obs_map), (next_obs_state, next_obs_map), action, reward.unsqueeze(2), idxs, weights, manipulability_reward.unsqueeze(2)


def linear_schedule(schdl, step):
	"""
	Outputs values following a linear decay schedule.
	Adapted from https://github.com/facebookresearch/drqv2
	"""
	try:
		return float(schdl)
	except ValueError:
		match = re.match(r'linear\((.+),(.+),(.+)\)', schdl)
		if match:
			init, final, duration = [float(g) for g in match.groups()]
			mix = np.clip(step / duration, 0.0, 1.0)
			return (1.0 - mix) * init + mix * final
	raise NotImplementedError(schdl)
