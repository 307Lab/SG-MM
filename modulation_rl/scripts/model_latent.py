import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from modulation.models.mapencoder import LocalMapCNN, LocalDoubleMapCNN
from ray.rllib.models.torch.misc import SlimFC
from torch import nn
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution, StateDependentNoiseDistribution

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)

class QNetwork(nn.Module):
    def __init__(self, feature_dim, action_space, hidden_dim):
        super(QNetwork, self).__init__()

        # --- 移除 CNN ---
        # OUT_CHANNELS = 10 if map_space.shape[2] <= 2 else 32
        # cnn_fn = LocalDoubleMapCNN
        # self.q1_cnn = cnn_fn(...)
        # self.q2_cnn = cnn_fn(...)
        # ----------------

        # 输入维度现在是：特征维度 + 动作维度
        concat_size = feature_dim
        if action_space is not None:
            concat_size += action_space.shape[0]

        # Q1 MLP 结构保持不变，输入层大小已调整
        q1_layers = []
        in_size = concat_size
        for hidden in hidden_dim:
            q1_layers.append(SlimFC(in_size=in_size, out_size=hidden, activation_fn="relu"))
            in_size = hidden
        self.q1_post_fc_stack = nn.Sequential(*q1_layers)
        self.q1_logits_layer = SlimFC(in_size=in_size, out_size=1, activation_fn=None)

        # Q2 MLP 结构保持不变
        q2_layers = []
        in_size = concat_size
        for hidden in hidden_dim:
            q2_layers.append(SlimFC(in_size=in_size, out_size=hidden, activation_fn="relu"))
            in_size = hidden
        self.q2_post_fc_stack = nn.Sequential(*q2_layers)
        self.q2_logits_layer = SlimFC(in_size=in_size, out_size=1, activation_fn=None)

        self.apply(weights_init_)

    # 修改 forward 方法，接收特征和动作
    def forward(self, features, action):
        # --- 移除 CNN 前向传播 ---
        # non_img_obs = [robot_state_space, action]
        # local_map = map_obs.permute([0, 3, 1, 2])...
        # q1_map_features = self.q1_cnn(local_map)
        # q2_map_features = self.q2_cnn(local_map)
        # -----------------------

        # 直接拼接特征和动作
        q_input = torch.cat([features, action], dim=-1)

        # Q1 forward
        q1_out = self.q1_post_fc_stack(q_input)
        q1 = self.q1_logits_layer(q1_out)

        # Q2 forward
        q2_out = self.q2_post_fc_stack(q_input)
        q2 = self.q2_logits_layer(q2_out)

        return q1, q2


class GaussianPolicy(nn.Module):
    def __init__(self, feature_dim, num_actions, hidden_dim, action_space=None, use_sde=True):
        super(GaussianPolicy, self).__init__()
        
        self.use_sde = use_sde
        concat_size = feature_dim

        # Optional post-concat FC-stack.
        layers = []
        in_size = concat_size
        for hidden in hidden_dim:
            layers.append(SlimFC(in_size=in_size, out_size=hidden, activation_fn="relu"))
            in_size = hidden
        self.post_fc_stack = nn.Sequential(*layers)


        if self.use_sde:
            self.action_dist = StateDependentNoiseDistribution(
                num_actions, full_std=True, use_expln=False, learn_features=True, squash_output=True
            )
            self.mean_linear, self.log_std_linear = self.action_dist.proba_distribution_net(
                latent_dim=hidden_dim[-1], latent_sde_dim=hidden_dim[-1], log_std_init=-3.67
            )
            # Avoid numerical issues by limiting the mean of the Gaussian
            # to be in [-clip_mean, clip_mean]
            self.mean_linear = nn.Sequential(self.mean_linear, nn.Hardtanh(min_val=-2.0, max_val=2.0))
        else:
            self.action_dist = SquashedDiagGaussianDistribution(num_actions)
            self.mean_linear = nn.Linear(hidden_dim[-1], num_actions)
            self.log_std_linear = nn.Linear(hidden_dim[-1], num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def reset_noise(self, batch_size: int = 1) -> None:
        """
        Sample new weights for the exploration matrix, when using gSDE.

        :param batch_size:
        """
        msg = "reset_noise() is only available when using gSDE"
        assert isinstance(self.action_dist, StateDependentNoiseDistribution), msg
        self.action_dist.sample_weights(self.log_std_linear, batch_size=batch_size)

    def forward(self, features):
        # --- 移除 CNN 前向传播 ---
        # non_img_obs = [robot_state_space]
        # local_map = ...
        # map_features = self.cnn(local_map)
        # out = torch.cat(non_img_obs + [map_features], dim=-1)
        # -----------------------

        # 直接通过 MLP
        out = self.post_fc_stack(features)

        mean = self.mean_linear(out)
        if self.use_sde:
            return mean, self.log_std_linear, dict(latent_sde=out)

        log_std = self.log_std_linear(out)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std, {}

    def sample(self, features):
        mean_actions, log_std, kwargs = self.forward(features)
        return self.action_dist.log_prob_from_params(mean_actions, log_std, **kwargs)

        # std = log_std.exp()
        # normal = Normal(mean, std)
        # x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        # y_t = torch.tanh(x_t)
        # action = y_t * self.action_scale + self.action_bias
        # log_prob = normal.log_prob(x_t)
        # log_prob = torch.clamp(log_prob, -100, 100)
        # # Enforcing Action Bound
        # log_prob -= torch.log((1 - y_t.pow(2)) + epsilon)
        # log_prob = log_prob.sum(1, keepdim=True)
        # mean = torch.tanh(mean) * self.action_scale + self.action_bias
        # return action, log_prob, mean

    # 修改 predict 方法，接收特征
    def predict(self, features, eval = False):
        mean_actions, log_std, kwargs = self.forward(features)
        action = self.action_dist.actions_from_params(mean_actions, log_std, deterministic=eval, **kwargs)
        # 注意：检查你的 action_dist 是否已经包含了 tanh 和缩放，如果包含了这里就不要重复了
        return torch.tanh(action) * self.action_scale + self.action_bias
    
    # 修改 sample_with_action 方法，接收特征
    def sample_with_action(self, features, actions):
        mean_actions, log_std, kwargs = self.forward(features)
        return self.action_dist.log_prob_from_params_given_action(mean_actions, log_std, actions, **kwargs)

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        # for StateDependentNoiseDistribution
        # self.action_dist.exploration_mat = self.action_dist.exploration_mat.to(device)
        return super(GaussianPolicy, self).to(device)


class DeterministicPolicy(nn.Module):
    def __init__(self, map_space, robot_state_space, num_actions, hidden_dim, action_space=None):
        super(DeterministicPolicy, self).__init__()

        self.noise = torch.Tensor(num_actions)

        OUT_CHANNELS = 10 if map_space.shape[2] <= 2 else 32    

        cnn_fn = LocalDoubleMapCNN
        self.cnn = cnn_fn(in_shape=map_space.shape, out_channels=OUT_CHANNELS, stride=1)
        self.cnn = self.cnn.to(memory_format=torch.channels_last)
        concat_size = robot_state_space.shape[0] + self.cnn.output_size

        # Optional post-concat FC-stack.
        layers = []
        in_size = concat_size
        for hidden in hidden_dim:
            layers.append(SlimFC(in_size=in_size, out_size=hidden, activation_fn="relu"))
            in_size = hidden
        self.post_fc_stack = nn.Sequential(*layers)

        self.mean_linear = nn.Linear(hidden_dim[-1], num_actions)
        self.log_std_linear = nn.Linear(hidden_dim[-1], num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, map_obs, robot_state_space):
        non_img_obs = [robot_state_space]
        local_map = map_obs.permute([0, 3, 1, 2]).contiguous(memory_format=torch.channels_last)

        map_features = self.cnn(local_map)
        out = torch.cat(non_img_obs + [map_features], dim=-1)
        out = self.post_fc_stack(out)

        mean = self.mean_linear(out)
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)