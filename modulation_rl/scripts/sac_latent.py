import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils import soft_update, hard_update
from model_latent import GaussianPolicy, QNetwork
import gym
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
import datetime

class SAC(object):
    def __init__(self, world_model, feature_dim, action_space, target_update_interval, args, num_steps):

        self.world_model = world_model # 保存 World Model 引用，用于 select_action
        self.feature_dim = feature_dim # 保存特征维度

        self.gamma = args["gamma"]
        self.tau = args["tau"]
        self.alpha = args["initial_alpha"]  
        # 限制alpha时使用
        self.max_alpha = 0.35
        self.min_alpha = 0.05
        self.lr_start = 1e-4
        self.lr_end = 1e-6
        self.grad_clip = 10
        self.use_sde = False
        self.num_steps = num_steps

        self.policy_type = "Gaussian"
        self.target_update_interval = target_update_interval
        self.automatic_entropy_tuning = True

        self.device = torch.device("cuda")

        self.critic = QNetwork(feature_dim, action_space, args["model"]["custom_model_config"]["post_fcnet_hiddens"]).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr_start)

        self.critic_target = QNetwork(feature_dim, action_space, args["model"]["custom_model_config"]["post_fcnet_hiddens"]).to(self.device)
        hard_update(self.critic_target, self.critic)
        # 定义学习率调度器
        lr_lambda = lambda step: (self.lr_end / self.lr_start) ** (step / self.num_steps)
        self.scheduler_critic_optim = LambdaLR(self.critic_optim, lr_lambda)

        if self.policy_type == "Gaussian":
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                # 限制alpha时使用
                self.alpha_optim = Adam([self.log_alpha], lr=self.lr_start * 0.01)
                # self.alpha_optim = Adam([self.log_alpha], lr=self.lr_start)
                # self.scheduler_alpha_optim = LambdaLR(self.alpha_optim, lr_lambda)
            self.policy = GaussianPolicy(feature_dim, action_space.shape[0], args["model"]["custom_model_config"]["post_fcnet_hiddens"], action_space, self.use_sde).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=self.lr_start)
            self.scheduler_policy_optim = LambdaLR(self.policy_optim, lr_lambda)
       

    def select_action(self, state, eval = False):
        map_state = torch.FloatTensor(state[1]).to(self.device).unsqueeze(0)
        rob_state = torch.FloatTensor(state[0]).to(self.device).unsqueeze(0)
        with torch.no_grad():
            # 注意图像维度，假设编码器需要 (B, C, H, W)
            if map_state.shape[-1] < map_state.shape[1]:
                 map_tensor = map_state.permute(0, 3, 1, 2).contiguous()
            
            # 调用 World Model 的辅助编码函数 (假设您在 World Model 中实现了类似 _encode_state 的方法)
            # 这里的 features 维度应该是 (1, feature_dim)
            features = self.world_model._encode_state(map_tensor, rob_state)
        action = self.policy.predict(features, eval)
        return action.detach().cpu().numpy()[0]

    # 修改：彻底改变输入签名，不再接收 memory，而是接收想象数据张量
    def update_parameters(self, features_b, actions_b, rewards_b, next_features_b, masks_b, updates):
        features_b = features_b.detach()
        next_features_b = next_features_b.detach()
        if self.use_sde:
            self.policy.reset_noise(features_b.shape[0])

        with torch.no_grad():
            # Actor 输入特征，采样下一时刻动作
            next_state_action, next_state_log_pi = self.policy.sample(next_features_b)
            
            # Target Critic 输入下一时刻特征和动作
            qf1_next_target, qf2_next_target = self.critic_target(next_features_b, next_state_action)
            
            # 计算 Target Q 值 (使用新的 alpha 更新逻辑，见我上一个回答)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi.unsqueeze(1)
            next_q_value = rewards_b + masks_b * self.gamma * (min_qf_next_target)
        # Critic 输入当前特征和动作
        qf1, qf2 = self.critic(features_b, actions_b)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()

        for param_group in self.critic_optim.param_groups:
        # Make sure we only pass params with grad != None into torch
        # clip_grad_norm_. Would fail otherwise.
            params = list(filter(lambda p: p.grad is not None, param_group["params"]))
            if params:
                nn.utils.clip_grad_norm_(params, self.grad_clip)

        self.critic_optim.step()
        self.scheduler_critic_optim.step()

        pi, log_pi = self.policy.sample(features_b)

        # Critic 评估动作
        qf1_pi, qf2_pi = self.critic(features_b, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        # 计算 Policy Loss
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()

        for param_group in self.policy_optim.param_groups:
        # Make sure we only pass params with grad != None into torch
        # clip_grad_norm_. Would fail otherwise.
            params = list(filter(lambda p: p.grad is not None, param_group["params"]))
            if params:
                nn.utils.clip_grad_norm_(params, self.grad_clip)        

        self.policy_optim.step()
        self.scheduler_policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            
            for param_group in self.alpha_optim.param_groups:
            # Make sure we only pass params with grad != None into torch
            # clip_grad_norm_. Would fail otherwise.
                params = list(filter(lambda p: p.grad is not None, param_group["params"]))
                if params:
                    nn.utils.clip_grad_norm_(params, self.grad_clip)
            
            self.alpha_optim.step()
            # current_alpha = self.log_alpha.exp()
            # self.alpha = torch.clamp(current_alpha, min=self.min_alpha, max=self.max_alpha).detach()
            # self.scheduler_alpha_optim.step()

            # self.alpha = self.log_alpha.exp()
            # 限制最大alpha为0.35
            self.alpha = self.log_alpha.exp() * self.max_alpha
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs

            # alpha降低至0.05时，不再降低了
            if self.alpha <= self.min_alpha:
                self.alpha = torch.tensor(self.min_alpha)
                # self.automatic_entropy_tuning = False
                alpha_tlogs = self.alpha
            # alpha_tlogs = self.alpha
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()


    def update_parameters_modified(self, memory, batch_size, updates, prior_agent):
            # Sample a batch from memory
            map_state_batch, rob_state_batch, action_batch, reward_batch, next_map_state_batch, next_rob_state_batch, mask_batch = memory.sample(batch_size=batch_size)

            map_state_batch = torch.FloatTensor(map_state_batch).to(self.device)
            rob_state_batch = torch.FloatTensor(rob_state_batch).to(self.device)
            next_map_state_batch = torch.FloatTensor(next_map_state_batch).to(self.device)
            next_rob_state_batch = torch.FloatTensor(next_rob_state_batch).to(self.device)
            action_batch = torch.FloatTensor(action_batch).to(self.device)
            reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
            mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

            if self.use_sde:
                self.policy.reset_noise(batch_size)

            with torch.no_grad():

                state_action, state_log_pi = self.policy.sample(map_state_batch, rob_state_batch)

                prior_log_prob = prior_agent.policy.sample_with_action(map_state_batch, rob_state_batch, state_action)
                prior_log_prob = prior_log_prob.reshape(-1, 1)

                next_state_action, next_state_log_pi = self.policy.sample(next_map_state_batch, next_rob_state_batch)
                next_state_log_pi = next_state_log_pi.reshape(-1, 1)

                next_prior_log_prob = prior_agent.policy.sample_with_action(next_map_state_batch, next_rob_state_batch, next_state_action)
                next_prior_log_prob = next_prior_log_prob.reshape(-1, 1)

                qf1_next_target, qf2_next_target = self.critic_target(next_map_state_batch, next_rob_state_batch, next_state_action)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) + self.alpha * next_prior_log_prob - self.alpha * next_state_log_pi

                next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
            qf1, qf2 = self.critic(map_state_batch, rob_state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
            qf1_loss = F.mse_loss(qf1, next_q_value)
            qf2_loss = F.mse_loss(qf2, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            self.critic_optim.zero_grad()
            qf_loss.backward()

            for param_group in self.critic_optim.param_groups:
            # Make sure we only pass params with grad != None into torch
            # clip_grad_norm_. Would fail otherwise.
                params = list(filter(lambda p: p.grad is not None, param_group["params"]))
                if params:
                    nn.utils.clip_grad_norm_(params, self.grad_clip)

            self.critic_optim.step()
            self.scheduler_critic_optim.step()

            qf1_pi, qf2_pi = self.critic(map_state_batch, rob_state_batch, state_action)
            min_qf_pi = torch.min(qf1_pi, qf2_pi)

            policy_loss = (self.alpha * state_log_pi - min_qf_pi - self.alpha * prior_log_prob).mean()

            self.policy_optim.zero_grad()
            policy_loss.backward()

            for param_group in self.policy_optim.param_groups:
            # Make sure we only pass params with grad != None into torch
            # clip_grad_norm_. Would fail otherwise.
                params = list(filter(lambda p: p.grad is not None, param_group["params"]))
                if params:
                    nn.utils.clip_grad_norm_(params, self.grad_clip)        

            self.policy_optim.step()
            self.scheduler_policy_optim.step()

            if self.automatic_entropy_tuning:
                alpha_loss = -(self.log_alpha * (state_log_pi + self.target_entropy).detach()).mean()

                self.alpha_optim.zero_grad()
                alpha_loss.backward()
                
                for param_group in self.alpha_optim.param_groups:
                # Make sure we only pass params with grad != None into torch
                # clip_grad_norm_. Would fail otherwise.
                    params = list(filter(lambda p: p.grad is not None, param_group["params"]))
                    if params:
                        nn.utils.clip_grad_norm_(params, self.grad_clip)
                
                self.alpha_optim.step()
                # self.scheduler_alpha_optim.step()

                # self.alpha = self.log_alpha.exp()
                # 限制最大alpha为0.35
                self.alpha = self.log_alpha.exp() * self.max_alpha
                alpha_tlogs = self.alpha.clone() # For TensorboardX logs

                # alpha降低至0.05时，不再降低了
                if self.alpha <= self.min_alpha:
                    self.alpha = torch.tensor(self.min_alpha)
                    self.automatic_entropy_tuning = False
                    alpha_tlogs = self.alpha
                # alpha_tlogs = self.alpha
            else:
                alpha_loss = torch.tensor(0.).to(self.device)
                alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


            if updates % self.target_update_interval == 0:
                soft_update(self.critic_target, self.critic, self.tau)

            return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    # Save model parameters
    def save_checkpoint(self, env_name, suffix="", ckpt_path=None):
        if not os.path.exists('modified_ik_checkpoints/'):
            os.makedirs('modified_ik_checkpoints/')
        if ckpt_path is None:
            ckpt_path = "modified_ik_checkpoints/sac_checkpoint_{}_{}_{}".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), env_name, suffix)
        print('Saving models to {}'.format(ckpt_path))
        torch.save({'policy_state_dict': self.policy.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optim.state_dict(),
                    'policy_optimizer_state_dict': self.policy_optim.state_dict()}, ckpt_path)

    # Load model parameters
    def load_checkpoint(self, ckpt_path, evaluate=False):
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])

            if evaluate:
                self.policy.eval()
                self.critic.eval()
                self.critic_target.eval()
            else:
                self.policy.train()
                self.critic.train()
                self.critic_target.train()


    # Load and freeze model parameters
    def load_and_freeze_checkpoint(self, ckpt_path):
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        else:
            print("Invaid models !!!")

        for param in self.policy.parameters():
            param.requires_grad = False
        for param in self.critic.parameters():
            param.requires_grad = False
        for param in self.critic_target.parameters():
            param.requires_grad = False

