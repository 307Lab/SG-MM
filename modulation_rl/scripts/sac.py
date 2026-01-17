import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils import soft_update, hard_update
from model import GaussianPolicy, QNetwork, DeterministicPolicy
import gym
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
import datetime

class SAC(object):
    def __init__(self, obs_space, action_space, target_update_interval, args, num_steps):

        original_space = obs_space.original_space if hasattr(obs_space, "original_space") else obs_space
        assert isinstance(original_space, gym.spaces.Tuple)
        robot_state_space, map_space = original_space

        self.gamma = args["gamma"]
        self.tau = args["tau"]
        self.alpha = args["initial_alpha"]  
        # 限制alpha时使用
        self.max_alpha = 0.35
        self.min_alpha = 0.10
        self.lr_start = 1e-4
        self.lr_end = 1e-6
        self.grad_clip = 10
        self.use_sde = False
        self.num_steps = num_steps

        self.policy_type = "Gaussian"
        self.target_update_interval = target_update_interval
        self.automatic_entropy_tuning = True

        self.device = torch.device("cuda")

        self.critic = QNetwork(map_space, robot_state_space,  action_space, args["model"]["custom_model_config"]["post_fcnet_hiddens"]).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr_start)

        self.critic_target = QNetwork(map_space, robot_state_space,  action_space, args["model"]["custom_model_config"]["post_fcnet_hiddens"]).to(self.device)
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
            self.policy = GaussianPolicy(map_space, robot_state_space, action_space.shape[0], args["model"]["custom_model_config"]["post_fcnet_hiddens"], action_space,self.use_sde).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=self.lr_start)
            self.scheduler_policy_optim = LambdaLR(self.policy_optim, lr_lambda)
        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(map_space, robot_state_space, action_space.shape[0], args["model"]["custom_model_config"]["post_fcnet_hiddens"], action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=self.lr_start)
            self.scheduler_policy_optim = LambdaLR(self.policy_optim, lr_lambda)
       

    def select_action(self, state, eval = False):
        map_state = torch.FloatTensor(state[1]).to(self.device).unsqueeze(0)
        rob_state = torch.FloatTensor(state[0]).to(self.device).unsqueeze(0)
        action = self.policy.predict(map_state, rob_state, eval)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory, batch_size, updates):
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
            next_state_action, next_state_log_pi = self.policy.sample(next_map_state_batch, next_rob_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_map_state_batch, next_rob_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi.unsqueeze(1)
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        qf1, qf2 = self.critic(map_state_batch, rob_state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
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

        pi, log_pi = self.policy.sample(map_state_batch, rob_state_batch)

        qf1_pi, qf2_pi = self.critic(map_state_batch, rob_state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

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

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item(), log_pi.mean().item(), min_qf_pi.mean().item()
    
    # def update_parameters(self, memory, batch_size, updates):
    #     # 1. Sample with Priority
    #     # 注意：这里 sample 返回三个值：transition batch, indices, weights
    #     (map_state_batch, rob_state_batch, action_batch, reward_batch, next_map_state_batch, next_rob_state_batch, mask_batch), idxs, is_weights = memory.sample(batch_size=batch_size)

    #     # 转换 Tensor
    #     map_state_batch = torch.FloatTensor(map_state_batch).to(self.device)
    #     rob_state_batch = torch.FloatTensor(rob_state_batch).to(self.device)
    #     next_map_state_batch = torch.FloatTensor(next_map_state_batch).to(self.device)
    #     next_rob_state_batch = torch.FloatTensor(next_rob_state_batch).to(self.device)
    #     action_batch = torch.FloatTensor(action_batch).to(self.device)
    #     reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
    #     mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
        
    #     # 权重也要转为 Tensor
    #     is_weights = torch.FloatTensor(is_weights).to(self.device).unsqueeze(1)

    #     if self.use_sde:
    #         self.policy.reset_noise(batch_size)

    #     # --- Critic Update ---
    #     with torch.no_grad():
    #         next_state_action, next_state_log_pi = self.policy.sample(next_map_state_batch, next_rob_state_batch)
    #         qf1_next_target, qf2_next_target = self.critic_target(next_map_state_batch, next_rob_state_batch, next_state_action)
    #         min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi.unsqueeze(1)
    #         next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)

    #     qf1, qf2 = self.critic(map_state_batch, rob_state_batch, action_batch)
        
    #     # 2. 计算 Element-wise Loss (不要 mean，因为要乘权重)
    #     # 这里的 F.mse_loss 必须设为 reduction='none' 才能返回每个样本的 loss
    #     qf1_loss_element = F.mse_loss(qf1, next_q_value, reduction='none') 
    #     qf2_loss_element = F.mse_loss(qf2, next_q_value, reduction='none')
        
    #     # 3. Apply Importance Sampling Weights
    #     # PER Loss = Mean(Weight * Loss)
    #     qf1_loss = (qf1_loss_element * is_weights).mean()
    #     qf2_loss = (qf2_loss_element * is_weights).mean()
    #     qf_loss = qf1_loss + qf2_loss

    #     self.critic_optim.zero_grad()
    #     qf_loss.backward()
        
    #     # Clip Grads
    #     for param_group in self.critic_optim.param_groups:
    #         params = list(filter(lambda p: p.grad is not None, param_group["params"]))
    #         if params:
    #             nn.utils.clip_grad_norm_(params, self.grad_clip)
                
    #     self.critic_optim.step()
    #     self.scheduler_critic_optim.step()

    #     # 4. 更新优先级 Update Priorities
    #     # 通常使用 TD-error 的绝对值作为优先级。这里可以用两个 critic 的 error 均值，或者直接用 qf1
    #     with torch.no_grad():
    #         td_errors = (torch.abs(qf1 - next_q_value) + torch.abs(qf2 - next_q_value)) / 2
    #         # 转回 numpy 存入 memory
    #         memory.update_priorities(idxs, td_errors.cpu().numpy().flatten())

    #     # --- Policy Update (SAC Standard) ---
    #     pi, log_pi = self.policy.sample(map_state_batch, rob_state_batch)

    #     qf1_pi, qf2_pi = self.critic(map_state_batch, rob_state_batch, pi)
    #     min_qf_pi = torch.min(qf1_pi, qf2_pi)

    #     policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

    #     self.policy_optim.zero_grad()
    #     policy_loss.backward()
        
    #     for param_group in self.policy_optim.param_groups:
    #         params = list(filter(lambda p: p.grad is not None, param_group["params"]))
    #         if params:
    #             nn.utils.clip_grad_norm_(params, self.grad_clip) 
                
    #     self.policy_optim.step()
    #     self.scheduler_policy_optim.step()

    #     # --- Alpha Tuning (Keep existing logic) ---
    #     if self.automatic_entropy_tuning:
    #         alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

    #         self.alpha_optim.zero_grad()
    #         alpha_loss.backward()
            
    #         for param_group in self.alpha_optim.param_groups:
    #             params = list(filter(lambda p: p.grad is not None, param_group["params"]))
    #             if params:
    #                 nn.utils.clip_grad_norm_(params, self.grad_clip)
            
    #         self.alpha_optim.step()
            
    #         # 用户原本的 Alpha 逻辑
    #         self.alpha = self.log_alpha.exp() * self.max_alpha * 0.3
    #         alpha_tlogs = self.alpha.clone()

    #         if self.alpha <= self.min_alpha:
    #             self.alpha = torch.tensor(self.min_alpha)
    #             alpha_tlogs = self.alpha
    #     else:
    #         alpha_loss = torch.tensor(0.).to(self.device)
    #         alpha_tlogs = torch.tensor(self.alpha)

    #     if updates % self.target_update_interval == 0:
    #         soft_update(self.critic_target, self.critic, self.tau)

    #     return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()


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

