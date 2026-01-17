import torch
import numpy as np
import re 

def linear_schedule(schdl, step):
    try:
        return float(schdl)
    except ValueError:
        match = re.match(r'linear\((.+),(.+),(.+)\)', schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
    raise NotImplementedError(f"Unknown scheduling format: {schdl}")

class Planner:
    def __init__(self, world_model, sac_agent, reward_oracle, action_space, 
                 num_samples=50, horizon=6, gamma=0.99, noise_std_schedule="linear(0.5, 0.1, 4500000)"):
        self.world_model = world_model
        self.actor = sac_agent.policy
        self.critic = sac_agent.critic
        
        self.num_samples = num_samples 
        self.horizon = horizon         
        self.horizon_linear_schedule = linear_schedule
        self.gamma = gamma
        self.action_dim = action_space.shape[0]
        self.device = next(world_model.parameters()).device
        
        self.noise_std_schedule_str = noise_std_schedule
        self.linear_schedule = linear_schedule 

    @torch.no_grad()
    def select_action(self, h_t, state_vec, total_numsteps):
        B = self.num_samples
        
        action_sequences = torch.empty(B, self.horizon, self.action_dim, device=self.device)
        
        vec_state_np, img_state_np = state_vec[0], state_vec[1]
        vec_tensor = torch.tensor(vec_state_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        img_tensor = torch.tensor(img_state_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        current_action_mean, _, _ = self.actor.forward(img_tensor, vec_tensor)
        current_noise_std = self.linear_schedule(self.noise_std_schedule_str, total_numsteps)
        self.horizon = int(round(self.horizon_linear_schedule("linear(1, 8, 4500000)", total_numsteps)))
        
        mean_action_seq = current_action_mean.repeat(B, self.horizon, 1)
        action_sequences = mean_action_seq + torch.randn_like(mean_action_seq) * current_noise_std
        action_sequences = torch.clamp(action_sequences, -1.0, 1.0) 

        h_current = h_t.repeat(B, 1)
        total_returns = torch.zeros(B, device=self.device)

        for k in range(self.horizon):
            actions_k = action_sequences[:, k, :] 
            
            h_next_pred = self.world_model.transition(h_current, actions_k)
            pred_reward = self.world_model.reward_decoder(h_current).squeeze()
            
            total_returns += (self.gamma ** k) * pred_reward
            h_current = h_next_pred 

        img_pred_T, vec_pred_T = self.world_model._decode_state(h_current)
        img_pred_T_for_actor = img_pred_T.permute(0, 2, 3, 1)
        terminal_actions, _ = self.actor.sample(img_pred_T_for_actor, vec_pred_T)
        terminal_q1, terminal_q2 = self.critic(img_pred_T_for_actor, vec_pred_T, terminal_actions)
        terminal_value = torch.min(terminal_q1, terminal_q2).squeeze() 
        
        total_returns += (self.gamma ** self.horizon) * terminal_value

        best_trajectory_idx = torch.argmax(total_returns)
        best_first_action = action_sequences[best_trajectory_idx, 0, :]
        
        return best_first_action.cpu().numpy() 