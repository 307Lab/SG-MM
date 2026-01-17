import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torchvision import transforms
from modulation.models.mapencoder import LocalMapCNN, LocalDoubleMapCNN
import datetime
import os

class ImageDecoder(nn.Module):
    def __init__(self, input_dim=512, cnn_feature_base=512, output_channels=1):
        super().__init__()
        self.input_dim = input_dim
        self.cnn_feature_base = cnn_feature_base
        self.fc = nn.Linear(input_dim, cnn_feature_base)
        
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(cnn_feature_base, 256, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=6, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, output_channels, kernel_size=8, stride=2)
        )
        
        self.final_crop = transforms.CenterCrop((120, 120))
        self.sigmoid = nn.Sigmoid() 

    def forward(self, h):
        x = self.fc(h)
        x = x.reshape(-1, self.cnn_feature_base, 1, 1)
        x = self.deconv(x)
        x = self.final_crop(x)
        x = self.sigmoid(x)
        return x

class VectorEncoder(nn.Module):
    def __init__(self, input_dim=41, output_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, output_dim)
        )
    
    def forward(self, x):
        return self.fc(x)

class VectorDecoder(nn.Module):
    def __init__(self, input_dim=512, output_dim=41):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, output_dim)
        )

    def forward(self, h):
        return self.fc(h)

class WorldModel(nn.Module):
    def __init__(self, 
                 observation_space: None,
                 action_space: None,
                 latent_dim: int = 512,
                 img_feature_dim: int = 1000, 
                 vec_dim: int = 41,
                 vec_feature_dim: int = 128):
        
        super().__init__()
        robot_state_space, map_space = observation_space
        hight, width, Channel = map_space.shape
        self.img_feature_dim = img_feature_dim
        self.vec_feature_dim = vec_feature_dim
        self.latent_dim = latent_dim
        self.action_dim = action_space.shape[0]
        
        self.img_encoder = LocalDoubleMapCNN(in_shape=map_space.shape, out_channels=10, stride=1)
        self.vec_encoder = VectorEncoder(input_dim=robot_state_space.shape[0], output_dim=vec_feature_dim)
        
        self.state_embed_dim = img_feature_dim + vec_feature_dim

        self.rnn = nn.GRU(
            input_size=self.state_embed_dim + self.action_dim, 
            hidden_size=latent_dim,
            batch_first=True # 输入张量形状为 (B, T, F)
        )

        self.transition_model = nn.Sequential(
            nn.Linear(latent_dim + self.action_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, latent_dim)
        )

        self.img_decoder = ImageDecoder(
            input_dim=latent_dim, 
            cnn_feature_base=self.img_feature_dim, 
            output_channels=Channel 
        )
        self.vec_decoder = VectorDecoder(
            input_dim=latent_dim, 
            output_dim=vec_dim
        )
        self.reward_decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1)
        )

    def _encode_state(self, image, vector):
        img_embed = self.img_encoder(image)
        vec_embed = self.vec_encoder(vector)
        z = torch.cat([img_embed, vec_embed], dim=1)
        return z

    def _decode_state(self, h):
        recon_img = self.img_decoder(h)
        recon_vec = self.vec_decoder(h)
        return recon_img, recon_vec

    def forward(self, images, vectors, actions, rewards):
        B, T_plus_1, C, H, W = images.shape
        T_act = actions.shape[1]
        V_dim = vectors.shape[2]
        
        images_flat = images.reshape(B * T_plus_1, C, H, W)
        vectors_flat = vectors.reshape(B * T_plus_1, V_dim)
        
        z_flat = self._encode_state(images_flat, vectors_flat)
        z_t = z_flat.reshape(B, T_plus_1, self.state_embed_dim)

        actions_prev = torch.cat([
            torch.zeros(B, 1, self.action_dim, device=actions.device),
            actions
        ], dim=1)
        
        rnn_input = torch.cat([z_t, actions_prev], dim=2)
        
        h_t, _ = self.rnn(rnn_input)

        h_flat = h_t.reshape(B * T_plus_1, self.latent_dim)
        recon_img, recon_vec = self._decode_state(h_flat)
        
        loss_recon_img_per_step = F.mse_loss(recon_img, images_flat, reduction='none')
        loss_recon_img_per_step = loss_recon_img_per_step.mean(dim=(1, 2, 3))
        loss_recon_img_per_sample = loss_recon_img_per_step.reshape(B, T_plus_1).mean(dim=1)

        loss_recon_vec_per_step = F.mse_loss(recon_vec, vectors_flat, reduction='none')
        loss_recon_vec_per_step = loss_recon_vec_per_step.mean(dim=1)
        loss_recon_vec_per_sample = loss_recon_vec_per_step.reshape(B, T_plus_1).mean(dim=1)
        
        h_t_to_predict = h_t[:, :-1, :]
        actions_t = actions            
        
        trans_input_flat = torch.cat([h_t_to_predict, actions_t], dim=2).reshape(B * T_act, -1)
        h_t_plus_1_pred = self.transition_model(trans_input_flat) 
        
        pred_img, pred_vec = self._decode_state(h_t_plus_1_pred) 
        
        target_images = images[:, 1:, ...].reshape(B * T_act, C, H, W)
        target_vectors = vectors[:, 1:, :].reshape(B * T_act, V_dim)
        
        loss_pred_img_per_step = F.mse_loss(pred_img, target_images, reduction='none') 
        loss_pred_img_per_step = loss_pred_img_per_step.mean(dim=(1, 2, 3))
        loss_pred_img_per_sample = loss_pred_img_per_step.reshape(B, T_act).mean(dim=1)

        loss_pred_vec_per_step = F.mse_loss(pred_vec, target_vectors, reduction='none')
        loss_pred_vec_per_step = loss_pred_vec_per_step.mean(dim=1)
        loss_pred_vec_per_sample = loss_pred_vec_per_step.reshape(B, T_act).mean(dim=1)
        
        h_for_reward = h_t[:, 1:, :]
        pred_rewards = self.reward_decoder(h_for_reward.reshape(B * T_act, -1))
        target_rewards = rewards.reshape(B * T_act, 1)
        
        loss_reward_per_step = F.mse_loss(pred_rewards, target_rewards, reduction='none')
        loss_reward_per_step = loss_reward_per_step.squeeze(dim=1)
        loss_reward_per_sample = loss_reward_per_step.reshape(B, T_act).mean(dim=1)

        return (
            loss_recon_img_per_sample,
            loss_recon_vec_per_sample, 
            loss_pred_img_per_sample,  
            loss_pred_vec_per_sample, 
            loss_reward_per_sample,   
            h_t
        )
        
    def transition(self, h, action):
        trans_input = torch.cat([h, action], dim=1)
        h_next_pred = self.transition_model(trans_input)
        return h_next_pred
    
    def save_checkpoint(self, world_model, env_name, optimizer, step, suffix="", ckpt_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
        if ckpt_path is None:
            ckpt_path = "checkpoints/world_model_checkpoint_{}_{}_{}".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), env_name, suffix)
        print('Saving models to {}'.format(ckpt_path))
        
        checkpoint_data = {
            'step': step,
            'model_state_dict': world_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        
        torch.save(checkpoint_data, ckpt_path)
        print(f"Checkpoint saved successfully at step {step} to {ckpt_path}")


    def load_checkpoint(self, filepath, optimizer, device):
        
        if not os.path.isfile(filepath):
            print(f"No checkpoint found at {filepath}. Starting from scratch.")
            return 0 

        print(f"Loading checkpoint from {filepath}...")
        checkpoint_data = torch.load(filepath, map_location=device)

        self.load_state_dict(checkpoint_data['model_state_dict'], strict=True)

        start_step = checkpoint_data.get('step', 0)

        print(f"Successfully loaded checkpoint. Resuming training from step {start_step}.")

        self.train()

        return start_step
    
def generate_imagination_data_from_hidden(world_model, agent, h_start, horizon):
    batch_size = h_start.shape[0]
    device = h_start.device

    imagined_features = []
    imagined_actions = []
    imagined_rewards = []

    world_model.eval()
    agent.policy.eval()

    with torch.no_grad():
        h_current = h_start
        imagined_features.append(h_current) # z_0

        for t in range(horizon):
            action, _ = agent.policy.sample(h_current)
            imagined_actions.append(action)

            h_next = world_model.transition(h_current, action)
            imagined_features.append(h_next)

            reward = world_model.reward_decoder(h_current)
            imagined_rewards.append(reward)

            h_current = h_next
    features_tensor = torch.stack(imagined_features[:-1])
    next_features_tensor = torch.stack(imagined_features[1:])
    actions_tensor = torch.stack(imagined_actions)
    rewards_tensor = torch.stack(imagined_rewards)

    feature_dim = features_tensor.shape[-1]
    action_dim = actions_tensor.shape[-1]

    features_b = features_tensor.reshape(-1, feature_dim)
    next_features_b = next_features_tensor.reshape(-1, feature_dim)
    actions_b = actions_tensor.reshape(-1, action_dim)
    rewards_b = rewards_tensor.reshape(-1, 1)
    
    masks_b = torch.ones_like(rewards_b)

    world_model.train()
    agent.policy.train()

    return features_b, actions_b, rewards_b, next_features_b, masks_b