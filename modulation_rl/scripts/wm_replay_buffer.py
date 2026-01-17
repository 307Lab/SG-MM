import numpy as np
import torch
from collections import deque
from wm_model import WorldModel
from sklearn.neighbors import LocalOutlierFactor
import random
import itertools
from sumtree import SumTree

class ReplayBuffer:
    def __init__(self, capacity, sequence_length, img_shape, vec_dim, action_dim, device,
                 alpha=0.6, beta_start=0.4, beta_frames=100000, epsilon=1e-6,
                 max_episode_len=2000):
        self.ik_fail_priority_bonus = 1
        self.capacity = capacity
        self.seq_len = sequence_length
        self.img_shape = img_shape
        self.vec_dim = vec_dim
        self.action_dim = action_dim
        self.device = device
        self.lof_priority_weight = 1.0

        self.alpha = alpha
        self.beta = beta_start
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.epsilon = epsilon
        self.frame_count = 0

        self.images = deque(maxlen=capacity)
        self.vectors = deque(maxlen=capacity)
        self.actions = deque(maxlen=capacity)
        self.rewards = deque(maxlen=capacity)
        self.ik_fail = deque(maxlen=capacity)
        self.dones = deque(maxlen=capacity)
        
        self.episode_lof_scores = deque(maxlen=capacity)
        
        self.episode_ids = deque(maxlen=capacity)
        self.next_episode_id = itertools.count()

        self.ep_id_to_deque_idx = {} # {global_ep_id: deque_idx}
        self.episode_segment_tree_indices = deque(maxlen=capacity)

        self.total_segment_capacity = capacity * (max_episode_len - self.seq_len + 1)
        if self.total_segment_capacity <= 0: self.total_segment_capacity = 1000
        self.sum_tree = SumTree(self.total_segment_capacity)
        
        self.current_episode_images = []
        self.current_episode_vectors = []
        self.current_episode_actions = []
        self.current_episode_rewards = []
        self.current_episode_ik_fail = []
        self.current_episode_dones = []
        self.current_episode_global_id = None

    def push_first_obs(self, image, vector):
        self.clear_current_episode()
        self.current_episode_global_id = next(self.next_episode_id)
        image = np.transpose(image, (2, 0, 1))
        self.current_episode_images.append(image.astype(np.uint8))
        vector = np.asarray(vector)
        self.current_episode_vectors.append(vector.astype(np.float32))

    def push_step(self, action, reward, done, next_image, next_vector, ik_fail):
        next_image = np.transpose(next_image, (2, 0, 1))
        
        self.current_episode_actions.append(action.astype(np.float32))
        self.current_episode_rewards.append(np.float32(reward))
        self.current_episode_ik_fail.append(bool(ik_fail))
        self.current_episode_dones.append(bool(done))
        self.current_episode_images.append(next_image.astype(np.uint8))
        next_vector = np.asarray(next_vector)
        self.current_episode_vectors.append(next_vector.astype(np.float32))

    def store_episode(self, world_model: WorldModel):
        if not self.current_episode_actions or len(self.current_episode_actions) < self.seq_len:
            self.clear_current_episode()
            self.current_episode_global_id = None
            return
            
        assert self.current_episode_global_id is not None
        
        if len(self.actions) == self.capacity:
            oldest_episode_id = self.episode_ids[0]
            if self.episode_segment_tree_indices:
                for tree_idx in self.episode_segment_tree_indices[0]:
                    self.sum_tree.update(tree_idx, 0.0)
            self.ep_id_to_deque_idx.pop(oldest_episode_id, None)

        images = np.array(self.current_episode_images)
        vectors = np.array(self.current_episode_vectors)
        actions = np.array(self.current_episode_actions)
        rewards = np.array(self.current_episode_rewards)
        ik_fail_flags = np.array(self.current_episode_ik_fail)
        self.ik_fail.append(ik_fail_flags)
        self.dones.append(np.array(self.current_episode_dones))
        self.episode_ids.append(self.current_episode_global_id)

        self.images.append(images)
        self.vectors.append(vectors)
        self.actions.append(actions)
        self.rewards.append(rewards)

        for i in range(len(self.episode_ids)):
            self.ep_id_to_deque_idx[self.episode_ids[i]] = i

        ep_len_act = len(self.current_episode_actions)
        segment_tree_indices_for_this_episode = []
        
        n_neighbors = 25
        if ep_len_act > n_neighbors:
            with torch.no_grad():
                img_tensor = torch.tensor(images, dtype=torch.float32, device=self.device)
                img_features = world_model.img_encoder(img_tensor).cpu().numpy()
            vec_features = vectors

            clf_img = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=False)
            clf_img.fit(img_features)
            raw_img_scores = -clf_img.negative_outlier_factor_
            clf_vec = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=False)
            clf_vec.fit(vec_features)
            raw_vec_scores = -clf_vec.negative_outlier_factor_
        
        def normalize_scores(scores):
            total_budget = 1.0 * ep_len_act
            sum_raw = np.sum(scores)
            if sum_raw < 1e-6: sum_raw = 1.0
            step_anomaly_factors = (scores / sum_raw) * total_budget
            return step_anomaly_factors
        norm_img_scores = normalize_scores(raw_img_scores)
        norm_vec_scores = normalize_scores(raw_vec_scores)
        w_img = 0.3 
        w_vec = 0.7
        step_lof_factors = w_img * norm_img_scores + w_vec * norm_vec_scores
        self.episode_lof_scores.append(step_lof_factors)
        
        ik_fail_indices = np.where(ik_fail_flags == True)[0]
        fail_segment_starts = set()
        for fail_idx in ik_fail_indices:
            start = max(0, fail_idx - self.seq_len + 1)
            end = min(fail_idx + 1, ep_len_act - self.seq_len + 1)
            for s in range(start, end):
                fail_segment_starts.add(s)

        for start_idx_act in range(0, ep_len_act - self.seq_len + 1):
            end_idx_obs = start_idx_act + self.seq_len
            current_segment_lof = np.mean(step_lof_factors[start_idx_act : end_idx_obs])
            if start_idx_act in fail_segment_starts:
                p_init = self.ik_fail_priority_bonus
            else:
                p_init = 0.0
            p_init += self.lof_priority_weight * min(max(0.0, current_segment_lof - 1.0), 5.0)
            initial_priority = (p_init + self.epsilon) ** self.alpha
            data_to_store = (self.current_episode_global_id, start_idx_act)
            tree_idx = self.sum_tree.add(initial_priority, data_to_store)
            segment_tree_indices_for_this_episode.append(tree_idx)
        
        self.episode_segment_tree_indices.append(segment_tree_indices_for_this_episode)
        self.clear_current_episode()
        self.current_episode_global_id = None

    def clear_current_episode(self):
        self.current_episode_images = []
        self.current_episode_vectors = []
        self.current_episode_actions = []
        self.current_episode_rewards = []
        self.current_episode_ik_fail = []
        self.current_episode_dones = []

    def _get_segment(self, global_ep_id, start_idx_act):
        deque_idx = self.ep_id_to_deque_idx.get(global_ep_id)
        if deque_idx is None:
            raise IndexError(f"Episode ID {global_ep_id} not found in buffer.")

        ep_len_act = len(self.actions[deque_idx])
        end_idx_act = start_idx_act + self.seq_len
        
        start_idx_obs = start_idx_act
        end_idx_obs = start_idx_act + self.seq_len + 1
        
        if start_idx_act < 0 or end_idx_act > ep_len_act:
             raise IndexError(f"Invalid segment index: ep_id={global_ep_id}, start_idx_act={start_idx_act}, ep_len_act={ep_len_act}")
        
        ik_fail_segment_flags = self.ik_fail[deque_idx][start_idx_act : start_idx_act + self.seq_len]

        return (
            self.images[deque_idx][start_idx_obs:end_idx_obs].astype(np.float32),
            self.vectors[deque_idx][start_idx_obs:end_idx_obs],
            self.actions[deque_idx][start_idx_act:end_idx_act],
            self.rewards[deque_idx][start_idx_act:end_idx_act][..., np.newaxis],
            ik_fail_segment_flags
        )

    def sample(self, batch_size):
        if self.sum_tree.n_entries() == 0:
            raise IndexError("Replay buffer (SumTree) is empty.")
            
        batch_imgs = np.empty((batch_size, self.seq_len + 1, *self.img_shape), dtype=np.float32)
        batch_vecs = np.empty((batch_size, self.seq_len + 1, self.vec_dim), dtype=np.float32)
        batch_acts = np.empty((batch_size, self.seq_len, self.action_dim), dtype=np.float32)
        batch_rewards = np.empty((batch_size, self.seq_len, 1), dtype=np.float32)
        batch_ik_fail_flags = np.empty((batch_size, self.seq_len), dtype=bool)

        tree_indices = np.empty(batch_size, dtype=np.int32)
        is_weights = np.empty(batch_size, dtype=np.float32)
        
        self.beta = min(1.0, self.beta + (1.0 - self.beta_start) / self.beta_frames * self.frame_count)
        self.frame_count += batch_size

        total_priority = self.sum_tree.total_priority()
        min_priority_val = self.sum_tree.min_priority()
        
        if total_priority <= 0 or self.sum_tree.n_entries() <= 0:
            max_weight = 1.0
        else:
            prob_min = min_priority_val / total_priority
            if prob_min <= 0:
                prob_min = 1e-9
            max_weight = (prob_min * self.sum_tree.n_entries()) ** (-self.beta)

        for i in range(batch_size):
            s = random.uniform(0, total_priority)
            tree_idx, priority, (global_ep_id, start_idx_act) = self.sum_tree.sample(s)
            
            p_sample = priority / total_priority
            if p_sample <= 0:
                is_weights[i] = 0.0 
            else:
                is_weights[i] = (p_sample * self.sum_tree.n_entries()) ** (-self.beta)
                is_weights[i] /= max_weight
            
            imgs, vecs, acts, rewards, ik_fail_flags = self._get_segment(global_ep_id, start_idx_act)
            
            batch_imgs[i] = imgs.astype(np.float32)
            batch_vecs[i] = vecs
            batch_acts[i] = acts
            batch_rewards[i] = rewards
            batch_ik_fail_flags[i] = ik_fail_flags

            tree_indices[i] = tree_idx

        return (
            torch.tensor(batch_imgs, device=self.device),
            torch.tensor(batch_vecs, device=self.device),
            torch.tensor(batch_acts, device=self.device),
            torch.tensor(batch_rewards, device=self.device),
            torch.tensor(batch_ik_fail_flags, device=self.device),
            tree_indices,
            torch.tensor(is_weights, device=self.device),
            total_priority
        )
        
    
    def update_priorities(self, tree_indices, wm_losses):
        alpha_loss_weight = 1.0
        gamma_ik_weight = 1.0
        lof_priority_weight = 1.0

        for i in range(len(tree_indices)):
            tree_idx = tree_indices[i]
            loss = wm_losses[i].item()
            
            data_idx = tree_idx - self.sum_tree.capacity + 1
            data_tuple = self.sum_tree.data[data_idx]
            if not isinstance(data_tuple, tuple): continue
            global_ep_id, start_idx_act = data_tuple
            deque_idx = self.ep_id_to_deque_idx.get(global_ep_id)
            if deque_idx is None: continue
            
            ik_flags = self.ik_fail[deque_idx]
            end_idx = min(start_idx_act + self.seq_len, len(ik_flags))
            contains_ik_fail = np.any(ik_flags[start_idx_act : end_idx])

            stored_lof_seq = self.episode_lof_scores[deque_idx]
            end_idx_obs = min(start_idx_act + self.seq_len, len(stored_lof_seq))
            segment_lof_val = np.mean(stored_lof_seq[start_idx_act : end_idx_obs])
            lof_bonus = max(0.0, segment_lof_val - 1.0)

            new_priority = (alpha_loss_weight * loss + 
                            lof_priority_weight * lof_bonus +
                            gamma_ik_weight * (1.0 if contains_ik_fail else 0.0) + 
                            self.epsilon) ** self.alpha
            
            self.sum_tree.update(tree_idx, new_priority)

    def __len__(self):
        return self.sum_tree.n_entries()