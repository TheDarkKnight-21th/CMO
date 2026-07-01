import numpy as np
from collections import deque
import torch

class PerPromptStatTracker:
    def __init__(self, global_std=False):
        self.global_std = global_std
        self.stats = {}
        self.history_prompts = set()

    def update(self, prompts, rewards, reward_tensors=None, type='grpo', attr_binding=False,temperature=0.5):
        prompts = np.array(prompts)
        rewards = np.array(rewards, dtype=np.float64)

        if reward_tensors is not None:

            reward_tensors = np.array(reward_tensors, dtype=np.float64)
            type='gdpo'

            if reward_tensors.ndim == 3:
                B, _ , T = reward_tensors.shape
                advantages = np.zeros((B, T), dtype=np.float64)
            else:
                advantages = np.empty_like(rewards, dtype=np.float64) * 0.0
        else:
             advantages = np.empty_like(rewards, dtype=np.float64) * 0.0

        unique = np.unique(prompts)
        # advantages = np.empty_like(rewards)*0.0
        for prompt in unique:
            prompt_rewards = rewards[prompts == prompt]
            if prompt not in self.stats:
                self.stats[prompt] = []
            self.stats[prompt].extend(prompt_rewards)
            self.history_prompts.add(hash(prompt))  # Add hash of prompt to history_prompts
        for prompt in unique:
            self.stats[prompt] = np.stack(self.stats[prompt])
            prompt_rewards = rewards[prompts == prompt]  # Fix: Recalculate prompt_rewards for each prompt
            mean = np.mean(self.stats[prompt], axis=0, keepdims=True)
            if self.global_std:
                std = np.std(rewards, axis=0, keepdims=True) + 1e-4  # Use global std of all rewards
            else:
                std = np.std(self.stats[prompt], axis=0, keepdims=True) + 1e-4
            if type=='grpo':
                advantages[prompts == prompt] = (prompt_rewards - mean) / std
            elif type == 'gdpo':
                if reward_tensors is None:
                    raise ValueError("GDPO requires 'reward_tensors' (detailed attributes), but it is None.")
             
                # -----------------------------------------------------------
                # group_tensors Shape: (GroupSize, Time, Attributes)
                # -----------------------------------------------------------
                group_tensors = reward_tensors[prompts == prompt] 

                pad_value = -1.0
                mask = (group_tensors != pad_value).astype(np.float64)

                valid_count = mask.sum(axis=0, keepdims=True)
                valid_count = np.maximum(valid_count, 1.0) 

                masked_val = group_tensors * mask
                mean = masked_val.sum(axis=0, keepdims=True) / valid_count

                variance = ((group_tensors - mean) ** 2 * mask).sum(axis=0, keepdims=True) / valid_count
                std = np.sqrt(variance + 1e-8)

                # Normalization
                # shape: (Group, Rewards, Time)
                adv_decoupled = ((group_tensors - mean) / std) * mask

                G, K, T = group_tensors.shape
                
                weights = np.ones((1, K, 1))

                is_clean_data = not (group_tensors == -1.0).any()

                if attr_binding and is_clean_data:
                    if T > 1:
                        r_proxy = group_tensors.mean(axis=2) # (G, K)
                    else:
                        r_proxy = group_tensors.squeeze(axis=2)

                    proxy_std = r_proxy.std(axis=0)
                    valid_std_mask = (proxy_std > 1e-6)

                    # Pearson Correlation Matrix (K, K)
                    with np.errstate(invalid='ignore'):
                        corr_matrix = np.corrcoef(r_proxy, rowvar=False)
                    
                    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
                    
                    #  Binding Score Computataion
                    sum_corr = corr_matrix.sum(axis=1) - 1.0
                    avg_binding = sum_corr / (K - 1) 

                    avg_binding[~valid_std_mask] = 1.0 

                    # Weighting
                    raw_weight = 1.0 - avg_binding
                    
                    temperature_ = temperature
                    exp_w = np.exp(raw_weight / temperature_)
                    valid_weights = exp_w / exp_w.sum()
                    
                    # Shape adjustment for broadcasting
                    weights = valid_weights.reshape(1, K, 1)

                    adv_sum = (adv_decoupled * weights).sum(axis=1)
                else:

                    adv_sum = adv_decoupled.sum(axis=1) 

                advantages[prompts == prompt] = adv_sum


            elif type=='rwr':
                # advantages[prompts == prompt] = (prompt_rewards - mean) / std
                advantages[prompts == prompt] = prompt_rewards
                # advantages[prompts == prompt] = torch.softmax(torch.tensor(prompt_rewards), dim=0).numpy()
            elif type=='sft':
                advantages[prompts == prompt] = (torch.tensor(prompt_rewards) == torch.max(torch.tensor(prompt_rewards))).float().numpy()
            elif type=='dpo':
                # Get the advantages of the current prompt
                prompt_advantages = torch.tensor(prompt_rewards)
                # Find the indices of the maximum and minimum values
                max_idx = torch.argmax(prompt_advantages)
                min_idx = torch.argmin(prompt_advantages)
                # If all rewards in a group are the same
                if max_idx == min_idx:
                    min_idx = 0
                    max_idx = 1
                result = torch.zeros_like(prompt_advantages).float()
                # Set the maximum index to 1, minimum index to -1
                result[max_idx] = 1.0
                result[min_idx] = -1.0
                advantages[prompts == prompt] = result.numpy()
                # print("reward difference one group", prompt_advantages[max_idx]-prompt_advantages[min_idx])
        if type=='gdpo':
            batch_mean = np.mean(advantages)
            batch_std = np.std(advantages) + 1e-8
            
            # 정규화 적용
            advantages = (advantages - batch_mean) / batch_std

        return advantages

    def get_stats(self):
        avg_group_size = sum(len(v) for v in self.stats.values()) / len(self.stats) if self.stats else 0
        history_prompts = len(self.history_prompts)
        return avg_group_size, history_prompts
    
    def clear(self):
        self.stats = {}

def main():
    tracker = PerPromptStatTracker()
    prompts = ['a', 'b', 'a', 'c', 'b', 'a']
    rewards = [1, 2, 3, 4, 5, 6]
    advantages = tracker.update(prompts, rewards)
    print("Advantages:", advantages)
    avg_group_size, history_prompts = tracker.get_stats()
    print("Average Group Size:", avg_group_size)
    print("History Prompts:", history_prompts)
    tracker.clear()
    print("Stats after clear:", tracker.stats)

if __name__ == "__main__":
    main()