# ==========================================================
# agents/ppo.py
# Proximal Policy Optimisation (PPO) Agent
#
# PPO baseline for comparison with Diffusion-QL and SAC.
# Uses identical observation/action spaces.
#
# Key differences from SAC/Diffusion-QL:
#   - ON-policy: no replay buffer
#   - Rollout buffer: collect N steps then update
#   - Clipped surrogate objective (epsilon=0.2)
#   - Value function baseline for variance reduction
#   - GAE (Generalised Advantage Estimation)
#   - Multiple epochs per rollout (PPO-epoch=10)
#
# Reference:
#   Schulman et al. "Proximal Policy Optimization
#   Algorithms." arXiv:1707.06347, 2017.
#
# CARLA 0.9.16 | PyTorch 2.x | Python 3.10
# ==========================================================

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

EPSILON = 1e-8


# ==========================================================
# ROLLOUT BUFFER
# ==========================================================

class RolloutBuffer:
    """
    On-policy rollout buffer for PPO.
    Stores N steps then computes returns + advantages.
    Cleared after each update — on-policy requirement.
    """

    def __init__(self, state_dim, action_dim,
                 buffer_size, device):
        self.buffer_size = buffer_size
        self.device      = device
        self.ptr         = 0
        self.size        = 0

        self.states      = np.zeros(
            (buffer_size, state_dim),  dtype=np.float32
        )
        self.actions     = np.zeros(
            (buffer_size, action_dim), dtype=np.float32
        )
        self.rewards     = np.zeros(
            (buffer_size, 1),          dtype=np.float32
        )
        self.values      = np.zeros(
            (buffer_size, 1),          dtype=np.float32
        )
        self.log_probs   = np.zeros(
            (buffer_size, 1),          dtype=np.float32
        )
        self.dones       = np.zeros(
            (buffer_size, 1),          dtype=np.float32
        )
        self.advantages  = np.zeros(
            (buffer_size, 1),          dtype=np.float32
        )
        self.returns     = np.zeros(
            (buffer_size, 1),          dtype=np.float32
        )

    def add(self, state, action, reward, value,
            log_prob, done):
        """Add one transition to buffer."""
        self.states[self.ptr]    = state
        self.actions[self.ptr]   = action
        self.rewards[self.ptr]   = reward
        self.values[self.ptr]    = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr]     = done
        self.ptr  = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def compute_returns_and_advantages(
        self, last_value, gamma=0.99, gae_lambda=0.95
    ):
        """
        Compute GAE advantages and discounted returns.
        last_value: V(s_T) for bootstrapping.
        """
        last_gae = 0.0
        for t in reversed(range(self.size)):
            if t == self.size - 1:
                next_value = last_value
                next_done  = 0.0
            else:
                next_value = self.values[t+1]
                next_done  = self.dones[t+1]

            delta = (
                self.rewards[t] +
                gamma * next_value * (1 - next_done) -
                self.values[t]
            )
            last_gae = (
                delta +
                gamma * gae_lambda * (1-next_done) * last_gae
            )
            self.advantages[t] = last_gae
            self.returns[t]    = last_gae + self.values[t]

        # Normalise advantages
        adv = self.advantages[:self.size]
        self.advantages[:self.size] = (
            (adv - adv.mean()) / (adv.std() + EPSILON)
        )

    def get_batches(self, batch_size):
        """Yield random minibatches from rollout."""
        indices = np.random.permutation(self.size)
        for start in range(0, self.size, batch_size):
            idx = indices[start:start+batch_size]
            yield (
                torch.FloatTensor(
                    self.states[idx]).to(self.device),
                torch.FloatTensor(
                    self.actions[idx]).to(self.device),
                torch.FloatTensor(
                    self.log_probs[idx]).to(self.device),
                torch.FloatTensor(
                    self.advantages[idx]).to(self.device),
                torch.FloatTensor(
                    self.returns[idx]).to(self.device),
            )

    def clear(self):
        """Clear buffer after PPO update."""
        self.ptr  = 0
        self.size = 0


# ==========================================================
# NETWORKS
# ==========================================================

class PPOActorCritic(nn.Module):
    """
    Shared backbone actor-critic network.
    Actor outputs Gaussian policy parameters.
    Critic outputs scalar state value V(s).
    Shared layers reduce parameters + improve data efficiency.
    """

    def __init__(self, state_dim, action_dim,
                 hidden_dim=256, max_action=1.0):
        super().__init__()

        self.max_action = max_action

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        # Actor head — outputs mean of Gaussian
        self.actor_mean    = nn.Linear(hidden_dim, action_dim)

        # Log std as learnable parameter (state-independent)
        self.actor_log_std = nn.Parameter(
            torch.zeros(action_dim)
        )

        # Critic head — outputs V(s)
        self.critic_value  = nn.Linear(hidden_dim, 1)

        # Initialise output layers with small weights
        nn.init.orthogonal_(self.actor_mean.weight,  gain=0.01)
        nn.init.orthogonal_(self.critic_value.weight, gain=1.0)

    def get_value(self, state):
        """Return V(s) for critic."""
        features = self.shared(state)
        return self.critic_value(features)

    def get_action_and_value(self, state, action=None):
        """
        Returns (action, log_prob, entropy, value).
        If action is None: samples new action.
        If action provided: evaluates given action.
        """
        features = self.shared(state)
        mean     = torch.tanh(
            self.actor_mean(features)
        ) * self.max_action

        std  = self.actor_log_std.exp().expand_as(mean)
        dist = Normal(mean, std)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(
            dim=-1, keepdim=True
        )
        entropy  = dist.entropy().sum(
            dim=-1, keepdim=True
        )
        value    = self.critic_value(features)

        return action, log_prob, entropy, value

    def get_action(self, state):
        """Deterministic action for inference."""
        features = self.shared(state)
        mean     = torch.tanh(
            self.actor_mean(features)
        ) * self.max_action
        return mean


# ==========================================================
# PPO AGENT
# ==========================================================

class PPO:
    """
    Proximal Policy Optimisation agent.

    Interface compatible with SAC and Diffusion-QL
    for fair comparison.

    Key difference from SAC/Diffusion-QL:
      Uses RolloutBuffer not ReplayBuffer.
      train_diffusion.py calls ppo.collect() each step
      then ppo.update() every ROLLOUT_SIZE steps.

    Hyperparameters:
      clip_epsilon : PPO clip ratio (0.2 standard)
      ppo_epochs   : update passes per rollout (10)
      vf_coef      : value function loss weight
      ent_coef     : entropy bonus weight
      gae_lambda   : GAE smoothing factor
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        max_action   = 1.0,
        device       = torch.device("cpu"),
        discount     = 0.99,
        lr           = 3e-4,      # PPO uses higher LR than SAC
        grad_norm    = 0.5,
        hidden_dim   = 256,
        clip_epsilon = 0.2,
        ppo_epochs   = 10,
        vf_coef      = 0.5,
        ent_coef     = 0.01,
        gae_lambda   = 0.95,
        rollout_size = 2048,
        batch_size   = 256,
    ):
        self.device       = device
        self.discount     = discount
        self.grad_norm    = grad_norm
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs   = ppo_epochs
        self.vf_coef      = vf_coef
        self.ent_coef     = ent_coef
        self.gae_lambda   = gae_lambda
        self.rollout_size = rollout_size
        self.batch_size   = batch_size

        # Actor-critic network
        self.policy = PPOActorCritic(
            state_dim, action_dim, hidden_dim, max_action
        ).to(device)

        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=lr, eps=1e-5
        )

        # Rollout buffer
        self.buffer = RolloutBuffer(
            state_dim, action_dim, rollout_size, device
        )

        self._step = 0

    def sample_action(self, state):
        """
        Sample action + collect rollout data.
        Returns (action, value, log_prob) for buffer storage.
        """
        state_t = torch.FloatTensor(
            state.reshape(1,-1)
        ).to(self.device)

        with torch.no_grad():
            action, log_prob, _, value = \
                self.policy.get_action_and_value(state_t)

        return (
            action.cpu().numpy().flatten(),
            value.cpu().numpy().flatten(),
            log_prob.cpu().numpy().flatten(),
        )

    def get_action(self, state):
        """Deterministic action for inference."""
        state_t = torch.FloatTensor(
            state.reshape(1,-1)
        ).to(self.device)
        with torch.no_grad():
            action = self.policy.get_action(state_t)
        return action.cpu().numpy().flatten()

    def collect(self, state, action, reward, value,
                log_prob, done):
        """Add transition to rollout buffer."""
        self.buffer.add(
            state, action, reward, value, log_prob, done
        )
        self._step += 1

    def ready_to_update(self):
        """True when rollout buffer is full."""
        return self._step >= self.rollout_size

    def update(self, last_state):
        """
        PPO update — called when rollout is full.

        1. Compute GAE advantages + returns
        2. Multiple epochs of minibatch updates
        3. Clipped surrogate loss
        4. Value function loss
        5. Entropy bonus
        6. Clear buffer
        """
        # Bootstrap value for last state
        last_state_t = torch.FloatTensor(
            last_state.reshape(1,-1)
        ).to(self.device)
        with torch.no_grad():
            last_value = self.policy.get_value(
                last_state_t
            ).cpu().numpy()

        self.buffer.compute_returns_and_advantages(
            last_value, self.discount, self.gae_lambda
        )

        metrics = {
            "actor_loss"  : [],
            "critic_loss" : [],
            "entropy"     : [],
            "total_loss"  : [],
        }

        for epoch in range(self.ppo_epochs):
            for (states, actions, old_log_probs,
                 advantages, returns) in \
                    self.buffer.get_batches(self.batch_size):

                _, new_log_probs, entropy, values = \
                    self.policy.get_action_and_value(
                        states, actions
                    )

                # Ratio for clipped surrogate
                ratio = (
                    new_log_probs - old_log_probs
                ).exp()

                # Clipped surrogate objective
                surr1 = ratio * advantages
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.clip_epsilon,
                    1.0 + self.clip_epsilon,
                ) * advantages

                actor_loss  = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(values, returns)
                ent_loss    = -entropy.mean()

                total_loss = (
                    actor_loss +
                    self.vf_coef  * critic_loss +
                    self.ent_coef * ent_loss
                )

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.grad_norm
                )
                self.optimizer.step()

                metrics["actor_loss"].append(
                    actor_loss.item()
                )
                metrics["critic_loss"].append(
                    critic_loss.item()
                )
                metrics["entropy"].append(
                    -ent_loss.item()
                )
                metrics["total_loss"].append(
                    total_loss.item()
                )

        # Clear buffer — on-policy requirement
        self.buffer.clear()
        self._step = 0

        return metrics

    def save_model(self, checkpoint_dir, id):
        """Save policy weights."""
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(
            self.policy.state_dict(),
            os.path.join(
                checkpoint_dir, f"ppo_policy_{id}.pth"
            )
        )

    def load_model(self, checkpoint_dir, id):
        """Load policy weights."""
        self.policy.load_state_dict(torch.load(
            os.path.join(
                checkpoint_dir, f"ppo_policy_{id}.pth"
            ),
            weights_only=True,
        ))
