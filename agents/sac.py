# ==========================================================
# agents/sac.py
# Soft Actor-Critic (SAC) Agent
#
# SAC baseline for comparison with Diffusion-QL.
# Uses identical observation/action spaces as Diffusion-QL.
#
# Key properties:
#   - Off-policy (uses same replay buffer as Diffusion-QL)
#   - Entropy regularisation (automatic temperature tuning)
#   - Twin Q-networks (clipped double Q — same as Diffusion-QL)
#   - Gaussian policy with reparameterisation trick
#   - Continuous action space [throttle, steer, brake]
#
# Reference:
#   Haarnoja et al. "Soft Actor-Critic: Off-Policy Maximum
#   Entropy Deep Reinforcement Learning with a Stochastic
#   Actor." ICML 2018. arXiv:1801.01290
#
# CARLA 0.9.16 | PyTorch 2.x | Python 3.10
# ==========================================================

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_STD_MAX =  2
LOG_STD_MIN = -5
EPSILON     = 1e-6


# ==========================================================
# NETWORKS
# ==========================================================

class SACCritic(nn.Module):
    """
    Twin Q-networks for clipped double Q learning.
    Q1 and Q2 trained independently.
    min(Q1, Q2) used for policy gradient — reduces overestimation.
    """

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()

        # Q1
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Q2
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, action):
        """Returns Q1(s,a) and Q2(s,a)."""
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)

    def q_min(self, state, action):
        """Returns min(Q1, Q2) for conservative value estimate."""
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)


class SACActor(nn.Module):
    """
    Gaussian policy with squashed output.
    Outputs mean and log_std of action distribution.
    Actions squashed to [-1, 1] via tanh.
    Reparameterisation trick enables gradient flow.
    """

    def __init__(self, state_dim, action_dim,
                 hidden_dim=256, max_action=1.0):
        super().__init__()

        self.max_action = max_action

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.mean_layer    = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        """
        Returns (mean, log_std) of action distribution.
        log_std clamped to [LOG_STD_MIN, LOG_STD_MAX].
        """
        features = self.net(state)
        mean     = self.mean_layer(features)
        log_std  = self.log_std_layer(features)
        log_std  = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, state):
        """
        Sample action using reparameterisation trick.
        Returns (action, log_prob) for entropy calculation.
        action is squashed to [-max_action, max_action].
        """
        mean, log_std = self.forward(state)
        std           = log_std.exp()
        dist          = Normal(mean, std)

        # Reparameterisation trick: a = tanh(mu + eps*sigma)
        x_t    = dist.rsample()
        y_t    = torch.tanh(x_t)
        action = y_t * self.max_action

        # Log prob with tanh correction
        log_prob = dist.log_prob(x_t)
        log_prob -= torch.log(
            self.max_action * (1 - y_t.pow(2)) + EPSILON
        )
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob

    def get_action(self, state):
        """
        Get deterministic action for inference.
        Uses mean of distribution — no sampling noise.
        """
        mean, _ = self.forward(state)
        return torch.tanh(mean) * self.max_action


# ==========================================================
# SAC AGENT
# ==========================================================

class SAC:
    """
    Soft Actor-Critic agent.

    Identical interface to Diffusion_QL for fair comparison:
      sample_action(state) → action
      train(replay, iterations, batch_size) → metrics
      save_model(dir, id)
      load_model(dir, id)

    Key hyperparameters:
      alpha      : entropy temperature (auto-tuned)
      tau        : soft update coefficient
      discount   : reward discount factor
      lr         : learning rate (same as Diffusion-QL)
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        max_action   = 1.0,
        device       = torch.device("cpu"),
        discount     = 0.99,
        tau          = 0.005,
        lr           = 3e-5,
        grad_norm    = 0.1,
        hidden_dim   = 256,
        auto_entropy = True,
    ):
        self.device      = device
        self.discount    = discount
        self.tau         = tau
        self.grad_norm   = grad_norm
        self.action_dim  = action_dim
        self.auto_entropy= auto_entropy

        # Actor
        self.actor = SACActor(
            state_dim, action_dim, hidden_dim, max_action
        ).to(device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=lr
        )

        # Critics
        self.critic = SACCritic(
            state_dim, action_dim, hidden_dim
        ).to(device)
        self.critic_target = SACCritic(
            state_dim, action_dim, hidden_dim
        ).to(device)
        self.critic_target.load_state_dict(
            self.critic.state_dict()
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=lr
        )

        # Entropy temperature — auto tuned
        if auto_entropy:
            # Target entropy = -action_dim (heuristic)
            self.target_entropy = -action_dim
            self.log_alpha      = torch.zeros(
                1, requires_grad=True, device=device
            )
            self.alpha          = self.log_alpha.exp().item()
            self.alpha_optimizer = torch.optim.Adam(
                [self.log_alpha], lr=lr
            )
        else:
            self.alpha = 0.2   # fixed temperature

    def sample_action(self, state):
        """
        Sample action from policy for training.
        Adds exploration noise via Gaussian sampling.
        """
        state  = torch.FloatTensor(
            state.reshape(1, -1)
        ).to(self.device)
        action, _ = self.actor.sample(state)
        return action.cpu().data.numpy().flatten()

    def get_action(self, state):
        """
        Get deterministic action for inference.
        No sampling noise — uses policy mean.
        """
        state  = torch.FloatTensor(
            state.reshape(1, -1)
        ).to(self.device)
        action = self.actor.get_action(state)
        return action.cpu().data.numpy().flatten()

    def train(self, replay_buffer, iterations=1,
              batch_size=256):
        """
        SAC update step.

        1. Sample batch from replay buffer
        2. Update critic (Q-function regression)
        3. Update actor (maximize Q + entropy)
        4. Update entropy temperature (if auto)
        5. Soft update target networks

        Returns metrics dict for TensorBoard logging.
        """
        metrics = {
            "critic_loss" : [],
            "actor_loss"  : [],
            "alpha_loss"  : [],
            "alpha"       : [],
        }

        for _ in range(iterations):
            state, action, next_state, reward, not_done = \
                replay_buffer.sample(batch_size, self.device)

            with torch.no_grad():
                # Sample next action + log prob
                next_action, next_log_pi = \
                    self.actor.sample(next_state)

                # Target Q value
                q1_next, q2_next = self.critic_target(
                    next_state, next_action
                )
                q_next  = torch.min(q1_next, q2_next)
                q_target = reward + not_done * self.discount * (
                    q_next - self.alpha * next_log_pi
                )

            # Critic update
            q1, q2 = self.critic(state, action)
            critic_loss = (
                F.mse_loss(q1, q_target) +
                F.mse_loss(q2, q_target)
            )

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(
                self.critic.parameters(), self.grad_norm
            )
            self.critic_optimizer.step()

            # Actor update
            new_action, log_pi = self.actor.sample(state)
            q1_new, q2_new     = self.critic(state, new_action)
            q_new              = torch.min(q1_new, q2_new)

            actor_loss = (
                self.alpha * log_pi - q_new
            ).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.grad_norm
            )
            self.actor_optimizer.step()

            # Entropy temperature update
            if self.auto_entropy:
                alpha_loss = -(
                    self.log_alpha.exp() * (
                        log_pi + self.target_entropy
                    ).detach()
                ).mean()

                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                self.alpha = self.log_alpha.exp().item()

                metrics["alpha_loss"].append(
                    alpha_loss.item()
                )

            # Soft update target networks
            for param, target_param in zip(
                self.critic.parameters(),
                self.critic_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data +
                    (1 - self.tau) * target_param.data
                )

            metrics["critic_loss"].append(critic_loss.item())
            metrics["actor_loss"].append(actor_loss.item())
            metrics["alpha"].append(self.alpha)

        return metrics

    def save_model(self, checkpoint_dir, id):
        """Save actor and critic weights."""
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(
            self.actor.state_dict(),
            os.path.join(checkpoint_dir, f"sac_actor_{id}.pth")
        )
        torch.save(
            self.critic.state_dict(),
            os.path.join(checkpoint_dir, f"sac_critic_{id}.pth")
        )

    def load_model(self, checkpoint_dir, id):
        """Load actor and critic weights."""
        self.actor.load_state_dict(torch.load(
            os.path.join(
                checkpoint_dir, f"sac_actor_{id}.pth"
            ),
            weights_only=True,
        ))
        self.critic.load_state_dict(torch.load(
            os.path.join(
                checkpoint_dir, f"sac_critic_{id}.pth"
            ),
            weights_only=True,
        ))
        self.critic_target.load_state_dict(
            self.critic.state_dict()
        )
