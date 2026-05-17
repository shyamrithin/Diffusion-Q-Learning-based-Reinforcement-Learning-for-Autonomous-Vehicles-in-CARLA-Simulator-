# ==========================================================
# agents/ql_diffusion.py
# Diffusion Q-Learning Agent v2
#
# Changes from v1:
#   - Separate LR for critic vs actor
#     critic_lr = lr * 10 (learns faster)
#     actor_lr  = lr      (stays stable)
#   - train_critic_only() method added
#     Pre-trains critic before BC/QL competition
#   - Periodic critic target reset support
#     agent.reset_critic_target() method added
#   - action_temperature in sample_action fixed
#   - weights_only=True in load_model (PyTorch 2.x)
#
# CARLA 0.9.16 | Python 3.10
# ==========================================================

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.logger import logger

from agents.diffusion import Diffusion
from agents.model     import MLP
from agents.helpers   import EMA


# ==========================================================
# CRITIC
# ==========================================================
class Critic(nn.Module):
    """
    Twin Q-network critic (clipped double Q-learning).
    Takes (state, action) → two independent Q-values.
    Uses min(Q1, Q2) as target to reduce overestimation.
    """

    def __init__(self, state_dim, action_dim,
                 hidden_dim=256):
        super().__init__()

        def _mlp():
            return nn.Sequential(
                nn.Linear(
                    state_dim + action_dim, hidden_dim
                ),
                nn.Mish(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Mish(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Mish(),
                nn.Linear(hidden_dim, 1),
            )

        self.q1_model = _mlp()
        self.q2_model = _mlp()

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x), self.q2_model(x)

    def q1(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x)

    def q_min(self, state, action):
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)


# ==========================================================
# DIFFUSION Q-LEARNING AGENT
# ==========================================================
class Diffusion_QL(object):
    """
    Online Diffusion Q-Learning agent v2.

    Key fixes vs v1:
      1. Separate LR for critic (10x higher than actor)
         Critic needs to learn Q-values faster than
         actor learns policy — prevents divergence

      2. train_critic_only() for pre-training phase
         Critic gets 20k steps of pure TD learning
         before BC/QL competition begins

      3. reset_critic_target() for periodic reset
         Prevents target network error accumulation
         over long training runs

    Training objective:
      L_actor = L_bc + eta * L_ql
      L_bc    = diffusion denoising loss
      L_ql    = -Q(s, a_new) normalised by |Q|
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        device,
        discount,
        tau,
        max_q_backup       = False,
        eta                = 1.0,
        beta_schedule      = "linear",
        n_timesteps        = 100,
        ema_decay          = 0.995,
        step_start_ema     = 1000,
        update_ema_every   = 5,
        lr                 = 3e-4,
        lr_decay           = False,
        lr_maxt            = 1000,
        grad_norm          = 1.0,
        action_temperature = 0.1,
    ):
        # --------------------------------------------------
        # Actor — diffusion policy
        # --------------------------------------------------
        self.model = MLP(
            state_dim  = state_dim,
            action_dim = action_dim,
            device     = device,
        )
        self.actor = Diffusion(
            state_dim     = state_dim,
            action_dim    = action_dim,
            model         = self.model,
            max_action    = max_action,
            beta_schedule = beta_schedule,
            n_timesteps   = n_timesteps,
        ).to(device)

        # Actor uses base LR — stays stable
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=lr
        )

        self.ema              = EMA(ema_decay)
        self.ema_model        = copy.deepcopy(self.actor)
        self.step_start_ema   = step_start_ema
        self.update_ema_every = update_ema_every

        # --------------------------------------------------
        # Critic — twin Q-networks
        # 🔥 Critic uses 10x higher LR than actor
        #    Critic needs to learn Q-values quickly
        #    Actor needs to stay stable
        # --------------------------------------------------
        self.critic        = Critic(
            state_dim, action_dim
        ).to(device)
        self.critic_target = copy.deepcopy(self.critic)

        # 🔥 Separate critic LR
        critic_lr = lr * 10
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr
        )

        logger.info(
            f"[DQL] Actor LR  : {lr}"
        )
        logger.info(
            f"[DQL] Critic LR : {critic_lr} (10x actor)"
        )

        # --------------------------------------------------
        # LR schedulers (optional)
        # --------------------------------------------------
        self.lr_decay = lr_decay
        if lr_decay:
            self.actor_lr_scheduler = CosineAnnealingLR(
                self.actor_optimizer,
                T_max=lr_maxt, eta_min=0.0
            )
            self.critic_lr_scheduler = CosineAnnealingLR(
                self.critic_optimizer,
                T_max=lr_maxt, eta_min=0.0
            )

        # --------------------------------------------------
        # Hyperparameters
        # --------------------------------------------------
        self.state_dim          = state_dim
        self.action_dim         = action_dim
        self.max_action         = max_action
        self.discount           = discount
        self.tau                = tau
        self.eta                = eta
        self.grad_norm          = grad_norm
        self.max_q_backup       = max_q_backup
        self.device             = device
        self.action_temperature = action_temperature
        self.step               = 0

    # ----------------------------------------------------------
    def step_ema(self):
        if self.step < self.step_start_ema:
            return
        self.ema.update_model_average(
            self.ema_model, self.actor
        )

    # ----------------------------------------------------------
    def train_critic_only(self, replay_buffer,
                          batch_size=256):
        """
        🔥 Pre-train critic WITHOUT actor updates.

        Called during the first PRE_TRAIN_CRITIC_STEPS
        steps to give critic a stable Q-value foundation
        before BC/QL competition begins.

        Only runs:
          - Critic forward pass
          - TD Bellman backup
          - Critic gradient step
          - Soft target update

        Actor is NOT updated — stays at random init.
        This gives critic time to learn the reward
        landscape before being pulled by BC/QL loss.
        """
        state, action, next_state, reward, not_done = \
            replay_buffer.sample(batch_size, self.device)

        with torch.no_grad():
            next_action      = self.ema_model(next_state)
            tq1, tq2         = self.critic_target(
                next_state, next_action
            )
            target_q = (
                reward + not_done * self.discount *
                torch.min(tq1, tq2)
            )

        current_q1, current_q2 = self.critic(
            state, action
        )
        critic_loss = (
            F.mse_loss(current_q1, target_q) +
            F.mse_loss(current_q2, target_q)
        )

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.grad_norm > 0:
            nn.utils.clip_grad_norm_(
                self.critic.parameters(),
                max_norm=self.grad_norm
            )
        self.critic_optimizer.step()

        # Soft update target
        for param, target_param in zip(
            self.critic.parameters(),
            self.critic_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data +
                (1 - self.tau) * target_param.data
            )

        return critic_loss.item()

    # ----------------------------------------------------------
    def reset_critic_target(self):
        """
        🔥 Hard reset of critic target network.

        Called periodically (every 10k steps) to prevent
        target network error accumulation over long runs.
        Copies current critic weights to target.
        """
        self.critic_target = copy.deepcopy(self.critic)
        logger.info(
            "[DQL] Critic target reset"
        )

    # ----------------------------------------------------------
    def train(self, replay_buffer, iterations,
              batch_size=100, log_writer=None):
        """
        Run iterations gradient steps.

        Full update:
          1. Critic TD backup
          2. Actor BC + QL loss
          3. EMA update
          4. Soft target update
        """
        metric = {
            "bc_loss"    : [],
            "ql_loss"    : [],
            "actor_loss" : [],
            "critic_loss": [],
        }

        for _ in range(iterations):

            state, action, next_state, reward, not_done = \
                replay_buffer.sample(batch_size, self.device)

            # ------------------------------------------
            # CRITIC UPDATE
            # ------------------------------------------
            current_q1, current_q2 = self.critic(
                state, action
            )

            if self.max_q_backup:
                ns_rpt   = torch.repeat_interleave(
                    next_state, repeats=10, dim=0
                )
                na_rpt   = self.ema_model(ns_rpt)
                tq1, tq2 = self.critic_target(
                    ns_rpt, na_rpt
                )
                tq1      = tq1.view(
                    batch_size, 10
                ).max(dim=1, keepdim=True)[0]
                tq2      = tq2.view(
                    batch_size, 10
                ).max(dim=1, keepdim=True)[0]
                target_q = torch.min(tq1, tq2)
            else:
                next_action      = self.ema_model(next_state)
                tq1, tq2         = self.critic_target(
                    next_state, next_action
                )
                target_q         = torch.min(tq1, tq2)

            target_q = (
                reward + not_done * self.discount * target_q
            ).detach()

            critic_loss = (
                F.mse_loss(current_q1, target_q) +
                F.mse_loss(current_q2, target_q)
            )

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            if self.grad_norm > 0:
                critic_grad_norms = \
                    nn.utils.clip_grad_norm_(
                        self.critic.parameters(),
                        max_norm=self.grad_norm,
                        norm_type=2
                    )
            self.critic_optimizer.step()

            # ------------------------------------------
            # ACTOR UPDATE
            # L = L_bc + eta * L_ql
            # ------------------------------------------
            bc_loss    = self.actor.loss(action, state)
            new_action = self.actor(state)

            q1_new, q2_new = self.critic(state, new_action)

            if np.random.uniform() > 0.5:
                q_loss = -q1_new.mean() / \
                    q2_new.abs().mean().detach()
            else:
                q_loss = -q2_new.mean() / \
                    q1_new.abs().mean().detach()

            actor_loss = bc_loss + self.eta * q_loss

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.grad_norm > 0:
                actor_grad_norms = \
                    nn.utils.clip_grad_norm_(
                        self.actor.parameters(),
                        max_norm=self.grad_norm,
                        norm_type=2
                    )
            self.actor_optimizer.step()

            # ------------------------------------------
            # TARGET NETWORK UPDATES
            # ------------------------------------------
            if self.step % self.update_ema_every == 0:
                self.step_ema()

            for param, target_param in zip(
                self.critic.parameters(),
                self.critic_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data +
                    (1 - self.tau) * target_param.data
                )

            self.step += 1

            metric["actor_loss"].append(
                actor_loss.item()
            )
            metric["bc_loss"].append(bc_loss.item())
            metric["ql_loss"].append(q_loss.item())
            metric["critic_loss"].append(
                critic_loss.item()
            )

        if self.lr_decay:
            self.actor_lr_scheduler.step()
            self.critic_lr_scheduler.step()

        return metric

    # ----------------------------------------------------------
    def sample_action(self, state):
        """
        Sample best action for given state.
        Draws 30 candidates from diffusion actor,
        scores with critic, selects via softmax.
        """
        state_t   = torch.FloatTensor(
            state.reshape(1, -1)
        ).to(self.device)
        state_rpt = torch.repeat_interleave(
            state_t, repeats=30, dim=0
        )

        with torch.no_grad():
            action  = self.actor.sample(state_rpt)
            q_value = self.critic_target.q_min(
                state_rpt, action
            ).flatten()

            probs = F.softmax(
                q_value / self.action_temperature, dim=0
            )
            idx   = torch.multinomial(
                probs, num_samples=1
            )

        return action[idx].cpu().data.numpy().flatten()

    # ----------------------------------------------------------
    def save_model(self, dir, id=None):
        suffix = f"_{id}" if id is not None else ""
        torch.save(
            self.actor.state_dict(),
            f"{dir}/actor{suffix}.pth"
        )
        torch.save(
            self.critic.state_dict(),
            f"{dir}/critic{suffix}.pth"
        )

    # ----------------------------------------------------------
    def load_model(self, dir, id=None):
        suffix = f"_{id}" if id is not None else ""
        self.actor.load_state_dict(
            torch.load(
                f"{dir}/actor{suffix}.pth",
                map_location=self.device,
                weights_only=True,
            )
        )
        self.critic.load_state_dict(
            torch.load(
                f"{dir}/critic{suffix}.pth",
                map_location=self.device,
                weights_only=True,
            )
        )