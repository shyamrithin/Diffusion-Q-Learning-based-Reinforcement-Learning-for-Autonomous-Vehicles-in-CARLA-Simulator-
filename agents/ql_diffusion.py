# ==========================================================
# agents/ql_diffusion.py
# Diffusion Q-Learning Agent v14 — Entropy Regularised
#                                  + Reward Scaling
#
# Changes from v11/v13:
#   - REWARD SCALING added (self.reward_scale, default 0.1)
#     Rationale: with gamma=0.99 and reward in [-15,15],
#     the value function magnitude reaches ~15/(1-0.99)
#     = 1500. MSE critic loss on Q-values of that scale
#     produces loss numbers in the thousands even when
#     the critic is only a few percent off. This is value
#     SCALE, not instability. Scaling rewards by 0.1 brings
#     Q-values to ~150 and critic loss into the tens/
#     hundreds, giving gentler TD targets and smoother
#     gradients. Applied ONLY inside the TD target — the
#     logged reward and reward function are unchanged so
#     reward curves stay interpretable.
#
#   - CONSERVATIVE (CQL) TERM REMOVED. CQL is an offline-RL
#     method; in the online setting (expanding replay
#     distribution) its suppression of OOD-action Q-values
#     fought the bellman update and made the critic loss
#     diverge worse (v13). Reverted.
#
#   - All v3 entropy-regularisation retained:
#     actor_loss = BC + eta*QL - alpha*entropy
#     auto-tuning alpha (SAC-style)
#     entropy-guided action selection
#
#   - All v2 stability fixes retained:
#     Separate critic LR (2x actor)
#     train_critic_only() pre-training
#     reset_critic_target() periodic reset
#
# CARLA 0.9.16 | Python 3.10
# ==========================================================

import copy
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import logging

logger = logging.getLogger(__name__)

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
# DIFFUSION Q-LEARNING AGENT v14
# ==========================================================
class Diffusion_QL(object):
    """
    Entropy-Regularised Online Diffusion Q-Learning
    with reward scaling for value-scale control.

    Novel contribution:
      Combines SAC-style entropy regularisation with
      diffusion policy's multimodal action distribution
      for online autonomous-driving RL.

    Entropy is applied in two ways:
      1. Action selection: diverse candidates bonus
      2. Actor loss: entropy maximisation term

    Reward scaling:
      Rewards are scaled by self.reward_scale inside the
      TD target only, keeping value magnitudes (and thus
      critic-loss magnitudes and TD-target gradients) in
      a numerically gentle range. The actor's normalised
      Q-loss ratio is scale-invariant, so it is unaffected.
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
        # Entropy regularisation parameters
        alpha              = 0.2,    # entropy coefficient
        auto_alpha         = True,   # auto-tune alpha
        target_entropy     = None,   # auto-computed if None
        # Reward scaling parameter
        reward_scale       = 0.1,    # scales TD-target reward
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

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=lr
        )

        self.ema              = EMA(ema_decay)
        self.ema_model        = copy.deepcopy(self.actor)
        self.step_start_ema   = step_start_ema
        self.update_ema_every = update_ema_every

        # --------------------------------------------------
        # Critic — separate LR (2x actor)
        # --------------------------------------------------
        self.critic        = Critic(
            state_dim, action_dim
        ).to(device)
        self.critic_target = copy.deepcopy(self.critic)

        critic_lr = lr * 2
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr
        )

        # --------------------------------------------------
        # Reward scaling (value-scale control)
        # --------------------------------------------------
        self.reward_scale = reward_scale

        # --------------------------------------------------
        # Entropy coefficient (alpha) — auto-tuning
        # --------------------------------------------------
        self.auto_alpha = auto_alpha

        if target_entropy is None:
            self.target_entropy = -float(action_dim)
        else:
            self.target_entropy = target_entropy

        if auto_alpha:
            self.log_alpha = torch.zeros(
                1,
                requires_grad=True,
                device=device
            )
            self.alpha = self.log_alpha.exp().item()
            self.alpha_optimizer = torch.optim.Adam(
                [self.log_alpha], lr=lr
            )
        else:
            self.alpha = alpha

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

        logger.info(f"[DQL] Actor LR      : {lr}")
        logger.info(f"[DQL] Critic LR     : {critic_lr}")
        logger.info(f"[DQL] Alpha         : {self.alpha}")
        logger.info(f"[DQL] Auto alpha    : {auto_alpha}")
        logger.info(f"[DQL] Reward scale  : {self.reward_scale}")
        logger.info(
            f"[DQL] Target entropy: {self.target_entropy}"
        )

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
        Pre-train critic WITHOUT actor updates.
        Gives critic stable Q-value foundation
        before BC/QL/entropy competition begins.
        Reward is scaled in the TD target.
        """
        state, action, next_state, reward, not_done = \
            replay_buffer.sample(batch_size, self.device)

        with torch.no_grad():
            next_action      = self.ema_model(next_state)
            tq1, tq2         = self.critic_target(
                next_state, next_action
            )
            target_q = (
                self.reward_scale * reward
                + not_done * self.discount *
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
        """Hard reset critic target — prevents divergence."""
        self.critic_target = copy.deepcopy(self.critic)
        logger.info("[DQL] Critic target reset")

    # ----------------------------------------------------------
    def _compute_entropy(self, state, n_samples=10):
        """
        Compute entropy of diffusion policy at state.
        Samples n_samples actions and measures variance.
        High variance = high entropy = diverse policy.
        """
        state_rpt = torch.repeat_interleave(
            state, repeats=n_samples, dim=0
        )

        with torch.no_grad():
            sampled = self.actor.sample(state_rpt)

        batch_size = state.shape[0]
        sampled    = sampled.view(
            batch_size, n_samples, self.action_dim
        )

        entropy = sampled.std(dim=1).mean(dim=1)
        return entropy

    # ----------------------------------------------------------
    def train(self, replay_buffer, iterations,
              batch_size=100, log_writer=None):
        """
        Full update with entropy regularisation.
        Reward is scaled in the TD target only.

        Steps:
          1. Critic TD backup (scaled reward)
          2. Compute diffusion entropy
          3. Actor: BC + QL - alpha*entropy
          4. Auto-tune alpha if enabled
          5. EMA + soft target update
        """
        metric = {
            "bc_loss"    : [],
            "ql_loss"    : [],
            "actor_loss" : [],
            "critic_loss": [],
            "entropy"    : [],
            "alpha"      : [],
        }

        for _ in range(iterations):

            state, action, next_state, reward, not_done = \
                replay_buffer.sample(batch_size, self.device)

            # ------------------------------------------
            # CRITIC UPDATE — scaled-reward TD target
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
                next_action = self.ema_model(next_state)
                tq1, tq2   = self.critic_target(
                    next_state, next_action
                )
                target_q   = torch.min(tq1, tq2)

            target_q = (
                self.reward_scale * reward
                + not_done * self.discount * target_q
            ).detach()

            critic_loss = (
                F.mse_loss(current_q1, target_q) +
                F.mse_loss(current_q2, target_q)
            )

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            if self.grad_norm > 0:
                nn.utils.clip_grad_norm_(
                    self.critic.parameters(),
                    max_norm=self.grad_norm,
                    norm_type=2
                )
            self.critic_optimizer.step()

            # ------------------------------------------
            # COMPUTE DIFFUSION ENTROPY
            # ------------------------------------------
            state_rpt_ent = torch.repeat_interleave(
                state, repeats=10, dim=0
            )
            sampled_actions = self.actor.sample(
                state_rpt_ent
            )
            sampled_actions = sampled_actions.view(
                batch_size, 10, self.action_dim
            )
            entropy = sampled_actions.std(
                dim=1
            ).mean()

            # ------------------------------------------
            # ACTOR UPDATE
            # L = L_bc + eta*L_ql - alpha*entropy
            # (q_loss is a normalised ratio → scale-invariant)
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

            actor_loss = (
                bc_loss
                + self.eta * q_loss
                - self.alpha * entropy
            )

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.grad_norm > 0:
                nn.utils.clip_grad_norm_(
                    self.actor.parameters(),
                    max_norm=self.grad_norm,
                    norm_type=2
                )
            self.actor_optimizer.step()

            # ------------------------------------------
            # AUTO-TUNE ALPHA
            # ------------------------------------------
            if self.auto_alpha:
                alpha_loss = -(
                    self.log_alpha.exp() * (
                        entropy.detach() +
                        self.target_entropy
                    )
                )
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                self.alpha = self.log_alpha.exp().item()

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
            metric["entropy"].append(entropy.item())
            metric["alpha"].append(self.alpha)

        if self.lr_decay:
            self.actor_lr_scheduler.step()
            self.critic_lr_scheduler.step()

        return metric

    # ----------------------------------------------------------
    def sample_action(self, state):
        """
        Entropy-guided action selection.
        score = Q(s,a) + alpha * diversity_bonus
        """
        state_t   = torch.FloatTensor(
            state.reshape(1, -1)
        ).to(self.device)
        state_rpt = torch.repeat_interleave(
            state_t, repeats=50, dim=0
        )

        with torch.no_grad():
            actions = self.actor.sample(state_rpt)

            q_values = self.critic_target.q_min(
                state_rpt, actions
            ).flatten()

            action_mean = actions.mean(
                dim=0, keepdim=True
            )
            action_std  = (
                actions.std(dim=0, keepdim=True) + 1e-6
            )
            diversity   = (
                (actions - action_mean) / action_std
            ).norm(dim=1)

            score = q_values + self.alpha * diversity

            probs = F.softmax(
                score / self.action_temperature, dim=0
            )
            idx   = torch.multinomial(
                probs, num_samples=1
            )

        return actions[idx].cpu().data.numpy().flatten()

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
        if self.auto_alpha:
            torch.save(
                self.log_alpha,
                f"{dir}/log_alpha{suffix}.pth"
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
        if self.auto_alpha:
            alpha_path = (
                f"{dir}/log_alpha{suffix}.pth"
            )
            if os.path.exists(alpha_path):
                self.log_alpha = torch.load(
                    alpha_path,
                    map_location=self.device
                )
                self.alpha = (
                    self.log_alpha.exp().item()
                )