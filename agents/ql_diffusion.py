# ==========================================================
# agents/ql_diffusion.py
# Diffusion Q-Learning Agent — RLCarla Edition
# Based on Diffusion-QL (Wang et al. 2022)
# Modified for online RL training in CARLA 0.9.16
#
# Changes from original:
#   - sample_action: softmax temperature added (fixes collapse)
#   - train: accepts RLCarla ReplayBuffer interface
#   - load_model: weights_only=True for PyTorch 2.x safety
#   - Removed offline RL assumptions throughout
#   - Full inline documentation added
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
    Takes (state, action) → two independent Q-value estimates.
    Training uses min(Q1, Q2) as target to reduce overestimation.
    """

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()

        def _mlp():
            return nn.Sequential(
                nn.Linear(state_dim + action_dim, hidden_dim),
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
        """Returns (Q1, Q2) for given state-action pair."""
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x), self.q2_model(x)

    def q1(self, state, action):
        """Returns Q1 only — used in policy gradient."""
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x)

    def q_min(self, state, action):
        """Returns min(Q1, Q2) — conservative estimate for target."""
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)


# ==========================================================
# DIFFUSION Q-LEARNING AGENT
# ==========================================================
class Diffusion_QL(object):
    """
    Online Diffusion Q-Learning agent for continuous control.

    The actor is a diffusion model — it learns a distribution
    over actions conditioned on state. At each training step:

      1. Critic is updated via Bellman backup (TD learning).
      2. Actor is updated via:
           L_actor = L_bc + eta * L_ql
         where:
           L_bc  = diffusion denoising loss (behaviour cloning term)
           L_ql  = -Q(s, a_new)  (Q-learning term)

    This balances staying close to observed behaviour (bc)
    with maximising expected return (ql).

    For action selection at inference, 50 candidate actions are
    sampled from the diffusion model and the one with highest
    Q-value is selected (with softmax temperature for exploration).
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        device,
        discount,
        tau,
        max_q_backup      = False,
        eta               = 1.0,
        beta_schedule     = "linear",
        n_timesteps       = 100,
        ema_decay         = 0.995,
        step_start_ema    = 1000,
        update_ema_every  = 5,
        lr                = 3e-4,
        lr_decay          = False,
        lr_maxt           = 1000,
        grad_norm         = 1.0,
        action_temperature= 0.1,    # 🔥 NEW: softmax temp for sample_action
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

        # EMA of actor weights — used for stable target actions
        self.ema             = EMA(ema_decay)
        self.ema_model       = copy.deepcopy(self.actor)
        self.step_start_ema  = step_start_ema
        self.update_ema_every= update_ema_every

        # --------------------------------------------------
        # Critic — twin Q-networks
        # --------------------------------------------------
        self.critic          = Critic(state_dim, action_dim).to(device)
        self.critic_target   = copy.deepcopy(self.critic)
        self.critic_optimizer= torch.optim.Adam(
            self.critic.parameters(), lr=3e-4
        )

        # --------------------------------------------------
        # LR schedulers (optional)
        # --------------------------------------------------
        self.lr_decay = lr_decay
        if lr_decay:
            self.actor_lr_scheduler = CosineAnnealingLR(
                self.actor_optimizer, T_max=lr_maxt, eta_min=0.0
            )
            self.critic_lr_scheduler = CosineAnnealingLR(
                self.critic_optimizer, T_max=lr_maxt, eta_min=0.0
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
        self.action_temperature = action_temperature   # 🔥 NEW
        self.step               = 0

    # ----------------------------------------------------------
    def step_ema(self):
        """Update EMA actor weights — called every update_ema_every steps."""
        if self.step < self.step_start_ema:
            return
        self.ema.update_model_average(self.ema_model, self.actor)

    # ----------------------------------------------------------
    def train(self, replay_buffer, iterations, batch_size=100,
              log_writer=None):
        """
        Run `iterations` gradient steps on a sampled minibatch.

        Args:
            replay_buffer : RLCarla ReplayBuffer (stores not_done correctly)
            iterations    : number of gradient steps per call
            batch_size    : minibatch size
            log_writer    : optional TensorBoard SummaryWriter

        Returns:
            metric dict with bc_loss, ql_loss, actor_loss, critic_loss
        """

        metric = {
            "bc_loss"    : [],
            "ql_loss"    : [],
            "actor_loss" : [],
            "critic_loss": [],
        }

        for _ in range(iterations):

            # ------------------------------------------
            # Sample minibatch from replay buffer
            # ------------------------------------------
            state, action, next_state, reward, not_done = \
                replay_buffer.sample(batch_size, self.device)

            # ------------------------------------------
            # CRITIC UPDATE
            # Bellman target: r + γ * (1 - done) * min_Q(s', a')
            # ------------------------------------------
            current_q1, current_q2 = self.critic(state, action)

            if self.max_q_backup:
                # Sample 10 next actions, take max Q (optimistic backup)
                ns_rpt   = torch.repeat_interleave(
                    next_state, repeats=10, dim=0
                )
                na_rpt   = self.ema_model(ns_rpt)
                tq1, tq2 = self.critic_target(ns_rpt, na_rpt)
                tq1      = tq1.view(batch_size, 10).max(dim=1, keepdim=True)[0]
                tq2      = tq2.view(batch_size, 10).max(dim=1, keepdim=True)[0]
                target_q = torch.min(tq1, tq2)
            else:
                # Standard: single next action from EMA actor
                next_action      = self.ema_model(next_state)
                tq1, tq2         = self.critic_target(next_state, next_action)
                target_q         = torch.min(tq1, tq2)

            # not_done = 1 during episode, 0 at terminal — CORRECT
            target_q = (reward + not_done * self.discount * target_q).detach()

            critic_loss = (
                F.mse_loss(current_q1, target_q) +
                F.mse_loss(current_q2, target_q)
            )

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            if self.grad_norm > 0:
                critic_grad_norms = nn.utils.clip_grad_norm_(
                    self.critic.parameters(),
                    max_norm=self.grad_norm, norm_type=2
                )
            self.critic_optimizer.step()

            # ------------------------------------------
            # ACTOR UPDATE
            # L = L_bc + eta * L_ql
            # L_bc  = diffusion denoising loss
            # L_ql  = -Q(s, actor(s))  normalised by |Q|
            # ------------------------------------------
            bc_loss    = self.actor.loss(action, state)
            new_action = self.actor(state)

            q1_new, q2_new = self.critic(state, new_action)

            # Randomly pick Q1 or Q2 to normalise by the other
            # (reduces coupling between the two critics)
            if np.random.uniform() > 0.5:
                q_loss = -q1_new.mean() / q2_new.abs().mean().detach()
            else:
                q_loss = -q2_new.mean() / q1_new.abs().mean().detach()

            actor_loss = bc_loss + self.eta * q_loss

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.grad_norm > 0:
                actor_grad_norms = nn.utils.clip_grad_norm_(
                    self.actor.parameters(),
                    max_norm=self.grad_norm, norm_type=2
                )
            self.actor_optimizer.step()

            # ------------------------------------------
            # TARGET NETWORK UPDATES
            # ------------------------------------------
            if self.step % self.update_ema_every == 0:
                self.step_ema()

            # Soft update critic target: θ' ← τθ + (1-τ)θ'
            for param, target_param in zip(
                self.critic.parameters(),
                self.critic_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data +
                    (1 - self.tau) * target_param.data
                )

            self.step += 1

            # ------------------------------------------
            # TENSORBOARD LOGGING
            # ------------------------------------------
            if log_writer is not None:
                if self.grad_norm > 0:
                    log_writer.add_scalar(
                        "grad/actor_norm",
                        actor_grad_norms.max().item(), self.step
                    )
                    log_writer.add_scalar(
                        "grad/critic_norm",
                        critic_grad_norms.max().item(), self.step
                    )
                log_writer.add_scalar(
                    "loss/bc",       bc_loss.item(),     self.step
                )
                log_writer.add_scalar(
                    "loss/ql",       q_loss.item(),      self.step
                )
                log_writer.add_scalar(
                    "loss/critic",   critic_loss.item(), self.step
                )
                log_writer.add_scalar(
                    "critic/target_q_mean",
                    target_q.mean().item(), self.step
                )

            metric["actor_loss"].append(actor_loss.item())
            metric["bc_loss"].append(bc_loss.item())
            metric["ql_loss"].append(q_loss.item())
            metric["critic_loss"].append(critic_loss.item())

        if self.lr_decay:
            self.actor_lr_scheduler.step()
            self.critic_lr_scheduler.step()

        return metric

    # ----------------------------------------------------------
    def sample_action(self, state):
        """
        Sample best action for a given state during training exploration.

        Draws 50 candidate actions from the diffusion actor,
        evaluates each with the critic, then selects via softmax
        sampling with temperature (not argmax — keeps exploration alive).

        Temperature:
            Low  (0.05) → near-deterministic, exploits learned policy
            High (1.0)  → near-uniform, pure exploration
            0.1         → good balance during training
        """
        state_t   = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        state_rpt = torch.repeat_interleave(state_t, repeats=50, dim=0)

        with torch.no_grad():
            action  = self.actor.sample(state_rpt)
            q_value = self.critic_target.q_min(
                state_rpt, action
            ).flatten()

            # 🔥 FIXED: temperature-scaled softmax
            # Original had no temperature → softmax collapsed to
            # near-argmax or near-uniform depending on Q scale
            probs = F.softmax(q_value / self.action_temperature, dim=0)
            idx   = torch.multinomial(probs, num_samples=1)

        return action[idx].cpu().data.numpy().flatten()

    # ----------------------------------------------------------
    def save_model(self, dir, id=None):
        """Save actor and critic weights to checkpoint directory."""
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
        """
        Load actor and critic weights from checkpoint directory.
        Uses weights_only=True for PyTorch 2.x security.
        """
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
