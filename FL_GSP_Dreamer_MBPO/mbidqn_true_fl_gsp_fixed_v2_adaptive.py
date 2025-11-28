"""
mbidqn_true_fl_gsp_fixed_v2_adaptive.py - True MBRL with Adaptive Control

核心改进：
1. ✅ Adaptive horizon: 从小horizon逐步增加
2. ✅ Quality-based control: WM质量不好时减少imagination步数
3. ✅ Warmup protection: 早期只用real data训练

区别于Dyna版本：
- True: Q在imagination中训练（梯度反向传播through WM）
- 不需要real/synthetic混合采样（因为不生成数据到buffer）
- 但需要动态控制imagination horizon

作者: Percy Zhang  
日期: 2025-11-28
版本: True Adaptive V2
"""

import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from copy import deepcopy
import pfrl
from pfrl.explorers import LinearDecayEpsilonGreedy
from pfrl.q_functions import DiscreteActionValueHead

from resco_benchmark.config.config import config as cfg
from resco_benchmark.agents.agent import IndependentAgent, Agent
from resco_benchmark.utils.utils import compute_safe_id

logger = logging.getLogger(__name__)


# ==================== Adaptive Imagination Controller ====================

class AdaptiveImaginationController:
    """
    自适应Imagination控制器 - For True MBRL
    
    核心策略：
    1. Horizon动态增长：从小horizon逐步增加
    2. Quality-based：WM质量差时减少horizon
    3. Warmup保护：早期不在imagination中训练
    """
    def __init__(self,
                 initial_horizon=1,
                 max_horizon=5,
                 horizon_increase_freq=10,
                 quality_threshold=0.5,
                 quality_scaling=True):
        
        self.initial_horizon = initial_horizon
        self.max_horizon = max_horizon
        self.horizon_increase_freq = horizon_increase_freq
        self.quality_threshold = quality_threshold
        self.quality_scaling = quality_scaling
        
        # 当前状态
        self.current_base_horizon = initial_horizon
        self.episode_count = 0
        
        # 质量历史
        self.quality_history = deque(maxlen=100)
        
        logger.info(f"AdaptiveImaginationController initialized:")
        logger.info(f"  horizon: {initial_horizon} → {max_horizon}")
        logger.info(f"  horizon_increase_freq={horizon_increase_freq}")
        logger.info(f"  quality_scaling={quality_scaling}")
    
    def update_quality(self, wm_quality):
        """更新WM质量"""
        self.quality_history.append(wm_quality)
    
    def get_current_horizon(self):
        """
        获取当前imagination horizon
        
        策略：
        - 基础horizon随episode增长
        - 如果quality_scaling=True，根据WM质量调整
        """
        if len(self.quality_history) == 0:
            return self.current_base_horizon
        
        avg_quality = np.mean(list(self.quality_history))
        
        if not self.quality_scaling:
            # 不使用质量缩放，直接返回基础horizon
            return self.current_base_horizon
        
        # 根据质量调整horizon
        if avg_quality < self.quality_threshold:
            # 质量差，使用更短的horizon
            adjusted_horizon = max(1, self.current_base_horizon - 2)
        elif avg_quality > 0.8:
            # 质量好，可以用更长的horizon
            adjusted_horizon = min(self.max_horizon, self.current_base_horizon + 1)
        else:
            # 质量中等，使用基础horizon
            adjusted_horizon = self.current_base_horizon
        
        return adjusted_horizon
    
    def should_use_imagination(self, step_count, warmup_steps):
        """判断是否应该在imagination中训练Q"""
        # Warmup期间不使用
        if step_count < warmup_steps:
            return False
        
        # 检查WM质量
        if len(self.quality_history) > 0:
            avg_quality = np.mean(list(self.quality_history))
            # 质量太差，不使用imagination训练
            if avg_quality < 0.2:
                logger.warning(f"WM quality too low ({avg_quality:.3f}), "
                             f"skipping imagination training")
                return False
        
        return True
    
    def on_episode_end(self):
        """Episode结束时调用"""
        self.episode_count += 1
        
        # 每N个episodes增加base horizon
        if self.episode_count % self.horizon_increase_freq == 0:
            if self.current_base_horizon < self.max_horizon:
                self.current_base_horizon += 1
                logger.info(f"📈 Base horizon increased to {self.current_base_horizon} "
                          f"(episode {self.episode_count})")
    
    def get_stats(self):
        """获取统计信息"""
        if len(self.quality_history) > 0:
            avg_quality = np.mean(list(self.quality_history))
        else:
            avg_quality = 0.0
        
        return {
            'episode': self.episode_count,
            'base_horizon': self.current_base_horizon,
            'actual_horizon': self.get_current_horizon(),
            'avg_wm_quality': avg_quality
        }


# ==================== True MB Agent with Adaptive Control ====================

class TrueMBAgent_Adaptive:
    """
    True Model-Based Agent with Adaptive Imagination Control
    
    核心特点：
    1. Q在imagination中训练（梯度反向传播）
    2. 使用adaptive controller动态调整horizon
    3. Quality-based控制策略
    """
    def __init__(self, agent_id, obs_space, act_space, coordinator=None):
        self.agent_id = agent_id
        self.obs_space = obs_space
        self.act_space = act_space
        self.coordinator = coordinator
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Dimensions
        if len(obs_space) == 1:
            obs_dim = obs_space[0]
        else:
            obs_dim = int(np.prod(obs_space))
        self.obs_dim = obs_dim
        self.act_dim = act_space
        
        # ⭐ 创建自适应imagination控制器
        self.adaptive_controller = AdaptiveImaginationController(
            initial_horizon=cfg.get('initial_horizon', 1),
            max_horizon=cfg.get('imagination_horizon', 5),
            horizon_increase_freq=cfg.get('horizon_increase_freq', 10),
            quality_threshold=cfg.get('min_wm_quality', 0.3),
            quality_scaling=cfg.get('quality_scaling', True)
        )
        
        # World Model
        self.world_model = RSSM_with_GSP(
            obs_dim=obs_dim,
            action_dim=act_space,
            hidden_dim=cfg.number_of_units,
            stoch_dim=32,
            deter_dim=cfg.number_of_units,
            global_dim=cfg.get('global_dim', 64)
        ).to(self.device)
        
        # Q-Network
        hidden_dim = cfg.number_of_units
        feature_dim = self.world_model.get_feature_size()
        self.q_network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            DiscreteActionValueHead(hidden_dim, act_space)
        ).to(self.device)
        
        self.target_q_network = deepcopy(self.q_network)
        
        # Optimizers
        self.wm_optimizer = torch.optim.Adam(
            self.world_model.parameters(),
            lr=cfg.get('lr_world_model', 1e-4)
        )
        self.q_optimizer = torch.optim.RMSprop(
            self.q_network.parameters(),
            lr=cfg.learning_rate,
            alpha=cfg.get('rmsprop_decay', 0.95),
            eps=cfg.get('rmsprop_epsilon', 1e-8),
            momentum=cfg.get('rmsprop_momentum', 0.0)
        )
        
        # Replay Buffer (只有real data)
        self.replay_buffer = ReplayBuffer(cfg.buffer_size)
        
        # Counters
        self.step_count = 0
        self.episode_count = 0
        self.global_step = 0
        
        # Config
        self.model_warmup_steps = cfg.get('model_warmup_steps', 1000)
        
        # FL
        self.fl_interval = cfg.get('fl_interval', 100)
        self.global_params = None
        
        logger.info(f"TrueMBAgent_Adaptive created for {agent_id}")
        logger.info(f"  Adaptive imagination control enabled")
        logger.info(f"  Initial horizon: {self.adaptive_controller.initial_horizon}")
        logger.info(f"  Max horizon: {self.adaptive_controller.max_horizon}")
    
    def act(self, obs):
        """选择动作"""
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            state = self._encode_obs(obs_t)
            feat = self.world_model.get_feature(state)
            q_values = self.q_network(feat).q_values
            action = q_values.argmax(dim=1).item()
        
        return action
    
    def observe(self, obs, action, reward, next_obs, done):
        """观察一个transition"""
        # 存储到buffer
        self.replay_buffer.push(obs, action, reward, next_obs, done)
        self.step_count += 1
        self.global_step += 1
        
        # 训练World Model
        if len(self.replay_buffer) >= cfg.batch_size:
            self._train_world_model()
            
            # ⭐ 更新WM质量
            wm_quality = self.get_quality_score()
            self.adaptive_controller.update_quality(wm_quality)
        
        # ⭐ Adaptive imagination training
        if self._should_use_imagination():
            self._train_q_in_imagination_adaptive()
        else:
            # 不用imagination时，用real data训练
            if len(self.replay_buffer) >= cfg.batch_size:
                self._train_q_on_real_data()
        
        # Upload FL
        if self.step_count % self.fl_interval == 0:
            self._upload_fl_params()
        
        # Upload GSP
        if self.coordinator:
            with torch.no_grad():
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                state = self._encode_obs(obs_t)
                G_hat = self.world_model.predict_global(state)
                self.coordinator.receive_gsp_prediction(
                    self.global_step, self.agent_id, G_hat
                )
    
    def _should_use_imagination(self):
        """判断是否应该在imagination中训练Q"""
        return self.adaptive_controller.should_use_imagination(
            self.step_count, self.model_warmup_steps
        )
    
    def _train_q_in_imagination_adaptive(self):
        """⭐ 在imagination中训练Q - 使用adaptive horizon"""
        if len(self.replay_buffer) < cfg.batch_size:
            return
        
        # ⭐ 获取当前的adaptive horizon
        horizon = self.adaptive_controller.get_current_horizon()
        
        # 从real buffer采样起始状态
        transitions = self.replay_buffer.sample_transitions(cfg.batch_size)
        obs = np.stack([t['obs'] for t in transitions])
        obs_t = torch.FloatTensor(obs).to(self.device)
        
        # Encode to latent state
        state = self._encode_obs(obs_t)
        
        # ⭐ Rollout in imagination with adaptive horizon
        total_loss = 0.0
        
        for step in range(horizon):
            # Select action with Q-network
            feat = self.world_model.get_feature(state)
            q_values = self.q_network(feat).q_values
            actions = q_values.argmax(dim=1)
            q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            
            # One-hot encode actions
            action_onehot = torch.zeros(
                len(actions), self.act_dim
            ).to(self.device)
            action_onehot.scatter_(1, actions.unsqueeze(1), 1.0)
            
            # Imagine next state
            next_state = self.world_model.imagine_step(state, action_onehot)
            
            # Get reward and next Q
            _, reward = self.world_model.decode(next_state)
            reward = reward.squeeze(-1)
            
            with torch.no_grad():
                next_feat = self.world_model.get_feature(next_state)
                next_q_values = self.target_q_network(next_feat)
                next_q = next_q_values.q_values.max(dim=1)[0]
            
            # TD loss
            target = reward + cfg.discount * next_q
            td_loss = F.mse_loss(q_selected, target)
            total_loss += td_loss
            
            state = next_state
        
        # Backward
        self.q_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.q_optimizer.step()
        
        # Update target network
        if self.step_count % cfg.target_update_steps == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())
    
    def _train_q_on_real_data(self):
        """在real data上训练Q（warmup阶段或WM质量差时）"""
        transitions = self.replay_buffer.sample_transitions(cfg.batch_size)
        
        obs = np.stack([t['obs'] for t in transitions])
        actions = np.array([t['action'] for t in transitions])
        rewards = np.array([t['reward'] for t in transitions])
        next_obs = np.stack([t['next_obs'] for t in transitions])
        dones = np.array([t['done'] for t in transitions])
        
        obs_t = torch.FloatTensor(obs).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_obs_t = torch.FloatTensor(next_obs).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)
        
        # Q-learning update
        with torch.no_grad():
            next_state = self._encode_obs(next_obs_t)
            next_feat = self.world_model.get_feature(next_state)
            next_q = self.target_q_network(next_feat).q_values.max(dim=1)[0]
            target = rewards_t + cfg.discount * (1 - dones_t) * next_q
        
        state = self._encode_obs(obs_t)
        feat = self.world_model.get_feature(state)
        q_values = self.q_network(feat).q_values
        q_selected = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)
        
        loss = F.mse_loss(q_selected, target)
        
        self.q_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.q_optimizer.step()
        
        # Update target
        if self.step_count % cfg.target_update_steps == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())
    
    def on_episode_end(self):
        """Episode结束时调用"""
        self.episode_count += 1
        
        # ⭐ 通知adaptive controller
        self.adaptive_controller.on_episode_end()
        
        # 打印统计信息
        stats = self.adaptive_controller.get_stats()
        logger.info(f"Agent {self.agent_id} Episode {stats['episode']}:")
        logger.info(f"  Base horizon: {stats['base_horizon']}")
        logger.info(f"  Actual horizon: {stats['actual_horizon']}")
        logger.info(f"  Avg WM quality: {stats['avg_wm_quality']:.3f}")
    
    def _encode_obs(self, obs):
        """Encode observation to latent state"""
        return self.world_model.encode(obs)
    
    def _train_world_model(self):
        """训练World Model"""
        if len(self.replay_buffer) < cfg.get('seq_length', 50):
            return
        
        # Sample sequence
        seq_data = self.replay_buffer.sample_sequence(cfg.get('seq_length', 50))
        
        obs = torch.FloatTensor(seq_data['obs']).to(self.device)
        actions = torch.FloatTensor(seq_data['actions']).to(self.device)
        rewards = torch.FloatTensor(seq_data['rewards']).unsqueeze(-1).to(self.device)
        
        # Forward
        recon_obs, pred_rewards, kl_loss = self.world_model(obs, actions)
        
        # Losses
        recon_loss = F.mse_loss(recon_obs, obs)
        reward_loss = F.mse_loss(pred_rewards, rewards)
        
        # GSP loss
        gsp_loss = torch.tensor(0.0).to(self.device)
        if self.coordinator:
            state = self.world_model.encode(obs[:, -1:])
            G_hat = self.world_model.predict_global(state)
            
            G_consensus = self.coordinator.get_latest_consensus()
            if G_consensus is not None:
                G_consensus_t = torch.FloatTensor(G_consensus).unsqueeze(0).to(self.device)
                gsp_loss = F.mse_loss(G_hat, G_consensus_t)
        
        # Total loss
        total_loss = (recon_loss + 
                     reward_loss + 
                     cfg.get('beta_kl', 1.0) * kl_loss +
                     cfg.get('alpha_contrastive', 0.1) * gsp_loss)
        
        # Backward
        self.wm_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), 10.0)
        self.wm_optimizer.step()
    
    def get_quality_score(self):
        """获取WM质量分数"""
        if len(self.replay_buffer) < cfg.batch_size:
            return 0.5
        
        transitions = self.replay_buffer.sample_transitions(
            min(cfg.batch_size, len(self.replay_buffer))
        )
        obs = np.stack([t['obs'] for t in transitions])
        obs_t = torch.FloatTensor(obs).to(self.device)
        
        with torch.no_grad():
            state = self._encode_obs(obs_t)
            Q_score = self.world_model.get_quality_score(state)
            return Q_score.mean().item()
    
    def _upload_fl_params(self):
        """上传FL参数"""
        if not self.coordinator:
            return
        
        params = {k: v.cpu().clone() for k, v in self.world_model.state_dict().items()}
        quality = self.get_quality_score()
        
        self.coordinator.receive_fl_update(self.agent_id, params, quality)
    
    def update_from_global(self, global_params):
        """从全局参数更新"""
        self.global_params = {k: v.clone().cpu() for k, v in global_params.items()}
        
        current_params = self.world_model.state_dict()
        updated_count = 0
        skipped_count = 0
        
        for key, global_param in global_params.items():
            if key in current_params:
                if current_params[key].shape == global_param.shape:
                    current_params[key] = global_param
                    updated_count += 1
                else:
                    skipped_count += 1
        
        self.world_model.load_state_dict(current_params)
        
        logger.info(f"Agent {self.agent_id}: Updated {updated_count} params, "
                   f"kept {skipped_count} local")


# ==================== RESCO Adapter ====================

class MBIDQN_True_FL_GSP_Fixed_V2_Adaptive(IndependentAgent):
    """
    RESCO Adapter for True MBRL with Adaptive Control
    """
    def __init__(self, obs_act):
        super().__init__(obs_act)
        
        logger.info("=" * 70)
        logger.info("🚀 MBIDQN True + FL + GSP (Adaptive V2)")
        logger.info("=" * 70)
        logger.info("Key features:")
        logger.info("1. ✅ Adaptive imagination horizon (1→5)")
        logger.info("2. ✅ Quality-based horizon scaling")
        logger.info("3. ✅ Warmup protection")
        logger.info("4. ✅ Gradual horizon increase")
        logger.info("=" * 70)
        
        # [创建global model, coordinator和agents的代码]
        # ...
        
        logger.info(f"Initialized {len(self.agents)} adaptive True MB agents")


# ==================== 配置示例 ====================
"""
MBIDQN_True_FL_GSP_Fixed_V2_Adaptive:
  module: action_value.mbidqn_true_fl_gsp_fixed_v2_adaptive
  state: drq
  reward: wait
  
  # Learning Rates
  learning_rate: 5e-4
  lr_world_model: 1e-4
  
  # FL配置
  fl_interval: 100
  aggregation_method: quality_weighted
  alpha_fedprox: 0.01
  
  # GSP配置
  global_dim: 64
  alpha_contrastive: 0.1
  contrastive_temperature: 0.1
  gsp_sync_threshold: 0.8
  
  # RSSM配置
  seq_length: 50
  model_train_freq: 1
  
  # ⭐ Adaptive Imagination配置 - NEW
  initial_horizon: 1                # 初始horizon
  imagination_horizon: 5            # 最大horizon
  horizon_increase_freq: 10         # 每10个episodes增加1步
  quality_scaling: True             # 根据WM质量动态调整horizon
  
  # Early阶段保护
  model_warmup_steps: 1000
  min_wm_quality: 0.3
  
  # Q-Network配置
  batch_size: 32
  discount: 0.99
  target_update_steps: 500
  number_of_layers: 3
  number_of_units: 128
  
  # Exploration
  epsilon_begin: 1.0
  epsilon_end: 0.1
  epsilon_decay_period: 100000
  
  # Buffer
  buffer_size: 50000
  
  # Optimizer
  rmsprop_decay: 0.95
  rmsprop_epsilon: 0.00001
  rmsprop_momentum: 0.0
"""
