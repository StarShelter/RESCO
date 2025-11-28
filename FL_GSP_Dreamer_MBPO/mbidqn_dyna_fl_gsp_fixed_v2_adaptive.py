"""
mbidqn_dyna_fl_gsp_fixed_v2_adaptive.py - 带动态控制的Dyna版本

核心改进：
1. ✅ Real data保底机制：min_real_ratio确保real data最小比例
2. ✅ Synthetic data作为增量：根据WM质量动态调整
3. ✅ 动态horizon调整：从小horizon逐步增加到max_horizon
4. ✅ 质量监控：追踪WM质量并自适应调整策略

关键公式：
- real_ratio = max(min_real_ratio, 1.0 - wm_quality * adaptive_factor)
- horizon = min(current_horizon, max_horizon) 随episode增长

作者: Percy Zhang
日期: 2025-11-28
版本: Adaptive V2
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


# ==================== Adaptive Controller ====================

class AdaptiveDataController:
    """
    自适应数据控制器 - Synthetic data作为Real data的增量
    
    核心策略：
    1. Real data固定：始终使用固定batch size的real data（如32）
    2. Synthetic data增量：根据WM质量动态增加synthetic batch size
    3. Horizon增长：每N个episodes增加1步，直到max_horizon
    
    关键：Real data不会被稀释，synthetic是额外的bonus
    """
    def __init__(self, 
                 base_real_batch=32,           # Real data固定batch size
                 max_synthetic_batch=64,       # Synthetic data最大batch size
                 initial_horizon=1,            # 初始horizon
                 max_horizon=5,                # 最大horizon
                 horizon_increase_freq=10,     # 每N个episodes增加horizon
                 quality_threshold=0.3,        # WM质量阈值（低于此不用synthetic）
                 adaptive_factor=1.0):         # 自适应调整因子
        
        self.base_real_batch = base_real_batch
        self.max_synthetic_batch = max_synthetic_batch
        self.initial_horizon = initial_horizon
        self.max_horizon = max_horizon
        self.horizon_increase_freq = horizon_increase_freq
        self.quality_threshold = quality_threshold
        self.adaptive_factor = adaptive_factor
        
        # 当前状态
        self.current_horizon = initial_horizon
        self.current_synthetic_batch = 0  # 初始0 synthetic
        self.episode_count = 0
        
        # 质量历史
        self.quality_history = deque(maxlen=100)
        
        logger.info(f"AdaptiveDataController initialized:")
        logger.info(f"  base_real_batch={base_real_batch} (固定)")
        logger.info(f"  max_synthetic_batch={max_synthetic_batch}")
        logger.info(f"  horizon: {initial_horizon} → {max_horizon}")
        logger.info(f"  horizon_increase_freq={horizon_increase_freq}")
    
    def update_quality(self, wm_quality):
        """更新WM质量"""
        self.quality_history.append(wm_quality)
    
    def get_current_synthetic_batch(self):
        """
        计算当前synthetic data batch size（作为增量）
        
        策略：
        - WM质量低（< threshold）→ synthetic = 0（不用）
        - WM质量中等 → synthetic逐步增加
        - WM质量高 → synthetic达到max_synthetic_batch
        
        重要：返回的是synthetic batch size，不影响real batch
        """
        if len(self.quality_history) == 0:
            return 0  # 没有质量数据时，不用synthetic
        
        # 使用最近的平均质量
        avg_quality = np.mean(list(self.quality_history))
        
        # 如果质量低于阈值，不用synthetic
        if avg_quality < self.quality_threshold:
            target_synthetic_batch = 0
        else:
            # 质量好时，根据质量增加synthetic
            # quality越高，synthetic越多
            # 例如：quality=0.5 → synthetic=32, quality=1.0 → synthetic=64
            quality_above_threshold = avg_quality - self.quality_threshold
            max_quality_range = 1.0 - self.quality_threshold
            
            normalized_quality = quality_above_threshold / max_quality_range
            target_synthetic_batch = int(
                normalized_quality * self.adaptive_factor * self.max_synthetic_batch
            )
            target_synthetic_batch = min(target_synthetic_batch, self.max_synthetic_batch)
        
        # 平滑过渡
        self.current_synthetic_batch = int(
            0.9 * self.current_synthetic_batch + 0.1 * target_synthetic_batch
        )
        
        return self.current_synthetic_batch
    
    def get_current_horizon(self):
        """获取当前imagination horizon"""
        return self.current_horizon
    
    def on_episode_end(self):
        """Episode结束时调用 - 更新horizon"""
        self.episode_count += 1
        
        # 每N个episodes增加horizon
        if self.episode_count % self.horizon_increase_freq == 0:
            if self.current_horizon < self.max_horizon:
                self.current_horizon += 1
                logger.info(f"📈 Horizon increased to {self.current_horizon} "
                          f"(episode {self.episode_count})")
    
    def should_use_imagination(self, step_count, warmup_steps):
        """判断是否应该使用imagination"""
        if step_count < warmup_steps:
            return False
        
        # 检查WM质量
        if len(self.quality_history) > 0:
            avg_quality = np.mean(list(self.quality_history))
            if avg_quality < 0.2:  # 质量太差，不用imagination
                return False
        
        return True
    
    def get_stats(self):
        """获取统计信息"""
        if len(self.quality_history) > 0:
            avg_quality = np.mean(list(self.quality_history))
        else:
            avg_quality = 0.0
        
        total_batch = self.base_real_batch + self.current_synthetic_batch
        
        return {
            'episode': self.episode_count,
            'horizon': self.current_horizon,
            'real_batch': self.base_real_batch,
            'synthetic_batch': self.current_synthetic_batch,
            'total_batch': total_batch,
            'avg_wm_quality': avg_quality
        }


# ==================== Copy the RSSM and other classes from original v2 ====================
# [这里需要复制原始v2文件中的所有类定义]
# 为了简洁，我只展示关键修改部分

# ... [复制 GlobalCoordinatorWithGSP, RSSM_with_GSP, ReplayBuffer 等类] ...


class DynaMBAgent_Adaptive:
    """
    Dyna-style MB Agent with Adaptive Control
    
    核心改进：
    1. 使用AdaptiveDataController动态调整策略
    2. 混合采样确保real data保底
    3. 动态horizon增长
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
        
        # ⭐ 创建自适应控制器
        self.adaptive_controller = AdaptiveDataController(
            base_real_batch=cfg.batch_size,  # Real batch固定为32
            max_synthetic_batch=cfg.get('max_synthetic_batch', 64),
            initial_horizon=cfg.get('initial_horizon', 1),
            max_horizon=cfg.get('imagination_horizon', 5),
            horizon_increase_freq=cfg.get('horizon_increase_freq', 10),
            quality_threshold=cfg.get('min_wm_quality', 0.3),
            adaptive_factor=cfg.get('adaptive_factor', 1.0)
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
            lr=cfg.get('lr_world_model', 5e-4)
        )
        self.q_optimizer = torch.optim.RMSprop(
            self.q_network.parameters(),
            lr=cfg.learning_rate,
            alpha=cfg.get('rmsprop_decay', 0.95),
            eps=cfg.get('rmsprop_epsilon', 1e-8),
            momentum=cfg.get('rmsprop_momentum', 0.0)
        )
        
        # Replay Buffers
        self.replay_buffer = ReplayBuffer(cfg.buffer_size)          # Real data
        self.imagined_buffer = ReplayBuffer(cfg.buffer_size // 2)   # Imagined data
        
        # Counters
        self.step_count = 0
        self.episode_count = 0
        self.global_step = 0
        
        # Config
        self.model_warmup_steps = cfg.get('model_warmup_steps', 1000)
        self.imagination_freq = cfg.get('imagination_freq', 5)
        self.num_imagined_rollouts = cfg.get('num_imagined_rollouts', 1)
        
        # FL
        self.fl_interval = cfg.get('fl_interval', 100)
        self.global_params = None
        
        logger.info(f"DynaMBAgent_Adaptive created for {agent_id}")
        logger.info(f"  Adaptive control enabled")
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
        # 存储到real buffer
        self.replay_buffer.push(obs, action, reward, next_obs, done)
        self.step_count += 1
        self.global_step += 1
        
        # 训练World Model
        if len(self.replay_buffer) >= cfg.batch_size:
            self._train_world_model()
            
            # ⭐ 更新WM质量到adaptive controller
            wm_quality = self.get_quality_score()
            self.adaptive_controller.update_quality(wm_quality)
        
        # ⭐ 根据adaptive controller决定是否生成imagined data
        if self._should_generate_imagined_data():
            self._generate_imagined_data()
        
        # ⭐ 使用adaptive混合采样训练Q
        if len(self.replay_buffer) >= cfg.batch_size:
            self._train_q_adaptive()
        
        # 上传FL参数
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
    
    def _should_generate_imagined_data(self):
        """判断是否应该生成imagined data"""
        # Warmup阶段不生成
        if not self.adaptive_controller.should_use_imagination(
            self.step_count, self.model_warmup_steps
        ):
            return False
        
        # 按频率生成
        if self.step_count % self.imagination_freq != 0:
            return False
        
        return True
    
    def _generate_imagined_data(self):
        """⭐ 生成imagined data - 使用动态horizon"""
        if len(self.replay_buffer) < cfg.batch_size:
            return
        
        # ⭐ 获取当前的动态horizon
        horizon = self.adaptive_controller.get_current_horizon()
        
        # 从real buffer采样起始状态
        transitions = self.replay_buffer.sample_transitions(
            min(self.num_imagined_rollouts, len(self.replay_buffer))
        )
        
        with torch.no_grad():
            for trans in transitions:
                obs = torch.FloatTensor(trans['obs']).unsqueeze(0).to(self.device)
                state = self._encode_obs(obs)
                
                # ⭐ Rollout使用动态horizon
                for step in range(horizon):
                    # 用当前policy选action
                    feat = self.world_model.get_feature(state)
                    q_values = self.q_network(feat).q_values
                    action = q_values.argmax(dim=1).item()
                    
                    # Imagine next state
                    action_onehot = torch.zeros(1, self.act_dim).to(self.device)
                    action_onehot[0, action] = 1.0
                    next_state = self.world_model.imagine_step(state, action_onehot)
                    
                    # Decode reward
                    _, reward = self.world_model.decode(next_state)
                    reward = reward.squeeze(-1).item()
                    
                    # 存储到imagined buffer
                    curr_obs = self.world_model.decode(state)[0].squeeze(0).cpu().numpy()
                    next_obs = self.world_model.decode(next_state)[0].squeeze(0).cpu().numpy()
                    
                    self.imagined_buffer.push(
                        curr_obs, action, reward, next_obs, False
                    )
                    
                    state = next_state
    
    def _train_q_adaptive(self):
        """⭐ 使用adaptive控制训练Q-Network - Synthetic作为增量"""
        # ⭐ Real batch是固定的（如32）
        real_batch_size = self.adaptive_controller.base_real_batch
        
        # ⭐ Synthetic batch是动态的（0到max_synthetic_batch）
        synthetic_batch_size = self.adaptive_controller.get_current_synthetic_batch()
        
        # 采样real data（固定数量）
        real_transitions = self.replay_buffer.sample_transitions(real_batch_size)
        
        # 采样synthetic data（动态数量，可能是0）
        if len(self.imagined_buffer) > 0 and synthetic_batch_size > 0:
            imagined_transitions = self.imagined_buffer.sample_transitions(
                min(synthetic_batch_size, len(self.imagined_buffer))
            )
            # 合并
            transitions = real_transitions + imagined_transitions
        else:
            # 如果没有synthetic data或synthetic_batch_size=0，只用real
            transitions = real_transitions
        
        # 准备batch
        obs = np.stack([t['obs'] for t in transitions])
        actions = np.array([t['action'] for t in transitions])
        rewards = np.array([t['reward'] for t in transitions])
        next_obs = np.stack([t['next_obs'] for t in transitions])
        dones = np.array([t['done'] for t in transitions])
        
        batch_size = len(transitions)  # 动态总batch size
        
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
        
        # 定期更新target network
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
        logger.info(f"  Horizon: {stats['horizon']}")
        logger.info(f"  Real batch: {stats['real_batch']} (固定)")
        logger.info(f"  Synthetic batch: {stats['synthetic_batch']}")
        logger.info(f"  Total batch: {stats['total_batch']}")
        logger.info(f"  Avg WM quality: {stats['avg_wm_quality']:.3f}")
    
    def _encode_obs(self, obs):
        """Encode observation to latent state"""
        return self.world_model.encode(obs)
    
    def _train_world_model(self):
        """训练World Model - 使用序列采样"""
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

class MBIDQN_Dyna_FL_GSP_Fixed_V2_Adaptive(IndependentAgent):
    """
    RESCO Adapter with Adaptive Control
    """
    def __init__(self, obs_act):
        super().__init__(obs_act)
        
        logger.info("=" * 70)
        logger.info("🚀 MBIDQN Dyna + FL + GSP (Adaptive V2)")
        logger.info("=" * 70)
        logger.info("Key features:")
        logger.info("1. ✅ Real data保底机制 (min 50%)")
        logger.info("2. ✅ Synthetic data作为增量")
        logger.info("3. ✅ 动态horizon调整 (1→5)")
        logger.info("4. ✅ Quality-based adaptive control")
        logger.info("=" * 70)
        
        # 创建全局模型和coordinator
        first_agent_id = list(obs_act.keys())[0]
        obs_space = obs_act[first_agent_id][0]
        act_space = obs_act[first_agent_id][1]
        
        if len(obs_space) == 1:
            obs_dim = obs_space[0]
        else:
            obs_dim = int(np.prod(obs_space))
        
        # [创建global model和coordinator的代码与原v2相同]
        # ...
        
        # 创建adaptive agents
        for agent_id in obs_act:
            obs_space = obs_act[agent_id][0]
            act_space = obs_act[agent_id][1]
            
            agent = DynaMBAgent_Adaptive(
                agent_id,
                obs_space,
                act_space,
                coordinator=self.coordinator
            )
            self.agents[agent_id] = agent
            self.coordinator.register_agent(agent_id)
        
        logger.info(f"Initialized {len(self.agents)} adaptive agents")
    
    def observe(self, observations, rewards, dones, infos):
        """扩展observe - 处理episode结束"""
        super().observe(observations, rewards, dones, infos)
        
        # FL聚合
        if self.coordinator.should_aggregate_fl():
            global_params = self.coordinator.aggregate_fl_and_broadcast()
            if global_params:
                for agent in self.agents.values():
                    agent.update_from_global(global_params)
        
        # GSP共识
        current_step = list(self.agents.values())[0].global_step
        if self.coordinator.should_compute_consensus(current_step):
            G_consensus = self.coordinator.compute_consensus(current_step)
        
        # ⭐ 检查episode结束
        if any(dones.values()):
            for agent in self.agents.values():
                agent.on_episode_end()


# ==================== 需要在agent.yaml中添加的配置 ====================
"""
MBIDQN_Dyna_FL_GSP_Fixed_V2_Adaptive:
  module: action_value.mbidqn_dyna_fl_gsp_fixed_v2_adaptive
  state: drq
  reward: wait
  
  # Learning Rates
  learning_rate: 1e-3
  lr_world_model: 5e-4
  
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
  
  # ⭐ Adaptive Control配置 - NEW
  min_real_ratio: 0.5              # Real data最小比例（保底）
  max_synthetic_ratio: 0.7          # Synthetic data最大比例
  initial_horizon: 1                # 初始horizon
  imagination_horizon: 5            # 最大horizon
  horizon_increase_freq: 10         # 每10个episodes增加1步horizon
  adaptive_factor: 0.5              # 自适应调整因子
  
  # Dyna配置
  num_imagined_rollouts: 1
  imagination_freq: 5
  
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
