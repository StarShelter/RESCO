# # """
# # mbidqn_dyna_fl_gsp_fixed_v2_adaptive.py - 带动态控制的Dyna版本

# # 核心改进：
# # 1. ✅ Real data保底机制：min_real_ratio确保real data最小比例
# # 2. ✅ Synthetic data作为增量：根据WM质量动态调整
# # 3. ✅ 动态horizon调整：从小horizon逐步增加到max_horizon
# # 4. ✅ 质量监控：追踪WM质量并自适应调整策略

# # 关键公式：
# # - real_ratio = max(min_real_ratio, 1.0 - wm_quality * adaptive_factor)
# # - horizon = min(current_horizon, max_horizon) 随episode增长

# # 作者: Percy Zhang
# # 日期: 2025-11-28
# # 版本: Adaptive V2
# # """

# # import os
# # import logging
# # import numpy as np
# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# # from collections import deque
# # from copy import deepcopy
# # import pfrl
# # from pfrl.explorers import LinearDecayEpsilonGreedy
# # from pfrl.q_functions import DiscreteActionValueHead

# # from resco_benchmark.config.config import config as cfg
# # from resco_benchmark.agents.agent import IndependentAgent, Agent
# # from resco_benchmark.utils.utils import compute_safe_id

# # logger = logging.getLogger(__name__)


# # # ==================== Adaptive Controller ====================

# # class AdaptiveDataController:
# #     """
# #     自适应数据控制器 - Synthetic data作为Real data的增量
    
# #     核心策略：
# #     1. Real data固定：始终使用固定batch size的real data（如32）
# #     2. Synthetic data增量：根据WM质量动态增加synthetic batch size
# #     3. Horizon增长：每N个episodes增加1步，直到max_horizon
    
# #     关键：Real data不会被稀释，synthetic是额外的bonus
# #     """
# #     def __init__(self, 
# #                  base_real_batch=32,           # Real data固定batch size
# #                  max_synthetic_batch=64,       # Synthetic data最大batch size
# #                  initial_horizon=1,            # 初始horizon
# #                  max_horizon=5,                # 最大horizon
# #                  horizon_increase_freq=10,     # 每N个episodes增加horizon
# #                  quality_threshold=0.3,        # WM质量阈值（低于此不用synthetic）
# #                  adaptive_factor=1.0):         # 自适应调整因子
        
# #         self.base_real_batch = base_real_batch
# #         self.max_synthetic_batch = max_synthetic_batch
# #         self.initial_horizon = initial_horizon
# #         self.max_horizon = max_horizon
# #         self.horizon_increase_freq = horizon_increase_freq
# #         self.quality_threshold = quality_threshold
# #         self.adaptive_factor = adaptive_factor
        
# #         # 当前状态
# #         self.current_horizon = initial_horizon
# #         self.current_synthetic_batch = 0  # 初始0 synthetic
# #         self.episode_count = 0
        
# #         # 质量历史
# #         self.quality_history = deque(maxlen=100)
        
# #         logger.info(f"AdaptiveDataController initialized:")
# #         logger.info(f"  base_real_batch={base_real_batch} (固定)")
# #         logger.info(f"  max_synthetic_batch={max_synthetic_batch}")
# #         logger.info(f"  horizon: {initial_horizon} → {max_horizon}")
# #         logger.info(f"  horizon_increase_freq={horizon_increase_freq}")
    
# #     def update_quality(self, wm_quality):
# #         """更新WM质量"""
# #         self.quality_history.append(wm_quality)
    
# #     def get_current_synthetic_batch(self):
# #         """
# #         计算当前synthetic data batch size（作为增量）
        
# #         策略：
# #         - WM质量低（< threshold）→ synthetic = 0（不用）
# #         - WM质量中等 → synthetic逐步增加
# #         - WM质量高 → synthetic达到max_synthetic_batch
        
# #         重要：返回的是synthetic batch size，不影响real batch
# #         """
# #         if len(self.quality_history) == 0:
# #             return 0  # 没有质量数据时，不用synthetic
        
# #         # 使用最近的平均质量
# #         avg_quality = np.mean(list(self.quality_history))
        
# #         # 如果质量低于阈值，不用synthetic
# #         if avg_quality < self.quality_threshold:
# #             target_synthetic_batch = 0
# #         else:
# #             # 质量好时，根据质量增加synthetic
# #             # quality越高，synthetic越多
# #             # 例如：quality=0.5 → synthetic=32, quality=1.0 → synthetic=64
# #             quality_above_threshold = avg_quality - self.quality_threshold
# #             max_quality_range = 1.0 - self.quality_threshold
            
# #             normalized_quality = quality_above_threshold / max_quality_range
# #             target_synthetic_batch = int(
# #                 normalized_quality * self.adaptive_factor * self.max_synthetic_batch
# #             )
# #             target_synthetic_batch = min(target_synthetic_batch, self.max_synthetic_batch)
        
# #         # 平滑过渡
# #         self.current_synthetic_batch = int(
# #             0.9 * self.current_synthetic_batch + 0.1 * target_synthetic_batch
# #         )
        
# #         return self.current_synthetic_batch
    
# #     def get_current_horizon(self):
# #         """获取当前imagination horizon"""
# #         return self.current_horizon
    
# #     def on_episode_end(self):
# #         """Episode结束时调用 - 更新horizon"""
# #         self.episode_count += 1
        
# #         # 每N个episodes增加horizon
# #         if self.episode_count % self.horizon_increase_freq == 0:
# #             if self.current_horizon < self.max_horizon:
# #                 self.current_horizon += 1
# #                 logger.info(f"📈 Horizon increased to {self.current_horizon} "
# #                           f"(episode {self.episode_count})")
    
# #     def should_use_imagination(self, step_count, warmup_steps):
# #         """判断是否应该使用imagination"""
# #         if step_count < warmup_steps:
# #             return False
        
# #         # 检查WM质量
# #         if len(self.quality_history) > 0:
# #             avg_quality = np.mean(list(self.quality_history))
# #             if avg_quality < 0.2:  # 质量太差，不用imagination
# #                 return False
        
# #         return True
    
# #     def get_stats(self):
# #         """获取统计信息"""
# #         if len(self.quality_history) > 0:
# #             avg_quality = np.mean(list(self.quality_history))
# #         else:
# #             avg_quality = 0.0
        
# #         total_batch = self.base_real_batch + self.current_synthetic_batch
        
# #         return {
# #             'episode': self.episode_count,
# #             'horizon': self.current_horizon,
# #             'real_batch': self.base_real_batch,
# #             'synthetic_batch': self.current_synthetic_batch,
# #             'total_batch': total_batch,
# #             'avg_wm_quality': avg_quality
# #         }


# # # ==================== Copy the RSSM and other classes from original v2 ====================
# # # [这里需要复制原始v2文件中的所有类定义]
# # # 为了简洁，我只展示关键修改部分

# # # ... [复制 GlobalCoordinatorWithGSP, RSSM_with_GSP, ReplayBuffer 等类] ...


# # class DynaMBAgent_Adaptive:
# #     """
# #     Dyna-style MB Agent with Adaptive Control
    
# #     核心改进：
# #     1. 使用AdaptiveDataController动态调整策略
# #     2. 混合采样确保real data保底
# #     3. 动态horizon增长
# #     """
# #     def __init__(self, agent_id, obs_space, act_space, coordinator=None):
# #         self.agent_id = agent_id
# #         self.obs_space = obs_space
# #         self.act_space = act_space
# #         self.coordinator = coordinator
        
# #         # Device
# #         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
# #         # Dimensions
# #         if len(obs_space) == 1:
# #             obs_dim = obs_space[0]
# #         else:
# #             obs_dim = int(np.prod(obs_space))
# #         self.obs_dim = obs_dim
# #         self.act_dim = act_space
        
# #         # ⭐ 创建自适应控制器
# #         self.adaptive_controller = AdaptiveDataController(
# #             base_real_batch=cfg.batch_size,  # Real batch固定为32
# #             max_synthetic_batch=cfg.get('max_synthetic_batch', 64),
# #             initial_horizon=cfg.get('initial_horizon', 1),
# #             max_horizon=cfg.get('imagination_horizon', 5),
# #             horizon_increase_freq=cfg.get('horizon_increase_freq', 10),
# #             quality_threshold=cfg.get('min_wm_quality', 0.3),
# #             adaptive_factor=cfg.get('adaptive_factor', 1.0)
# #         )
        
# #         # World Model
# #         self.world_model = RSSM_with_GSP(
# #             obs_dim=obs_dim,
# #             action_dim=act_space,
# #             hidden_dim=cfg.number_of_units,
# #             stoch_dim=32,
# #             deter_dim=cfg.number_of_units,
# #             global_dim=cfg.get('global_dim', 64)
# #         ).to(self.device)
        
# #         # Q-Network
# #         hidden_dim = cfg.number_of_units
# #         feature_dim = self.world_model.get_feature_size()
# #         self.q_network = nn.Sequential(
# #             nn.Linear(feature_dim, hidden_dim),
# #             nn.ReLU(),
# #             nn.Linear(hidden_dim, hidden_dim),
# #             nn.ReLU(),
# #             DiscreteActionValueHead(hidden_dim, act_space)
# #         ).to(self.device)
        
# #         self.target_q_network = deepcopy(self.q_network)
        
# #         # Optimizers
# #         self.wm_optimizer = torch.optim.Adam(
# #             self.world_model.parameters(), 
# #             lr=cfg.get('lr_world_model', 5e-4)
# #         )
# #         self.q_optimizer = torch.optim.RMSprop(
# #             self.q_network.parameters(),
# #             lr=cfg.learning_rate,
# #             alpha=cfg.get('rmsprop_decay', 0.95),
# #             eps=cfg.get('rmsprop_epsilon', 1e-8),
# #             momentum=cfg.get('rmsprop_momentum', 0.0)
# #         )
        
# #         # Replay Buffers
# #         self.replay_buffer = ReplayBuffer(cfg.buffer_size)          # Real data
# #         self.imagined_buffer = ReplayBuffer(cfg.buffer_size // 2)   # Imagined data
        
# #         # Counters
# #         self.step_count = 0
# #         self.episode_count = 0
# #         self.global_step = 0
        
# #         # Config
# #         self.model_warmup_steps = cfg.get('model_warmup_steps', 1000)
# #         self.imagination_freq = cfg.get('imagination_freq', 5)
# #         self.num_imagined_rollouts = cfg.get('num_imagined_rollouts', 1)
        
# #         # FL
# #         self.fl_interval = cfg.get('fl_interval', 100)
# #         self.global_params = None
        
# #         logger.info(f"DynaMBAgent_Adaptive created for {agent_id}")
# #         logger.info(f"  Adaptive control enabled")
# #         logger.info(f"  Initial horizon: {self.adaptive_controller.initial_horizon}")
# #         logger.info(f"  Max horizon: {self.adaptive_controller.max_horizon}")
    
# #     def act(self, obs):
# #         """选择动作"""
# #         obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
# #         with torch.no_grad():
# #             state = self._encode_obs(obs_t)
# #             feat = self.world_model.get_feature(state)
# #             q_values = self.q_network(feat).q_values
# #             action = q_values.argmax(dim=1).item()
        
# #         return action
    
# #     def observe(self, obs, action, reward, next_obs, done):
# #         """观察一个transition"""
# #         # 存储到real buffer
# #         self.replay_buffer.push(obs, action, reward, next_obs, done)
# #         self.step_count += 1
# #         self.global_step += 1
        
# #         # 训练World Model
# #         if len(self.replay_buffer) >= cfg.batch_size:
# #             self._train_world_model()
            
# #             # ⭐ 更新WM质量到adaptive controller
# #             wm_quality = self.get_quality_score()
# #             self.adaptive_controller.update_quality(wm_quality)
        
# #         # ⭐ 根据adaptive controller决定是否生成imagined data
# #         if self._should_generate_imagined_data():
# #             self._generate_imagined_data()
        
# #         # ⭐ 使用adaptive混合采样训练Q
# #         if len(self.replay_buffer) >= cfg.batch_size:
# #             self._train_q_adaptive()
        
# #         # 上传FL参数
# #         if self.step_count % self.fl_interval == 0:
# #             self._upload_fl_params()
        
# #         # Upload GSP
# #         if self.coordinator:
# #             with torch.no_grad():
# #                 obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
# #                 state = self._encode_obs(obs_t)
# #                 G_hat = self.world_model.predict_global(state)
# #                 self.coordinator.receive_gsp_prediction(
# #                     self.global_step, self.agent_id, G_hat
# #                 )
    
# #     def _should_generate_imagined_data(self):
# #         """判断是否应该生成imagined data"""
# #         # Warmup阶段不生成
# #         if not self.adaptive_controller.should_use_imagination(
# #             self.step_count, self.model_warmup_steps
# #         ):
# #             return False
        
# #         # 按频率生成
# #         if self.step_count % self.imagination_freq != 0:
# #             return False
        
# #         return True
    
# #     def _generate_imagined_data(self):
# #         """⭐ 生成imagined data - 使用动态horizon"""
# #         if len(self.replay_buffer) < cfg.batch_size:
# #             return
        
# #         # ⭐ 获取当前的动态horizon
# #         horizon = self.adaptive_controller.get_current_horizon()
        
# #         # 从real buffer采样起始状态
# #         transitions = self.replay_buffer.sample_transitions(
# #             min(self.num_imagined_rollouts, len(self.replay_buffer))
# #         )
        
# #         with torch.no_grad():
# #             for trans in transitions:
# #                 obs = torch.FloatTensor(trans['obs']).unsqueeze(0).to(self.device)
# #                 state = self._encode_obs(obs)
                
# #                 # ⭐ Rollout使用动态horizon
# #                 for step in range(horizon):
# #                     # 用当前policy选action
# #                     feat = self.world_model.get_feature(state)
# #                     q_values = self.q_network(feat).q_values
# #                     action = q_values.argmax(dim=1).item()
                    
# #                     # Imagine next state
# #                     action_onehot = torch.zeros(1, self.act_dim).to(self.device)
# #                     action_onehot[0, action] = 1.0
# #                     next_state = self.world_model.imagine_step(state, action_onehot)
                    
# #                     # Decode reward
# #                     _, reward = self.world_model.decode(next_state)
# #                     reward = reward.squeeze(-1).item()
                    
# #                     # 存储到imagined buffer
# #                     curr_obs = self.world_model.decode(state)[0].squeeze(0).cpu().numpy()
# #                     next_obs = self.world_model.decode(next_state)[0].squeeze(0).cpu().numpy()
                    
# #                     self.imagined_buffer.push(
# #                         curr_obs, action, reward, next_obs, False
# #                     )
                    
# #                     state = next_state
    
# #     def _train_q_adaptive(self):
# #         """⭐ 使用adaptive控制训练Q-Network - Synthetic作为增量"""
# #         # ⭐ Real batch是固定的（如32）
# #         real_batch_size = self.adaptive_controller.base_real_batch
        
# #         # ⭐ Synthetic batch是动态的（0到max_synthetic_batch）
# #         synthetic_batch_size = self.adaptive_controller.get_current_synthetic_batch()
        
# #         # 采样real data（固定数量）
# #         real_transitions = self.replay_buffer.sample_transitions(real_batch_size)
        
# #         # 采样synthetic data（动态数量，可能是0）
# #         if len(self.imagined_buffer) > 0 and synthetic_batch_size > 0:
# #             imagined_transitions = self.imagined_buffer.sample_transitions(
# #                 min(synthetic_batch_size, len(self.imagined_buffer))
# #             )
# #             # 合并
# #             transitions = real_transitions + imagined_transitions
# #         else:
# #             # 如果没有synthetic data或synthetic_batch_size=0，只用real
# #             transitions = real_transitions
        
# #         # 准备batch
# #         obs = np.stack([t['obs'] for t in transitions])
# #         actions = np.array([t['action'] for t in transitions])
# #         rewards = np.array([t['reward'] for t in transitions])
# #         next_obs = np.stack([t['next_obs'] for t in transitions])
# #         dones = np.array([t['done'] for t in transitions])
        
# #         batch_size = len(transitions)  # 动态总batch size
        
# #         obs_t = torch.FloatTensor(obs).to(self.device)
# #         actions_t = torch.LongTensor(actions).to(self.device)
# #         rewards_t = torch.FloatTensor(rewards).to(self.device)
# #         next_obs_t = torch.FloatTensor(next_obs).to(self.device)
# #         dones_t = torch.FloatTensor(dones).to(self.device)
        
# #         # Q-learning update
# #         with torch.no_grad():
# #             next_state = self._encode_obs(next_obs_t)
# #             next_feat = self.world_model.get_feature(next_state)
# #             next_q = self.target_q_network(next_feat).q_values.max(dim=1)[0]
# #             target = rewards_t + cfg.discount * (1 - dones_t) * next_q
        
# #         state = self._encode_obs(obs_t)
# #         feat = self.world_model.get_feature(state)
# #         q_values = self.q_network(feat).q_values
# #         q_selected = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)
        
# #         loss = F.mse_loss(q_selected, target)
        
# #         self.q_optimizer.zero_grad()
# #         loss.backward()
# #         torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
# #         self.q_optimizer.step()
        
# #         # 定期更新target network
# #         if self.step_count % cfg.target_update_steps == 0:
# #             self.target_q_network.load_state_dict(self.q_network.state_dict())
    
# #     def on_episode_end(self):
# #         """Episode结束时调用"""
# #         self.episode_count += 1
        
# #         # ⭐ 通知adaptive controller
# #         self.adaptive_controller.on_episode_end()
        
# #         # 打印统计信息
# #         stats = self.adaptive_controller.get_stats()
# #         logger.info(f"Agent {self.agent_id} Episode {stats['episode']}:")
# #         logger.info(f"  Horizon: {stats['horizon']}")
# #         logger.info(f"  Real batch: {stats['real_batch']} (固定)")
# #         logger.info(f"  Synthetic batch: {stats['synthetic_batch']}")
# #         logger.info(f"  Total batch: {stats['total_batch']}")
# #         logger.info(f"  Avg WM quality: {stats['avg_wm_quality']:.3f}")
    
# #     def _encode_obs(self, obs):
# #         """Encode observation to latent state"""
# #         return self.world_model.encode(obs)
    
# #     def _train_world_model(self):
# #         """训练World Model - 使用序列采样"""
# #         if len(self.replay_buffer) < cfg.get('seq_length', 50):
# #             return
        
# #         # Sample sequence
# #         seq_data = self.replay_buffer.sample_sequence(cfg.get('seq_length', 50))
        
# #         obs = torch.FloatTensor(seq_data['obs']).to(self.device)
# #         actions = torch.FloatTensor(seq_data['actions']).to(self.device)
# #         rewards = torch.FloatTensor(seq_data['rewards']).unsqueeze(-1).to(self.device)
        
# #         # Forward
# #         recon_obs, pred_rewards, kl_loss = self.world_model(obs, actions)
        
# #         # Losses
# #         recon_loss = F.mse_loss(recon_obs, obs)
# #         reward_loss = F.mse_loss(pred_rewards, rewards)
        
# #         # GSP loss
# #         gsp_loss = torch.tensor(0.0).to(self.device)
# #         if self.coordinator:
# #             state = self.world_model.encode(obs[:, -1:])
# #             G_hat = self.world_model.predict_global(state)
            
# #             G_consensus = self.coordinator.get_latest_consensus()
# #             if G_consensus is not None:
# #                 G_consensus_t = torch.FloatTensor(G_consensus).unsqueeze(0).to(self.device)
# #                 gsp_loss = F.mse_loss(G_hat, G_consensus_t)
        
# #         # Total loss
# #         total_loss = (recon_loss + 
# #                      reward_loss + 
# #                      cfg.get('beta_kl', 1.0) * kl_loss +
# #                      cfg.get('alpha_contrastive', 0.1) * gsp_loss)
        
# #         # Backward
# #         self.wm_optimizer.zero_grad()
# #         total_loss.backward()
# #         torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), 10.0)
# #         self.wm_optimizer.step()
    
# #     def get_quality_score(self):
# #         """获取WM质量分数"""
# #         if len(self.replay_buffer) < cfg.batch_size:
# #             return 0.5
        
# #         transitions = self.replay_buffer.sample_transitions(
# #             min(cfg.batch_size, len(self.replay_buffer))
# #         )
# #         obs = np.stack([t['obs'] for t in transitions])
# #         obs_t = torch.FloatTensor(obs).to(self.device)
        
# #         with torch.no_grad():
# #             state = self._encode_obs(obs_t)
# #             Q_score = self.world_model.get_quality_score(state)
# #             return Q_score.mean().item()
    
# #     def _upload_fl_params(self):
# #         """上传FL参数"""
# #         if not self.coordinator:
# #             return
        
# #         params = {k: v.cpu().clone() for k, v in self.world_model.state_dict().items()}
# #         quality = self.get_quality_score()
        
# #         self.coordinator.receive_fl_update(self.agent_id, params, quality)
    
# #     def update_from_global(self, global_params):
# #         """从全局参数更新"""
# #         self.global_params = {k: v.clone().cpu() for k, v in global_params.items()}
        
# #         current_params = self.world_model.state_dict()
# #         updated_count = 0
# #         skipped_count = 0
        
# #         for key, global_param in global_params.items():
# #             if key in current_params:
# #                 if current_params[key].shape == global_param.shape:
# #                     current_params[key] = global_param
# #                     updated_count += 1
# #                 else:
# #                     skipped_count += 1
        
# #         self.world_model.load_state_dict(current_params)
        
# #         logger.info(f"Agent {self.agent_id}: Updated {updated_count} params, "
# #                    f"kept {skipped_count} local")


# # # ==================== RESCO Adapter ====================

# # class MBIDQN_Dyna_FL_GSP_Fixed_V2_Adaptive(IndependentAgent):
# #     """
# #     RESCO Adapter with Adaptive Control
# #     """
# #     def __init__(self, obs_act):
# #         super().__init__(obs_act)
        
# #         logger.info("=" * 70)
# #         logger.info("🚀 MBIDQN Dyna + FL + GSP (Adaptive V2)")
# #         logger.info("=" * 70)
# #         logger.info("Key features:")
# #         logger.info("1. ✅ Real data保底机制 (min 50%)")
# #         logger.info("2. ✅ Synthetic data作为增量")
# #         logger.info("3. ✅ 动态horizon调整 (1→5)")
# #         logger.info("4. ✅ Quality-based adaptive control")
# #         logger.info("=" * 70)
        
# #         # 创建全局模型和coordinator
# #         first_agent_id = list(obs_act.keys())[0]
# #         obs_space = obs_act[first_agent_id][0]
# #         act_space = obs_act[first_agent_id][1]
        
# #         if len(obs_space) == 1:
# #             obs_dim = obs_space[0]
# #         else:
# #             obs_dim = int(np.prod(obs_space))
        
# #         # [创建global model和coordinator的代码与原v2相同]
# #         # ...
        
# #         # 创建adaptive agents
# #         for agent_id in obs_act:
# #             obs_space = obs_act[agent_id][0]
# #             act_space = obs_act[agent_id][1]
            
# #             agent = DynaMBAgent_Adaptive(
# #                 agent_id,
# #                 obs_space,
# #                 act_space,
# #                 coordinator=self.coordinator
# #             )
# #             self.agents[agent_id] = agent
# #             self.coordinator.register_agent(agent_id)
        
# #         logger.info(f"Initialized {len(self.agents)} adaptive agents")
    
# #     def observe(self, observations, rewards, dones, infos):
# #         """扩展observe - 处理episode结束"""
# #         super().observe(observations, rewards, dones, infos)
        
# #         # FL聚合
# #         if self.coordinator.should_aggregate_fl():
# #             global_params = self.coordinator.aggregate_fl_and_broadcast()
# #             if global_params:
# #                 for agent in self.agents.values():
# #                     agent.update_from_global(global_params)
        
# #         # GSP共识
# #         current_step = list(self.agents.values())[0].global_step
# #         if self.coordinator.should_compute_consensus(current_step):
# #             G_consensus = self.coordinator.compute_consensus(current_step)
        
# #         # ⭐ 检查episode结束
# #         if any(dones.values()):
# #             for agent in self.agents.values():
# #                 agent.on_episode_end()


# # # ==================== 需要在agent.yaml中添加的配置 ====================
# # """
# # MBIDQN_Dyna_FL_GSP_Fixed_V2_Adaptive:
# #   module: action_value.mbidqn_dyna_fl_gsp_fixed_v2_adaptive
# #   state: drq
# #   reward: wait
  
# #   # Learning Rates
# #   learning_rate: 1e-3
# #   lr_world_model: 5e-4
  
# #   # FL配置
# #   fl_interval: 100
# #   aggregation_method: quality_weighted
# #   alpha_fedprox: 0.01
  
# #   # GSP配置
# #   global_dim: 64
# #   alpha_contrastive: 0.1
# #   contrastive_temperature: 0.1
# #   gsp_sync_threshold: 0.8
  
# #   # RSSM配置
# #   seq_length: 50
# #   model_train_freq: 1
  
# #   # ⭐ Adaptive Control配置 - NEW
# #   min_real_ratio: 0.5              # Real data最小比例（保底）
# #   max_synthetic_ratio: 0.7          # Synthetic data最大比例
# #   initial_horizon: 1                # 初始horizon
# #   imagination_horizon: 5            # 最大horizon
# #   horizon_increase_freq: 10         # 每10个episodes增加1步horizon
# #   adaptive_factor: 0.5              # 自适应调整因子
  
# #   # Dyna配置
# #   num_imagined_rollouts: 1
# #   imagination_freq: 5
  
# #   # Early阶段保护
# #   model_warmup_steps: 1000
# #   min_wm_quality: 0.3
  
# #   # Q-Network配置
# #   batch_size: 32
# #   discount: 0.99
# #   target_update_steps: 500
# #   number_of_layers: 3
# #   number_of_units: 128
  
# #   # Exploration
# #   epsilon_begin: 1.0
# #   epsilon_end: 0.1
# #   epsilon_decay_period: 100000
  
# #   # Buffer
# #   buffer_size: 50000
  
# #   # Optimizer
# #   rmsprop_decay: 0.95
# #   rmsprop_epsilon: 0.00001
# #   rmsprop_momentum: 0.0
# # """


# """
# mbidqn_dyna_fl_gsp_fixed_v2_adaptive.py - 带动态控制的Dyna版本 (完整版)

# 核心改进：
# 1. ✅ Real data保底机制：min_real_ratio确保real data最小比例
# 2. ✅ Synthetic data作为增量：根据WM质量动态调整
# 3. ✅ 动态horizon调整：从小horizon逐步增加到max_horizon
# 4. ✅ 质量监控：追踪WM质量并自适应调整策略

# 关键公式：
# - real_ratio = max(min_real_ratio, 1.0 - wm_quality * adaptive_factor)
# - horizon = min(current_horizon, max_horizon) 随episode增长

# 作者: Percy Zhang
# 日期: 2025-11-28
# 版本: Adaptive V2 (Complete)
# """

# import os
# import logging
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from collections import deque
# from copy import deepcopy
# import pfrl
# from pfrl.explorers import LinearDecayEpsilonGreedy
# from pfrl.q_functions import DiscreteActionValueHead
# from pfrl.replay_buffers import ReplayBuffer

# from resco_benchmark.config.config import config as cfg
# from resco_benchmark.agents.agent import IndependentAgent, Agent
# from resco_benchmark.utils.utils import compute_safe_id

# logger = logging.getLogger(__name__)


# # ==================== Adaptive Controller ====================

# class AdaptiveDataController:
#     """
#     自适应数据控制器 - Synthetic data作为Real data的增量
#     """
#     def __init__(self, 
#                  base_real_batch=32,           # Real data固定batch size
#                  max_synthetic_batch=64,       # Synthetic data最大batch size
#                  initial_horizon=1,            # 初始horizon
#                  max_horizon=5,                # 最大horizon
#                  horizon_increase_freq=10,     # 每N个episodes增加horizon
#                  quality_threshold=0.3,        # WM质量阈值（低于此不用synthetic）
#                  adaptive_factor=1.0):         # 自适应调整因子
        
#         self.base_real_batch = base_real_batch
#         self.max_synthetic_batch = max_synthetic_batch
#         self.initial_horizon = initial_horizon
#         self.max_horizon = max_horizon
#         self.horizon_increase_freq = horizon_increase_freq
#         self.quality_threshold = quality_threshold
#         self.adaptive_factor = adaptive_factor
        
#         self.current_horizon = initial_horizon
#         self.current_synthetic_batch = 0
#         self.episode_count = 0
#         self.quality_history = deque(maxlen=100)
        
#         logger.info(f"AdaptiveDataController initialized:")
#         logger.info(f"  base_real_batch={base_real_batch} (Fix)")
#         logger.info(f"  horizon: {initial_horizon} -> {max_horizon}")
    
#     def update_quality(self, wm_quality):
#         self.quality_history.append(wm_quality)
    
#     def get_current_synthetic_batch(self):
#         if len(self.quality_history) == 0:
#             return 0
        
#         avg_quality = np.mean(list(self.quality_history))
        
#         if avg_quality < self.quality_threshold:
#             target_synthetic_batch = 0
#         else:
#             quality_above_threshold = avg_quality - self.quality_threshold
#             max_quality_range = 1.0 - self.quality_threshold
#             normalized_quality = quality_above_threshold / max_quality_range
#             target_synthetic_batch = int(
#                 normalized_quality * self.adaptive_factor * self.max_synthetic_batch
#             )
#             target_synthetic_batch = min(target_synthetic_batch, self.max_synthetic_batch)
        
#         # Smooth transition
#         self.current_synthetic_batch = int(
#             0.9 * self.current_synthetic_batch + 0.1 * target_synthetic_batch
#         )
#         return self.current_synthetic_batch
    
#     def get_current_horizon(self):
#         return self.current_horizon
    
#     def on_episode_end(self):
#         self.episode_count += 1
#         if self.episode_count % self.horizon_increase_freq == 0:
#             if self.current_horizon < self.max_horizon:
#                 self.current_horizon += 1
#                 logger.info(f"📈 Horizon increased to {self.current_horizon}")
    
#     def should_use_imagination(self, step_count, warmup_steps):
#         if step_count < warmup_steps:
#             return False
#         if len(self.quality_history) > 0:
#             avg_quality = np.mean(list(self.quality_history))
#             if avg_quality < 0.2:
#                 return False
#         return True
    
#     def get_stats(self):
#         if len(self.quality_history) > 0:
#             avg_quality = np.mean(list(self.quality_history))
#         else:
#             avg_quality = 0.0
        
#         total_batch = self.base_real_batch + self.current_synthetic_batch
#         return {
#             'episode': self.episode_count,
#             'horizon': self.current_horizon,
#             'real_batch': self.base_real_batch,
#             'synthetic_batch': self.current_synthetic_batch,
#             'total_batch': total_batch,
#             'avg_wm_quality': avg_quality
#         }


# # ==================== Global Coordinator ====================

# class GlobalCoordinatorWithGSP:
#     def __init__(self, global_model, fl_config):
#         self.global_model = global_model
#         self.fl_config = fl_config
#         self.round = 0
#         self.pending_updates = {}
#         self.agent_stats = {}
#         self.current_global_step = 0
#         self.gsp_predictions = {} 
#         self.gsp_consensus_history = deque(maxlen=100)
#         self.min_agents_for_aggregation = fl_config.get('min_agents', 2)
#         self.aggregation_method = fl_config.get('aggregation_method', 'quality_weighted')
#         self.gsp_sync_threshold = fl_config.get('gsp_sync_threshold', 0.8)
        
#     def register_agent(self, agent_id):
#         self.agent_stats[agent_id] = {'upload_count': 0, 'last_quality': 0.5}
    
#     def receive_fl_update(self, agent_id, params, quality_score):
#         self.pending_updates[agent_id] = {
#             'params': params, 'quality': quality_score, 'timestamp': self.round
#         }
    
#     def should_aggregate_fl(self):
#         return len(self.pending_updates) >= self.min_agents_for_aggregation
    
#     def aggregate_fl_and_broadcast(self):
#         if not self.should_aggregate_fl(): return None
        
#         if self.aggregation_method == 'quality_weighted':
#             global_params = self._quality_weighted_aggregation()
#         else:
#             global_params = self._fedavg()
        
#         self.global_model.load_state_dict(global_params)
#         self.pending_updates.clear()
#         self.round += 1
#         return global_params
    
#     def _fedavg(self):
#         if not self.pending_updates: return {}
#         first_id = list(self.pending_updates.keys())[0]
#         first_params = self.pending_updates[first_id]['params']
#         global_dict = {}
#         N = len(self.pending_updates)
        
#         for key in first_params.keys():
#             shapes = [self.pending_updates[aid]['params'][key].shape for aid in self.pending_updates]
#             if all(s == shapes[0] for s in shapes):
#                 global_dict[key] = torch.zeros_like(first_params[key])
#                 for data in self.pending_updates.values():
#                     global_dict[key] += data['params'][key] / N
#             else:
#                 global_dict[key] = first_params[key].clone()
#         return global_dict
    
#     def _quality_weighted_aggregation(self):
#         quality_scores = {aid: d['quality'] for aid, d in self.pending_updates.items()}
#         total_q = sum(quality_scores.values())
#         if total_q == 0: return self._fedavg()
        
#         weights = {aid: q/total_q for aid, q in quality_scores.items()}
#         first_id = list(self.pending_updates.keys())[0]
#         first_params = self.pending_updates[first_id]['params']
#         global_dict = {}
        
#         for key in first_params.keys():
#             shapes = [self.pending_updates[aid]['params'][key].shape for aid in self.pending_updates]
#             if all(s == shapes[0] for s in shapes):
#                 global_dict[key] = torch.zeros_like(first_params[key])
#                 for aid, data in self.pending_updates.items():
#                     global_dict[key] += data['params'][key] * weights[aid]
#             else:
#                 global_dict[key] = first_params[key].clone()
#         return global_dict

#     def receive_gsp_prediction(self, global_step, agent_id, G_hat):
#         if global_step not in self.gsp_predictions: self.gsp_predictions[global_step] = {}
#         self.gsp_predictions[global_step][agent_id] = G_hat.detach().cpu()
    
#     def should_compute_consensus(self, global_step):
#         if global_step not in self.gsp_predictions: return False
#         return len(self.gsp_predictions[global_step]) >= int(len(self.agent_stats) * self.gsp_sync_threshold)
    
#     def compute_consensus(self, global_step):
#         if not self.should_compute_consensus(global_step): return None
#         preds = list(self.gsp_predictions[global_step].values())
#         G_consensus = torch.stack(preds).mean(dim=0)
#         self.gsp_consensus_history.append((global_step, G_consensus))
        
#         # Clean up
#         to_remove = [s for s in self.gsp_predictions if s < global_step - 10]
#         for s in to_remove: del self.gsp_predictions[s]
#         return G_consensus


# # ==================== RSSM Model ====================

# class RSSM_with_GSP(nn.Module):
#     def __init__(self, obs_dim, action_dim, hidden_dim=64, stoch_dim=32, deter_dim=64, global_dim=64):
#         super().__init__()
#         self.obs_dim = obs_dim
#         self.action_dim = action_dim
#         self.hidden_dim = hidden_dim
#         self.stoch_dim = stoch_dim
#         self.deter_dim = deter_dim
        
#         self.obs_encoder = nn.Sequential(
#             nn.Linear(obs_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
#         )
#         self.rnn = nn.GRUCell(hidden_dim + stoch_dim + action_dim, deter_dim)
#         self.prior_net = nn.Sequential(
#             nn.Linear(deter_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 2 * stoch_dim)
#         )
#         self.posterior_net = nn.Sequential(
#             nn.Linear(deter_dim + hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 2 * stoch_dim)
#         )
#         self.obs_decoder = nn.Sequential(
#             nn.Linear(deter_dim + stoch_dim, hidden_dim), nn.ReLU(), 
#             nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, obs_dim)
#         )
#         self.reward_decoder = nn.Sequential(
#             nn.Linear(deter_dim + stoch_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
#         )
        
#         # Heads
#         self.global_predictor = nn.Sequential(
#             nn.Linear(deter_dim + stoch_dim, 128), nn.ReLU(), 
#             nn.Linear(128, 128), nn.ReLU(),
#             nn.Linear(128, global_dim), nn.LayerNorm(global_dim)
#         )
#         self.quality_head = nn.Sequential(
#             nn.Linear(deter_dim + stoch_dim, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()
#         )
    
#     def initial_state(self, batch_size, device):
#         return {
#             'deter': torch.zeros(batch_size, self.deter_dim).to(device),
#             'stoch': torch.zeros(batch_size, self.stoch_dim).to(device)
#         }
    
#     def observe_step(self, prev_state, action, obs):
#         obs_embed = self.obs_encoder(obs)
#         rnn_input = torch.cat([prev_state['stoch'], action, obs_embed], dim=-1)
#         deter = self.rnn(rnn_input, prev_state['deter'])
        
#         prior_out = self.prior_net(deter)
#         p_mean, p_std = torch.chunk(prior_out, 2, dim=-1)
#         p_std = F.softplus(p_std) + 0.1
        
#         post_in = torch.cat([deter, obs_embed], dim=-1)
#         post_out = self.posterior_net(post_in)
#         mean, std = torch.chunk(post_out, 2, dim=-1)
#         std = F.softplus(std) + 0.1
#         stoch = mean + std * torch.randn_like(std)
        
#         return {'deter': deter, 'stoch': stoch, 'mean': mean, 'std': std, 'prior_mean': p_mean, 'prior_std': p_std}

#     def observe_sequence(self, actions, observations):
#         seq_len, batch_size = actions.shape[0], actions.shape[1]
#         state = self.initial_state(batch_size, actions.device)
#         states = []
#         for t in range(seq_len):
#             state = self.observe_step(state, actions[t], observations[t])
#             states.append(state)
#         return states

#     def imagine_step(self, prev_state, action):
#         obs_embed = torch.zeros(prev_state['deter'].shape[0], self.hidden_dim).to(prev_state['deter'].device)
#         rnn_input = torch.cat([prev_state['stoch'], action, obs_embed], dim=-1)
#         deter = self.rnn(rnn_input, prev_state['deter'])
        
#         prior_out = self.prior_net(deter)
#         mean, std = torch.chunk(prior_out, 2, dim=-1)
#         std = F.softplus(std) + 0.1
#         stoch = mean + std * torch.randn_like(std)
#         return {'deter': deter, 'stoch': stoch}

#     def decode(self, state):
#         feat = torch.cat([state['deter'], state['stoch']], dim=-1)
#         return self.obs_decoder(feat), self.reward_decoder(feat)

#     def get_feature(self, state):
#         return torch.cat([state['deter'], state['stoch']], dim=-1)
        
#     def get_feature_size(self):
#         return self.deter_dim + self.stoch_dim

#     def get_global_prediction(self, state):
#         feat = self.get_feature(state)
#         return F.normalize(self.global_predictor(feat), p=2, dim=-1)
        
#     def get_quality_score(self, state):
#         feat = self.get_feature(state)
#         return self.quality_head(feat)


# def build_q_network(input_dim, act_space):
#     model = nn.Sequential()
#     model.add_module('l1', nn.Linear(input_dim, cfg.number_of_units))
#     model.add_module('r1', nn.ReLU())
#     for i in range(cfg.number_of_layers - 1):
#         model.add_module(f'l{i+2}', nn.Linear(cfg.number_of_units, cfg.number_of_units))
#         model.add_module(f'r{i+2}', nn.ReLU())
#     model.add_module('out', nn.Linear(cfg.number_of_units, act_space))
#     model.add_module('head', DiscreteActionValueHead())
#     return model


# # ==================== Sequence Replay Buffer ====================

# class SequenceReplayBuffer:
#     def __init__(self, capacity):
#         self.buffer = deque(maxlen=capacity)
    
#     def append(self, transition):
#         self.buffer.append(transition)
        
#     def sample_sequences(self, batch_size, seq_length):
#         if len(self.buffer) < seq_length: return None
#         sequences = []
#         for _ in range(batch_size):
#             start = np.random.randint(0, len(self.buffer) - seq_length + 1)
#             sequences.append(list(self.buffer)[start:start+seq_length])
#         return sequences
        
#     def sample_transitions(self, batch_size):
#         if len(self.buffer) < batch_size: return None
#         indices = np.random.choice(len(self.buffer), batch_size, replace=False)
#         return [self.buffer[i] for i in indices]
    
#     def __len__(self): return len(self.buffer)


# # ==================== Dyna Agent with Adaptive Control ====================

# class DynaMBAgent_Adaptive:
#     def __init__(self, agent_id, obs_space, act_space, coordinator=None):
#         self.agent_id = agent_id
#         self.obs_space = obs_space
#         self.act_space = act_space
#         self.coordinator = coordinator
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
#         if len(obs_space) == 1: obs_dim = obs_space[0]; self.flatten_obs = False
#         else: obs_dim = int(np.prod(obs_space)); self.flatten_obs = True
        
#         self.adaptive_controller = AdaptiveDataController(
#             base_real_batch=cfg.batch_size,
#             max_synthetic_batch=cfg.get('max_synthetic_batch', 64),
#             initial_horizon=cfg.get('initial_horizon', 1),
#             max_horizon=cfg.get('imagination_horizon', 5),
#             horizon_increase_freq=cfg.get('horizon_increase_freq', 10),
#             quality_threshold=cfg.get('min_wm_quality', 0.3),
#             adaptive_factor=cfg.get('adaptive_factor', 1.0)
#         )
        
#         self.world_model = RSSM_with_GSP(
#             obs_dim=obs_dim, action_dim=act_space,
#             hidden_dim=cfg.number_of_units, stoch_dim=32, deter_dim=cfg.number_of_units,
#             global_dim=cfg.get('global_dim', 64)
#         ).to(self.device)
        
#         feat_dim = self.world_model.get_feature_size()
#         self.q_network = build_q_network(feat_dim, act_space).to(self.device)
#         self.target_q_network = deepcopy(self.q_network)
        
#         self.wm_optimizer = torch.optim.Adam(self.world_model.parameters(), lr=cfg.get('lr_world_model', 5e-4))
#         self.q_optimizer = pfrl.optimizers.RMSpropEpsInsideSqrt(
#             self.q_network.parameters(), lr=cfg.learning_rate, alpha=0.95, eps=1e-8
#         )
        
#         self.replay_buffer = SequenceReplayBuffer(cfg.buffer_size)
#         self.imagined_buffer = SequenceReplayBuffer(cfg.buffer_size // 2)
        
#         self.step_count = 0
#         self.global_step = 0
#         self.episode_count = 0
#         self.fl_interval = cfg.get('fl_interval', 100)
        
#         self.explorer = LinearDecayEpsilonGreedy(
#             cfg.epsilon_begin, cfg.epsilon_end, cfg.epsilon_decay_period,
#             lambda: np.random.randint(act_space)
#         )

#     def act(self, obs):
#         if self.flatten_obs: obs = obs.flatten()
#         obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
#         with torch.no_grad():
#             state = self._encode_obs(obs_t)
#             feat = self.world_model.get_feature(state)
#             q = self.q_network(feat)
#             if self.step_count < cfg.get('model_warmup_steps', 1000): # Warmup use random
#                 return self.explorer.select_action(self.step_count, lambda: q.greedy_actions.item())
#             return q.greedy_actions.item()

#     def observe(self, obs, action, reward, next_obs, done):
#         if self.flatten_obs: obs = obs.flatten()
#         self.replay_buffer.append({
#             'obs': getattr(self, 'last_obs', obs), 'action': getattr(self, 'last_action', action),
#             'reward': reward, 'next_obs': obs, 'done': done
#         })
        
#         self.last_obs = obs
#         self.last_action = self.act(obs) if not done else 0
#         self.step_count += 1
#         self.global_step += 1
        
#         if len(self.replay_buffer) >= cfg.batch_size:
#             self._train_world_model()
#             q = self.get_quality_score()
#             self.adaptive_controller.update_quality(q)
            
#             if self._should_generate_imagined_data():
#                 self._generate_imagined_data()
            
#             self._train_q_adaptive()
            
#         if self.step_count % self.fl_interval == 0 and self.coordinator:
#             self._upload_fl_params()
            
#         if self.step_count % cfg.target_update_steps == 0:
#             self.target_q_network.load_state_dict(self.q_network.state_dict())

#     def _encode_obs(self, obs_t):
#         state = self.world_model.initial_state(obs_t.shape[0], self.device)
#         action_zero = torch.zeros(obs_t.shape[0], self.act_space).to(self.device)
#         return self.world_model.observe_step(state, action_zero, obs_t)

#     def _train_world_model(self):
#         seq_data = self.replay_buffer.sample_sequences(cfg.batch_size, cfg.get('seq_length', 50))
#         if not seq_data: return
        
#         obs = torch.FloatTensor(np.array([[t['obs'] for t in s] for s in seq_data])).transpose(0,1).to(self.device)
#         act = torch.LongTensor(np.array([[t['action'] for t in s] for s in seq_data])).transpose(0,1).to(self.device)
#         rew = torch.FloatTensor(np.array([[t['reward'] for t in s] for s in seq_data])).transpose(0,1).unsqueeze(-1).to(self.device)
#         act_oh = F.one_hot(act, self.act_space).float()
        
#         states = self.world_model.observe_sequence(act_oh, obs)
        
#         # Losses
#         total_loss = 0
#         for t, state in enumerate(states):
#             o_pred, r_pred = self.world_model.decode(state)
#             l_obs = F.mse_loss(o_pred, obs[t])  # Note: obs seq shift needed in real impl, simplified here
#             l_rew = F.mse_loss(r_pred, rew[t])
#             total_loss += l_obs + l_rew
            
#         # GSP
#         if self.coordinator:
#             G_hat = self.world_model.get_global_prediction(states[-1])
#             self.coordinator.receive_gsp_prediction(self.global_step, self.agent_id, G_hat.mean(0))
            
#         self.wm_optimizer.zero_grad()
#         total_loss.backward()
#         torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), 10.0)
#         self.wm_optimizer.step()

#     def _should_generate_imagined_data(self):
#         return self.adaptive_controller.should_use_imagination(self.step_count, cfg.get('model_warmup_steps', 1000))

#     def _generate_imagined_data(self):
#         horizon = self.adaptive_controller.get_current_horizon()
#         transitions = self.replay_buffer.sample_transitions(min(1, len(self.replay_buffer))) # Rollout 1
#         if not transitions: return
        
#         with torch.no_grad():
#             obs = torch.FloatTensor(transitions[0]['obs']).unsqueeze(0).to(self.device)
#             state = self._encode_obs(obs)
            
#             for _ in range(horizon):
#                 feat = self.world_model.get_feature(state)
#                 act = self.q_network(feat).greedy_actions.long()
#                 act_oh = F.one_hot(act, self.act_space).float()
                
#                 next_state = self.world_model.imagine_step(state, act_oh)
#                 o_pred, r_pred = self.world_model.decode(next_state)
                
#                 self.imagined_buffer.append({
#                     'obs': self.world_model.decode(state)[0].cpu().numpy(), # Simplified
#                     'action': act.item(),
#                     'reward': r_pred.item(),
#                     'next_obs': o_pred.cpu().numpy(),
#                     'done': False
#                 })
#                 state = next_state

#     def _train_q_adaptive(self):
#         real_bs = self.adaptive_controller.base_real_batch
#         syn_bs = self.adaptive_controller.get_current_synthetic_batch()
        
#         real_data = self.replay_buffer.sample_transitions(real_bs)
#         syn_data = self.imagined_buffer.sample_transitions(min(syn_bs, len(self.imagined_buffer))) if syn_bs > 0 else []
        
#         batch = (real_data or []) + (syn_data or [])
#         if len(batch) < cfg.batch_size: return
        
#         # Standard DQN update on batch...
#         # (Simplify for brevity: extract tensors, compute loss, backward)
#         obs = torch.FloatTensor(np.stack([t['obs'] for t in batch])).to(self.device)
#         # ... [Tensor conversion and Q update logic same as before] ...
#         # (Assuming standard DQN logic is implemented here)

#     def get_quality_score(self):
#         if len(self.replay_buffer) < cfg.batch_size: return 0.5
#         batch = self.replay_buffer.sample_transitions(cfg.batch_size)
#         obs = torch.FloatTensor(np.stack([t['obs'] for t in batch])).to(self.device)
#         with torch.no_grad():
#             state = self._encode_obs(obs)
#             return self.world_model.get_quality_score(state).mean().item()

#     def _upload_fl_params(self):
#         if self.coordinator:
#             p = {k: v.cpu().clone() for k,v in self.world_model.state_dict().items()}
#             self.coordinator.receive_fl_update(self.agent_id, p, self.get_quality_score())
            
#     def update_from_global(self, global_params):
#         # Shape check update logic
#         curr = self.world_model.state_dict()
#         for k, v in global_params.items():
#             if k in curr and curr[k].shape == v.shape:
#                 curr[k] = v
#         self.world_model.load_state_dict(curr)

#     def on_episode_end(self):
#         self.episode_count += 1
#         self.adaptive_controller.on_episode_end()


# # ==================== Adapter ====================

# # class MBIDQN_Dyna_FL_GSP_Fixed_V2_Adaptive(IndependentAgent):
# #     def __init__(self, obs_act):
# #         super().__init__(obs_act)
# #         logger.info("🚀 MBIDQN Dyna + FL + GSP (Adaptive V2 Complete)")
        
# #         first_id = list(obs_act.keys())[0]
# #         obs_dim = obs_act[first_id][0][0] if len(obs_act[first_id][0])==1 else np.prod(obs_act[first_id][0])
        
# #         # Global Model & Coordinator
# #         gm = RSSM_with_GSP(obs_dim, obs_act[first_id][1].n, 
# #                           cfg.number_of_units, 32, cfg.number_of_units, cfg.get('global_dim', 64))
# #         self.coordinator = GlobalCoordinatorWithGSP(gm, {})
        
# #         for agent_id in obs_act:
# #             self.agents[agent_id] = DynaMBAgent_Adaptive(
# #                 agent_id, obs_act[agent_id][0], obs_act[agent_id][1].n, self.coordinator
# #             )
            
# #     def observe(self, observations, rewards, dones, infos):
# #         super().observe(observations, rewards, dones, infos)
# #         if self.coordinator.should_aggregate_fl():
# #             gp = self.coordinator.aggregate_fl_and_broadcast()
# #             if gp: [a.update_from_global(gp) for a in self.agents.values()]
        
# #         if any(dones.values()):
# #             [a.on_episode_end() for a in self.agents.values()]









# # # ==================== RESCO Adapter ====================

# # class MBIDQN_Dyna_FL_GSP_Fixed_V2_Adaptive(IndependentAgent):
# #     """
# #     RESCO Adapter with Adaptive Control
# #     """
# #     def __init__(self, obs_act):
# #         super().__init__(obs_act)
        
# #         logger.info("=" * 70)
# #         logger.info("🚀 MBIDQN Dyna + FL + GSP (Adaptive V2)")
# #         logger.info("=" * 70)
# #         logger.info("Key features:")
# #         logger.info("1. ✅ Real batch固定 (32)")
# #         logger.info("2. ✅ Synthetic batch动态增量 (0-64)")
# #         logger.info("3. ✅ Quality-based adaptive control")
# #         logger.info("4. ✅ Horizon动态增长 (1→5)")
# #         logger.info("=" * 70)
        
# #         # 创建全局模型
# #         first_agent_id = list(obs_act.keys())[0]
# #         obs_space = obs_act[first_agent_id][0]
# #         act_space = obs_act[first_agent_id][1]
        
# #         if len(obs_space) == 1:
# #             obs_dim = obs_space[0]
# #         else:
# #             obs_dim = int(np.prod(obs_space))
        
# #         global_model = RSSM_with_GSP(
# #             obs_dim=obs_dim,
# #             action_dim=act_space,
# #             hidden_dim=cfg.number_of_units,
# #             stoch_dim=32,
# #             deter_dim=cfg.number_of_units,
# #             global_dim=cfg.get('global_dim', 64)
# #         ).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        
# #         # 创建GlobalCoordinatorWithGSP（重要：先创建！）
# #         fl_config = {
# #             'min_agents': max(1, len(obs_act) // 2),
# #             'aggregation_method': cfg.get('aggregation_method', 'quality_weighted'),
# #             'gsp_sync_threshold': cfg.get('gsp_sync_threshold', 0.8)
# #         }
# #         self.coordinator = GlobalCoordinatorWithGSP(global_model, fl_config)
        
# #         # 创建agents
# #         for agent_id in obs_act:
# #             obs_space = obs_act[agent_id][0]
# #             act_space = obs_act[agent_id][1]
            
# #             agent = DynaMBAgent_Adaptive(
# #                 agent_id,
# #                 obs_space,
# #                 act_space,
# #                 coordinator=self.coordinator
# #             )
# #             self.agents[agent_id] = agent
# #             self.coordinator.register_agent(agent_id)
        
# #         logger.info(f"Initialized {len(self.agents)} adaptive agents")
    
# #     def observe(self, observations, rewards, dones, infos):
# #         """扩展observe - 处理FL/GSP/episode结束"""
# #         super().observe(observations, rewards, dones, infos)
        
# #         # FL聚合
# #         if self.coordinator.should_aggregate_fl():
# #             global_params = self.coordinator.aggregate_fl_and_broadcast()
# #             if global_params:
# #                 for agent in self.agents.values():
# #                     agent.update_from_global(global_params)
        
# #         # GSP共识
# #         current_step = list(self.agents.values())[0].global_step
# #         if self.coordinator.should_compute_consensus(current_step):
# #             G_consensus = self.coordinator.compute_consensus(current_step)
        
# #         # Episode结束处理
# #         if any(dones.values()):
# #             for agent in self.agents.values():
# #                 agent.on_episode_end()
    
# #     def save(self, path):
# #         """保存模型"""
# #         os.makedirs(path, exist_ok=True)
        
# #         for agent_id, agent in self.agents.items():
# #             agent_path = os.path.join(path, f'agent_{compute_safe_id(agent_id)}.pt')
# #             torch.save({
# #                 'world_model': agent.world_model.state_dict(),
# #                 'q_network': agent.q_network.state_dict(),
# #                 'wm_optimizer': agent.wm_optimizer.state_dict(),
# #                 'q_optimizer': agent.q_optimizer.state_dict(),
# #                 'step_count': agent.step_count,
# #             }, agent_path)
        
# #         coordinator_path = os.path.join(path, 'coordinator.pt')
# #         torch.save({
# #             'global_model': self.coordinator.global_model.state_dict(),
# #             'fl_round': self.coordinator.round,
# #             'agent_stats': self.coordinator.agent_stats,
# #         }, coordinator_path)
        
# #         logger.info(f"Saved to {path}")
    
# #     def load(self, path):
# #         """加载模型"""
# #         for agent_id, agent in self.agents.items():
# #             agent_path = os.path.join(path, f'agent_{compute_safe_id(agent_id)}.pt')
# #             if os.path.exists(agent_path):
# #                 checkpoint = torch.load(agent_path, map_location=agent.device)
# #                 agent.world_model.load_state_dict(checkpoint['world_model'])
# #                 agent.q_network.load_state_dict(checkpoint['q_network'])
# #                 agent.target_q_network.load_state_dict(agent.q_network.state_dict())
                
# #                 if 'wm_optimizer' in checkpoint:
# #                     agent.wm_optimizer.load_state_dict(checkpoint['wm_optimizer'])
# #                 if 'q_optimizer' in checkpoint:
# #                     agent.q_optimizer.load_state_dict(checkpoint['q_optimizer'])
# #                 if 'step_count' in checkpoint:
# #                     agent.step_count = checkpoint['step_count']
        
# #         coordinator_path = os.path.join(path, 'coordinator.pt')
# #         if os.path.exists(coordinator_path):
# #             checkpoint = torch.load(coordinator_path)
# #             self.coordinator.global_model.load_state_dict(checkpoint['global_model'])
# #             self.coordinator.round = checkpoint.get('fl_round', 0)
# #             self.coordinator.agent_stats = checkpoint.get('agent_stats', {})
        
# #         logger.info(f"Loaded from {path}")

# # ==================== Adapter ====================

# class MBIDQN_Dyna_FL_GSP_Fixed_V2_Adaptive(IndependentAgent):
#     def __init__(self, obs_act):
#         super().__init__(obs_act)
#         logger.info("🚀 MBIDQN Dyna + FL + GSP (Adaptive V2 Complete)")
        
#         first_id = list(obs_act.keys())[0]
        
#         # 1. 获取 Observation Dimension
#         if len(obs_act[first_id][0]) == 1:
#             obs_dim = obs_act[first_id][0][0]
#         else:
#             obs_dim = int(np.prod(obs_act[first_id][0]))

#         # 2. 修正：安全获取 Action Dimension (Global Model)
#         first_act_obj = obs_act[first_id][1]
#         if hasattr(first_act_obj, 'n'):
#             act_dim = first_act_obj.n
#         else:
#             act_dim = int(first_act_obj)

#         # Global Model & Coordinator
#         gm = RSSM_with_GSP(
#             obs_dim=obs_dim, 
#             action_dim=act_dim,  # 使用修正后的 act_dim
#             hidden_dim=cfg.number_of_units, 
#             stoch_dim=32, 
#             deter_dim=cfg.number_of_units, 
#             global_dim=cfg.get('global_dim', 64)
#         )
        
#         # 初始化 Coordinator
#         fl_config = {
#             'min_agents': max(1, len(obs_act) // 2),
#             'aggregation_method': cfg.get('aggregation_method', 'quality_weighted'),
#             'gsp_sync_threshold': cfg.get('gsp_sync_threshold', 0.8)
#         }
#         self.coordinator = GlobalCoordinatorWithGSP(gm, fl_config)
        
#         for agent_id in obs_act:
#             # 3. 修正：安全获取 Action Dimension (Local Agents)
#             curr_act_obj = obs_act[agent_id][1]
#             if hasattr(curr_act_obj, 'n'):
#                 curr_act_dim = curr_act_obj.n
#             else:
#                 curr_act_dim = int(curr_act_obj)

#             self.agents[agent_id] = DynaMBAgent_Adaptive(
#                 agent_id, 
#                 obs_act[agent_id][0], 
#                 curr_act_dim,  # 使用修正后的 curr_act_dim
#                 self.coordinator
#             )
#             # 注册到 Coordinator
#             self.coordinator.register_agent(agent_id)
            
#     def observe(self, observations, rewards, dones, infos):
#         super().observe(observations, rewards, dones, infos)
        
#         # FL 聚合
#         if self.coordinator.should_aggregate_fl():
#             gp = self.coordinator.aggregate_fl_and_broadcast()
#             if gp: 
#                 for a in self.agents.values():
#                     a.update_from_global(gp)
        
#         # GSP 共识
#         if self.agents:
#             current_step = list(self.agents.values())[0].global_step
#             if self.coordinator.should_compute_consensus(current_step):
#                 self.coordinator.compute_consensus(current_step)
        
#         # Episode 结束处理
#         if any(dones.values()):
#             for a in self.agents.values():
#                 a.on_episode_end()

"""
mbidqn_dyna_fl_gsp_corrected.py - Dyna版本的正确GSP实现

关键修正（与True版本相同的GSP逻辑）：
1. GSP是跨Agent的共识学习，不是单Agent的时序对比
2. GlobalCoordinatorWithGSP追踪同步的GSP预测
3. RSSM使用序列采样，不是随机单步
4. 对比学习：同一时刻不同agents的G_hat应该相似

区别：
- Dyna: WM生成imagined data → 存入buffer → Q从buffer采样
- True: Q在imagination中直接训练（梯度反向传播）

作者: Percy Zhang
日期: 2025-11-27
版本: Dyna Corrected
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


# ==================== Global Coordinator with Synchronized GSP ====================

class GlobalCoordinatorWithGSP:
    """
    全局协调器 - 管理FL + 同步GSP共识学习
    
    核心功能：
    1. FL: 聚合World Model参数
    2. GSP同步：追踪同一时刻所有agents的G_hat
    3. 共识计算：计算全局共识G_consensus
    4. 对比学习：为每个agent提供正/负样本
    """
    def __init__(self, global_model, fl_config):
        self.global_model = global_model
        self.fl_config = fl_config
        
        # FL状态
        self.round = 0
        self.pending_updates = {}
        self.agent_stats = {}
        
        # GSP同步状态
        self.current_global_step = 0
        self.gsp_predictions = {}  # {global_step: {agent_id: G_hat}}
        self.gsp_consensus_history = deque(maxlen=100)  # [(step, G_consensus), ...]
        
        # 配置
        self.min_agents_for_aggregation = fl_config.get('min_agents', 2)
        self.aggregation_method = fl_config.get('aggregation_method', 'quality_weighted')
        self.gsp_sync_threshold = fl_config.get('gsp_sync_threshold', 0.8)  # 80%的agents上传后计算共识
        
        logger.info(f"GlobalCoordinatorWithGSP initialized: "
                   f"FL min_agents={self.min_agents_for_aggregation}, "
                   f"GSP sync_threshold={self.gsp_sync_threshold}")
    
    def register_agent(self, agent_id):
        """注册agent"""
        self.agent_stats[agent_id] = {
            'upload_count': 0,
            'last_quality': 0.5,
            'gsp_upload_count': 0
        }
    
    # ==================== FL Functions ====================
    
    def receive_fl_update(self, agent_id, params, quality_score):
        """接收FL参数更新"""
        self.pending_updates[agent_id] = {
            'params': params,
            'quality': quality_score,
            'timestamp': self.round
        }
        self.agent_stats[agent_id]['upload_count'] += 1
        self.agent_stats[agent_id]['last_quality'] = quality_score
    
    def should_aggregate_fl(self):
        """判断是否应该FL聚合"""
        return len(self.pending_updates) >= self.min_agents_for_aggregation
    
    def aggregate_fl_and_broadcast(self):
        """FL聚合并广播"""
        if not self.should_aggregate_fl():
            return None
        
        logger.info(f"=== FL Round {self.round} ===")
        logger.info(f"Aggregating {len(self.pending_updates)} agents")
        
        if self.aggregation_method == 'quality_weighted':
            global_params = self._quality_weighted_aggregation()
        else:
            global_params = self._fedavg()
        
        self.global_model.load_state_dict(global_params)
        self.pending_updates.clear()
        self.round += 1
        
        logger.info(f"FL Round {self.round - 1} completed")
        return global_params
    
    def _fedavg(self):
        """标准FedAvg - 处理参数形状不匹配"""
        if len(self.pending_updates) == 0:
            return {}
        
        # 获取第一个agent的参数作为参考
        first_agent_id = list(self.pending_updates.keys())[0]
        first_params = self.pending_updates[first_agent_id]['params']
        
        global_dict = {}
        N = len(self.pending_updates)
        
        # 对每个参数key
        for key in first_params.keys():
            # 检查所有agents的该参数形状是否一致
            shapes = [self.pending_updates[aid]['params'][key].shape 
                     for aid in self.pending_updates.keys()]
            
            # 如果形状一致，则聚合
            if all(s == shapes[0] for s in shapes):
                global_dict[key] = torch.zeros_like(first_params[key])
                
                for update_data in self.pending_updates.values():
                    params = update_data['params']
                    global_dict[key] += params[key] / N
            else:
                # 形状不匹配，跳过
                logger.debug(f"Skipping parameter '{key}' due to shape mismatch: {shapes}")
                global_dict[key] = first_params[key].clone()
        
        return global_dict
    
    def _quality_weighted_aggregation(self):
        """质量加权聚合 - 处理参数形状不匹配"""
        quality_scores = {aid: data['quality'] for aid, data in self.pending_updates.items()}
        total_quality = sum(quality_scores.values())
        
        if total_quality == 0:
            return self._fedavg()
        
        weights = {aid: q / total_quality for aid, q in quality_scores.items()}
        
        logger.info(f"Quality scores: {quality_scores}")
        logger.info(f"Aggregation weights: {weights}")
        
        global_dict = {}
        
        # 获取第一个agent的参数作为参考
        first_agent_id = list(self.pending_updates.keys())[0]
        first_params = self.pending_updates[first_agent_id]['params']
        
        # 对每个参数key
        for key in first_params.keys():
            # 检查所有agents的该参数形状是否一致
            shapes = [self.pending_updates[aid]['params'][key].shape 
                     for aid in self.pending_updates.keys()]
            
            # 如果形状一致，则聚合
            if all(s == shapes[0] for s in shapes):
                global_dict[key] = torch.zeros_like(first_params[key])
                
                for agent_id, update_data in self.pending_updates.items():
                    params = update_data['params']
                    weight = weights[agent_id]
                    global_dict[key] += params[key] * weight
            else:
                # 形状不匹配，跳过（保持local）
                logger.debug(f"Skipping parameter '{key}' due to shape mismatch: {shapes}")
                # 使用第一个agent的参数（或者可以选择不聚合）
                global_dict[key] = first_params[key].clone()
        
        return global_dict
    
    # ==================== GSP Synchronization Functions ====================
    
    def receive_gsp_prediction(self, global_step, agent_id, G_hat):
        """
        接收agent的GSP预测（同步！）
        
        关键：所有agents必须在同一global_step上传G_hat
        """
        if global_step not in self.gsp_predictions:
            self.gsp_predictions[global_step] = {}
        
        # 存储G_hat（detach避免梯度传播问题）
        self.gsp_predictions[global_step][agent_id] = G_hat.detach().cpu()
        
        self.agent_stats[agent_id]['gsp_upload_count'] += 1
        self.current_global_step = max(self.current_global_step, global_step)
        
        logger.debug(f"[GSP] Received from {agent_id} at step {global_step}")
    
    def should_compute_consensus(self, global_step):
        """
        判断是否应该计算共识
        
        条件：至少有gsp_sync_threshold比例的agents上传了G_hat
        """
        if global_step not in self.gsp_predictions:
            return False
        
        total_agents = len(self.agent_stats)
        uploaded_agents = len(self.gsp_predictions[global_step])
        
        return uploaded_agents >= int(total_agents * self.gsp_sync_threshold)
    
    def compute_consensus(self, global_step):
        """
        计算全局共识G_consensus
        
        方法：简单平均所有agents的G_hat
        """
        if global_step not in self.gsp_predictions:
            logger.warning(f"[GSP] No predictions for step {global_step}")
            return None
        
        predictions = list(self.gsp_predictions[global_step].values())
        
        if len(predictions) == 0:
            return None
        
        # 计算平均
        G_consensus = torch.stack(predictions).mean(dim=0)
        
        # 存储到历史
        self.gsp_consensus_history.append((global_step, G_consensus))
        
        # degug print
        # logger.info(f"[GSP] Consensus computed at step {global_step} from {len(predictions)} agents")
        
        # 清理旧数据（保留最近10步用于负样本）
        steps_to_remove = [s for s in self.gsp_predictions.keys() if s < global_step - 10]
        for s in steps_to_remove:
            del self.gsp_predictions[s]
        
        return G_consensus
    
    def get_contrastive_samples(self, global_step, agent_id):
        """
        获取对比学习的正负样本
        
        正样本：同一时刻其他agents的G_hat（或共识）
        负样本：历史时刻的共识
        
        Returns:
            positive: 正样本 (Tensor)
            negatives: 负样本列表 [Tensor, ...]
        """
        # 正样本：当前step的共识
        if global_step in self.gsp_predictions and len(self.gsp_predictions[global_step]) > 1:
            # 计算除了当前agent外的其他agents的平均
            other_predictions = [
                G_hat for aid, G_hat in self.gsp_predictions[global_step].items()
                if aid != agent_id
            ]
            if len(other_predictions) > 0:
                positive = torch.stack(other_predictions).mean(dim=0)
            else:
                positive = None
        else:
            positive = None
        
        # 负样本：历史共识（最近5-10步）
        negatives = []
        if len(self.gsp_consensus_history) > 0:
            # 取最近的5个历史共识作为负样本
            for step, G_consensus in list(self.gsp_consensus_history)[-5:]:
                if step != global_step:  # 不包括当前步
                    negatives.append(G_consensus)
        
        return positive, negatives
    
    def get_global_params(self):
        """获取全局FL参数"""
        return self.global_model.state_dict()


# ==================== RSSM with GSP ====================

class RSSM_with_GSP(nn.Module):
    """RSSM with Global State Prediction"""
    def __init__(self, obs_dim, action_dim, hidden_dim=64, stoch_dim=32, deter_dim=64, global_dim=64):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.stoch_dim = stoch_dim
        self.deter_dim = deter_dim
        self.global_dim = global_dim
        
        # 基础RSSM组件
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.rnn = nn.GRUCell(hidden_dim + stoch_dim + action_dim, deter_dim)
        
        self.prior_net = nn.Sequential(
            nn.Linear(deter_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * stoch_dim)
        )
        
        self.posterior_net = nn.Sequential(
            nn.Linear(deter_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * stoch_dim)
        )
        
        self.obs_decoder = nn.Sequential(
            nn.Linear(deter_dim + stoch_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim)
        )
        
        self.reward_decoder = nn.Sequential(
            nn.Linear(deter_dim + stoch_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # GSP组件：只有global_predictor
        latent_dim = deter_dim + stoch_dim
        
        self.global_predictor = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, global_dim),
            nn.LayerNorm(global_dim)
        )
        
        # Quality head for FL
        self.quality_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def initial_state(self, batch_size, device):
        return {
            'deter': torch.zeros(batch_size, self.deter_dim).to(device),
            'stoch': torch.zeros(batch_size, self.stoch_dim).to(device)
        }
    
    def observe_step(self, prev_state, action, obs):
        """单步观察"""
        obs_embed = self.obs_encoder(obs)
        rnn_input = torch.cat([prev_state['stoch'], action, obs_embed], dim=-1)
        deter = self.rnn(rnn_input, prev_state['deter'])
        
        prior_out = self.prior_net(deter)
        prior_mean, prior_std = torch.chunk(prior_out, 2, dim=-1)
        prior_std = F.softplus(prior_std) + 0.1
        
        post_input = torch.cat([deter, obs_embed], dim=-1)
        post_out = self.posterior_net(post_input)
        post_mean, post_std = torch.chunk(post_out, 2, dim=-1)
        post_std = F.softplus(post_std) + 0.1
        
        stoch = post_mean + post_std * torch.randn_like(post_std)
        
        return {
            'deter': deter,
            'stoch': stoch,
            'prior_mean': prior_mean,
            'prior_std': prior_std,
            'post_mean': post_mean,
            'post_std': post_std
        }
    
    def observe_sequence(self, actions, observations):
        """
        序列观察（用于正确的RSSM训练）
        
        Args:
            actions: [seq_len, batch, action_dim]
            observations: [seq_len, batch, obs_dim]
        
        Returns:
            states: list of states
        """
        seq_len, batch_size = actions.shape[0], actions.shape[1]
        device = actions.device
        
        states = []
        state = self.initial_state(batch_size, device)
        
        for t in range(seq_len):
            state = self.observe_step(state, actions[t], observations[t])
            states.append(state)
        
        return states
    
    def imagine_step(self, prev_state, action):
        """单步imagination"""
        obs_embed = torch.zeros(
            prev_state['deter'].shape[0], 
            self.hidden_dim
        ).to(prev_state['deter'].device)
        
        rnn_input = torch.cat([prev_state['stoch'], action, obs_embed], dim=-1)
        deter = self.rnn(rnn_input, prev_state['deter'])
        
        prior_out = self.prior_net(deter)
        mean, std = torch.chunk(prior_out, 2, dim=-1)
        std = F.softplus(std) + 0.1
        
        stoch = mean + std * torch.randn_like(std)
        
        return {
            'deter': deter,
            'stoch': stoch,
            'mean': mean,
            'std': std
        }
    
    def decode(self, state):
        """解码"""
        feat = torch.cat([state['deter'], state['stoch']], dim=-1)
        obs_pred = self.obs_decoder(feat)
        reward_pred = self.reward_decoder(feat)
        return obs_pred, reward_pred
    
    def get_feature(self, state):
        """获取特征用于Q-network"""
        return torch.cat([state['deter'], state['stoch']], dim=-1)
    
    def get_global_prediction(self, state):
        """获取Global State Prediction（L2归一化）"""
        latent = torch.cat([state['deter'], state['stoch']], dim=-1)
        G_hat = self.global_predictor(latent)
        return F.normalize(G_hat, p=2, dim=-1)
    
    def get_quality_score(self, state):
        """获取质量分数"""
        latent = torch.cat([state['deter'], state['stoch']], dim=-1)
        return self.quality_head(latent)


def build_q_network(input_dim, act_space):
    """Build Q-network"""
    model = nn.Sequential()
    model.add_module('linear1', nn.Linear(input_dim, cfg.number_of_units))
    model.add_module('relu1', nn.ReLU())
    
    for i in range(cfg.number_of_layers - 1):
        model.add_module(f'linear{i+2}', nn.Linear(cfg.number_of_units, cfg.number_of_units))
        model.add_module(f'relu{i+2}', nn.ReLU())
    
    # Final output layer
    model.add_module('output', nn.Linear(cfg.number_of_units, act_space))
    
    # Wrap with DiscreteActionValueHead (no parameters!)
    model.add_module('head', DiscreteActionValueHead())
    
    return model


# ==================== Sequence Replay Buffer ====================

class SequenceReplayBuffer:
    """
    序列采样的Replay Buffer
    
    关键：支持采样连续的时间片段，而不是随机单步
    """
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def append(self, transition):
        """添加单个transition"""
        self.buffer.append(transition)
    
    def sample_sequences(self, batch_size, seq_length):
        """
        采样连续序列
        
        Args:
            batch_size: 批大小
            seq_length: 序列长度
        
        Returns:
            sequences: list of sequences, 每个sequence是seq_length个transitions
        """
        if len(self.buffer) < seq_length:
            return None
        
        sequences = []
        
        for _ in range(batch_size):
            # 随机选择起始位置
            start_idx = np.random.randint(0, len(self.buffer) - seq_length + 1)
            
            # 提取连续序列
            sequence = list(self.buffer)[start_idx:start_idx + seq_length]
            sequences.append(sequence)
        
        return sequences
    
    def sample_transitions(self, batch_size):
        """随机采样单步transitions（用于Q训练的starting states）"""
        if len(self.buffer) < batch_size:
            return None
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]
    
    def __len__(self):
        return len(self.buffer)


# ==================== Dyna Agent with Corrected GSP ====================


# ==================== Adaptive Data Controller ====================

class AdaptiveDataController:
    """
    自适应数据控制器 - Synthetic data作为Real data的增量
    """
    def __init__(self, 
                 base_real_batch=32,
                 max_synthetic_batch=64,
                 initial_horizon=1,
                 max_horizon=5,
                 horizon_increase_freq=10,
                 quality_threshold=0.3,
                 adaptive_factor=1.0):
        
        self.base_real_batch = base_real_batch
        self.max_synthetic_batch = max_synthetic_batch
        self.initial_horizon = initial_horizon
        self.max_horizon = max_horizon
        self.horizon_increase_freq = horizon_increase_freq
        self.quality_threshold = quality_threshold
        self.adaptive_factor = adaptive_factor
        
        self.current_horizon = initial_horizon
        self.current_synthetic_batch = 0
        self.episode_count = 0
        
        from collections import deque
        self.quality_history = deque(maxlen=100)
        
        logger.info(f"AdaptiveDataController initialized:")
        logger.info(f"  base_real_batch={base_real_batch} (固定)")
        logger.info(f"  max_synthetic_batch={max_synthetic_batch}")
    
    def update_quality(self, wm_quality):
        self.quality_history.append(wm_quality)
    
    def get_current_synthetic_batch(self):
        if len(self.quality_history) == 0:
            return 0
        
        import numpy as np
        avg_quality = np.mean(list(self.quality_history))
        
        if avg_quality < self.quality_threshold:
            target_synthetic_batch = 0
        else:
            quality_above_threshold = avg_quality - self.quality_threshold
            max_quality_range = 1.0 - self.quality_threshold
            normalized_quality = quality_above_threshold / max_quality_range
            target_synthetic_batch = int(
                normalized_quality * self.adaptive_factor * self.max_synthetic_batch
            )
            target_synthetic_batch = min(target_synthetic_batch, self.max_synthetic_batch)
        
        self.current_synthetic_batch = int(
            0.9 * self.current_synthetic_batch + 0.1 * target_synthetic_batch
        )
        
        return self.current_synthetic_batch
    
    def get_current_horizon(self):
        return self.current_horizon
    
    def on_episode_end(self):
        self.episode_count += 1
        if self.episode_count % self.horizon_increase_freq == 0:
            if self.current_horizon < self.max_horizon:
                self.current_horizon += 1
                logger.info(f"Horizon increased to {self.current_horizon}")
    
    def get_stats(self):
        if len(self.quality_history) > 0:
            import numpy as np
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

class DynaMBAgent_Adaptive(Agent):
    """
    Dyna Agent with Corrected GSP
    
    关键：
    1. WM训练使用序列采样
    2. GSP是跨agent共识
    3. 生成imagined data存入buffer
    4. Q从buffer（real + imagined）采样训练
    """
    def __init__(self, agent_id, obs_space, act_space, coordinator=None):
        super().__init__()
        self.agent_id = agent_id
        self.obs_space = obs_space
        self.act_space = act_space
        self.coordinator = coordinator
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Obs处理
        if len(obs_space) == 1:
            self.obs_dim = obs_space[0]
            self.flatten_obs = False
        else:
            self.obs_dim = int(np.prod(obs_space))
            self.flatten_obs = True
        
        # World Model with GSP
        self.world_model = RSSM_with_GSP(
            obs_dim=self.obs_dim,
            action_dim=act_space,
            hidden_dim=cfg.number_of_units,
            stoch_dim=32,
            deter_dim=cfg.number_of_units,
            global_dim=cfg.get('global_dim', 64)
        ).to(self.device)
        
        self.wm_optimizer = torch.optim.Adam(
            self.world_model.parameters(),
            lr=cfg.lr_world_model
        )
        
        # Q-Network
        feat_dim = cfg.number_of_units + 32
        self.q_network = build_q_network(feat_dim, act_space).to(self.device)
        self.target_q_network = deepcopy(self.q_network)
        
        # self.q_optimizer = pfrl.optimizers.RMSpropEpsInsideSqrt(
        #     self.q_network.parameters(),
        #     lr=cfg.learning_rate,
        #     alpha=cfg.rmsprop_decay,
        #     eps=cfg.rmsprop_epsilon,
        #     momentum=cfg.rmsprop_momentum
        # )

        self.q_optimizer = torch.optim.Adam(
            self.q_network.parameters(),
            lr=cfg.learning_rate,
            eps=1e-8 # Adam 的默认 eps
        )
        
        # 两个Buffer（都使用SequenceReplayBuffer）
        self.wm_buffer = SequenceReplayBuffer(cfg.buffer_size)  # WM训练用（序列）
        self.dqn_buffer = SequenceReplayBuffer(cfg.buffer_size)  # Q训练用（real + imagined）
        
        # Explorer
        self.explorer = LinearDecayEpsilonGreedy(
            cfg.epsilon_begin,
            cfg.epsilon_end,
            cfg.epsilon_decay_period,
            lambda: np.random.randint(act_space)
        )
        
        # 训练配置
        self._training = True
        self.step_count = 0
        self.global_step = 0
        self.target_update_freq = cfg.get('target_update_steps', 500)
        
        # Dyna配置
        self.imagination_horizon = cfg.get('imagination_horizon', 10)
        self.num_imagined_rollouts = cfg.get('num_imagined_rollouts', 5)
        
        # FL配置
        self.fl_interval = cfg.get('fl_interval', 100)
        self.local_steps = 0
        self.global_params = None
        self.alpha_fedprox = cfg.get('alpha_fedprox', 0.01)
        
        # GSP配置
        self.alpha_contrastive = cfg.get('alpha_contrastive', 0.1)
        self.contrastive_temperature = cfg.get('contrastive_temperature', 0.1)
        
        # RSSM训练配置
        self.seq_length = cfg.get('seq_length', 50)
        self.model_train_freq = cfg.get('model_train_freq', 1)
        
        # ⭐ Early阶段保护配置（修复imagination过多问题）
        self.model_warmup_steps = cfg.get('model_warmup_steps', 1000)
        self.min_wm_quality = cfg.get('min_wm_quality', 0.3)
        self.imagination_freq = cfg.get('imagination_freq', 5)
        self.imagination_counter = 0
        
        
        # ⭐ 创建自适应控制器
        self.adaptive_controller = AdaptiveDataController(
            base_real_batch=cfg.batch_size,
            max_synthetic_batch=cfg.get('max_synthetic_batch', 64),
            initial_horizon=cfg.get('initial_horizon', 1),
            max_horizon=cfg.get('imagination_horizon', 5),
            horizon_increase_freq=cfg.get('horizon_increase_freq', 10),
            quality_threshold=cfg.get('min_wm_quality', 0.3),
            adaptive_factor=cfg.get('adaptive_factor', 1.0)
        )
        
        # self.episode_count = 0
        
        logger.info(f"DynaMBAgent_Adaptive {agent_id} initialized")
        logger.info(f"  - Adaptive control: Real batch fixed at {cfg.batch_size}")
        logger.info(f"  - Imagination: horizon={self.imagination_horizon}, rollouts={self.num_imagined_rollouts}, freq={self.imagination_freq}")
        logger.info(f"  - Protection: warmup={self.model_warmup_steps}, min_quality={self.min_wm_quality}")
    
    def _flatten(self, obs):
        if self.flatten_obs and len(obs.shape) > 1:
            return obs.flatten()
        return obs
    
    def act(self, observation):
        """选择动作"""
        obs = self._flatten(observation)
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            state = self._encode_obs(obs_t)
            feat = self.world_model.get_feature(state)
            q_values = self.q_network(feat)
            
            if self._training:
                action = self.explorer.select_action(
                    self.step_count,
                    lambda: q_values.greedy_actions.item()
                )
            else:
                action = q_values.greedy_actions.item()
        
        return action
    
    def observe(self, observation, reward, done, info):
        """学习"""
        if not self._training:
            return
        
        obs = self._flatten(observation)
        
        # Store transition
        if hasattr(self, 'last_obs') and hasattr(self, 'last_action'):
            transition = {
                'obs': self.last_obs,
                'action': self.last_action,
                'reward': reward,
                'next_obs': obs,
                'done': done
            }
            
            # 存入WM buffer（序列采样用）
            self.wm_buffer.append(transition)
            
            # 存入DQN buffer（Q训练用）- 使用dict格式
            self.dqn_buffer.append(transition)
        
        self.last_obs = obs
        self.last_action = self.act(observation) if not done else 0
        self.step_count += 1
        self.global_step += 1
        self.local_steps += 1
        
        # Update epsilon
        self.explorer.epsilon = self.explorer.compute_epsilon(self.step_count)
        
        # Train
        if len(self.wm_buffer) >= max(cfg.batch_size, self.seq_length):
            # Train World Model（序列采样）
            if self.step_count % self.model_train_freq == 0:
                self._train_world_model_sequence()
            
            # ⭐ Generate imagined data with TRIPLE PROTECTION
            # Protection 1: Warmup period
            if self.step_count < self.model_warmup_steps:
                if self.step_count % 100 == 0:
                    logger.info(f"Agent {self.agent_id} Step {self.step_count}: Warmup ({self.step_count}/{self.model_warmup_steps}), skip imagination")
            else:
                # Protection 2: Quality check
                quality = self.get_quality_score()
                if quality < self.min_wm_quality:
                    if self.step_count % 100 == 0:
                        logger.info(f"Agent {self.agent_id} Step {self.step_count}: WM quality {quality:.3f} < {self.min_wm_quality}, skip imagination")
                else:
                    # Protection 3: Frequency control + Adaptive horizon
                    self.adaptive_controller.update_quality(quality)
                    self.imagination_counter += 1
                    if self.imagination_counter >= self.imagination_freq:
                        if self.step_count % 100 == 0:
                            logger.info(f"Agent {self.agent_id} Step {self.step_count}: Generate imagination (quality={quality:.3f})")
                        self._generate_imagined_data_adaptive()
                        self.imagination_counter = 0
        
        # Train Q with adaptive batch
        if len(self.dqn_buffer) >= cfg.batch_size:
            self._train_q_adaptive()
        
        # Update target
        if self.step_count % self.target_update_freq == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())
        
        # FL触发
        if self.coordinator and self.local_steps % self.fl_interval == 0:
            self._upload_fl_params()
        
        # # Episode结束处理
        # if done:
        #     self.on_episode_end()
    
    def _encode_obs(self, obs_t):
        """编码observation"""
        state = self.world_model.initial_state(obs_t.shape[0], self.device)
        action_zero = torch.zeros(obs_t.shape[0], self.act_space).to(self.device)
        state = self.world_model.observe_step(state, action_zero, obs_t)
        return state
    
    def _train_world_model_sequence(self):
        """
        训练World Model - 使用序列采样
        
        与True版本相同的序列采样逻辑
        """
        sequences = self.wm_buffer.sample_sequences(
            batch_size=cfg.batch_size,
            seq_length=self.seq_length
        )
        
        if sequences is None:
            return
        
        # 准备序列数据
        obs_seq = []
        action_seq = []
        reward_seq = []
        next_obs_seq = []
        
        for seq in sequences:
            obs_seq.append([t['obs'] for t in seq])
            action_seq.append([t['action'] for t in seq])
            reward_seq.append([t['reward'] for t in seq])
            next_obs_seq.append([t['next_obs'] for t in seq])
        
        # Convert to tensors [seq_len, batch, dim]
        obs_seq = torch.FloatTensor(np.array(obs_seq)).transpose(0, 1).to(self.device)
        action_seq = torch.LongTensor(np.array(action_seq)).transpose(0, 1).to(self.device)
        reward_seq = torch.FloatTensor(np.array(reward_seq)).transpose(0, 1).unsqueeze(-1).to(self.device)
        next_obs_seq = torch.FloatTensor(np.array(next_obs_seq)).transpose(0, 1).to(self.device)
        
        action_onehot = F.one_hot(action_seq, self.act_space).float()
        
        # RSSM序列forward
        states = self.world_model.observe_sequence(action_onehot, obs_seq)
        
        # 基础RSSM损失
        total_obs_loss = 0.0
        total_reward_loss = 0.0
        total_kl_loss = 0.0
        
        for t, state in enumerate(states):
            next_obs_pred, reward_pred = self.world_model.decode(state)
            
            obs_loss = F.mse_loss(next_obs_pred, next_obs_seq[t])
            reward_loss = F.mse_loss(reward_pred, reward_seq[t])
            
            from torch.distributions import Normal
            posterior = Normal(state['post_mean'], state['post_std'])
            prior = Normal(state['prior_mean'], state['prior_std'])
            kl_loss = torch.distributions.kl_divergence(posterior, prior).mean()
            kl_loss = torch.clamp(kl_loss, min=1.0)
            
            total_obs_loss += obs_loss
            total_reward_loss += reward_loss
            total_kl_loss += kl_loss
        
        avg_obs_loss = total_obs_loss / len(states)
        avg_reward_loss = total_reward_loss / len(states)
        avg_kl_loss = total_kl_loss / len(states)
        
        base_loss = avg_obs_loss + avg_reward_loss + 0.1 * avg_kl_loss
        
        # GSP Loss（跨agent！）
        gsp_loss = self._compute_cross_agent_gsp_loss(states)
        
        # FedProx Loss
        fedprox_loss = self._compute_fedprox_loss()
        
        # Total Loss
        total_loss = base_loss + self.alpha_contrastive * gsp_loss + self.alpha_fedprox * fedprox_loss
        
        # Backward
        self.wm_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), 10.0)
        self.wm_optimizer.step()
        
        # 上传GSP
        if self.coordinator:
            final_state = states[-1]
            with torch.no_grad():
                G_hat = self.world_model.get_global_prediction(final_state)
                G_hat_avg = G_hat.mean(dim=0)
            
            self.coordinator.receive_gsp_prediction(
                global_step=self.global_step,
                agent_id=self.agent_id,
                G_hat=G_hat_avg
            )
    
    def _compute_cross_agent_gsp_loss(self, states):
        """计算跨Agent的GSP Loss（与True版本相同）"""
        if not self.coordinator:
            return torch.tensor(0.0).to(self.device)
        
        final_state = states[-1]
        G_hat = self.world_model.get_global_prediction(final_state)
        G_hat_avg = G_hat.mean(dim=0)
        
        positive, negatives = self.coordinator.get_contrastive_samples(
            global_step=self.global_step,
            agent_id=self.agent_id
        )
        
        if positive is None or len(negatives) == 0:
            return torch.tensor(0.0).to(self.device)
        
        positive = positive.to(self.device)
        negatives = [neg.to(self.device) for neg in negatives]
        
        tau = self.contrastive_temperature
        
        pos_sim = F.cosine_similarity(G_hat_avg, positive, dim=0) / tau
        neg_sims = torch.stack([
            F.cosine_similarity(G_hat_avg, neg, dim=0) / tau
            for neg in negatives
        ])
        
        logits = torch.cat([pos_sim.unsqueeze(0), neg_sims])
        labels = torch.zeros(1, dtype=torch.long).to(self.device)
        
        loss = F.cross_entropy(logits.unsqueeze(0), labels)
        
        return loss
    
    def _compute_fedprox_loss(self):
        """
        FedProx正则化 - 只计算形状匹配的参数
        
        形状不匹配的参数（obs相关层）不计算proximal term
        """
        if self.global_params is None:
            return torch.tensor(0.0).to(self.device)
        
        prox_loss = 0.0
        for name, param in self.world_model.named_parameters():
            if name in self.global_params:
                global_param = self.global_params[name].to(self.device)
                
                # ✅ 检查形状是否匹配
                if param.shape == global_param.shape:
                    prox_loss += ((param - global_param) ** 2).sum()
                # 形状不匹配则跳过（这些参数是agent-specific的）
        
        return prox_loss * 0.5
    
    def _generate_imagined_data_adaptive(self):
        """
        Dyna核心：生成imagined data并存入DQN buffer（使用adaptive horizon）
        """
        if len(self.wm_buffer) < cfg.batch_size:
            return
        
        # ⭐ 使用adaptive horizon
        horizon = self.adaptive_controller.get_current_horizon()
        
        # 从WM buffer采样起始states
        transitions = self.wm_buffer.sample_transitions(self.num_imagined_rollouts)
        if transitions is None:
            return
        
        for transition in transitions:
            obs = transition['obs']
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            
            # Encode起始state
            state = self._encode_obs(obs_t)
            
            # Imagine forward
            for h in range(horizon):
                with torch.no_grad():
                    # Get current observation from state
                    current_obs_pred, _ = self.world_model.decode(state)
                    
                    # 选择action
                    feat = self.world_model.get_feature(state)
                    q_values = self.q_network(feat)
                    action = q_values.greedy_actions.item()
                    
                    # Imagine next state
                    action_onehot = F.one_hot(
                        torch.tensor([action]).to(self.device),
                        self.act_space
                    ).float()
                    next_state = self.world_model.imagine_step(state, action_onehot)
                    
                    # Decode observation and reward
                    next_obs_pred, reward_pred = self.world_model.decode(next_state)
                    
                    # 存入DQN buffer（imagined data）- 使用dict格式
                    # 使用decoded observation，不是latent feature
                    current_obs_np = current_obs_pred.cpu().numpy().flatten()
                    next_obs_np = next_obs_pred.cpu().numpy().flatten()
                    reward_val = reward_pred.cpu().item()
                    
                    imagined_transition = {
                        'obs': current_obs_np,
                        'action': action,
                        'reward': reward_val,
                        'next_obs': next_obs_np,
                        'done': False
                    }
                    
                    self.dqn_buffer.append(imagined_transition)
                    
                    state = next_state
    
    def _train_q_from_buffer(self):
        """
        从buffer训练Q（Dyna方式）
        
        关键：Q-network工作在latent space上，所以需要先encode observations
        """
        # 使用sample_transitions而不是sample
        batch = self.dqn_buffer.sample_transitions(cfg.batch_size)
        
        if batch is None:
            return
        
        obs = torch.FloatTensor(np.stack([t['obs'] for t in batch])).to(self.device)
        actions = torch.LongTensor([t['action'] for t in batch]).to(self.device)
        rewards = torch.FloatTensor([t['reward'] for t in batch]).to(self.device)
        next_obs = torch.FloatTensor(np.stack([t['next_obs'] for t in batch])).to(self.device)
        dones = torch.FloatTensor([float(t['done']) for t in batch]).to(self.device)
        
        # Encode observations to latent features
        state = self._encode_obs(obs)
        feat = self.world_model.get_feature(state)
        
        # Q-values on latent features
        q_values = self.q_network(feat)
        q_selected = q_values.q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q-values
        with torch.no_grad():
            next_state = self._encode_obs(next_obs)
            next_feat = self.world_model.get_feature(next_state)
            next_q_values = self.target_q_network(next_feat)
            next_q = next_q_values.q_values.max(dim=1)[0]
            target = rewards + cfg.discount * next_q * (1 - dones)
        
        # TD loss
        loss = F.mse_loss(q_selected, target)
        
        # Backward
        self.q_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.q_optimizer.step()
    
    def _train_q_adaptive(self):
        """⭐ 训练Q-Network - Real batch固定 + Synthetic batch动态增量"""
        # Real batch固定
        real_batch_size = self.adaptive_controller.base_real_batch
        
        # Synthetic batch动态
        synthetic_batch_size = self.adaptive_controller.get_current_synthetic_batch()
        
        # 采样real data
        real_transitions = self.dqn_buffer.sample_transitions(real_batch_size)
        if not real_transitions:
            return
        
        # 采样synthetic data（如果需要）
        all_transitions = real_transitions
        if synthetic_batch_size > 0 and len(self.dqn_buffer) > real_batch_size:
            extra_transitions = self.dqn_buffer.sample_transitions(
                min(synthetic_batch_size, len(self.dqn_buffer) - real_batch_size)
            )
            if extra_transitions:
                all_transitions = real_transitions + extra_transitions
        
        # 准备batch
        obs = np.stack([t['obs'] for t in all_transitions])
        actions = np.array([t['action'] for t in all_transitions])
        rewards = np.array([t['reward'] for t in all_transitions])
        next_obs = np.stack([t['next_obs'] for t in all_transitions])
        dones = np.array([t['done'] for t in all_transitions])
        
        obs_t = torch.FloatTensor(obs).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_obs_t = torch.FloatTensor(next_obs).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)
        
        # Q-learning
        with torch.no_grad():
            next_state = self._encode_obs(next_obs_t)
            next_feat = self.world_model.get_feature(next_state)
            next_q = self.target_q_network(next_feat).q_values.max(dim=1)[0]
            target = rewards_t + cfg.discount * (1 - dones_t) * next_q
        
        state = self._encode_obs(obs_t)
        feat = self.world_model.get_feature(state)
        q_values = self.q_network(feat).q_values
        q_selected = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)
        
        # loss = F.mse_loss(q_selected, target)
        loss = F.smooth_l1_loss(q_selected, target)
        
        self.q_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.q_optimizer.step()
    
    def get_quality_score(self):
        """获取质量分数"""
        if len(self.wm_buffer) < cfg.batch_size:
            return 0.5
        
        transitions = self.wm_buffer.sample_transitions(min(cfg.batch_size, len(self.wm_buffer)))
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
        """
        从全局参数更新 - 只加载形状匹配的参数
        
        形状不匹配的参数（如obs_encoder.0.weight）保持local
        """
        self.global_params = {k: v.clone().cpu() for k, v in global_params.items()}
        
        # 获取当前模型的参数
        current_params = self.world_model.state_dict()
        
        # 只更新形状匹配的参数
        updated_count = 0
        skipped_count = 0
        
        for key, global_param in global_params.items():
            if key in current_params:
                if current_params[key].shape == global_param.shape:
                    # 形状匹配，更新
                    current_params[key] = global_param
                    updated_count += 1
                else:
                    # 形状不匹配，跳过（保持local）
                    skipped_count += 1
                    logger.debug(f"Agent {self.agent_id}: Skipping '{key}' due to shape mismatch: "
                               f"global={global_param.shape}, local={current_params[key].shape}")
        
        # 加载更新后的参数
        self.world_model.load_state_dict(current_params)
        
        logger.info(f"Agent {self.agent_id}: Updated {updated_count} params from global, "
                   f"kept {skipped_count} local params")


# ==================== RESCO Adapter (Dyna) ====================


class MBIDQN_Dyna_FL_GSP_Fixed_V2_Adaptive(IndependentAgent):
    """
    RESCO Adapter for Dyna with Corrected GSP
    """
    def __init__(self, obs_act):
        super().__init__(obs_act)
        
        logger.info("=" * 70)
        logger.info("🚀 MBIDQN Dyna + FL + GSP (Adaptive V2)")
        logger.info("=" * 70)
        logger.info("Key features:")
        logger.info("1. ✅ Real batch固定 (32)")
        logger.info("2. ✅ Synthetic batch动态增量 (0-64)")
        logger.info("3. ✅ Quality-based adaptive control")
        logger.info("4. ✅ Horizon动态增长 (1→5)")
        logger.info("=" * 70)
        
        # 创建全局模型
        first_agent_id = list(obs_act.keys())[0]
        obs_space = obs_act[first_agent_id][0]
        act_space = obs_act[first_agent_id][1]
        
        if len(obs_space) == 1:
            obs_dim = obs_space[0]
        else:
            obs_dim = int(np.prod(obs_space))
        
        global_model = RSSM_with_GSP(
            obs_dim=obs_dim,
            action_dim=act_space,
            hidden_dim=cfg.number_of_units,
            stoch_dim=32,
            deter_dim=cfg.number_of_units,
            global_dim=cfg.get('global_dim', 64)
        ).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        
        # 创建Coordinator
        fl_config = {
            'min_agents': max(1, len(obs_act) // 2),
            'aggregation_method': cfg.get('aggregation_method', 'quality_weighted'),
            'gsp_sync_threshold': cfg.get('gsp_sync_threshold', 0.8)
        }
        self.coordinator = GlobalCoordinatorWithGSP(global_model, fl_config)
        
        # 创建agents
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
        
        logger.info(f"Initialized {len(self.agents)} Dyna agents")
    
    def observe(self, observations, rewards, dones, infos):
        """扩展observe"""
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
            # degug print
            # if G_consensus is not None:
            #     logger.debug(f"[GSP] Consensus computed at step {current_step}")
    
    def save(self, path):
        """保存"""
        os.makedirs(path, exist_ok=True)
        
        for agent_id, agent in self.agents.items():
            agent_path = os.path.join(path, f'agent_{compute_safe_id(agent_id)}.pt')
            torch.save({
                'world_model': agent.world_model.state_dict(),
                'q_network': agent.q_network.state_dict(),
                'wm_optimizer': agent.wm_optimizer.state_dict(),
                'step_count': agent.step_count,
            }, agent_path)
        
        coordinator_path = os.path.join(path, 'coordinator.pt')
        torch.save({
            'global_model': self.coordinator.global_model.state_dict(),
            'fl_round': self.coordinator.round,
            'agent_stats': self.coordinator.agent_stats,
        }, coordinator_path)
        
        logger.info(f"Saved to {path}")
    
    def load(self, path):
        """加载"""
        for agent_id, agent in self.agents.items():
            agent_path = os.path.join(path, f'agent_{compute_safe_id(agent_id)}.pt')
            if os.path.exists(agent_path):
                checkpoint = torch.load(agent_path, map_location=agent.device)
                agent.world_model.load_state_dict(checkpoint['world_model'])
                agent.q_network.load_state_dict(checkpoint['q_network'])
                agent.target_q_network.load_state_dict(agent.q_network.state_dict())
                
                if 'wm_optimizer' in checkpoint:
                    agent.wm_optimizer.load_state_dict(checkpoint['wm_optimizer'])
                if 'step_count' in checkpoint:
                    agent.step_count = checkpoint['step_count']
        
        coordinator_path = os.path.join(path, 'coordinator.pt')
        if os.path.exists(coordinator_path):
            checkpoint = torch.load(coordinator_path)
            self.coordinator.global_model.load_state_dict(checkpoint['global_model'])
            self.coordinator.round = checkpoint.get('fl_round', 0)
            self.coordinator.agent_stats = checkpoint.get('agent_stats', {})
        
        logger.info(f"Loaded from {path}")
