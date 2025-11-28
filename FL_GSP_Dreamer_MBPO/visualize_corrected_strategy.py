"""
visualize_corrected_strategy.py - 可视化修正后的adaptive策略

展示：
1. Real batch固定不变
2. Synthetic batch动态增加（作为增量）
3. Total batch随WM质量增长
"""

import numpy as np
import matplotlib.pyplot as plt

# 参数
num_episodes = 100
base_real_batch = 32
max_synthetic_batch = 64
quality_threshold = 0.3
adaptive_factor = 1.0

# 模拟WM质量
def simulate_wm_quality(episodes):
    quality = []
    for ep in range(episodes):
        base_quality = 0.9 / (1 + np.exp(-(ep - 30) / 10))
        noise = np.random.normal(0, 0.05)
        q = np.clip(base_quality + noise, 0, 1)
        quality.append(q)
    return quality

# 计算synthetic batch size
def compute_synthetic_batch(quality):
    if quality < quality_threshold:
        return 0
    quality_above = quality - quality_threshold
    max_range = 1.0 - quality_threshold
    normalized = quality_above / max_range
    synthetic = int(normalized * adaptive_factor * max_synthetic_batch)
    return min(synthetic, max_synthetic_batch)

# 生成数据
episodes = np.arange(num_episodes)
wm_quality = simulate_wm_quality(num_episodes)

# 计算batch sizes
synthetic_batches = [compute_synthetic_batch(q) for q in wm_quality]
real_batches = [base_real_batch] * num_episodes
total_batches = [base_real_batch + s for s in synthetic_batches]

# 创建图表
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Corrected Adaptive Strategy: Synthetic as Increment', 
             fontsize=16, fontweight='bold')

# ==================== 子图1: Batch Size Evolution ====================
ax1 = axes[0, 0]
ax1.fill_between(episodes, 0, real_batches, alpha=0.3, color='blue', label='Real Batch (固定32)')
ax1.fill_between(episodes, real_batches, total_batches, alpha=0.3, color='orange', label='Synthetic Batch (0~64)')
ax1.plot(episodes, real_batches, 'b-', linewidth=2, label='Real Batch Line')
ax1.plot(episodes, total_batches, 'r-', linewidth=2, label='Total Batch')
ax1.axhline(y=base_real_batch, color='blue', linestyle='--', alpha=0.5)
ax1.axhline(y=base_real_batch + max_synthetic_batch, color='red', linestyle='--', alpha=0.5)
ax1.set_xlabel('Episode', fontsize=12)
ax1.set_ylabel('Batch Size', fontsize=12)
ax1.set_title('Batch Size: Real (Fixed) + Synthetic (Dynamic)', fontsize=13, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0, 100])

# 标注
ax1.text(10, 40, 'Real = 32\n(Always!)', ha='center', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
ax1.text(70, 85, 'Total = 32 + Synthetic\n(Up to 96)', ha='center', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))

# ==================== 子图2: Synthetic Batch Growth ====================
ax2 = axes[0, 1]
ax2.plot(episodes, synthetic_batches, 'orange', linewidth=2.5, label='Synthetic Batch')
ax2.fill_between(episodes, 0, synthetic_batches, alpha=0.2, color='orange')
ax2.axhline(y=max_synthetic_batch, color='red', linestyle='--', linewidth=2, 
           label=f'Max Synthetic ({max_synthetic_batch})')
ax2.set_xlabel('Episode', fontsize=12)
ax2.set_ylabel('Synthetic Batch Size', fontsize=12)
ax2.set_title('Synthetic Batch Growth (Increment Only)', fontsize=13, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, 70])

# 标注不同阶段
ax2.axvspan(0, 20, alpha=0.1, color='red')
ax2.text(10, 60, 'WM Poor\nSynthetic=0', ha='center', fontsize=9)
ax2.axvspan(20, 50, alpha=0.1, color='yellow')
ax2.text(35, 60, 'WM Improving\nSynthetic Growing', ha='center', fontsize=9)
ax2.axvspan(50, 100, alpha=0.1, color='green')
ax2.text(75, 60, 'WM Good\nSynthetic≈Max', ha='center', fontsize=9)

# ==================== 子图3: WM Quality ====================
ax3 = axes[1, 0]
ax3.plot(episodes, wm_quality, 'g-', linewidth=2, label='WM Quality')
ax3.axhline(y=quality_threshold, color='orange', linestyle='--', 
           linewidth=2, label=f'Threshold ({quality_threshold})')
ax3.fill_between(episodes, 0, wm_quality, alpha=0.2, color='green')
ax3.set_xlabel('Episode', fontsize=12)
ax3.set_ylabel('WM Quality Score', fontsize=12)
ax3.set_title('World Model Quality Evolution', fontsize=13, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_ylim([0, 1])

# ==================== 子图4: Sample Efficiency ====================
ax4 = axes[1, 1]

# 计算cumulative samples
real_cumulative = np.cumsum(real_batches)
synthetic_cumulative = np.cumsum(synthetic_batches)
total_cumulative = np.cumsum(total_batches)

# Only real baseline
only_real_cumulative = np.cumsum([base_real_batch] * num_episodes)

ax4.plot(episodes, total_cumulative, 'r-', linewidth=2.5, label='Adaptive (Real+Synthetic)')
ax4.plot(episodes, only_real_cumulative, 'b--', linewidth=2, label='Only Real Baseline')
ax4.fill_between(episodes, only_real_cumulative, total_cumulative, 
                 alpha=0.2, color='orange', label='Synthetic Bonus')
ax4.set_xlabel('Episode', fontsize=12)
ax4.set_ylabel('Cumulative Samples', fontsize=12)
ax4.set_title('Cumulative Sample Efficiency', fontsize=13, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 计算改进
final_improvement = (total_cumulative[-1] - only_real_cumulative[-1]) / only_real_cumulative[-1] * 100
ax4.text(50, total_cumulative[-1] * 0.5, 
        f'Adaptive Gain:\n+{final_improvement:.1f}% samples',
        ha='center', fontsize=11, 
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/corrected_adaptive_strategy.png', dpi=300, bbox_inches='tight')
print("✅ Visualization saved to: /mnt/user-data/outputs/corrected_adaptive_strategy.png")

# ==================== 打印统计 ====================
print("\n" + "="*60)
print("CORRECTED ADAPTIVE STRATEGY STATISTICS")
print("="*60)

print(f"\n📊 Episode 0-20 (Early Stage):")
print(f"  Real batch: {np.mean(real_batches[:20]):.0f} (固定)")
print(f"  Synthetic batch: {np.mean(synthetic_batches[:20]):.1f}")
print(f"  Total batch: {np.mean(total_batches[:20]):.1f}")
print(f"  Avg WM Quality: {np.mean(wm_quality[:20]):.3f}")

print(f"\n📊 Episode 20-50 (Growth Stage):")
print(f"  Real batch: {np.mean(real_batches[20:50]):.0f} (固定)")
print(f"  Synthetic batch: {np.mean(synthetic_batches[20:50]):.1f}")
print(f"  Total batch: {np.mean(total_batches[20:50]):.1f}")
print(f"  Avg WM Quality: {np.mean(wm_quality[20:50]):.3f}")

print(f"\n📊 Episode 50-100 (Mature Stage):")
print(f"  Real batch: {np.mean(real_batches[50:]):.0f} (固定)")
print(f"  Synthetic batch: {np.mean(synthetic_batches[50:]):.1f}")
print(f"  Total batch: {np.mean(total_batches[50:]):.1f}")
print(f"  Avg WM Quality: {np.mean(wm_quality[50:]):.3f}")

print(f"\n💡 Sample Efficiency:")
print(f"  Adaptive Total: {total_cumulative[-1]:,.0f} samples")
print(f"  Only Real: {only_real_cumulative[-1]:,.0f} samples")
print(f"  Improvement: +{final_improvement:.1f}%")

print("\n" + "="*60)
print("KEY FEATURES")
print("="*60)
print("""
✅ Real Batch = 32 (固定不变)
   - 每次训练都用32个real samples
   - 训练信号强度不会被稀释

✅ Synthetic Batch = 0~64 (动态增量)
   - Early: 0 (WM质量差，不用)
   - Mid: ~25 (WM质量提升，开始用)
   - Late: ~55 (WM质量好，充分用)

✅ Total Batch = 32 + Synthetic
   - Early: 32 (只用real)
   - Late: 87 (real + synthetic bonus)
   - Sample efficiency提升: +{:.1f}%

✅ Quality Protection
   - WM质量 < 0.3 → Synthetic = 0
   - WM质量 > 0.8 → Synthetic ≈ 64
""".format(final_improvement))

print("\n" + "="*60)

plt.show()
