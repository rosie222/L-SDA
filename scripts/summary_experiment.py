import time
import json
from collections import defaultdict
import numpy as np
class SimplifiedStats:
    """A simplified statistics container (includes speed statistics)."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all statistics."""
        self.decision_times = []  # Time cost per decision
        self.llm_times = []       # LLM decision time
        self.mcts_times = []      # MCTS decision time
        
        # Speed-related statistics
        self.speeds = []          # Vehicle speed history
        self.x_positions = []     # X position history
        self.y_positions = []     # Y position history
        self.headings = []        # Vehicle heading history
        
        self.action_counts = defaultdict(int)  # Final MCTS action counts
        self.llm_action_counts = defaultdict(int)  # LLM recommended action counts
        self.decision_agreements = 0  # Number of times LLM and MCTS agree
        self.total_decisions = 0
        
        # Lane-change frequency statistics
        self.llm_lane_changes = 0  # Lane changes recommended by LLM
        self.mcts_lane_changes = 0  # Lane changes executed by MCTS
        
        # TTC safety mechanism statistics
        self.ttc_interventions = 0  # Number of TTC interventions
        self.ttc_values = []  # TTC value history
        self.min_ttc_per_step = []  # Minimum TTC per step
        
        self.start_time = None
        self.end_time = None
        
        # Total program runtime
        self.program_start_time = None
        self.program_end_time = None
        
        # Per-episode distance statistics
        self.episode_distances = []  # Travel distance per episode
        self.episode_net_distances = []  # Net forward distance per episode
        self.episode_steps = []  # Steps per episode
    
    def start_simulation(self):
        """Mark the start of a simulation."""
        self.start_time = time.time()
    
    def end_simulation(self):
        """Mark the end of a simulation."""
        self.end_time = time.time()
    
    def start_program(self):
        """Mark the start of the program."""
        self.program_start_time = time.time()
    
    def end_program(self):
        """Mark the end of the program."""
        self.program_end_time = time.time()
    
    def record_decision_time(self, total_time, llm_time, mcts_time):
        """Record decision timing."""
        self.decision_times.append(total_time)
        self.llm_times.append(llm_time)
        self.mcts_times.append(mcts_time)
    
    def record_vehicle_state(self, vehicle):
        """Record vehicle state."""
        self.speeds.append(vehicle.speed)
        self.x_positions.append(vehicle.position[0])
        self.y_positions.append(vehicle.position[1])
        self.headings.append(vehicle.heading)
    
    def record_action_decision(self, llm_action, mcts_action):
        """Record action decisions (separating LLM vs MCTS actions)."""
        self.action_counts[mcts_action] += 1  # Final MCTS action
        self.llm_action_counts[llm_action] += 1  # LLM recommended action
        self.total_decisions += 1
        
        # Count lane-change frequency (action 0 = left lane change, action 2 = right lane change)
        if llm_action in [0, 2]:  # LLM recommended a lane change
            self.llm_lane_changes += 1
        if mcts_action in [0, 2]:  # MCTS executed a lane change
            self.mcts_lane_changes += 1
        
        if llm_action == mcts_action:
            self.decision_agreements += 1
    
    def record_ttc_intervention(self, llm_intervened, min_ttc):
        """Record TTC intervention statistics."""
        if llm_intervened:
            self.ttc_interventions += 1
        if min_ttc != np.inf:
            self.ttc_values.append(min_ttc)
        self.min_ttc_per_step.append(min_ttc if min_ttc != np.inf else 100.0)
    
    def calculate_episode_distance(self, x_positions, y_positions):
        """Compute travel distance for a single episode."""
        if len(x_positions) < 2:
            return 0
        
        total_distance = 0
        for i in range(1, len(x_positions)):
            dx = x_positions[i] - x_positions[i-1]
            dy = y_positions[i] - y_positions[i-1]
            total_distance += np.sqrt(dx*dx + dy*dy)
        
        return total_distance
    
    def calculate_episode_net_distance(self, x_positions):
        """Compute net forward distance for a single episode."""
        if len(x_positions) < 2:
            return 0
        return x_positions[-1] - x_positions[0]
    
    def add_episode_stats(self, episode_stats):
        """Add statistics from a single episode."""
        # Compute episode distances
        episode_distance = self.calculate_episode_distance(
            episode_stats.x_positions, episode_stats.y_positions
        )
        episode_net_distance = self.calculate_episode_net_distance(
            episode_stats.x_positions
        )
        
        # Compute episode steps (total decision count)
        episode_steps = episode_stats.total_decisions
        
        # Append to global stats
        self.episode_distances.append(episode_distance)
        self.episode_net_distances.append(episode_net_distance)
        self.episode_steps.append(episode_steps)
        
        # Merge other data
        self.speeds.extend(episode_stats.speeds)
        self.decision_times.extend(episode_stats.decision_times)
        self.llm_times.extend(episode_stats.llm_times)
        self.mcts_times.extend(episode_stats.mcts_times)
        
        for action, count in episode_stats.action_counts.items():
            self.action_counts[action] += count
        
        for action, count in episode_stats.llm_action_counts.items():
            self.llm_action_counts[action] += count
        
        # Merge lane-change statistics
        self.llm_lane_changes += episode_stats.llm_lane_changes
        self.mcts_lane_changes += episode_stats.mcts_lane_changes
        
        # Merge TTC statistics
        self.ttc_interventions += episode_stats.ttc_interventions
        self.ttc_values.extend(episode_stats.ttc_values)
        self.min_ttc_per_step.extend(episode_stats.min_ttc_per_step)
    
    def get_summary(self):
        """Get a summary of collected statistics."""
        if not self.decision_times:
            return None
        
        total_simulation_time = self.end_time - self.start_time if self.end_time and self.start_time else 0
        total_program_time = self.program_end_time - self.program_start_time if self.program_end_time and self.program_start_time else 0
        
        summary = {
            # Time statistics
            'total_simulation_time': total_simulation_time,
            'total_program_time': total_program_time,
            'avg_decision_time': np.mean(self.decision_times),
            'max_decision_time': np.max(self.decision_times),
            'min_decision_time': np.min(self.decision_times),
            'avg_llm_time': np.mean(self.llm_times),
            'avg_mcts_time': np.mean(self.mcts_times),
            'max_llm_time': np.max(self.llm_times) if self.llm_times else 0,
            'min_llm_time': np.min(self.llm_times) if self.llm_times else 0,
            'max_mcts_time': np.max(self.mcts_times) if self.mcts_times else 0,
            'min_mcts_time': np.min(self.mcts_times) if self.mcts_times else 0,
            'episodes_count': len(self.episode_steps),
            
            # Speed statistics
            'avg_speed': np.mean(self.speeds) if self.speeds else 0,
            'max_speed': np.max(self.speeds) if self.speeds else 0,
            'min_speed': np.min(self.speeds) if self.speeds else 0,
            'std_speed': np.std(self.speeds) if self.speeds else 0,
            
            # Decision statistics
            'decision_agreement_rate': self.decision_agreements / max(self.total_decisions, 1),
            'mcts_action_distribution': dict(self.action_counts),
            'llm_action_distribution': dict(self.llm_action_counts),
            
            # Lane-change frequency statistics (computed from action distributions)
            'llm_lane_change_frequency': (self.llm_action_counts.get(0, 0) + self.llm_action_counts.get(2, 0)) / max(sum(self.llm_action_counts.values()), 1),
            'mcts_lane_change_frequency': (self.action_counts.get(0, 0) + self.action_counts.get(2, 0)) / max(sum(self.action_counts.values()), 1),
            'llm_lane_changes_total': self.llm_action_counts.get(0, 0) + self.llm_action_counts.get(2, 0),
            'mcts_lane_changes_total': self.action_counts.get(0, 0) + self.action_counts.get(2, 0),
            
            # Keep net forward distance statistics only
            'avg_net_distance_per_episode': np.mean(self.episode_net_distances) if self.episode_net_distances else 0,
            'max_net_distance_per_episode': np.max(self.episode_net_distances) if self.episode_net_distances else 0,
            'min_net_distance_per_episode': np.min(self.episode_net_distances) if self.episode_net_distances else 0,
            
            # Step statistics (decision frequency = 1Hz, steps ~= driving time)
            'avg_steps_per_episode': np.mean(self.episode_steps) if self.episode_steps else 0,
            'max_steps_per_episode': int(np.max(self.episode_steps)) if self.episode_steps else 0,
            'min_steps_per_episode': int(np.min(self.episode_steps)) if self.episode_steps else 0,
            'episode_steps_list': self.episode_steps,
            
            # TTC safety mechanism statistics
            'ttc_interventions': self.ttc_interventions,
            'ttc_intervention_rate': self.ttc_interventions / max(self.total_decisions, 1),
            'avg_ttc': np.mean(self.ttc_values) if self.ttc_values else np.inf,
            'min_ttc': np.min(self.ttc_values) if self.ttc_values else np.inf,
            'max_ttc': np.max(self.ttc_values) if self.ttc_values else np.inf,
            
            # Detailed data
            'decision_times': self.decision_times,
            'llm_times': self.llm_times,
            'mcts_times': self.mcts_times,
            'speeds': self.speeds,
            'positions': list(zip(self.x_positions, self.y_positions)) if self.x_positions and self.y_positions else [],
            'episode_net_distances': self.episode_net_distances,
            'ttc_values': self.ttc_values,
            'min_ttc_per_step': self.min_ttc_per_step,
        }
        
        return summary
    
    def _calculate_total_distance(self):
        """Compute total travel distance."""
        if len(self.x_positions) < 2:
            return 0
        
        total_distance = 0
        for i in range(1, len(self.x_positions)):
            dx = self.x_positions[i] - self.x_positions[i-1]
            dy = self.y_positions[i] - self.y_positions[i-1]
            total_distance += np.sqrt(dx*dx + dy*dy)
        
        return total_distance
def save_stats_to_file(stats_data, filepath):
    """Save statistics data to a file."""
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.float64):
            return float(obj)
        elif isinstance(obj, np.int64):
            return int(obj)
        return obj
    
    def clean_data(data):
        if isinstance(data, dict):
            return {str(convert_numpy(k)): clean_data(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [clean_data(v) for v in data]
        else:
            return convert_numpy(data)
    
    clean_stats = clean_data(stats_data)
    
    with open(filepath, 'w') as f:
        json.dump(clean_stats, f, indent=2)

def generate_stats_report(stats_data, filepath):
    """Generate a plain-text statistics report."""
    if not stats_data or 'avg_decision_time' not in stats_data:
        return
    
    action_names = {
        0: "Turn-left",
        1: "IDLE", 
        2: "Turn-right",
        3: "Acceleration",
        4: "Deceleration"
    }
    
    report_lines = []
    report_lines.append("=" * 50)
    report_lines.append(f"L-SDA 决策统计报告")
    report_lines.append(f"Episodes数量: {stats_data.get('episodes_count', stats_data.get('episodes_num', 'N/A'))}")
    if 'total_program_time' in stats_data and stats_data['total_program_time'] > 0:
        program_time_minutes = stats_data['total_program_time'] / 60
        report_lines.append(f"程序总运行时间: {stats_data['total_program_time']:.1f}秒 ({program_time_minutes:.1f}分钟)")
    report_lines.append("=" * 50)
    
    report_lines.append("\n【决策时间统计】")
    report_lines.append(f"  平均决策时间: {stats_data['avg_decision_time']:.3f}秒")
    report_lines.append(f"  最大决策时间: {stats_data['max_decision_time']:.3f}秒")
    report_lines.append(f"  最小决策时间: {stats_data['min_decision_time']:.3f}秒")
    report_lines.append(f"  LLM平均决策时间: {stats_data['avg_llm_time']:.3f}秒 (最大: {stats_data.get('max_llm_time', 0):.3f}秒, 最小: {stats_data.get('min_llm_time', 0):.3f}秒)")
    report_lines.append(f"  MCTS平均决策时间: {stats_data['avg_mcts_time']:.3f}秒 (最大: {stats_data.get('max_mcts_time', 0):.3f}秒, 最小: {stats_data.get('min_mcts_time', 0):.3f}秒)")
    
    # Add speed statistics
    if 'avg_speed' in stats_data:
        report_lines.append("\n【速度统计】")
        report_lines.append(f"  平均速度: {stats_data['avg_speed']:.2f} m/s")
        report_lines.append(f"  最大速度: {stats_data['max_speed']:.2f} m/s")
        report_lines.append(f"  最小速度: {stats_data['min_speed']:.2f} m/s")
        report_lines.append(f"  速度标准差: {stats_data['std_speed']:.2f} m/s")
    
    # Keep net forward distance statistics only
    if 'avg_net_distance_per_episode' in stats_data:
        report_lines.append("\n【距离统计】")
        report_lines.append(f"  平均每Episode净前进距离: {stats_data['avg_net_distance_per_episode']:.2f} m")
        report_lines.append(f"  最长Episode净前进距离: {stats_data['max_net_distance_per_episode']:.2f} m")
        report_lines.append(f"  最短Episode净前进距离: {stats_data['min_net_distance_per_episode']:.2f} m")
    
    # Add step statistics
    if 'avg_steps_per_episode' in stats_data:
        report_lines.append("\n【Episode步数统计】")
        report_lines.append(f"  平均每Episode成功步数: {stats_data['avg_steps_per_episode']:.1f}步")
        report_lines.append(f"  最长Episode步数: {stats_data['max_steps_per_episode']}步")
        report_lines.append(f"  最短Episode步数: {stats_data['min_steps_per_episode']}步")
        if 'episode_steps_list' in stats_data:
            # Show per-episode step details
            report_lines.append("  各Episode步数详情:")
            for i, steps in enumerate(stats_data['episode_steps_list']):
                report_lines.append(f"    Episode {i+1}: {steps}步")
    # Driving time
    if 'avg_steps_per_episode' in stats_data:
        report_lines.append("\n【驾驶时间统计】")
        report_lines.append(f"  平均驾驶时间: {stats_data['avg_steps_per_episode']:.2f}秒")
        report_lines.append(f"  最长Episode驾驶时间: {stats_data['max_steps_per_episode']}秒")
        report_lines.append(f"  最短Episode驾驶时间: {stats_data['min_steps_per_episode']}秒")
    # Add lane-change frequency statistics
    if 'llm_lane_change_frequency' in stats_data and 'mcts_lane_change_frequency' in stats_data:
        report_lines.append("\n【变道频率统计】")
        report_lines.append(f"  LLM推荐变道频率: {stats_data['llm_lane_change_frequency']:.3f} ({stats_data['llm_lane_changes_total']}次变道/{sum(stats_data['llm_action_distribution'].values())}次决策)")
        report_lines.append(f"  MCTS执行变道频率: {stats_data['mcts_lane_change_frequency']:.3f} ({stats_data['mcts_lane_changes_total']}次变道/{sum(stats_data['mcts_action_distribution'].values())}次决策)")
        
        # Compute lane-change differences
        lane_change_diff = stats_data['mcts_lane_changes_total'] - stats_data['llm_lane_changes_total']
        if lane_change_diff > 0:
            report_lines.append(f"  MCTS比LLM多选择了{lane_change_diff}次变道")
        elif lane_change_diff < 0:
            report_lines.append(f"  LLM比MCTS多推荐了{-lane_change_diff}次变道")
        else:
            report_lines.append(f"  LLM和MCTS的变道次数完全一致")
    
    # Per-episode detailed distance section removed
    
    report_lines.append("\n【MCTS最终动作分布统计】")
    total_mcts_actions = sum(stats_data['mcts_action_distribution'].values())
    for action_id, count in stats_data['mcts_action_distribution'].items():
        action_name = action_names.get(int(action_id), f"未知({action_id})")
        percentage = count / max(total_mcts_actions, 1) * 100
        report_lines.append(f"  {action_name}(动作{action_id}): {count}次 ({percentage:.1f}%)")
    
    report_lines.append("\n【LLM推荐动作分布统计】")
    total_llm_actions = sum(stats_data['llm_action_distribution'].values())
    for action_id, count in stats_data['llm_action_distribution'].items():
        action_name = action_names.get(int(action_id), f"未知({action_id})")
        percentage = count / max(total_llm_actions, 1) * 100
        report_lines.append(f"  {action_name}(动作{action_id}): {count}次 ({percentage:.1f}%)")
    
    # Add TTC safety mechanism statistics
    if 'ttc_interventions' in stats_data:
        report_lines.append("\n【TTC安全机制统计】")
        report_lines.append(f"  TTC介入次数: {stats_data['ttc_interventions']}次")
        report_lines.append(f"  TTC介入率: {stats_data['ttc_intervention_rate']:.3f} ({stats_data['ttc_interventions']}/{total_mcts_actions})")
        
        if stats_data['avg_ttc'] != np.inf:
            report_lines.append(f"  平均TTC: {stats_data['avg_ttc']:.2f}秒")
            report_lines.append(f"  最小TTC: {stats_data['min_ttc']:.2f}秒")
            report_lines.append(f"  最大TTC: {stats_data['max_ttc']:.2f}秒")
        else:
            report_lines.append(f"  未检测到碰撞风险（所有TTC均为无穷大）")
    
    report_lines.append(f"\n【决策一致性统计】")
    report_lines.append(f"  LLM-MCTS决策一致率: {stats_data['decision_agreement_rate']:.3f}")
    
    # Compute action selection differences
    action_differences = []
    all_actions = set(stats_data['mcts_action_distribution'].keys()) | set(stats_data['llm_action_distribution'].keys())
    for action_id in all_actions:
        mcts_count = stats_data['mcts_action_distribution'].get(action_id, 0)
        llm_count = stats_data['llm_action_distribution'].get(action_id, 0)
        if mcts_count != llm_count:
            action_name = action_names.get(int(action_id), f"未知({action_id})")
            diff = mcts_count - llm_count
            if diff > 0:
                action_differences.append(f"  MCTS选择{action_name}比LLM多{diff}次")
            else:
                action_differences.append(f"  LLM推荐{action_name}比MCTS多{-diff}次")
    
    if action_differences:
        report_lines.append("\n【动作选择差异】")
        report_lines.extend(action_differences)
    
    report_lines.append("\n" + "=" * 50)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
