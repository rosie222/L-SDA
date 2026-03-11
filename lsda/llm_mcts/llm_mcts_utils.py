"""
LLM-MCTS utility functions and policy classes.
Includes policy classes, reward wrappers, and helper functions.
"""

import copy
import numpy as np
import os
import sys
from rich import print
import gymnasium as gym
from highway_env import utils

# Add rl-agents path (repo root when this file is under lsda/llm_mcts/)
_reporoot = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(_reporoot, 'rl-agents-master'))
from rl_agents.agents.tree_search.mcts import MCTSAgent
from highway_env.envs.highway_env import HighwayEnv, Action
from typing import Dict

class HighwayEnvWithLaneChange(HighwayEnv):
    """
    Extend HighwayEnv to compute lane-change rewards directly in the environment.
    This ensures lane-change penalties take effect correctly during MCTS rollout.
    """
    
    def _rewards(self, action: Action) -> Dict[str, float]:
        """
        Override reward calculation to add lane-change detection and a custom speed reward.
        
        Args:
            action: Action in various formats (int, array, tuple, etc.)
            
        Returns:
            Reward dictionary including lane-change reward.
        """
        # Get original reward items (excluding lane_change_reward)
        rewards = super()._rewards(action)

        # Detect if lane change action (support int discrete actions, arrays/tuples, etc.)
        lane_change = 0.0
        try:
            act = action
            if isinstance(action, (list, tuple, np.ndarray)):
                # Multiple agents or array, take first agent action as example
                act = action[0]
            if isinstance(act, (int, np.integer)):
                idx = getattr(self.action_type, "actions_indexes", {})
                # Support multiple naming formats: LANE_LEFT/LANE_RIGHT or Left/Right
                if idx.get("LANE_LEFT") == int(act) or idx.get("LANE_RIGHT") == int(act):
                    lane_change = 1.0
                    # Only output during lane change to reduce log noise
                    # print(f"[red]⚠️ Lane change action: {act}[/red]")
        except Exception as e:
            print(f"[yellow]⚠️ 变道检测异常: {e}[/yellow]")
            lane_change = 0.0

        rewards["lane_change_reward"] = lane_change
        
        # Custom speed reward function: r = 1.0 / (1.0 + exp(-k * (v - speed_target)))
        # k=0.2, speed_target=25
        if hasattr(self, 'vehicle') and self.vehicle:
            current_speed = self.vehicle.speed  # Get current speed (m/s)
            k = 0.2
            speed_target = 25.0
            
            # Sigmoid speed reward function
            speed_reward = 1.0 / (1.0 + np.exp(-k * (current_speed - speed_target)))
            
            # Replace original high_speed_reward
            rewards["high_speed_reward"] = speed_reward
            
            # Optional: add debug information
            # print(f"[cyan]🚄 Speed reward: v={current_speed:.2f}m/s -> r={speed_reward:.4f}[/cyan]")
        
        return rewards

# Strategy class definition
class CachedPriorPolicy:
    def __init__(self, actions, probabilities):
        self.actions = np.array(actions)
        self.probabilities = np.array(probabilities)
        
    def __call__(self, state, observation):
        """Return the cached LLM probability distribution."""
        return self.actions, self.probabilities
    
    def get_policy_info(self):
        """Return a policy info string."""
        return "LLM prior"

class CachedRolloutPolicy:
    def __init__(self, actions):
        self.actions = np.array(actions)
        self.probabilities = np.ones(len(actions)) / len(actions)
        
    def __call__(self, state, observation):
        """Return a uniform-distribution rollout policy."""
        return self.actions, self.probabilities
    
    def get_policy_info(self):
        """Return a policy info string."""
        return "Uniform rollout"

def get_action_name(action_id):
    """Get action name in English"""
    action_names = {
        0: "Turn-left",
        1: "IDLE", 
        2: "Turn-right",
        3: "Acceleration",
        4: "Deceleration"
    }
    return action_names.get(action_id, f"UNKNOWN({action_id})")


def analyze_action_decision(llm_action, mcts_action, mcts_action_values=None, mcts_search_stats=None):
    """Analyze action decisions."""
    llm_name = get_action_name(llm_action)
    mcts_name = get_action_name(mcts_action)
    
    print(f"[bold green]LLM Decision: {llm_action}({llm_name}), MCTS Decision: {mcts_action}({mcts_name})[/bold green]")
