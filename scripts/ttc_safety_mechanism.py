"""
TTC (Time To Collision) Safety Mechanism Module
Implements TTC-based LLM intervention mechanism that takes over decision-making when potential collision risks are detected
"""

import numpy as np
from typing import Tuple, Optional, List, Dict, Any, TYPE_CHECKING
from rich import print

if TYPE_CHECKING:
    from lsda.scenario.envScenario import EnvScenario


class TTCSafetyMechanism:
    """
    TTC Safety Mechanism Class
    Determines whether LLM should take over decision-making based on TTC (Time To Collision)
    
    Formula:
    TTC(ego, target) = {
        ||P_ego - P_target||_2 / (v_ego - v_target),  if v_ego > v_target
        ∞,                                             otherwise
    }
    
    a^Applied = (1-I) * a^MCTS + I * a^LLM
    Where I is safety indicator:
        I = 1, if TTC < threshold
        I = 0, otherwise
    """
    
    def __init__(self, ttc_threshold: float = 3.0, enable: bool = True, verbose: bool = False):
        """
        Initialize TTC Safety Mechanism
        
        Args:
            ttc_threshold: TTC threshold (seconds), LLM takes over when below this value
            enable: Whether to enable TTC safety mechanism
            verbose: Whether to print initialization information
        """
        self.ttc_threshold = ttc_threshold
        self.enable = enable
        self.last_ttc_info = None
        
        # Only print in verbose mode or first initialization
        if verbose:
            print(f"[blue]🛡️ TTC Safety Mechanism Initialized: Threshold={ttc_threshold}s, Enabled={enable}[/blue]")
    
    def calculate_ttc(self, ego_pos: np.ndarray, ego_vel: np.ndarray, 
                     target_pos: np.ndarray, target_vel: np.ndarray) -> float:
        """
        Calculate TTC (Time To Collision)
        
        Args:
            ego_pos: Ego vehicle position [x, y]
            ego_vel: Ego vehicle velocity [vx, vy]
            target_pos: Target vehicle position [x, y]
            target_vel: Target vehicle velocity [vx, vy]
            
        Returns:
            TTC value (seconds), returns np.inf if no collision will occur
        """
        # Calculate position distance
        distance = np.linalg.norm(ego_pos - target_pos)
        
        # Calculate velocity magnitude
        ego_speed = np.linalg.norm(ego_vel)
        target_speed = np.linalg.norm(target_vel)
        
        # Calculate relative velocity (projection on connection line)
        # More accurate method: calculate relative velocity component on vehicle connection line
        if distance > 0:
            # Unit vector from ego to target vehicle
            direction = (target_pos - ego_pos) / distance
            
            # Calculate relative velocity projection on connection line
            # Positive value means vehicles are approaching, negative means separating
            relative_speed_projected = np.dot(ego_vel - target_vel, direction)
            
            # Calculate TTC only when ego is approaching target vehicle
            if relative_speed_projected > 0.1:  # Add small threshold to avoid division by zero
                ttc = distance / relative_speed_projected
                return ttc
        
        # No collision case
        return np.inf
    
    def calculate_min_ttc_from_observation(self, observation: np.ndarray) -> Tuple[float, Optional[int]]:
        """
        Calculate minimum TTC from observation data
        
        Args:
            observation: Observation data with shape (n_vehicles, 5)
                        Each row: [presence, x, y, vx, vy]
            
        Returns:
            (min_ttc, critical_vehicle_index)
            min_ttc: Minimum TTC value
            critical_vehicle_index: Index of most dangerous vehicle (if exists)
        """
        if observation is None or len(observation) == 0:
            return np.inf, None
        
        # Ensure observation is numpy array
        obs = np.array(observation)
        
        # First vehicle is ego vehicle
        if obs.shape[0] < 2:
            return np.inf, None
        
        ego_presence = obs[0, 0]
        if ego_presence <= 0:
            return np.inf, None
        
        ego_pos = obs[0, 1:3]
        ego_vel = obs[0, 3:5]
        
        min_ttc = np.inf
        critical_vehicle_idx = None
        
        # Traverse other vehicles
        for i in range(1, obs.shape[0]):
            presence = obs[i, 0]
            if presence <= 0:
                continue
            
            target_pos = obs[i, 1:3]
            target_vel = obs[i, 3:5]
            
            ttc = self.calculate_ttc(ego_pos, ego_vel, target_pos, target_vel)
            
            if ttc < min_ttc:
                min_ttc = ttc
                critical_vehicle_idx = i
        
        return min_ttc, critical_vehicle_idx
    
    def calculate_action_specific_ttc(self, env_scenario, action: int) -> Tuple[float, Optional[int]]:
        """
        Calculate TTC for the relevant lane based on action type
        Uses laneRelative to determine lane, consistent with envScenario.py
        Only considers closest front and rear vehicles for each lane (filtered by envScenario methods)
        
        Action-Vehicle Filtering Logic:
        - Keep/Acceleration/Deceleration (1/3/4): Only consider closest front vehicle in current lane
        - Left Lane Change (0): Only consider closest front and rear vehicles in left lane
        - Right Lane Change (2): Only consider closest front and rear vehicles in right lane
        
        Args:
            env_scenario: EnvScenario instance for surrounding vehicle info and lane judgment
            action: Action ID (0=left, 1=keep, 2=right, 3=Acceleration, 4=Deceleration)
            
        Returns:
            (min_ttc, critical_vehicle_index)
            min_ttc: Minimum TTC value for relevant lane
            critical_vehicle_index: Index of most dangerous vehicle
        """
        try:
            from highway_env.road.lane import StraightLane
            from highway_env.vehicle.behavior import IDMVehicle
        except ImportError:
            # Fallback to observation-based method if import fails
            return self.calculate_action_specific_ttc_from_observation(None, action)
        
        # Get ego vehicle and road network info
        ego = env_scenario.ego
        road_network = env_scenario.network
        
        if ego is None or road_network is None:
            return np.inf, None
        
        ego_pos = ego.position
        ego_vel = np.array([ego.velocity[0], ego.velocity[1]])
        current_lane_idx = ego.lane_index
        
        # Get surrounding vehicles
        surrounding_vehicles = env_scenario.getSurrendVehicles(vehicles_count=10)
        
        if not surrounding_vehicles:
            return np.inf, None
        
        min_ttc = np.inf
        critical_vehicle_idx = None
        vehicles_checked = 0  # Track checked vehicles
        vehicles_matched = 0  # Track matched vehicles
        
        # Get adjacent lane information
        side_lanes = road_network.all_side_lanes(current_lane_idx)
        next_lane = road_network.next_lane(
            current_lane_idx, ego.route, ego.position
        )
        
        # Classify vehicles by lane
        current_lane_vehicles = []
        left_lane_vehicles = []
        right_lane_vehicles = []
        
        for sv in surrounding_vehicles:
            sv_lane_idx = sv.lane_index
            
            if sv_lane_idx == current_lane_idx:
                # Current lane
                current_lane_vehicles.append(sv)
            elif sv_lane_idx in side_lanes:
                # Use laneRelative to determine left/right lane
                lane_relative = sv_lane_idx[2] - current_lane_idx[2]
                if lane_relative == -1:  # Left lane
                    left_lane_vehicles.append(sv)
                elif lane_relative == 1:  # Right lane
                    right_lane_vehicles.append(sv)
        
        # Filter closest front and rear vehicles for each lane (using envScenario methods)
        def get_closest_front_rear(vehicles):
            """Return closest front and rear vehicles"""
            if not vehicles:
                return None, None
            
            ahead_vehicles = []
            behind_vehicles = []
            
            for sv in vehicles:
                # Use envScenario's getSVRelativeState method
                relative_state = env_scenario.getSVRelativeState(sv)
                if relative_state == 'is ahead of you':
                    ahead_vehicles.append(sv)
                else:  # 'is behind of you'
                    behind_vehicles.append(sv)
            
            # Use envScenario's getClosestSV method
            ahead_closest = env_scenario.getClosestSV(ahead_vehicles)
            behind_closest = env_scenario.getClosestSV(behind_vehicles)
            
            return ahead_closest, behind_closest
        
        # Select vehicles to check based on action type
        vehicles_to_check = []
        
        if action in [1, 3, 4]:  # Keep, Acceleration, Deceleration: only current lane front vehicle
            ahead_closest, _ = get_closest_front_rear(current_lane_vehicles)
            if ahead_closest is not None:
                vehicles_to_check.append(ahead_closest)
                
        elif action == 0:  # Left lane change: left lane front and rear vehicles
            ahead_closest, behind_closest = get_closest_front_rear(left_lane_vehicles)
            if ahead_closest is not None:
                vehicles_to_check.append(ahead_closest)
            if behind_closest is not None:
                vehicles_to_check.append(behind_closest)
                
        elif action == 2:  # Right lane change: right lane front and rear vehicles
            ahead_closest, behind_closest = get_closest_front_rear(right_lane_vehicles)
            if ahead_closest is not None:
                vehicles_to_check.append(ahead_closest)
            if behind_closest is not None:
                vehicles_to_check.append(behind_closest)
        
        # Calculate TTC for filtered vehicles
        min_ttc = np.inf
        critical_vehicle_idx = None
        vehicles_matched = len(vehicles_to_check)
        vehicles_checked = len(surrounding_vehicles)
        
        for sv in vehicles_to_check:
            target_pos = sv.position
            target_vel = np.array([sv.velocity[0], sv.velocity[1]])
            
            ttc = self.calculate_ttc(ego_pos, ego_vel, target_pos, target_vel)
            if ttc < min_ttc:
                min_ttc = ttc
                # Get this vehicle's index in original list
                try:
                    critical_vehicle_idx = surrounding_vehicles.index(sv)
                except ValueError:
                    critical_vehicle_idx = None
        
        # Store debug info (including reason diagnosis)
        if not hasattr(self, 'last_action_info'):
            self.last_action_info = {}
        
        # Diagnose TTC=∞ reason
        ttc_inf_reason = None
        if min_ttc == np.inf:
            if not surrounding_vehicles:
                ttc_inf_reason = "No vehicles nearby"
            elif vehicles_matched == 0:
                if action in [1, 3, 4]:
                    ttc_inf_reason = "No vehicle in front of current lane"
                elif action == 0:
                    ttc_inf_reason = "No vehicle in left lane"
                elif action == 2:
                    ttc_inf_reason = "No vehicle in right lane"
            else:
                ttc_inf_reason = "Vehicles are separating"
        
        self.last_action_info[action] = {
            'vehicles_checked': vehicles_checked,
            'vehicles_matched': vehicles_matched,
            'min_ttc': min_ttc,
            'ttc_inf_reason': ttc_inf_reason
        }
        
        return min_ttc, critical_vehicle_idx
    
    def calculate_action_specific_ttc_from_observation(self, observation: np.ndarray, action: int) -> Tuple[float, Optional[int]]:
        """
        Calculate action-specific TTC from observation data (backup method based on lane width)
        
        Args:
            observation: Observation data with shape (n_vehicles, 5)
                        Each row: [presence, x, y, vx, vy]
            action: Action ID (0=left, 1=keep, 2=right, 3=Acceleration, 4=Deceleration)
            
        Returns:
            (min_ttc, critical_vehicle_index)
        """
        if observation is None or len(observation) == 0:
            return np.inf, None
        
        obs = np.array(observation)
        if obs.shape[0] < 2:
            return np.inf, None
        
        ego_presence = obs[0, 0]
        if ego_presence <= 0:
            return np.inf, None
        
        ego_pos = obs[0, 1:3]
        ego_vel = obs[0, 3:5]
        ego_y = ego_pos[1]  # Lateral position
        
        # Lane width approximately 4 meters
        LANE_WIDTH = 4.0
        
        min_ttc = np.inf
        critical_vehicle_idx = None
        
        # Filter relevant vehicles based on action type
        for i in range(1, obs.shape[0]):
            presence = obs[i, 0]
            if presence <= 0:
                continue
            
            target_pos = obs[i, 1:3]
            target_vel = obs[i, 3:5]
            target_y = target_pos[1]
            target_x = target_pos[0]
            ego_x = ego_pos[0]
            
            # Calculate lateral distance (lane judgment)
            lateral_dist = target_y - ego_y
            lateral_dist_abs = abs(lateral_dist)
            
            # Determine if vehicle is in attention lane range
            should_consider = False
            
            if action in [1, 3, 4]:  # Keep, Acceleration, Deceleration: only current lane front vehicle
                # Current lane: lateral distance less than half lane width
                in_same_lane = lateral_dist_abs < LANE_WIDTH / 2
                # Front vehicle: longitudinal position ahead of ego
                is_front = target_x > ego_x
                should_consider = in_same_lane and is_front
                
            elif action == 0:  # Left lane change: left lane front and rear vehicles
                # Left lane: lateral distance in [0.5 lane width, 1.5 lane width]
                in_left_lane = lateral_dist < 0 and LANE_WIDTH * 0.5 < lateral_dist_abs < LANE_WIDTH * 1.5
                should_consider = in_left_lane
                
            elif action == 2:  # Right lane change: right lane front and rear vehicles
                # Right lane: lateral distance in [0.5 lane width, 1.5 lane width]
                in_right_lane = lateral_dist > 0 and LANE_WIDTH * 0.5 < lateral_dist_abs < LANE_WIDTH * 1.5
                should_consider = in_right_lane
            
            # Only calculate TTC for relevant vehicles
            if should_consider:
                ttc = self.calculate_ttc(ego_pos, ego_vel, target_pos, target_vel)
                if ttc < min_ttc:
                    min_ttc = ttc
                    critical_vehicle_idx = i
        
        return min_ttc, critical_vehicle_idx
    
    def should_llm_intervene(self, env_scenario=None, action: int = None, verbose: bool = True, observation: np.ndarray = None) -> Tuple[bool, float, Optional[int]]:
        """
        Determine if LLM should take over (based on action-related TTC)
        
        Args:
            env_scenario: EnvScenario instance (preferred), for precise lane judgment
            action: Action selected by MCTS
            verbose: Whether to print detailed information
            observation: Observation data (backup if env_scenario unavailable)
            
        Returns:
            (should_intervene, min_ttc, critical_vehicle_idx)
            should_intervene: Whether should intervene
            min_ttc: Minimum TTC value for relevant lane
            critical_vehicle_idx: Critical vehicle index
        """
        if not self.enable or action is None:
            return False, np.inf, None
        
        # Prioritize env_scenario for precise lane judgment
        if env_scenario is not None:
            min_ttc, critical_vehicle_idx = self.calculate_action_specific_ttc(env_scenario, action)
        else:
            # Fall back to observation-based method
            min_ttc, critical_vehicle_idx = self.calculate_action_specific_ttc_from_observation(observation, action)
        
        # Save info for subsequent analysis
        ttc_inf_reason = None
        if hasattr(self, 'last_action_info') and action in self.last_action_info:
            ttc_inf_reason = self.last_action_info[action].get('ttc_inf_reason')
        
        self.last_ttc_info = {
            'min_ttc': min_ttc,
            'critical_vehicle_idx': critical_vehicle_idx,
            'threshold': self.ttc_threshold,
            'action': action,
            'ttc_inf_reason': ttc_inf_reason
        }
        
        should_intervene = min_ttc < self.ttc_threshold
        
        # Action name mapping
        action_names = {0: "Turn-left", 1: "IDLE", 2: "Turn-right", 3: "Acceleration", 4: "Deceleration"}
        action_name = action_names.get(action, f"Unknown({action})")
        
        if verbose and should_intervene:
            print(f"[red]⚠️ TTC Warning [{action_name}]: Min TTC={min_ttc:.2f}s < Threshold {self.ttc_threshold}s, LLM Override[/red]")
            if critical_vehicle_idx is not None:
                print(f"[red]🚨 Critical Vehicle Index: {critical_vehicle_idx}[/red]")
        elif verbose and min_ttc < self.ttc_threshold * 1.5 and min_ttc < np.inf:
            # Give alert when TTC approaches threshold but hasn't triggered yet
            print(f"[yellow]⚡ TTC Alert [{action_name}]: Min TTC={min_ttc:.2f}s, Approaching Threshold {self.ttc_threshold}s[/yellow]")
        
        return should_intervene, min_ttc, critical_vehicle_idx
    
    def select_final_action(self, llm_action: int, mcts_action: int, 
                          env_scenario=None, observation: np.ndarray = None, verbose: bool = True) -> Tuple[int, bool, float]:
        """
        Select final action based on TTC (Time To Collision) considering MCTS action's lane
        
        Improved Strategy:
        - If TTC < threshold AND MCTS action is not Deceleration(4), use LLM decision
        - If TTC < threshold BUT MCTS action is already Deceleration(4), keep MCTS decision (avoid redundant intervention)
        - If TTC >= threshold, use MCTS decision
        
        Formula: a_Applied = (1-I) * a_MCTS + I * a_LLM
        Where I is the safety indicator:
            I = 1, if TTC < threshold AND mcts_action != 4 (Deceleration)
            I = 0, otherwise
        
        Args:
            llm_action: Action recommended by LLM
            mcts_action: Action selected by MCTS
            env_scenario: EnvScenario instance (preferred for accurate lane judgment)
            observation: Current observation (backup)
            verbose: Whether to print detailed information
            
        Returns:
            (final_action, llm_intervened, min_ttc)
            final_action: Final selected action
            llm_intervened: Whether LLM intervention occurred
            min_ttc: Minimum TTC value for the lane related to MCTS action
        """
        # Calculate TTC for relevant lane based on MCTS action
        should_intervene, min_ttc, critical_vehicle_idx = self.should_llm_intervene(
            env_scenario=env_scenario, action=mcts_action, verbose=False, observation=observation
        )
        
        # Action name mapping
        action_names = {0: "Turn-left", 1: "IDLE", 2: "Turn-right", 3: "Acceleration", 4: "Deceleration"}
        
        # New logic: if TTC < threshold but MCTS already selected Deceleration, keep MCTS decision
        if should_intervene and mcts_action == 4:
            # TTC < threshold, but MCTS is already decelerating, no LLM override needed
            final_action = mcts_action
            if verbose:
                mcts_name = action_names.get(mcts_action, f"Action{mcts_action}")
                print(f"[yellow]⚠️ TTC Warning (TTC={min_ttc:.2f}s < {self.ttc_threshold}s)[/yellow]")
                print(f"[yellow]   But MCTS selected [Deceleration], keeping MCTS decision, no LLM override needed[/yellow]")
            return final_action, False, min_ttc  # llm_intervened = False
        
        if should_intervene:
            # I = 1: TTC < threshold AND MCTS is not Deceleration, use LLM decision
            final_action = llm_action
            if verbose:
                mcts_name = action_names.get(mcts_action, f"Action{mcts_action}")
                llm_name = action_names.get(llm_action, f"Action{llm_action}")
                print(f"[red]🛡️ TTC Safety Mechanism Activated: Using LLM Decision (TTC={min_ttc:.2f}s < {self.ttc_threshold}s)[/red]")
                print(f"[red]   MCTS Action={mcts_action}({mcts_name}) → LLM Action={llm_action}({llm_name})[/red]")
        else:
            # I = 0: use MCTS decision
            final_action = mcts_action
            if verbose:
                mcts_name = action_names.get(mcts_action, f"Action{mcts_action}")
                if min_ttc < np.inf:
                    print(f"[green]✅ Safe State [{mcts_name}]: Using MCTS Decision (TTC={min_ttc:.2f}s ≥ {self.ttc_threshold}s)[/green]")
                else:
                    # Get specific reason for TTC=∞
                    ttc_inf_reason = "No relevant vehicles"
                    if hasattr(self, 'last_action_info') and mcts_action in self.last_action_info:
                        reason = self.last_action_info[mcts_action].get('ttc_inf_reason')
                        if reason:
                            ttc_inf_reason = reason
                    print(f"[green]✅ Safe State [{mcts_name}]: Using MCTS Decision (TTC=∞, Reason: {ttc_inf_reason})[/green]")
        
        return final_action, should_intervene, min_ttc
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics information"""
        if self.last_ttc_info is None:
            return {
                'enabled': self.enable,
                'threshold': self.ttc_threshold,
                'last_ttc': None
            }
        
        return {
            'enabled': self.enable,
            'threshold': self.ttc_threshold,
            'last_ttc': self.last_ttc_info['min_ttc'],
            'last_critical_vehicle': self.last_ttc_info['critical_vehicle_idx']
        }
    
    def calculate_detailed_ttc_info(self, observation: np.ndarray) -> List[Dict[str, Any]]:
        """
        Calculate detailed TTC information (for debugging and analysis)
        
        Returns:
            List containing TTC information for each vehicle
        """
        if observation is None or len(observation) == 0:
            return []
        
        obs = np.array(observation)
        if obs.shape[0] < 2:
            return []
        
        ego_pos = obs[0, 1:3]
        ego_vel = obs[0, 3:5]
        ego_speed = np.linalg.norm(ego_vel)
        
        ttc_info_list = []
        
        for i in range(1, obs.shape[0]):
            presence = obs[i, 0]
            if presence <= 0:
                continue
            
            target_pos = obs[i, 1:3]
            target_vel = obs[i, 3:5]
            target_speed = np.linalg.norm(target_vel)
            
            distance = np.linalg.norm(ego_pos - target_pos)
            ttc = self.calculate_ttc(ego_pos, ego_vel, target_pos, target_vel)
            
            # Calculate lateral and longitudinal distance
            lateral_dist = target_pos[1] - ego_pos[1]  # y direction
            longitudinal_dist = target_pos[0] - ego_pos[0]  # x direction
            
            ttc_info_list.append({
                'vehicle_idx': i,
                'ttc': ttc,
                'distance': distance,
                'lateral_dist': lateral_dist,
                'longitudinal_dist': longitudinal_dist,
                'ego_speed': ego_speed,
                'target_speed': target_speed,
                'relative_speed': ego_speed - target_speed
            })
        
        # Sort by TTC
        ttc_info_list.sort(key=lambda x: x['ttc'])
        
        return ttc_info_list


def create_ttc_safety_mechanism(config: Dict[str, Any], verbose: bool = False) -> TTCSafetyMechanism:
    """
    Create TTC safety mechanism based on configuration
    
    Args:
        config: Configuration dictionary
        verbose: Whether to print initialization info (default False, avoid printing for each simulation)
        
    Returns:
        TTCSafetyMechanism instance
    """
    enable = config.get('enable_ttc_safety', True)
    threshold = config.get('ttc_threshold', 3.0)
    
    return TTCSafetyMechanism(ttc_threshold=threshold, enable=enable, verbose=verbose)
