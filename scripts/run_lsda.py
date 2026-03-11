import copy
import numpy as np
import yaml
import os
import sys
from rich import print
from rich.console import Console
import argparse
from math import sqrt
import logging
from datetime import datetime

import time
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import glob
import shutil

from highway_env import utils
# Add rl-agents path (repo root = parent of scripts/)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'rl-agents-master'))

from lsda.scenario.envScenario import EnvScenario
from lsda.driver_agent.driverAgent import DriverAgent
from lsda.driver_agent.vectorStore import DrivingMemory
from lsda.driver_agent.reflectionAgent import ReflectionAgent

# Use standard MCTS from rl-agents
from rl_agents.agents.tree_search.mcts import MCTSAgent
from rl_agents.agents.tree_search.graphics import TreeGraphics, MCTSGraphics, TreePlot
from rl_agents.agents.common.graphics import AgentGraphics
from summary_experiment import SimplifiedStats
from summary_experiment import save_stats_to_file, generate_stats_report

# Import LLM-MCTS modules from lsda package
from lsda.llm_mcts import (
    create_llm_prompt_engine,
    CachedPriorPolicy,
    CachedRolloutPolicy,
    HighwayEnvWithLaneChange,
    get_action_name,
    analyze_action_decision,
)

# Import TTC safety mechanism
from ttc_safety_mechanism import create_ttc_safety_mechanism

# Import TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    print("[yellow]Warning: TensorBoard not available. Install with: pip install tensorboard torch[/yellow]")
    TENSORBOARD_AVAILABLE = False

def load_config_with_llm_provider():
    """
    Load configuration and set LLM configuration (GPT only)
    """
    # Support specifying config file via environment variable
    config_path = os.environ.get('LSDA_CONFIG', 'config.yaml')
    config = yaml.load(open(config_path), Loader=yaml.FullLoader)
    
    # Use GPT configuration (only supported provider)
    gpt_config = config.get('gpt_config', {})
    config['OPENAI_KEY'] = gpt_config.get('OPENAI_KEY', '')
    config['OPENAI_CHAT_MODEL'] = gpt_config.get('OPENAI_CHAT_MODEL', 'gpt-4o-mini')
    config['OPENAI_API_BASE'] = gpt_config.get('OPENAI_API_BASE', 'https://api.openai.com/v1')
    print(f"[blue]LLM Config: Using GPT ({config['OPENAI_CHAT_MODEL']})[/blue]")
    
    return config

test_list_seed = [5838, 2421, 7294, 9650, 4176, 6382, 8765, 1348, 5678, 8587, 7523, 6321, 5214, 31, 9876, 3456, 2345, 3756, 768, 318]
                

def get_test_seeds(config):
    """
    Get test seed list based on configuration
    Supports specifying specific seeds for testing
    """
    # If specific test seeds are specified in config
    if 'test_specific_seeds' in config and config['test_specific_seeds']:
        specific_seeds = config['test_specific_seeds']
        print(f"[yellow]Using specified test seeds: {specific_seeds}[/yellow]")
        return specific_seeds
    
    # Otherwise use default seed list
    episodes_num = config.get("episodes_num", 1)
    selected_seeds = test_list_seed[:episodes_num]
    print(f"[blue]Using default seed list: {selected_seeds}[/blue]")
    return selected_seeds


def setup_env(config):
    # Set OpenAI API environment variables
    os.environ["OPENAI_API_TYPE"] = 'openai'
    os.environ["OPENAI_API_KEY"] = config['OPENAI_KEY']
    os.environ["OPENAI_CHAT_MODEL"] = config['OPENAI_CHAT_MODEL']
    os.environ["OPENAI_API_BASE"] = config['OPENAI_API_BASE']

    # Environment configuration
    env_config = {
        'highway-v0':
        {
            "observation": {
                "type": "Kinematics",
                "features": ["presence", "x", "y", "vx", "vy"],
                "absolute": True,
                "normalize": False,
                "vehicles_count": config["vehicle_count"],
                "see_behind": True,
            },
            "action": {
                "type": "DiscreteMetaAction",
                "target_speeds": np.linspace(5, 32, 9),
            },
            "lanes_count": config["lanes_count"],
            "other_vehicles_type": config["other_vehicle_type"],
            "duration": config["simulation_duration"],
            "vehicles_density": config["vehicles_density"],
            "show_trajectories": False,
            "render_agent": True,
            "scaling": 5,
            'initial_lane_id': None,
            "ego_spacing": 4,
            # Fixed render size to ensure consistent video frames
            "screen_width": 600,
            "screen_height": 150,
            # Reward configuration
            "collision_reward": -1,
            "right_lane_reward": 0,
            "high_speed_reward": 0.4,
            "lane_change_reward": -0.3,
            "reward_speed_range": [20, 30],
            # Reward normalization configuration
            "normalize_reward": config.get("normalize_reward", True),
        }
    }

    return env_config

def create_llm_optimized_mcts_agent(env, scenario, llm_engine, config):
    """
    Create LLM-optimized MCTS agent
    - Use probability distribution generated by LLM as prior policy
    - Support critical vehicle-guided search strategy
    - Support configured LLM prior probability guidance
    
    :param env: Environment object
    :param scenario: Scenario object  
    :param llm_engine: LLM prompt engine
    :param config: Configuration object
    :return: Optimized MCTS agent
    """
    # MCTS configuration (unified config, lane change rewards already built into environment)
    mcts_config = {
        "__class__": "<class 'rl_agents.agents.tree_search.mcts.MCTSAgent'>",
        "env_preprocessors": [
            {"method": "simplify"}  # Only apply simplification preprocessor, lane change reward already in environment
        ],
        "display_tree": config.get("display_tree", False),
        "rollout_debug": config.get("rollout_debug", False),        
        # Fixed search budget for consistency
        "budget": config.get("mcts_budget", 100),
        "prior_policy": {
            "type": "random_available"
        },
        "rollout_policy": {
            "type": "random_available"
        },
    }
    
    # Environment already has lane change rewards built-in, no additional wrapper needed
    
    # Check wrapper hierarchy, find correct environment to pass to MCTS
    current_env = env
    target_env = env.unwrapped  # Default to unwrapped
    
    # Traverse wrapper hierarchy, find environment containing _custom_reward
    while hasattr(current_env, 'env'):
        if hasattr(current_env, '_custom_reward'):
            target_env = current_env
            break
        current_env = current_env.env
    
    # Create MCTS agent using found correct environment
    mcts_agent = MCTSAgent(target_env, mcts_config)
    
    # Note: act_with_cached_strategies function needs to be defined before calling this function
    # Not setting it here, will set after function definition
    return mcts_agent


def setup_output_redirect(run_folder):
    """
    Setup logging output in log format (file only, no console formatting)
    
    Args:
        run_folder: Dedicated folder for this run (already contains timestamp)
        
    Returns:
        output_file_path: Output file path
        output_file: File object for output
    """
    # Suppress ffmpeg/moviepy logging
    import subprocess
    import logging as py_logging
    py_logging.getLogger('moviepy').setLevel(py_logging.ERROR)
    
    # Create output filename with .log extension
    output_filename = 'output.log'
    output_file_path = os.path.join(run_folder, output_filename)
    
    # Ensure run_folder exists before creating the log file
    os.makedirs(run_folder, exist_ok=True)
    
    # Open log file for writing
    output_file = open(output_file_path, 'w', encoding='utf-8')
    
    # Create custom wrapper for structured log output
    class StructuredLogWriter:
        """Wrapper to write structured logs without console overhead"""
        def __init__(self, file_obj):
            self.file = file_obj
            self.terminal = sys.stdout
            
        def write(self, message):
            import re
            
            # Remove ANSI color codes
            ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
            clean_message = ansi_escape.sub('', str(message)).strip()
            
            # Write to file without timestamp
            # Include empty lines (blank print statements)
            if clean_message:
                self.file.write(f'{clean_message}\n')
            elif str(message).strip() == '':
                # Write blank line if message is empty
                self.file.write('\n')
            self.file.flush()
                
        def flush(self):
            self.file.flush()
            
        def close(self):
            self.file.close()
    
    # Create wrapped file object
    wrapped_file = StructuredLogWriter(output_file)
    
    # Redirect stdout and stderr to file only
    sys.stdout = wrapped_file
    sys.stderr = wrapped_file
    
    return output_file_path, wrapped_file


def save_config_to_file(config, run_folder):
    """
    Save all configuration parameters to YAML file
    
    Args:
        config: Configuration dictionary
        run_folder: Run directory
        
    Returns:
        config_file_path: Configuration file path
    """
    config_filename = 'config_params.yaml'
    config_file_path = os.path.join(run_folder, config_filename)
    
    # Create a config copy without sensitive information and nested configs
    safe_config = {}
    for key, value in config.items():
        # Skip nested config objects
        if key in ['gpt_config']:
            continue
        # Redact API keys
        if key == 'OPENAI_KEY':
            safe_config[key] = '***REDACTED***'
        else:
            safe_config[key] = value
    
    # Save as YAML format
    with open(config_file_path, 'w', encoding='utf-8') as f:
        yaml.dump(safe_config, f, default_flow_style=False, allow_unicode=True)
    
    return config_file_path


def main_lsda():
    """
    LSDA main function
    Uses episodes_num setting from config file
    Supports environment variable override (LSDA_ENABLE_TTC, LSDA_RESULT_SUFFIX, LSDA_TIMESTAMP)
    """
    import warnings
    warnings.filterwarnings("ignore")

    # Load config with LLM provider switching
    config = load_config_with_llm_provider()
    
    # Read environment variables to support dual experiment mode
    enable_ttc_env = os.environ.get('LSDA_ENABLE_TTC', '').lower()
    result_suffix_env = os.environ.get('LSDA_RESULT_SUFFIX', '')
    timestamp_env = os.environ.get('LSDA_TIMESTAMP', '')
    result_folder_env = os.environ.get('LSDA_RESULT_FOLDER', '')  # New: support specifying results directory
    
    # Override config from environment variables if TTC status is specified
    if enable_ttc_env in ['true', 'false']:
        config['enable_ttc_safety'] = (enable_ttc_env == 'true')
        print(f"[yellow]📌 Read from environment variable: enable_ttc_safety = {config['enable_ttc_safety']}[/yellow]")
    
    # Modify result_folder if result file suffix is specified
    # Priority: LSDA_RESULT_FOLDER > config result_folder
    original_result_folder = result_folder_env if result_folder_env else config.get('result_folder', 'results')
    if result_suffix_env:
        if not timestamp_env:
            timestamp_env = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_folder = os.path.join(original_result_folder, f"exp_{timestamp_env}_{result_suffix_env}")
        config['result_folder'] = exp_folder
        print(f"[yellow]📌 Set result_folder from environment variable = {exp_folder}[/yellow]")
    else:
        # If result_suffix_env is not set, still update config when LSDA_RESULT_FOLDER is provided
        if result_folder_env:
            config['result_folder'] = result_folder_env
            print(f"[yellow]📌 Set result_folder from environment variable = {result_folder_env}[/yellow]")
    
    episodes_num = config.get('episodes_num', 1)
    
    env_config = setup_env(config)

    REFLECTION = config["reflection_module"]
    memory_path = config["memory_path"]
    few_shot_num = config["few_shot_num"]
    result_folder = config["result_folder"]
    
    # Get test seeds list
    selected_seeds = get_test_seeds(config) 
    
    # Manage results directory based on configuration and calling context
    clear_results = config.get('clear_results_before_run', False)
    
    # Check if being called by run_experiments.py (result_folder already set to exp_${timestamp}_${suffix})
    is_called_by_run_experiments = 'LSDA_RESULT_SUFFIX' in os.environ
    
    if is_called_by_run_experiments:
        # Called by run_experiments.py - result_folder is already the final destination
        # Use result_folder directly without creating sub-directories
        run_folder = result_folder
        print(f"[cyan]📁 Using result folder from run_experiments: {run_folder}[/cyan]")
    elif clear_results:
        # Mode 1: Clear results and run directly in results folder
        if os.path.exists(result_folder):
            try:
                shutil.rmtree(result_folder)
                os.makedirs(result_folder, exist_ok=True)
                print(f"[yellow]🗑️  Cleared {result_folder} directory (clear_results_before_run=True)[/yellow]")
            except Exception as e:
                print(f"[red]❌ Failed to clear {result_folder}: {e}[/red]")
        else:
            os.makedirs(result_folder, exist_ok=True)
            print(f"[cyan]📁 Created {result_folder} directory[/cyan]")
        run_folder = result_folder
    else:
        # Mode 2: Create timestamped subdirectory in results folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(result_folder, exist_ok=True)
        run_folder = os.path.join(result_folder, f"run_{timestamp}")
        os.makedirs(run_folder, exist_ok=True)
        print(f"[cyan]📁 Created new run directory: {run_folder} (clear_results_before_run=False)[/cyan]")
    
    # Generate run directory with timestamp
    lanes_count = env_config['highway-v0']['lanes_count']
    vehicles_density = config['vehicles_density']
    
    # Display run directory in console before setting output redirect
    print(f"🚀 LSDA starts running (Episodes: {episodes_num})")
    print("-" * 60)
    
    # Setup output redirect (output to log file with timestamps)
    actual_output_path, output_file = setup_output_redirect(run_folder)
    
    # Save all config parameters to file
    config_file_path = save_config_to_file(config, run_folder)
    print(f"Config parameters saved to: {config_file_path}")
    print(f"Log file saved to: {actual_output_path}")
    
    # Write config information
    print("==========LSDA Run Configuration==========")
    print(f"memory_path: {memory_path}")
    print(f"result_folder: {result_folder}")
    print(f"run_folder: {run_folder}")
    print(f"clear_results_before_run: {clear_results}")
    print(f"few_shot_num: {few_shot_num}")
    print(f"lanes_count: {lanes_count}")
    print(f"vehicles_density: {vehicles_density}")
    print(f"episodes_num: {episodes_num}")
    print(f"LLM model: {config.get('OPENAI_CHAT_MODEL', 'N/A')}")
    print(f"TTC Safety Mechanism: {'Enabled' if config.get('enable_ttc_safety', False) else 'Disabled'}")
    print(f"TTC threshold: {config.get('ttc_threshold', 'N/A')}s")
    print()

    agent_memory = DrivingMemory(db_path=memory_path)
    if REFLECTION:
        updated_memory = DrivingMemory(db_path=memory_path + "_updated")
        updated_memory.combineMemory(agent_memory)

    # Global statistics
    global_stats = SimplifiedStats()
    global_stats.start_program()  # Start program runtime timing

    # Initialize TensorBoard - save to run directory
    writer = None
    total_steps_across_episodes = 0  # Accumulate total steps of all episodes
    if TENSORBOARD_AVAILABLE and getattr(__builtins__, 'ENABLE_TENSORBOARD', False):
        tb_dir = os.path.join(run_folder, "tb_logs")
        os.makedirs(tb_dir, exist_ok=True)
        writer = SummaryWriter(tb_dir)
        # print(f"[green]📊 TensorBoard logs saved to: {tb_dir}[/green]")
        # print(f"[cyan]💡 Start TensorBoard: tensorboard --logdir={tb_dir}[/cyan]")

    episode = 0
    while episode < episodes_num:
        # 1. Create base environment (use gym.make to ensure correct spec)
        envType = 'highway-v0'
        # Use custom HighwayEnvWithLaneChange instead of default highway-v0
        env = HighwayEnvWithLaneChange()
        env.configure(env_config[envType])
        
        # Set render mode based on display_tree config
        if config.get("display_tree", False):
            env.render_mode = "human"  # Real-time display mode (show window with MCTS tree)
        else:
            env.render_mode = "rgb_array"  # Off-screen rendering        
        # 2. Environment already has lane change reward calculation, no additional wrapper needed

        # 3. Reset environment first, configure viewer then enable recording to avoid frame size mismatch
        result_prefix = f"highway_{episode + 1}"
        seed = selected_seeds[episode % len(selected_seeds)]
        obs, info = env.reset(seed=seed)
        
        # scenario and driver agent setting - save to run directory
        database_path = os.path.join(run_folder, result_prefix + ".db")
        sce = EnvScenario(env, envType, seed, database_path)
        DA = DriverAgent(sce, verbose=True)
        
        # Create LLM prompt engine
        llm_engine = create_llm_prompt_engine(config)
        
        # Create TTC safety mechanism
        ttc_safety = create_ttc_safety_mechanism(config)
        
        # First define act_with_cached_strategies function
        def act_with_cached_strategies(observation, cached_prior_policy, cached_rollout_policy):
            """Execute MCTS planning using cached strategy"""
            try:
                # Display strategy information
                # prior_info = cached_prior_policy.get_policy_info()
                # rollout_info = cached_rollout_policy.get_policy_info()
                # print(f"[blue]🌳 MCTS using strategy: prior={prior_info}, rollout={rollout_info}[/blue]")
                
                # Temporary replacement strategy
                original_prior = mcts_agent.planner.prior_policy
                original_rollout = mcts_agent.planner.rollout_policy
                
                mcts_agent.planner.prior_policy = cached_prior_policy
                mcts_agent.planner.rollout_policy = cached_rollout_policy
                
                # Execute MCTS directly, using default config and OLOP auto-allocation
                action = mcts_agent.act(observation)
                
                # Print MCTS debug info (based on config)
                if config.get('mcts_debug', False):
                    try:
                        if hasattr(mcts_agent.planner, 'root') and mcts_agent.planner.root:
                            root = mcts_agent.planner.root
                            if hasattr(root, 'children') and root.children:
                                temperature = mcts_agent.planner.config.get('temperature', 1.0)
                                print(f"[MCTS DEBUG] Root node stats after simulation (total visits: {root.count}), sqrt(N): {sqrt(root.count):.4f}")
                                for child_action, child in root.children.items():
                                    value = child.get_value()
                                    prior = child.prior
                                    exploration_base = sqrt(root.count) / (child.count + 1)
                                    exploration_term = temperature * prior * exploration_base
                                    count = child.count
                                    puct_score = child.selection_strategy(temperature)
                                    # print(f"  action={child_action}({get_action_name(child_action)}), value={value:.3f}, explore={exploration_term:.2f}, count={count}, prior={prior:.3f}, sqrt(N)/(n+1)={exploration_base:.2f}, puct={puct_score:.2f}")
                                    print(f"  action={child_action}({get_action_name(child_action)}), value={value:.3f}, count={count}, prior={prior:.3f}")

                    except Exception as debug_e:
                        print(f"[MCTS DEBUG] Print error: {debug_e}")
                
                return action, {}            
            except Exception as e:
                import traceback
                print(f"[red]❌ MCTS Internal Error: {e}[/red]")
                print(f"[red]Error Stack:\n{traceback.format_exc()}[/red]")
                # Return LLM recommended action
                actions, probabilities = cached_prior_policy(None, None)
                return actions[np.argmax(probabilities)], {}
            finally:
                # Restore original strategy
                try:
                    mcts_agent.planner.prior_policy = original_prior
                    mcts_agent.planner.rollout_policy = original_rollout
                except:
                    pass
        
        
        # Use unified MCTS config function to avoid duplicate configuration
        # Only create MCTS agent if MCTS is enabled
        if config.get("enable_mcts", True):
            mcts_agent = create_llm_optimized_mcts_agent(env, sce, llm_engine, config)
            
            # Bind TensorBoard writer to MCTS agent for tree visualization
            if writer and config.get("display_tree", False):
                mcts_agent.writer = writer
            
            # Set MCTS random seed to ensure deterministic decisions with same seed
            # Set numpy global random state
            np.random.seed(seed)
            
            # Set MCTS agent random state
            if hasattr(mcts_agent, 'seed'):
                mcts_agent.seed(seed)
            
            # Set MCTS planner random state (compatible with new and old numpy versions)
            if hasattr(mcts_agent.planner, 'np_random'):
                # New numpy uses Generator, old numpy uses RandomState
                if hasattr(mcts_agent.planner.np_random, 'bit_generator'):
                    # New numpy.random.Generator
                    from numpy.random import default_rng
                    mcts_agent.planner.np_random = default_rng(seed)
            
            # Set environment random state (already set in reset, ensure consistency here)
            if hasattr(env, 'seed'):
                env.seed(seed)
            if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'seed'):
                env.unwrapped.seed(seed)
            
            # Set method after function definition
            mcts_agent.act_with_cached_strategies = act_with_cached_strategies
        else:
            # MCTS disabled: create dummy agent object (not used)
            mcts_agent = None
            np.random.seed(seed)

        # Set MCTS agent display in environment viewer (refer to evaluation.py implementation)
        # Implement dual-screen synthesis: driving process + MCTS search tree
        if config.get("enable_agent_display", False):
            try:
                # Pre-render once to create viewer, then immediately configure agent display to ensure fixed size from first frame
                if env.render_mode is not None:
                    env.render()
                    # Set environment viewer directory
                    env.unwrapped.viewer.directory = result_folder
                    # Set agent display callback function (tree display below)
                    env.unwrapped.viewer.set_agent_display(
                        lambda agent_surface, sim_surface: AgentGraphics.display(mcts_agent, agent_surface, sim_surface))
                else:
                    print("[yellow]Cannot enable agent display when render_mode=None[/yellow]")
            except AttributeError:
                print("[yellow]⚠️ Environment viewer does not support agent rendering[/yellow]")
            except Exception as e:
                print(f"[yellow]⚠️ Failed to set agent display: {e}[/yellow]")
        else:
            print("[cyan]Agent display disabled (enable_agent_display=False)[/cyan]")

        # 4. Wrap video recording and reset to ensure consistent first frame size
        # Video saved to run directory (disable moviepy progress output)
        env = RecordVideo(env, run_folder, name_prefix=result_prefix, disable_logger=True)
        env.unwrapped.set_record_video_wrapper(env)
        obs, info = env.reset(seed=seed)
        # Configure auto frame capture (note: don't call function itself)
        try:
            env.unwrapped.automatic_rendering_callback = env.video_recorder.capture_frame
        except Exception:
            pass

        # Sync scenario object references after reset to avoid stale ego state
        sce.sync_environment(env.unwrapped)

        if REFLECTION:
            RA = ReflectionAgent(verbose=True, debug_reflection_prompts=config.get('debug_reflection_prompts', False), config=config)

        # Episode statistics
        episode_stats = SimplifiedStats()
        episode_stats.start_simulation()
        
        response = "Not available"
        action = "Not available"
        docs = []
        collision_frame = -1
        decision_step = 0  # Add decision counter

        # Output simulation start information
        print()
        print(f"==========SIMULATION {episode + 1}/{episodes_num} Started - Seed: {seed}==========")
        
        try:
            already_decision_steps = 0
            for i in range(0, config["simulation_duration"]):
                decision_step += 1  # Increment decision number
                print()
                print(f"\n--- Simulation {episode + 1} - Decision Step {decision_step} ---")
                print()  # Add blank line after decision step header
                
                # Reset random seed at each step to ensure deterministic MCTS search
                step_seed = seed + i  # Generate unique seed based on episode seed and step
                np.random.seed(step_seed)
                
                # Terminal output ego vehicle speed (m/s)
                try:
                    ego_speed = getattr(env.unwrapped.vehicle, 'speed', None)
                    if ego_speed is not None:
                        print(f"[cyan]🚄 Ego speed: {ego_speed:.2f} m/s[/cyan]")
                except Exception:
                    pass
                obs = np.array(obs, dtype=float)

                # Decide whether to retrieve few-shot memory based on config
                if config.get("enable_llm_prior_guidance", True):
                    # print("[cyan]Retreive similar memories...[/cyan]")
                    fewshot_results = agent_memory.retriveMemory(
                        sce, i, few_shot_num) if few_shot_num > 0 else []
                    fewshot_messages = []
                    fewshot_answers = []
                    fewshot_actions = []
                    for fewshot_result in fewshot_results:
                        fewshot_messages.append(fewshot_result["human_question"])
                        fewshot_answers.append(fewshot_result["LLM_response"])
                        fewshot_actions.append(fewshot_result["action"])
                        mode_action = max(
                            set(fewshot_actions), key=fewshot_actions.count)
                        mode_action_count = fewshot_actions.count(mode_action)
                    # if few_shot_num == 0:
                    #     print("[yellow]Now in the zero-shot mode, no few-shot memories.[/yellow]")
                    # else:
                    #     print("[green4]Successfully find[/green4]", len(
                    #         fewshot_actions), "[green4]similar memories![/green4]")
                else:
                    # When LLM guidance is disabled, few-shot memory is not needed
                    fewshot_results = []
                    fewshot_messages = []
                    fewshot_answers = []
                    fewshot_actions = []

                sce_descrip = sce.describe(i)
                avail_action = sce.availableActionsDescription()
                #print('[cyan]Scenario description: [/cyan]\n', sce_descrip)
                # Start timing decision process
                decision_start_time = time.time()

                # 1. Decide whether to call LLM based on config
                llm_start_time = time.time()
                
                # Check if LLM prior probability guidance is enabled
                if config.get("enable_llm_prior_guidance", True):
                    try:
                        # Get available actions from environment for current frame
                        available_actions = env.unwrapped.get_available_actions()
                                            
                        # LLM uses environment available actions to generate probability distribution and get logprobs
                        actions, probabilities, llm_chosen_action, logprobs_list = llm_engine.get_action_probabilities(
                            sce, i, available_actions,  # Use environment actions rather than fixed actions
                            print_response=True, sce_descrip=sce_descrip,
                            avail_action=avail_action, fewshot_messages=fewshot_messages,
                            fewshot_answers=fewshot_answers, fewshot_actions=fewshot_actions,
                            return_logprobs=True  # Request to return logprobs
                        )
                        
                        # Use LLM's directly selected action instead of highest probability
                        llm_action = llm_chosen_action
                        chosen_prob = probabilities[actions.index(llm_action)] if llm_action in actions else 0.0
                        response = f"LLM optimized decision: select action {llm_action}"
                        human_question = f"Frame {i} driving decision"
                        fewshot_answer = response
                        
                        print(f"[green]LLM decision completed: Action {llm_action}({get_action_name(llm_action)}) (confidence: {chosen_prob:.3f})[/green]")
                        print(f"[blue]Environment available actions: {available_actions}[/blue]")
                        
                    except Exception as e:
                        print(f"[yellow]Warning: LLM decision failed, using default: {e}[/yellow]")
                        llm_action = 1
                        chosen_prob = 0.2  # Default probability
                        actions = [0, 1, 2, 3, 4]
                        probabilities = [0.2, 0.2, 0.2, 0.2, 0.2]  # Uniform distribution
                        logprobs_list = [0.0, 0.0, 0.0, 0.0, 0.0]  # Default logprobs
                        response = "Fallback decision: IDLE"
                        human_question = f"Frame {i} driving decision"
                        fewshot_answer = response
                else:
                    # Disable LLM guidance, use random policy
                    # print(f"[yellow]LLM guidance disabled, using random policy[/yellow]")
                    
                    # Get available actions from environment
                    available_actions = env.unwrapped.get_available_actions()
                    print(f"[blue]Environment available actions: {available_actions}[/blue]")
                                       
                    response = "MCTS decision: MCTS built-in policy"
                    human_question = f"Frame {i} driving decision"
                    fewshot_answer = response
                    
                    # Set default values for compatibility
                    llm_action = 1  # Default IDLE action
                    chosen_prob = 0.0  # Not used when LLM disabled
                    actions = available_actions
                    probabilities = [1.0/len(actions)] * len(actions)
                    logprobs_list = [0.0] * len(actions)

                llm_end_time = time.time()
                llm_time = llm_end_time - llm_start_time

                # 2. Prepare strategy for MCTS (using LLM-guided prior policy)
                action_names_list = [get_action_name(action_id) for action_id in available_actions]
                print(f"[blue]Available action names: {action_names_list}[/blue]")

                # Use LLM-guided prior strategy
                if config.get("enable_llm_prior_guidance", True):
                    # Use basic LLM strategy
                    cached_prior_policy = CachedPriorPolicy(actions, probabilities)
                    
                    cached_rollout_policy = CachedRolloutPolicy(actions)
                    
                    # Get adjusted probability distribution for display
                    test_actions, guided_probs = cached_prior_policy(None, None)
                    print(f"[cyan]📊 LLM-Guided Prior Probabilities: {[f'{p:.3f}' for p in probabilities]}[/cyan]")
                else:
                    # Use MCTS built-in random strategy
                    cached_prior_policy = None  # Let MCTS use prior_policy from config
                    cached_rollout_policy = None  # Let MCTS use rollout_policy from config

                # 3. MCTS Planning or Direct LLM Decision
                mcts_start_time = time.time()
                action_values = {}
                
                # Check if MCTS is enabled
                enable_mcts = config.get("enable_mcts", True)
                
                if enable_mcts:
                    # Traditional MCTS planning
                    try:
                        print(f"[blue]Starting MCTS decision...[/blue]")

                        # Execute MCTS decision
                        if config.get("enable_llm_prior_guidance", True):
                            # Use custom strategy
                            mcts_action, action_values = mcts_agent.act_with_cached_strategies(
                                obs, cached_prior_policy, cached_rollout_policy
                            )
                        else:
                            # Use MCTS built-in strategy
                            mcts_action = mcts_agent.act(obs)
                            action_values = {}
                            
                            # Output MCTS debug info (when LLM disabled, based on config)
                            if config.get('mcts_debug', False):
                                try:
                                    if hasattr(mcts_agent.planner, 'root') and mcts_agent.planner.root:
                                        root = mcts_agent.planner.root
                                        if hasattr(root, 'children') and root.children:
                                            temperature = mcts_agent.planner.config.get('temperature', 1.0)
                                            print(f"[MCTS DEBUG] Root node stats after simulation (total visits: {root.count}), sqrt(N): {sqrt(root.count):.4f}")
                                            for child_action, child in root.children.items():
                                                value = child.get_value()
                                                prior = child.prior
                                                exploration_base = sqrt(root.count) / (child.count + 1)
                                                exploration_term = temperature * prior * exploration_base
                                                count = child.count
                                                puct_score = child.selection_strategy(temperature)
                                                # print(f"  action={child_action}({get_action_name(child_action)}), value={value:.3f}, count={count}, prior={prior:.3f}")
                                                print(f"  action={child_action}({get_action_name(child_action)}), value={value:.3f}, count={count}, prior={prior:.3f}")

                                except Exception as debug_e:
                                    print(f"[MCTS DEBUG] Print error: {debug_e}")
                        
                        print(f"[green]MCTS decision completed: Action {mcts_action}({get_action_name(mcts_action)})[/green]")                                        
                        # Visualize MCTS search tree
                        # visualize_mcts_tree(mcts_agent, decision_step, config)
                    except Exception as e:
                        print(f"[red]Error: MCTS decision failed: {e}[/red]")
                        if config.get("enable_llm_prior_guidance", True) and llm_action is not None:
                            # When LLM enabled, fallback to LLM decision
                            mcts_action = llm_action
                            action_values = {}  # Clear action values on failure
                            print(f"[yellow]Warning: Falling back to LLM decision: Action {mcts_action}({get_action_name(mcts_action)})[/yellow]")
                        else:
                            # When LLM disabled, use default action
                            mcts_action = 1  # Default IDLE action
                            action_values = {}  # Clear action values on failure
                            print(f"[yellow]Warning: MCTS failed, using default action: Action {mcts_action}({get_action_name(mcts_action)})[/yellow]")
                else:
                    # Only LLM decision mode (skip MCTS)
                    print(f"[cyan]MCTS disabled - using LLM decision directly[/cyan]")
                    if config.get("enable_llm_prior_guidance", True) and llm_action is not None:
                        mcts_action = llm_action
                        print(f"[green]Using LLM decision: Action {mcts_action}({get_action_name(mcts_action)})[/green]")
                    else:
                        # No LLM either, use default action
                        mcts_action = 1  # Default IDLE action
                        print(f"[yellow]Warning: Both MCTS and LLM disabled, using default action: {get_action_name(mcts_action)}[/yellow]")

                mcts_end_time = time.time()
                mcts_time = mcts_end_time - mcts_start_time

                # End decision timing
                decision_end_time = time.time()
                total_decision_time = decision_end_time - decision_start_time

                # 4. Record statistics
                if config.get("enable_llm_prior_guidance", True):
                    if config.get("enable_mcts", True):
                        # MCTS enabled: record both LLM and MCTS times
                        episode_stats.record_decision_time(total_decision_time, llm_time, mcts_time)
                    else:
                        # MCTS disabled: only record LLM time
                        episode_stats.record_decision_time(total_decision_time, llm_time, 0.0)
                    episode_stats.record_action_decision(llm_action, mcts_action)
                else:
                    # When LLM disabled
                    if config.get("enable_mcts", True):
                        # MCTS enabled: record MCTS time
                        episode_stats.record_decision_time(total_decision_time, 0.0, mcts_time)
                    else:
                        # Both disabled: record 0 for both
                        episode_stats.record_decision_time(total_decision_time, 0.0, 0.0)
                    episode_stats.record_action_decision(mcts_action, mcts_action)  # Use same action for both

                # TTC safety mechanism: determine which action to use based on TTC
                # a^Applied = (1-I) * a^MCTS + I * a^LLM
                # Where I is safety indicator: I=1 when TTC < threshold, otherwise I=0
                # Note: even without LLM enabled, TTC mechanism still needs to be computed (for final_action)
                if config.get("enable_ttc_safety", True):
                    # Pass env_scenario for accurate lane judgment (use laneRelative instead of lane width)
                    final_action, llm_intervened, min_ttc = ttc_safety.select_final_action(
                        llm_action, mcts_action, env_scenario=sce, observation=obs, 
                        verbose=config.get("ttc_verbose", True)
                    )
                else:
                    # When TTC safety mechanism is not enabled, use MCTS action directly
                    final_action = mcts_action
                    llm_intervened = False
                    min_ttc = np.inf

                # Record ego vehicle speed and position
                episode_stats.record_vehicle_state(env.unwrapped.vehicle)
                
                # Record TTC statistics (only when TTC safety mechanism enabled)
                if config.get("enable_ttc_safety", True):
                    episode_stats.record_ttc_intervention(llm_intervened, min_ttc)

                # Add decision analysis (only show when both LLM and MCTS are used)
                if config.get("enable_llm_prior_guidance", True) and config.get("enable_mcts", True):
                    analyze_action_decision(llm_action, mcts_action)

                # Display final decision clearly
                final_action_name = get_action_name(final_action)
                print(f"[bold]>>> Final Action: [green]{final_action}[/green]({final_action_name})[/bold]")

                # Display time statistics
                if config.get("enable_llm_prior_guidance", True):
                    if config.get("enable_mcts", True):
                        print(f"[white]Decision time: Total {total_decision_time:.3f}s (LLM: {llm_time:.3f}s, MCTS: {mcts_time:.3f}s)[/white]\n")
                    else:
                        print(f"[white]Decision time: LLM only {llm_time:.3f}s[/white]\n")
                else:
                    if config.get("enable_mcts", True):
                        print(f"[white]Decision time: MCTS: {mcts_time:.3f}s[/white]\n")
                    else:
                        print(f"[white]Decision time: {total_decision_time:.3f}s (No LLM/MCTS)[/white]\n")
                
                docs.append({
                    "sce_descrip": sce_descrip,
                    "avail_action": avail_action,  # Store available actions description
                    "human_question": human_question,
                    "response": response,
                    "action": final_action,  # Final executed action (after TTC safety mechanism)
                    "llm_action": llm_action if config.get("enable_llm_prior_guidance", True) else None,  # LLM selected action
                    "mcts_action": mcts_action if config.get("enable_mcts", True) else None,  # MCTS selected action (None if MCTS disabled)
                    "final_action": final_action,  # Final action sent to environment
                    "llm_intervened": llm_intervened,  # TTC intervention flag
                    "min_ttc": min_ttc,  # TTC value record
                    "frame": i,  # Frame index
                    "sce": copy.deepcopy(sce)
                })

                # Execute final selected action (determined by TTC safety mechanism)
                obs, reward, done, truncated, info = env.step(final_action)
                
                # TensorBoard logging
                if writer:
                    step_global = total_steps_across_episodes + i
                    
                    # Time statistics
                    writer.add_scalar("time/decision_total", total_decision_time, step_global)
                    if config.get("enable_mcts", True):
                        writer.add_scalar("time/mcts", mcts_time, step_global)
                    if config.get("enable_llm_prior_guidance", True):
                        writer.add_scalar("time/llm", llm_time, step_global)
                    
                    # Vehicle state
                    ego_speed = getattr(env.unwrapped.vehicle, 'speed', 0)
                    writer.add_scalar("ego/speed", ego_speed, step_global)
                    writer.add_scalar("ego/reward", reward, step_global)
                    
                    # Action statistics (only log MCTS action if MCTS enabled)
                    if config.get("enable_mcts", True):
                        writer.add_scalar("action/mcts", mcts_action, step_global)
                    writer.add_scalar("action/final", final_action, step_global)
                    if config.get("enable_llm_prior_guidance", True):
                        writer.add_scalar("action/llm", llm_action, step_global)
                        if config.get("enable_mcts", True):
                            writer.add_scalar("action/agreement", int(llm_action == mcts_action), step_global)
                    
                    # TTC safety mechanism statistics
                    if config.get("enable_ttc_safety", True):
                        writer.add_scalar("ttc/min_ttc", min_ttc if min_ttc != np.inf else 100.0, step_global)
                        writer.add_scalar("ttc/llm_intervened", int(llm_intervened), step_global)
                        if config.get("enable_mcts", True):
                            writer.add_scalar("ttc/action_changed", int(final_action != mcts_action), step_global)
                    
                    # MCTS root node statistics (only when MCTS enabled)
                    if config.get("enable_mcts", True) and hasattr(mcts_agent, 'planner') and hasattr(mcts_agent.planner, 'root') and mcts_agent.planner.root:
                        root = mcts_agent.planner.root
                        writer.add_scalar("mcts/root_visits", root.count, step_global)
                        writer.add_scalar("mcts/root_value", root.get_value(), step_global)
                        writer.add_scalar("mcts/children_count", len(root.children), step_global)
                        
                        # Visit statistics for each action
                        for action_id, child in root.children.items():
                            action_name = get_action_name(action_id)
                            writer.add_scalar(f"mcts_action_visits/{action_name}", child.count, step_global)
                            writer.add_scalar(f"mcts_action_values/{action_name}", child.get_value(), step_global)
                    
                    # Reward decomposition (if provided by environment)
                    reward_info = getattr(env.unwrapped, '_last_reward_info', None)
                    if reward_info:
                        for name, value in reward_info.get('raw_components', {}).items():
                            writer.add_scalar(f"reward_raw/{name}", value, step_global)
                        for name, value in reward_info.get('weighted_components', {}).items():
                            writer.add_scalar(f"reward_weighted/{name}", value, step_global)
                        writer.add_scalar("reward/total_weighted", reward_info.get('total_weighted', 0), step_global)
                        writer.add_scalar("reward/final", reward_info.get('final_reward', 0), step_global)
                        writer.add_scalar("reward/normalized_enabled", int(reward_info.get('normalized_enabled', False)), step_global)
                
                # Print reward decomposition and final reward for each decision (for audit)
                # Prioritize using info['reward_breakdown']; if not available, calculate from environment
                # if "reward_breakdown" in info:
                #     breakdown = info["reward_breakdown"]
                # else:
                #     try:
                #         breakdown = env.unwrapped._rewards(mcts_action)
                #     except Exception:
                #         breakdown = {}

                # if breakdown:
                #     # Calculate weighted value item by item using env.unwrapped.config weights
                #     cfg = getattr(env.unwrapped, 'config', {})
                #     lines = ["[bold yellow]Total reward calculation process:[/bold yellow]"]
                #     total_weighted = 0.0
                #     for name, val in breakdown.items():
                #         weight = cfg.get(name, 0)
                #         weighted = weight * val
                #         lines.append(f"  - {name}: value={val:.6f}, weight={weight}, weighted={weighted:.6f}")
                #         total_weighted += weighted
                #     lines.append(f"  Weighted sum = {total_weighted:.6f}")
                #     normalized = total_weighted
                #     if cfg.get('normalize_reward', False):
                #         min_reward = cfg.get('collision_reward', 0) + cfg.get('lane_change_reward', 0)
                #         max_reward = cfg.get('high_speed_reward', 0) + cfg.get('right_lane_reward', 0)
                #         if max_reward != min_reward:
                #             normalized = utils.lmap(total_weighted, [min_reward, max_reward], [-1, 1])
                #         lines.append(f"  Normalized(normalize_reward=True) -> {normalized:.6f}")
                #     lines.append(f"  on_road_reward = {breakdown.get('on_road_reward', 1.0)}")
                #     final_reward = normalized if cfg.get('normalize_reward', False) else total_weighted
                #     final_reward = final_reward * breakdown.get('on_road_reward', 1.0)
                #     lines.append(f"  Final total reward = {final_reward:.6f}")
                #     print("\n".join(lines))
                
                already_decision_steps += 1
                
                # Build complete few-shot string for visualization
                fewshot_display = ""
                if config.get("enable_llm_prior_guidance", True):
                    if fewshot_messages and len(fewshot_messages) > 0:
                        fewshot_display = "Few-shot Examples:\n"
                        for idx, (msg, answer, action) in enumerate(zip(fewshot_messages, fewshot_answers, fewshot_actions)):
                            fewshot_display += f"Example {idx+1}:\n"
                            fewshot_display += f"Scenario: {msg}\n"
                            fewshot_display += f"Decision: {answer}\n"
                            fewshot_display += f"Action: {action}\n\n"
                    else:
                        fewshot_display = "Zero-shot mode: No few-shot examples used."

                # Build complete decision description for visualization
                # Display decision analysis
                if config.get("enable_llm_prior_guidance", True):
                    decision_display = f"""LLM Decision Analysis:
- LLM selected action: {llm_action} ({get_action_name(llm_action)})
- LLM confidence: {chosen_prob:.3f}
- MCTS final action: {mcts_action} ({get_action_name(mcts_action)})
- Decision time: LLM {llm_time:.3f}s, MCTS {mcts_time:.3f}s, Total {total_decision_time:.3f}s

Decision Reasoning:
LLM recommended action {llm_action}, MCTS searched and {'confirmed' if llm_action == mcts_action else 'corrected to'} action {mcts_action}."""
                else:
                    decision_display = f"""MCTS Decision Analysis:
- MCTS final action: {mcts_action} ({get_action_name(mcts_action)})
- Decision time: MCTS {mcts_time:.3f}s, Total {total_decision_time:.3f}s
"""

                # Only render if render_mode is not None
                if env.render_mode is not None:
                    env.render()
                sce.promptsCommit(i, None, done, sce_descrip,
                                  fewshot_display, decision_display)
                        # Callback already set as function reference at episode start, no need to repeat here

                if done:
                    collision_frame = i + 1  # Collision frame counted from 1
                    print()
                    print(f"[red]Collision detected, Episode ended at step {collision_frame}[/red]")
                    break

        finally:
            # End episode statistics
            episode_stats.end_simulation()
            
            # Save episode results to global statistics
            episode_summary = episode_stats.get_summary()
            if episode_summary:
                # Use new method to merge episode statistics
                global_stats.add_episode_stats(episode_stats)
                
                # Merge decision statistics
                global_stats.decision_agreements += episode_stats.decision_agreements
                global_stats.total_decisions += episode_stats.total_decisions

            # Write episode summary to log
            logging.info(f"Simulation {episode + 1} | Seed {seed} | Drive Time: {already_decision_steps}s | File prefix: {result_prefix}")
            logging.info("-" * 40)

            if REFLECTION:
                print()
                print("[yellow]Now running reflection agent...[/yellow]")
                try:
                    if collision_frame != -1:  # End with collision
                        # collision_frame is 1-based, convert to 0-based for docs array access
                        collision_index = collision_frame - 1
                        print(f"[cyan]Analyzing collision scenario (collision frame: {collision_frame})[/cyan]")
                        
                        # Check if docs list is valid
                        if not docs or collision_index >= len(docs):
                            print(f"[yellow]Warning: Invalid collision frame index: collision_frame={collision_frame}, len(docs)={len(docs)}[/yellow]")
                        else:
                            # Traverse backward from collision frame to find first non-decelerate action
                            found_reflection_case = False
                            for i in range(collision_index, -1, -1):
                                try:
                                    if docs[i]["action"] != 4:  # not Deceleration
                                        print(f"[cyan]Found decision step {i+1} for reflection (action={docs[i]['action']})[/cyan]")                           
                                        
                                        # Use new reflection method with complete scenario info
                                        reflection_result = RA.reflection_on_failure(
                                            scenario_description=docs[i]["sce_descrip"],
                                            available_actions=docs[i].get("avail_action", "Not available"),
                                            executed_action=docs[i]["action"],
                                            frame_index=i + 1,
                                            collision_frame=collision_frame,
                                            debug_reflection_prompts=config.get('debug_reflection_prompts', False)
                                        )
                                        found_reflection_case = True
                                                                       
                                        print(f"[green]Analysis: {reflection_result.get('analysis', 'N/A')[:200]}...[/green]")
                                        print(f"[green]Correct Action: {reflection_result.get('correct_action', 'N/A')}[/green]")
                                        print(f"[green]One-Sentence Rule: {reflection_result.get('one_sentence_rule', 'N/A')}[/green]\n")
                                        
                                        # Check if auto-save is enabled
                                        auto_save = config.get('auto_save_reflection', True)
                                        if auto_save:
                                            choice = 'Y'
                                            print("[cyan]Auto-saving reflection result (auto_save_reflection=True)...[/cyan]")
                                        else:
                                            choice = input("Do you want to add this new memory item to update memory module? (Y/N): ").strip().upper()
                                        
                                        if choice == 'Y':
                                            # Store: scenario + decision + one-sentence rule (unified format)
                                            # Format: #### Scenario: ... [NEWLINE] #### Failure Analysis: ...
                                            # Explicitly convert action to native Python int to avoid NumPy type issues
                                            memory_content = reflection_result.get('memory_content', '')
                                            # Ensure consistent format: include scenario in memory_content if not already there
                                            if not memory_content.startswith('####'):
                                                memory_content = f"#### Scenario:\n{docs[i]['sce_descrip']}\n{memory_content}"
                                            
                                            updated_memory.addMemory(
                                                docs[i]["sce_descrip"],
                                                docs[i]["sce_descrip"],  # Use scenario as human_question for retrieval
                                                memory_content,
                                                int(docs[i]["action"]),  # ✅ Explicitly convert to Python int
                                                docs[i]["sce"],
                                                comments="collision-reflection"
                                            )
                                            print(f"[green]Successfully added new memory item. Database now has {len(updated_memory.scenario_memory._collection.get(include=['embeddings'])['embeddings'])} items.[/green]")
                                        else:
                                            print("[blue]Ignored this new memory item[/blue]")
                                        break
                                except (KeyError, IndexError) as e:
                                    print(f"[yellow]Warning: Error accessing docs at step {i}: {e}[/yellow]")
                                    continue
                                except Exception as e:
                                    print(f"[red]Error: Reflection analysis failed: {e}[/red]")
                                    import traceback
                                    print(f"Stack trace: {traceback.format_exc()}")
                                    print("[yellow]Stopping reflection attempts due to API failure[/yellow]")
                                    break  # Changed from continue to break - stop retrying on failure
                            
                            if not found_reflection_case:
                                print("[yellow]Warning: No decision step found for reflection (all steps are Deceleration actions)[/yellow]")
                    else:
                        # Success case: Check if success reflection is enabled
                        enable_success_reflection = config.get('enable_success_reflection', True)
                        
                        if enable_success_reflection:
                            # Sample decision steps based on configuration
                            # Only sample if episode completed fully (length == simulation_duration)
                            num_samples = config.get('success_experience_samples_per_episode', 3)
                            simulation_duration = config.get('simulation_duration', 30)
                            
                            if num_samples > 0 and len(docs) > 0 and already_decision_steps == simulation_duration:
                                # Only sample when episode reaches full duration (no early termination)
                                print(f"[cyan]Episode reached full duration ({already_decision_steps}s), sampling {num_samples} experiences...[/cyan]")
                                
                                # Calculate sampling interval to get approximately num_samples items
                                if len(docs) <= num_samples:
                                    # If fewer docs than desired samples, take all
                                    sampled_indices = list(range(len(docs)))
                                else:
                                    # Uniformly sample num_samples items from docs
                                    sample_interval = len(docs) // num_samples
                                    sampled_indices = [i for i in range(0, len(docs)) if i % sample_interval == 0][:num_samples]
                            else:
                                sampled_indices = []
                                if len(docs) > 0 and already_decision_steps != simulation_duration:
                                    print(f"[yellow]Episode ended early ({already_decision_steps}s < {simulation_duration}s), skipping memory sampling[/yellow]")
                            
                            if sampled_indices:
                                print(f"[green]Episode completed successfully.[/green]")
                                print(f"[cyan]Storing {len(sampled_indices)} successful decision steps to memory (without API reflection)...[/cyan]")
                                
                                cnt = 0
                                for idx in sampled_indices:
                                    try:
                                        doc = docs[idx]
                                        print(f"[cyan]Processing decision step {idx + 1}...[/cyan]")
                                        
                                        # Get action name for reference
                                        action_names = {0: "Turn-left", 1: "IDLE", 2: "Turn-right", 3: "Acceleration", 4: "Deceleration"}
                                        action_name = action_names.get(doc["action"], f"UNKNOWN_{doc['action']}")
                                        
                                        # Build unified memory content format (without duplicating scenario)
                                        # Format: #### Scenario: ... [NEWLINE] #### Successful Action: ...
                                        memory_content = f"#### Scenario:\n{doc['sce_descrip']}\n\n#### Successful Action:\nAction: {doc['action']} ({action_name})"
                                        
                                        # Store: only memory_content (no scenario duplication)
                                        # Explicitly convert action to native Python int to avoid NumPy type issues
                                        updated_memory.addMemory(
                                            doc["sce_descrip"],
                                            doc["sce_descrip"],  # Use scenario as human_question for retrieval
                                            memory_content,  # Contains both Scenario and Successful Decision
                                            int(doc["action"]),  # ✅ Explicitly convert to Python int
                                            doc["sce"],
                                            comments="success-experience"
                                        )
                                        cnt += 1
                                        print(f"[green]✓ Added successful experience from step {idx + 1} (Action: {action_name})[/green]")
                                        
                                    except Exception as e:
                                        print(f"[yellow]Warning: Error processing step {idx}: {e}[/yellow]")
                                        continue
                                
                                print(f"[green]Successfully stored {cnt} new memory items to memory module.[/green] Now the database has {len(updated_memory.scenario_memory._collection.get(include=['embeddings'])['embeddings'])} items.")
                            else:
                                print("[green]Episode completed successfully. No decision steps to sample.[/green]")
                        else:
                            print("[yellow]Success reflection is disabled in config (enable_success_reflection=False)[/yellow]")
                            
                except Exception as e:
                    print(f"[red]Error: Reflection module execution failed: {e}[/red]")
                    import traceback
                    print(f"Stack trace: {traceback.format_exc()}")

            # TensorBoard episode-level statistics
            if writer and episode_summary:
                writer.add_scalar("episode/duration", already_decision_steps, episode)
                writer.add_scalar("episode/collision", int(collision_frame != -1), episode)
                writer.add_scalar("episode/total_reward", episode_summary.get('total_reward', 0), episode)
                writer.add_scalar("episode/avg_decision_time", episode_summary.get('avg_decision_time', 0), episode)
                writer.add_scalar("episode/avg_speed", episode_summary.get('avg_speed', 0), episode)
                if config.get("enable_llm_prior_guidance", True):
                    writer.add_scalar("episode/llm_mcts_agreement", episode_summary.get('agreement_rate', 0), episode)
            
            # Update cumulative steps
            total_steps_across_episodes += already_decision_steps
            
            # Output simulation end information
            print(f"==========SIMULATION {episode + 1}/{episodes_num} Completed - Total steps: {already_decision_steps}==========")
            # if collision_frame != -1:
            #     print(f"Warning: Collision occurred at frame {collision_frame}")
            
            #print("==========Simulation {} Done==========".format(episode))
            episode += 1
            env.close()
            # Delete meta.json files generated by RecordVideo/Monitor (e.g.: highway_0-episode-0.meta.json)
            try:
                meta_pattern = os.path.join(run_folder, f"{result_prefix}-episode-*.meta.json")
                for meta_file in glob.glob(meta_pattern):
                    try:
                        os.remove(meta_file)
                    except Exception:
                        # Ignore deletion errors, continue execution
                        pass
            except Exception:
                pass
            
    # End program runtime timing
    global_stats.end_program()
    
    # Save global statistics results to run directory
    stats_filename = 'statistics.json'
    report_filename = 'statistics_report.txt'
    
    save_stats_to_file(global_stats.get_summary(), os.path.join(run_folder, stats_filename))
    generate_stats_report(global_stats.get_summary(), os.path.join(run_folder, report_filename))
    
    # Merge updated memory back to main memory (if reflection is enabled)
    if REFLECTION:
        try:
            print("\n[cyan]Merging reflection experiences back to main memory database...[/cyan]")
            agent_memory.combineMemory(updated_memory)
            agent_memory.persist()  # Explicitly persist to disk
            print(f"[green]✅ Reflection experiences successfully saved to {memory_path}[/green]")
        except Exception as e:
            print(f"[yellow]⚠️ Warning: Failed to merge updated memory: {e}[/yellow]")
    
    # Close TensorBoard
    if writer:
        writer.close()
        print(f"📊 TensorBoard logs saved, run to view: tensorboard --logdir={run_folder}/tb_logs")
    
    # # Write completion information
    # print("=" * 40)
    # print(f"✅ All episodes completed")
    # print(f"📊 Statistics file: {stats_filename}")
    # print(f"📄 Report file: {report_filename}")
    # print(f"📝 Output file: {actual_output_path}")
    # print("=" * 40)
    
    # Close output file and restore stdout to console
    if hasattr(sys.stdout, 'close'):
        sys.stdout.close()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    
    # Display completion information on console

    print(f"LSDA run completed!")
    if writer:
        print(f"📊 TensorBoard: tensorboard --logdir={run_folder}/tb_logs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LSDA Autonomous Driving Simulation')
    parser.add_argument('--enable-tensorboard', action='store_true', help='Enable TensorBoard logging')
    parser.add_argument('--config', type=str, default='config.yaml', help='Configuration file path (default: config.yaml)')
    
    args = parser.parse_args()
    
    # Set global configuration
    import builtins
    builtins.ENABLE_TENSORBOARD = args.enable_tensorboard
    
    # Run main program with graceful Ctrl+C handling
    try:
        main_lsda()
    except KeyboardInterrupt:
        # Suppress file path traceback on Ctrl+C, restore stdout first
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        print("\n[yellow]⚠️ LSDA interrupted by user (Ctrl+C)[/yellow]")
    except Exception as e:
        # Restore stdout/stderr for error output
        if sys.stdout != sys.__stdout__:
            sys.stdout = sys.__stdout__
        if sys.stderr != sys.__stderr__:
            sys.stderr = sys.__stderr__
        
        print(f"\n[red]❌ LSDA execution failed with error: {type(e).__name__}[/red]")
        print(f"[red]Error message: {str(e)[:200]}[/red]")
        
        import traceback
        print("\n[yellow]Stack trace:[/yellow]")
        traceback.print_exc()
        
        # Try to output partial statistics even on failure
        try:
            from summary_experiment import generate_stats_report, save_stats_to_file
            
            # Attempt to get partial statistics (may be incomplete)
            if 'global_stats' in globals():
                stats_summary = global_stats.get_summary()
                if 'run_folder' in globals() and run_folder:
                    print(f"\n[cyan]Attempting to save partial statistics to {run_folder}...[/cyan]")
                    
                    try:
                        stats_file = os.path.join(run_folder, 'statistics_partial.json')
                        save_stats_to_file(stats_summary, stats_file)
                        print(f"[green]✅ Partial statistics saved to: {stats_file}[/green]")
                    except Exception as save_e:
                        print(f"[yellow]⚠️ Could not save JSON statistics: {save_e}[/yellow]")
                    
                    try:
                        report_file = os.path.join(run_folder, 'statistics_report_partial.txt')
                        generate_stats_report(stats_summary, report_file)
                        print(f"[green]✅ Partial report saved to: {report_file}[/green]")
                    except Exception as report_e:
                        print(f"[yellow]⚠️ Could not save report: {report_e}[/yellow]")
        except Exception as cleanup_e:
            print(f"[yellow]⚠️ Could not save partial statistics: {cleanup_e}[/yellow]")
