"""
LLM Prompt Template Module
Used to generate action probability distributions and select critical vehicles
"""
import yaml
import os
import time
import numpy as np
from typing import List, Dict, Tuple, Any, TYPE_CHECKING
from rich import print
if TYPE_CHECKING:
    from lsda.scenario.envScenario import EnvScenario

# Import openai
try:
    import openai
except ImportError:
    print("Warning: OpenAI package not found. LLM features will be disabled.")
    openai = None


# LLM Retry Configuration Constants
LLM_RETRY_CONFIG = {
    'max_retries': 5,           # Maximum number of retries
    'initial_delay': 1.0,       # Initial delay (seconds)
    'max_delay': 30.0,          # Maximum delay (seconds)
    'backoff_factor': 2.0,      # Exponential backoff factor
    'timeout': 30.0,            # API call timeout (seconds)
}


class LLMPromptEngine:
    """LLM Prompt Engine for handling action decisions and critical vehicle selection"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.debug_prompts = config.get("debug_prompts", False)  # Debug switch: whether to print prompts
        self.action_mapping = {
            0: "Turn-left",
            1: "IDLE", 
            2: "Turn-right",
            3: "Acceleration",
            4: "Deceleration"
        }
        
        # Load retry configuration from config or use defaults
        self.retry_config = {
            'max_retries': config.get('api_max_retries', LLM_RETRY_CONFIG['max_retries']),
            'initial_delay': config.get('api_initial_delay', LLM_RETRY_CONFIG['initial_delay']),
            'max_delay': config.get('api_max_delay', LLM_RETRY_CONFIG['max_delay']),
            'backoff_factor': config.get('api_backoff_factor', LLM_RETRY_CONFIG['backoff_factor']),
            'timeout': config.get('api_timeout', LLM_RETRY_CONFIG['timeout']),
        }
        
        if self.debug_prompts:
            print("[blue]🔧 LLM engine: debug mode enabled, will print all prompts[/blue]")
    
    def _exponential_backoff_retry(self, func, *args, max_retries=None, **kwargs):
        """
        Execute function with exponential backoff retry mechanism
        
        Args:
            func: Function to execute
            *args: Positional arguments
            max_retries: Maximum number of retries
            **kwargs: Keyword arguments
            
        Returns:
            Function execution result
            
        Raises:
            Exception: If all retries fail
        """
        if max_retries is None:
            max_retries = self.retry_config['max_retries']
        
        delay = self.retry_config['initial_delay']
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    print(f"[cyan]🔄 LLM API Retry (Attempt {attempt + 1}/{max_retries})...[/cyan]")
                return func(*args, **kwargs)
            except (openai.error.APIConnectionError, openai.error.APITimeoutError, openai.error.Timeout, TimeoutError) as e:
                last_exception = e
                if attempt < max_retries - 1:
                    print(f"[yellow]⚠️ LLM API Connection Error: {type(e).__name__} - {str(e)[:100]}[/yellow]")
                    print(f"[yellow]   Waiting {delay:.1f}s before retry... (Attempt {attempt + 1}/{max_retries})[/yellow]")
                    time.sleep(delay)
                    # Exponential backoff: delay = min(delay * backoff_factor, max_delay)
                    delay = min(delay * self.retry_config['backoff_factor'], self.retry_config['max_delay'])
                else:
                    print(f"[red]❌ LLM API failed after {max_retries} retries[/red]")
            except openai.error.RateLimitError as e:
                last_exception = e
                if attempt < max_retries - 1:
                    # Rate limit: use longer delay
                    rate_limit_delay = min(delay * 2, 60.0)
                    print(f"[yellow]⚠️ Rate limit reached. Waiting {rate_limit_delay:.1f}s before retry...[/yellow]")
                    time.sleep(rate_limit_delay)
                    delay = rate_limit_delay
                else:
                    print(f"[red]❌ LLM API rate limit: exhausted all retries[/red]")
            except Exception as e:
                print(f"[red]❌ Unexpected error in LLM API: {type(e).__name__} - {str(e)[:100]}[/red]")
                raise
        
        # All retries failed
        raise last_exception or Exception("LLM API call failed: Unknown error after all retries")
    
    def _call_openai_api(self, prompt: str, max_tokens: int = 10, logprobs: bool = False, top_logprobs: int = 5) -> Dict[str, Any]:
        """
        Unified interface for calling OpenAI API with retry mechanism
        Supports GPT special configurations
        """
        if openai is None:
            raise Exception("OpenAI package not available")
        
        # Check API key
        api_key = os.environ.get("OPENAI_API_KEY") or self.config.get("OPENAI_KEY")
        if not api_key:
            raise Exception("No API key provided. You can set your API key in code using 'openai.api_key = <API-KEY>', or you can set the environment variable OPENAI_API_KEY=<API-KEY>")
        
        openai.api_key = api_key
        openai.api_base = self.config.get("OPENAI_API_BASE")
        openai.request_timeout = self.retry_config['timeout']  # Set timeout
        
        # Get model name
        model_name = self.config.get("OPENAI_CHAT_MODEL", "gpt-3.5-turbo-0125")
        
        # Base parameter dict
        def call_api():
            completion_params = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": self.config.get("llm_sampling_temperature", 0.0),  # Add sampling temperature parameter
                "request_timeout": self.retry_config['timeout'],
            }
            
            if logprobs:
                completion_params["logprobs"] = True  # GPT logprobs parameter is also boolean
                completion_params["top_logprobs"] = top_logprobs  # GPT uses top_logprobs to specify quantity
          
            return openai.ChatCompletion.create(**completion_params)
        
        # Execute with retry mechanism
        response = self._exponential_backoff_retry(call_api)
        return response
    
    def get_action_probabilities(self, scenario: 'EnvScenario', frame: int, available_actions: List[int], 
                            critical_vehicles: List[int] = None, print_response: bool = True, 
                            sce_descrip: str = None, avail_action: str = None, 
                            fewshot_messages: List[str] = None, fewshot_answers: List[str] = None, 
                            fewshot_actions: List[int] = None, return_logprobs: bool = False) -> Tuple[List[int], List[float], int, List[float]]:
        """
        Generate action probability distribution using LLM
        
        Args:
            scenario: Scenario object
            frame: Current frame number
            available_actions: List of available actions
            critical_vehicles: List of critical vehicle indices (deprecated)
            print_response: Whether to print LLM response
            sce_descrip: Scenario description
            avail_action: Available actions description
            fewshot_messages: List of few-shot example questions
            fewshot_answers: List of few-shot example answers
            fewshot_actions: List of few-shot example actions
            return_logprobs: return logprobs
            
        Returns:
            (actions, probabilities, llm_chosen_action, logprobs) if return_logprobs else (actions, probabilities, llm_chosen_action)
        """
        scenario_description = scenario.describe(frame)
        
        # Build few-shot examples - use experiences from memory
        fewshot_examples = ""
        if (fewshot_messages and fewshot_answers and fewshot_actions and 
            len(fewshot_messages) == len(fewshot_answers) == len(fewshot_actions) > 0):
            fewshot_examples += "## Reference Experiences\n"
            fewshot_examples += "The following are lessons learned from similar driving scenarios, You can refer to these experiences when making decisions:\n\n"
            for i, (msg, answer, action) in enumerate(zip(fewshot_messages, fewshot_answers, fewshot_actions)):
                fewshot_examples += f"**Experience {i+1}:**\n"
                fewshot_examples += f"{answer}\n\n"
            fewshot_examples += "---\n\n"

        # System prompt following prompt.txt structure
        system_prompt = """# Role: Autonomous Driving Decision Strategy Generator

## Profile
- language: English
- description: You are a strategy decision expert for an autonomous driving agent. Your responsibility is to generate the optimal action (0-4) based on the observation information of the highway-env v0 environment, helping the vehicle to safely and efficiently travel in the highway environment.

## Skills
- Thoroughly understand the state observation structure of highway-env v0 (such as distance, speed, relative positions of surrounding vehicles, etc.).
- Be familiar with the discrete action space of highway-env: 0 = Turn-left, 1 = IDLE, 2 = Turn-right, 3 = Acceleration, 4 = Deceleration.
- Be able to generate safe, reasonable and explainable autonomous driving decisions based on real-time observations.
- Be capable of structuring complex traffic scenarios and extracting core information for decision-making.

## Goals
- Based on the environmental observation, output an integer action SELECTED FROM AVAILABLE ACTIONS ONLY.
- Safety comes first, followed by efficiency. Unnecessary lane changes should be avoided.
- When the risk of collision is high, strategies such as deceleration or staying in the lane should be adopted first.

## Rules
1. Always output only legal actions: 0, 1, 2, 3, 4.
2. If unsure, prioritize safe actions (such as IDLE or SLOWER).
3. Decisions must be based on observations and no fabricated information should be included.
4. Do not output unnecessary text.

## Workflows
1. Receive the observation and analyze the positions, speeds and risks of the vehicle and surrounding vehicles.
2. Determine if there is a collision risk:
   - If the longitudinal distance is too close (< 20m) → Slow down (Action 4) or change lanes
   - If the lateral distance is insufficient → Avoid changing lanes
3. Consider efficiency in a safe manner:
   - If the preceding vehicle is moving slowly → Consider lane change if safe
   - If the road is clear (> 40m ahead) → Can accelerate (Action 3)
4. Output the optimal action from the Available Actions.
"""

        prompt = f"""{system_prompt}
{fewshot_examples}## Current Observation
{scenario_description}
## Available Actions
{avail_action}
## Output Format
Output only a single integer (0, 1, 2, 3, or 4) from the Available Actions. No explanation needed.
"""
        # Print complete prompt to terminal (for debugging)
        if self.debug_prompts:
            print(f"\n[cyan]{'='*80}[/cyan]")
            print(f"[cyan]📝 LLM 输入 Prompt (Frame {frame}):[/cyan]")
            print(f"[cyan]{'='*80}[/cyan]")
            print(f"[cyan]{prompt}[/cyan]")
            print(f"[cyan]{'='*80}[/cyan]\n")
        
        try:
            response = self._call_openai_api(prompt, max_tokens=10, logprobs=True, top_logprobs=5)
            
            # Print raw LLM response
            response_text = response.choices[0].message.content.strip()
            print(f"[green]🤖LLM Response:'{response_text}'[/green]")
            
            # Extract logprobs information
            logprobs_data = response.choices[0].logprobs if hasattr(response.choices[0], 'logprobs') else None
            
            # Check logprobs availability (simplified output)
            if not (logprobs_data and hasattr(logprobs_data, 'content') and logprobs_data.content):
                print("[yellow]⚠️ No logprobs data, using heuristic method[/yellow]")
            
            if logprobs_data and hasattr(logprobs_data, 'content') and logprobs_data.content:
                # Parse LLM selected action
                response_text = response.choices[0].message.content.strip()
                import re
                numbers = re.findall(r'\b[0-4]\b', response_text)
                if numbers:
                    llm_chosen_action = int(numbers[0])
                else:
                    all_numbers = re.findall(r'\d', response_text)
                    valid_numbers = [int(n) for n in all_numbers if int(n) in available_actions]
                    llm_chosen_action = valid_numbers[0] if valid_numbers else available_actions[0]
                
                # Unified logprobs handling (GPT and OpenAI format same)
                if return_logprobs:
                    probabilities, logprobs_list = self._parse_logprobs(logprobs_data, available_actions, return_logprobs=True)
                else:
                    probabilities = self._parse_logprobs(logprobs_data, available_actions, return_logprobs=False)
                
                if probabilities:
                    # Convert dict format to list format for MCTS compatibility (only available actions)
                    if isinstance(probabilities, dict):
                        prob_list = [probabilities.get(action, 0.0) for action in available_actions]
                        if return_logprobs:
                            return available_actions, prob_list, llm_chosen_action, logprobs_list
                        else:
                            return available_actions, prob_list, llm_chosen_action
                    else:
                        if return_logprobs:
                            return available_actions, probabilities, llm_chosen_action, logprobs_list
                        else:
                            return available_actions, probabilities, llm_chosen_action
            
            # If logprobs extraction fails, use response content parsing method
            response_text = response.choices[0].message.content.strip()
            
            # Parse LLM selected action
            import re
            numbers = re.findall(r'\b[0-4]\b', response_text)
            if numbers:
                predicted_action = int(numbers[0])
            else:
                all_numbers = re.findall(r'\d', response_text)
                valid_numbers = [int(n) for n in all_numbers if int(n) in available_actions]
                predicted_action = valid_numbers[0] if valid_numbers else available_actions[0]
            
            # Return uniform distribution as backup
            uniform_prob = 1.0 / len(available_actions)
            probabilities = [uniform_prob] * len(available_actions)
            uniform_logprobs = [0.0] * len(available_actions)  # Logprobs of uniform distribution are 0
            
            if return_logprobs:
                return available_actions, probabilities, predicted_action, uniform_logprobs
            else:
                return available_actions, probabilities, predicted_action
            
        except Exception as e:
            print(f"[red]❌ LLM action probability generation failed: {type(e).__name__} - {str(e)[:150]}[/red]")
            print(f"[yellow]⚠️  Using fallback uniform distribution strategy[/yellow]")
            # Return uniform distribution
            uniform_prob = 1.0 / len(available_actions)
            probabilities = [uniform_prob] * len(available_actions)
            uniform_logprobs = [0.0] * len(available_actions)
            default_action = available_actions[0] if available_actions else 1  # Default to IDLE if no actions
            print(f"[yellow]📌 Default action: {default_action} ({self.action_mapping.get(default_action, 'Unknown')})[/yellow]")
            if return_logprobs:
                return available_actions, probabilities, default_action, uniform_logprobs
            else:
                return available_actions, probabilities, default_action
    
    def _parse_logprobs(self, logprobs_data, available_actions, return_logprobs=False):
        """
        Parse logprobs in a unified format (compatible with GPT and OpenAI).
        Returns a probability distribution over available actions only. For available actions
        without logprobs, a default logprob value is assigned (configurable).
        """
        try:
            # Extract logprobs and build probability distribution
            top_logprobs = logprobs_data.content[0].top_logprobs
            action_logprobs_dict = {}
            
            # Process raw logprobs list (comment detailed output, preserve core logic)
            # print(f"[cyan]🔍 LLM returned raw logprobs list:[/cyan]")
            for idx, logprob_entry in enumerate(top_logprobs):
                token = str(logprob_entry.token).strip()
                logprob_value = logprob_entry.logprob
                # print(f"[cyan]  #{idx+1}: token='{token}', logprob={logprob_value:.4f}, prob={np.exp(logprob_value):.4f}[/cyan]")
                
                if token.isdigit():
                    action_id = int(token)
                    if action_id in available_actions:
                        action_logprobs_dict[action_id] = logprob_entry.logprob
                        # print(f"[green]    ✅ Action {action_id} valid, record logprob={logprob_value:.4f}[/green]")
                    # else:
                        # print(f"[yellow]    ❌ Action {action_id} not in available actions {available_actions}[/yellow]")
                # else:
                    # print(f"[gray]    ❌ token '{token}' is not valid action number[/gray]")
            
            # Build complete available action logprobs distribution
            complete_logprobs = {}
            for action_id in available_actions:
                if action_id in action_logprobs_dict:
                    # Use logprobs returned by LLM
                    complete_logprobs[action_id] = action_logprobs_dict[action_id]
                else:
                    # Assign default value from config to available actions without logprobs
                    default_logprob = self.config.get("default_missing_logprob", -25.0)
                    complete_logprobs[action_id] = default_logprob
            
            # Extract logprobs values for softmax
            logprobs_list = [complete_logprobs[action] for action in available_actions]
            
            # Output available action logprobs list (preserve key information)
            print(f"[cyan]📊 logprobs: {[f'{logprob:.3f}' for logprob in logprobs_list]}[/cyan]")
            
            # Softmax conversion to probability distribution
            temperature = self.config.get("softmax_temperature", 20.0)
            probabilities = self._softmax_with_temperature(logprobs_list, temperature)
            
            # Build final probability distribution (only available actions)
            final_probabilities = {}
            for i, action_id in enumerate(available_actions):
                final_probabilities[action_id] = probabilities[i]
            
            # Simplify state information
            model_name = self.config.get("OPENAI_CHAT_MODEL", "gpt-4o-mini")
            provider = "GPT" if "gpt" in model_name.lower() else "OpenAI"
            
            has_logprobs_count = len(action_logprobs_dict)
            missing_logprobs_count = len(available_actions) - has_logprobs_count
            
            # print(f"[green]✅ {provider} probability distribution generated: {has_logprobs_count} actions have logprobs, {missing_logprobs_count} actions assigned -10.0[/green]")
            # print(f"[green]✅ {provider} logprobs: {has_logprobs_count}/{len(available_actions)} actions[/green]")
            
            if return_logprobs:
                return final_probabilities, logprobs_list
            else:
                return final_probabilities
        except Exception as e:
            print(f"[yellow]⚠️ logprobs解析失败，使用均匀分布: {e}[/yellow]")
            # Return uniform distribution (only available actions)
            uniform_prob = 1.0 / len(available_actions)
            uniform_logprobs = [0.0] * len(available_actions)  # Logprobs of uniform distribution
            if return_logprobs:
                return {action_id: uniform_prob for action_id in available_actions}, uniform_logprobs
            else:
                return {action_id: uniform_prob for action_id in available_actions}
                
        except Exception as e:
            print(f"[yellow]⚠️ logprobs解析失败: {e}[/yellow]")
            return None
    
    def _softmax_with_temperature(self, logprobs, temperature=5.0):
        """Softmax with temperature parameter"""
        logprobs_array = np.array(logprobs) / temperature
        exp_logprobs = np.exp(logprobs_array - np.max(logprobs_array))
        return exp_logprobs / np.sum(exp_logprobs)


def create_llm_prompt_engine(config: Dict[str, Any]) -> LLMPromptEngine:
    """Create an LLM prompt engine instance."""
    return LLMPromptEngine(config)
