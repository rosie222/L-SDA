import os
import textwrap
import time
import re
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.callbacks.base import BaseCallbackHandler

from rich import print


class StreamingCallbackHandler(BaseCallbackHandler):
    """Callback handler for streaming LLM output"""
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Print token in real-time as it arrives"""
        print(token, end="", flush=True)


# Action name mapping
ACTION_NAMES = {
    0: "Turn-left",
    1: "IDLE", 
    2: "Turn-right",
    3: "Acceleration",
    4: "Deceleration"
}


def get_action_name(action_id):
    """Get action name in English"""
    return ACTION_NAMES.get(action_id, f"UNKNOWN({action_id})")


class ReflectionAgent:
    def __init__(
        self, temperature: float = 0.0, verbose: bool = False, debug_reflection_prompts: bool = False, config: dict = None
    ) -> None:
        self.debug_reflection_prompts = debug_reflection_prompts
        # Only support OpenAI API (Azure removed)
        # Get model name from config.yaml first, then from environment variable, then use default
        if config is None:
            config = {}
        model_name = config.get("REFLECTION_OPENAI_MODEL") 
        
        # Get API key and base URL
        api_key = os.getenv("OPENAI_API_KEY") or config.get("gpt_config", {}).get("OPENAI_KEY")
        api_base = config.get("gpt_config", {}).get("OPENAI_API_BASE")
        
        # Create callback handler for streaming
        self.streaming_handler = StreamingCallbackHandler()
        
        self.llm = ChatOpenAI(
            temperature=temperature,
            model_name=model_name,
            max_tokens=8000,
            request_timeout=300,  # Increase timeout
            streaming=False,  # Disable streaming for stability
            openai_api_key=api_key,
            openai_api_base=api_base,
        )

    def reflection_on_failure(self, scenario_description: str, available_actions: str,
                               executed_action: int, frame_index: int,
                               collision_frame: int, debug_reflection_prompts: bool = None) -> dict:
        """
        Reflection on a collision case.
        Returns a dict with keys: 'analysis', 'correct_action', 'one_sentence_rule', 'memory_content'
        """
        if debug_reflection_prompts is None:
            debug_reflection_prompts = self.debug_reflection_prompts
            
        delimiter = "####"
        action_name = get_action_name(executed_action)
        
        system_prompt = """# Role: Autonomous Driving Collision Analyst

## Profile
- language: English
- description: You are an expert driving instructor analyzing collision incidents in highway-env v0. Your responsibility is to identify the root cause and extract actionable driving rules to prevent similar accidents.

## Skills
- Analyze collision incidents based on vehicle positions, speeds, accelaration and relative distances.
- Identify decision errors that led to collisions.
- Extract memorable, actionable driving rules from failure cases.
- Be familiar with the discrete action space:  0 = Turn-left, 1 = IDLE, 2 = Turn-right, 3 = Acceleration, 4 = Deceleration.

## Goals
- Identify the root cause of the collision.
- Determine the CORRECT ACTION that SHOULD HAVE BEEN TAKEN from the available actions ONLY.
- Extract a concise, memorable driving rule to prevent this mistake.

## Rules
1. Analysis must be based on the provided observation data.
2. The correct action must be one of the available actions (0-4).
3. The driving rule should be concise and actionable.
4. Do not fabricate information not present in the scenario.
"""

        human_message = f"""\n# Collision Incident

## Current Observation (at decision frame {frame_index})
{scenario_description}
## Available Actions
{available_actions}
## Decision
Action {executed_action} ({action_name}) was executed.

## Outcome
COLLISION occurred at frame {collision_frame} .

## Analysis Required

Please analyze and provide:
1. Root cause: Why did this action lead to collision?
2. Correct action: What should have been done instead?
3. Driving rule: A concise rule to prevent this mistake.

## OutputFormat
{delimiter} Analysis:
<Explain the decision error - focus on distances, speeds, and timing>

{delimiter} Correct Action:
<Action_id (0-4): Action_name - Brief justification>

{delimiter} One-Sentence Rule:
<A clear, actionable driving rule>
"""

        if debug_reflection_prompts:
            print(f"\n{'='*80}")
            print(f"[magenta]Reflection Module - LLM Input:[/magenta]")
            print(f"{'='*40}")
            print(f"{system_prompt}")
            print(f"{human_message}")
            print(f"{'='*80}\n")
        
        print()
        print("[cyan]Running self-reflection on collision case...[/cyan]")
        print()
        start_time = time.time()
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_message),
        ]
        response = self.llm(messages)
        
        print(f"\n{'='*80}")
        print(f"[yellow]Reflection Module - LLM Output:[/yellow]")
        print(f"{'='*40}")
        print(f"Response length: {len(response.content)} chars")
        print(f"{response.content}")
        if hasattr(response, 'response_metadata'):
            print(f"[dim]Metadata: {response.response_metadata}[/dim]")
        print(f"{'='*80}\n")
        
        # Parse the response
        result = self._parse_reflection_response(response.content, delimiter)
        
        # Build memory content: scenario + decision + one-sentence rule
        memory_content = self._build_failure_memory(
            scenario_description, executed_action, action_name, result
        )
        result['memory_content'] = memory_content
        
        print(f"[green]Reflection completed. Time taken: {time.time() - start_time:.2f}s[/green]")
        
        return result

    def reflection_on_success(self, scenario_description: str, available_actions: str,
                               executed_action: int, frame_index: int,
                               min_ttc: float, debug_reflection_prompts: bool = None) -> dict:
        """
        Reflection on a successful TTC-triggered intervention.
        Returns a dict with keys: 'analysis', 'one_sentence_rule', 'memory_content'
        """
        if debug_reflection_prompts is None:
            debug_reflection_prompts = self.debug_reflection_prompts
            
        delimiter = "####"
        action_name = get_action_name(executed_action)
        
        system_prompt = """# Role: Autonomous Driving Experience Extractor

## Profile
- language: English
- description: You are an expert driving instructor documenting successful driving decisions in highway-env v0. Your responsibility is to extract valuable experience that can guide future decisions.

## Skills
- Analyze successful driving decisions based on vehicle positions, speeds, acceleration, and safety margins.
- Identify what made a decision effective and safe.
- Extract memorable, reusable driving guidelines from success cases.
- Be familiar with the discrete action space: 0 = Turn-left, 1 = IDLE, 2 = Turn-right, 3 = Acceleration, 4 = Deceleration.

## Goals
- Understand why the selected action was appropriate.
- Extract a concise, reusable driving guideline.

## Rules
1. Analysis must be based on the provided observation data.
2. Focus on what made this decision effective.
3. The driving rule should be concise and generalizable.
4. Do not fabricate information not present in the scenario.
"""

        human_message = f"""{delimiter} Successful Driving Decision

## Current Observation (at frame {frame_index})
{scenario_description}

## Available Actions
{available_actions}

## Driver's Decision
Action {executed_action} ({action_name}) was executed.

## Outcome
SUCCESS - This action effectively handled the traffic situation.

{delimiter} Experience Extraction

Please analyze and provide:
1. Success analysis: Why was this the right decision?
2. Driving rule: A memorable guideline for similar situations.

## OutputFormat
{delimiter} Analysis:
<Explain why this action was appropriate - consider distances, speeds, and safety margins>

{delimiter} One-Sentence Rule:
<A clear, reusable driving guideline>
"""

        if debug_reflection_prompts:
            print(f"\n{'='*80}")
            print(f"[magenta]Reflection Module (Success) - LLM Input:[/magenta]")
            print(f"{'='*40}")
            print(f"{system_prompt}")
            print(f"{human_message}")
            print(f"{'='*80}\n")
        print()
        print("[cyan]Running self-reflection on successful case...[/cyan]")
        print()
        start_time = time.time()
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_message),
        ]
        response = self.llm(messages)
        
        print(f"\n{'='*80}")
        print(f"[yellow]Reflection Module (Success) - LLM Output:[/yellow]")
        print(f"{'='*40}")
        print(f"Response length: {len(response.content)} chars")
        print(f"{response.content}")
        if hasattr(response, 'response_metadata'):
            print(f"[dim]Metadata: {response.response_metadata}[/dim]")
        print(f"{'='*80}\n")
        
        # Parse the response
        result = self._parse_success_response(response.content, delimiter)
        
        # Build memory content: scenario + decision + one-sentence rule
        memory_content = self._build_success_memory(
            scenario_description, executed_action, action_name, result
        )
        result['memory_content'] = memory_content
        
        print(f"[green]Reflection completed. Time taken: {time.time() - start_time:.2f}s[/green]")
        
        return result

    def _parse_reflection_response(self, response_content: str, delimiter: str) -> dict:
        """Parse the reflection response for failure case"""
        result = {
            'analysis': '',
            'correct_action': '',
            'correct_action_id': None,
            'one_sentence_rule': ''
        }
        
        # Extract Analysis
        analysis_match = re.search(
            rf'{delimiter}\s*Analysis:\s*(.+?)(?={delimiter}|$)', 
            response_content, re.DOTALL | re.IGNORECASE
        )
        if analysis_match:
            result['analysis'] = analysis_match.group(1).strip()
        
        # Extract Correct Action
        correct_match = re.search(
            rf'{delimiter}\s*Correct Action:\s*(.+?)(?={delimiter}|$)', 
            response_content, re.DOTALL | re.IGNORECASE
        )
        if correct_match:
            result['correct_action'] = correct_match.group(1).strip()
            # Try to extract action ID
            action_id_match = re.search(r'(\d)', result['correct_action'])
            if action_id_match:
                result['correct_action_id'] = int(action_id_match.group(1))
        
        # Extract One-Sentence Rule
        rule_match = re.search(
            rf'{delimiter}\s*One-Sentence Rule:\s*(.+?)(?={delimiter}|$)', 
            response_content, re.DOTALL | re.IGNORECASE
        )
        if rule_match:
            result['one_sentence_rule'] = rule_match.group(1).strip()
        
        return result

    def _parse_success_response(self, response_content: str, delimiter: str) -> dict:
        """Parse the reflection response for success case"""
        result = {
            'analysis': '',
            'one_sentence_rule': ''
        }
        
        # Extract Analysis
        analysis_match = re.search(
            rf'{delimiter}\s*Analysis:\s*(.+?)(?={delimiter}|$)', 
            response_content, re.DOTALL | re.IGNORECASE
        )
        if analysis_match:
            result['analysis'] = analysis_match.group(1).strip()
        
        # Extract One-Sentence Rule
        rule_match = re.search(
            rf'{delimiter}\s*One-Sentence Rule:\s*(.+?)(?={delimiter}|$)', 
            response_content, re.DOTALL | re.IGNORECASE
        )
        if rule_match:
            result['one_sentence_rule'] = rule_match.group(1).strip()
        
        return result

    def _build_failure_memory(self, scenario_description: str, executed_action: int,
                               action_name: str, reflection_result: dict) -> str:
        """Build memory content for failure case: scenario + decision + one-sentence rule"""
        correct_action = reflection_result.get('correct_action', 'Not specified.')
        correct_action_id = reflection_result.get('correct_action_id', None)
        
        memory = f"""\
#### Scenario:
{scenario_description}
#### Wrong Action: {executed_action} ({action_name}) → Led to COLLISION
#### Correct Action: {correct_action}
#### Lesson: {reflection_result.get('one_sentence_rule', 'No rule extracted.')}"""
        return memory

    def _build_success_memory(self, scenario_description: str, executed_action: int,
                               action_name: str, reflection_result: dict) -> str:
        """Build memory content for success case: scenario + decision + one-sentence rule"""
        memory = f"""\
#### Scenario:
{scenario_description}
#### Successful Action: {executed_action} ({action_name})
#### Experience: {reflection_result.get('one_sentence_rule', 'No rule extracted.')}"""
        return memory

    # Keep the old method for backward compatibility
    def reflection(self, human_message: str, llm_response: str, debug_reflection_prompts: bool = None) -> str:
        """Legacy reflection method - kept for backward compatibility"""
        if debug_reflection_prompts is None:
            debug_reflection_prompts = self.debug_reflection_prompts
            
        delimiter = "####"
        system_message = textwrap.dedent(f"""\
        You are ChatGPT, a large language model trained by OpenAI. Now you act as a mature driving assistant, who can give accurate and correct advice for human driver in complex urban driving scenarios.
        You will be given a detailed description of the driving scenario of current frame along with the available actions allowed to take. 

        Your response should use the following format:
        <reasoning>
        <reasoning>
        <repeat until you have a decision>
        Response to user:{delimiter} <only output one `Action_id` as a int number of you decision, without any action name or explanation. The output decision must be unique and not ambiguous, for example if you decide to decelearate, then output `4`> 

        Make sure to include {delimiter} to separate every step.
        """)
        human_message_full = textwrap.dedent(f"""\
            ``` Human Message ```
            {human_message}
            ``` ChatGPT Response ```
            {llm_response}

            Now, you know this action ChatGPT output cause a collison after taking this action, which means there are some mistake in ChatGPT resoning and cause the wrong action.    
            Please carefully check every reasoning in ChatGPT response and find out the mistake in the reasoning process of ChatGPT, and also output your corrected version of ChatGPT response.
            Your answer should use the following format:
            {delimiter} Analysis of the mistake:
            <Your analysis of the mistake in ChatGPT reasoning process>
            {delimiter} What should ChatGPT do to avoid such errors in the future:
            <Your answer>
            {delimiter} Corrected version of ChatGPT response:
            <Your corrected version of ChatGPT response>
        """)

        if debug_reflection_prompts:
            print(f"\n{'='*80}")
            print(f"[magenta]Reflection Module - LLM Input:[/magenta]")
            print(f"{'='*80}")
            print(f"[magenta]System Message:[/magenta]")
            print(f"{system_message}")
            print(f"\n[magenta]Human Message:[/magenta]")
            print(f"{human_message_full}")
            print(f"{'='*80}\n")

        print("[cyan]Self-reflection is running, may take time...[/cyan]")
        start_time = time.time()
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=human_message_full),
        ]
        response = self.llm(messages)
        
        print(f"\n{'='*80}")
        print(f"[yellow]Reflection Module - LLM Output:[/yellow]")
        print(f"{'='*80}")
        print(f"{response.content}")
        print(f"{'='*80}\n")
        
        target_phrase = f"{delimiter} What should ChatGPT do to avoid such errors in the future:"
        substring = response.content[response.content.find(
            target_phrase)+len(target_phrase):].strip()
        corrected_memory = f"{delimiter} I have made a misake before and below is my self-reflection:\n{substring}"
        print(f"[green]Reflection done. Time taken: {time.time() - start_time:.2f}s[/green]")
        print("corrected_memory:", corrected_memory)

        return corrected_memory
