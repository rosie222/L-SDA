#!/usr/bin/env python3
"""
Generic experiment runner - supports sequential runs with multiple configurations.
Usage:
python run_experiments.py --configs config1.yaml config2.yaml --params "enable_ttc_safety=[true,false]" "mcts_budget=[50,100,200]"
"""

import subprocess
import sys
import os
import argparse
import itertools
from datetime import datetime
import yaml

def load_config(config_path):
    """Load a YAML config file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def generate_config_combinations(params_str_list):
    """Generate configuration combinations from parameter strings."""
    param_dict = {}
    for param_str in params_str_list:
        if '=' not in param_str:
            continue
        key, values_str = param_str.split('=', 1)
        # Parse a list of values, e.g. [true,false] or [50,100,200]
        if values_str.startswith('[') and values_str.endswith(']'):
            values = [v.strip() for v in values_str[1:-1].split(',')]
            # Convert types
            typed_values = []
            for v in values:
                if v.lower() in ('true', 'false'):
                    typed_values.append(v.lower() == 'true')
                elif v.isdigit():
                    typed_values.append(int(v))
                elif v.replace('.', '').isdigit():
                    typed_values.append(float(v))
                else:
                    typed_values.append(v.strip('"\''))
            param_dict[key] = typed_values
        else:
            # Single value
            param_dict[key] = [values_str]

    # Generate all combinations
    keys = list(param_dict.keys())
    values = list(param_dict.values())
    combinations = list(itertools.product(*values))

    configs = []
    for combo in combinations:
        config = {}
        for i, key in enumerate(keys):
            config[key] = combo[i]
        configs.append(config)

    return configs

def run_experiment_with_config(base_config_path, param_overrides, exp_name, timestamp, script_name='run_lsda.py'):
    """Run an experiment with a specific configuration.
    
    :param base_config_path: Base config file path
    :param param_overrides: Parameter override dictionary
    :param exp_name: Experiment name
    :param timestamp: Timestamp
    :param script_name: Python script name to run (default: 'run_lsda.py')
    """
    print(f"\n{'='*60}")
    print(f"🚀 运行实验: {exp_name} (使用脚本: {script_name})")
    print(f"{'='*60}")

    # Load base configuration
    config = load_config(base_config_path)

    # Apply parameter overrides
    for key, value in param_overrides.items():
        config[key] = value
        print(f"  {key} = {value}")

    # Create a temporary config file
    temp_config_path = f"temp_config_{exp_name.replace('/', '_')}.yaml"
    with open(temp_config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False)

    # Set environment variables
    env = os.environ.copy()
    env['LSDA_CONFIG'] = temp_config_path
    env['LSDA_TIMESTAMP'] = timestamp
    
    # Create different result folders for different scripts to avoid overwriting results.
    # Format: script_name_without_extension/exp_name
    script_prefix = os.path.splitext(script_name)[0]  # Strip the .py extension
    exp_name_with_script = f"{script_prefix}/{exp_name}"
    
    env['LSDA_RESULT_SUFFIX'] = exp_name_with_script
    # Set the results directory to results_exp (to distinguish results from run_experiments.py vs run_lsda.py)
    env['LSDA_RESULT_FOLDER'] = 'results_exp'

    try:
        # Run the experiment using the specified script
        result = subprocess.run(
            [sys.executable, script_name],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            check=True,
            capture_output=False,
            env=env
        )
        print(f"✅ 实验完成: {exp_name}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ 实验失败: {exp_name} (错误码: {e.returncode})")
        return False
    except KeyboardInterrupt:
        print(f"⚠️ 实验中断: {exp_name}")
        return False
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)

def main():
    parser = argparse.ArgumentParser(description='通用实验运行器')
    parser.add_argument('--configs', nargs='+', help='配置文件列表')
    parser.add_argument('--params', nargs='+', help='参数覆盖，如 "enable_ttc_safety=[true,false]"')
    parser.add_argument('--base-config', default='config.yaml', help='基础配置文件')
    parser.add_argument('--script', default='run_lsda.py', help='要运行的Python脚本，默认为 run_lsda.py')
    parser.add_argument('--scripts', nargs='+', help='多个脚本列表，会为每个脚本运行所有实验')
    parser.add_argument('--dry-run', action='store_true', help='仅显示实验计划，不实际运行')

    args = parser.parse_args()

    if not args.configs and not args.params:
        print("使用示例:")
        print("1. 使用不同配置文件:")
        print("   python run_experiments.py --configs config1.yaml config2.yaml")
        print("2. 使用参数组合:")
        print("   python run_experiments.py --params 'enable_ttc_safety=[true,false]' 'mcts_budget=[50,100]'")
        print("3. 混合使用:")
        print("   python run_experiments.py --base-config config.yaml --params 'enable_ttc_safety=[true,false]'")
        print("4. 使用特定脚本:")
        print("   python run_experiments.py --script run_lsda_seed.py --params 'enable_ttc_safety=[true,false]'")
        print("5. 运行多个脚本:")
        print("   python run_experiments.py --scripts run_lsda.py run_lsda_seed.py --params 'enable_ttc_safety=[true,false]'")
        print("6. 预览实验计划:")
        print("   python run_experiments.py --params 'enable_ttc_safety=[true,false]' --dry-run")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not args.dry_run:
        print(f"📅 实验开始时间: {timestamp}")

    # Determine which scripts to use
    if args.scripts:
        script_list = args.scripts
        print(f"📋 使用脚本列表: {', '.join(script_list)}")
    else:
        script_list = [args.script]
        print(f"📋 使用脚本: {args.script}")

    experiments = []
    results = []

    # Process the config file list
    if args.configs:
        for i, config_path in enumerate(args.configs):
            exp_name = f"config_{i+1}_{os.path.basename(config_path)}"
            experiments.append((config_path, {}, exp_name))

    # Process parameter combinations
    if args.params:
        param_configs = generate_config_combinations(args.params)
        for i, param_config in enumerate(param_configs):
            # Generate an experiment name
            param_str = '_'.join([f"{k}={v}" for k, v in param_config.items()])
            exp_name = f"params_{i+1}_{param_str}"
            experiments.append((args.base_config, param_config, exp_name))

    # Build the experiment list for each script
    all_experiments_with_scripts = []
    for script in script_list:
        for config_path, param_overrides, exp_name in experiments:
            all_experiments_with_scripts.append((config_path, param_overrides, exp_name, script))

    print(f"\n🎯 计划运行 {len(all_experiments_with_scripts)} 个实验:")
    for i, (_, _, exp_name, script) in enumerate(all_experiments_with_scripts, 1):
        # Compute the actual folder name
        script_prefix = os.path.splitext(script)[0]
        exp_name_with_script = f"{script_prefix}/{exp_name}"
        result_folder = f"results_exp/exp_{timestamp}_{exp_name_with_script}"
        
        print(f"  {i}. {exp_name} (脚本: {script})")
        print(f"     📁 结果文件夹: {result_folder}")

    if args.dry_run:
        print(f"\n🔍 这是预览模式，不会实际运行实验")
        print(f"💡 移除 --dry-run 参数来实际运行实验")
        return

    # Run all experiments
    for config_path, param_overrides, exp_name, script in all_experiments_with_scripts:
        success = run_experiment_with_config(config_path, param_overrides, exp_name, timestamp, script)
        results.append((exp_name, script, success))

    # Summarize results
    print(f"\n{'🌟'*6}")
    print("实验运行总结")
    print(f"{'🌟'*6}")

    successful = sum(1 for _, _, success in results if success)
    total = len(results)

    for exp_name, script, success in results:
        status = "✅ 成功" if success else "❌ 失败"
        print(f"  {status}: {exp_name} ({script})")

    print(f"\n📊 总计: {successful}/{total} 个实验成功")

    if successful == total:
        print("🎉 所有实验成功完成！")
    elif successful > 0:
        print("⚠️ 部分实验成功")
    else:
        print("❌ 所有实验失败")

if __name__ == '__main__':
    main()