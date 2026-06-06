#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试CIFAR-100权重在不同时间步T下的性能
对比原始IF模型和SignedIF（SNM）模型的性能
"""

import subprocess
import sys
import os
from datetime import datetime
import re

def run_test(T, test_type='if', dataset='cifar100', arch='vgg16', identifier='cifar100-vgg16-l8-example', device='0'):
    """
    运行单个T值的测试
    
    Args:
        T: 时间步数
        test_type: 'if' 使用main_test.py (原始IF), 'signed' 使用main_test_signed.py (SignedIF/SNM)
        dataset: 数据集名称
        arch: 模型架构
        identifier: 权重文件标识符
        device: GPU设备号
    """
    if test_type == 'if':
        cmd = [
            'python', 'main_test.py',
            '-data', dataset,
            '-arch', arch,
            '-id', identifier,
            '-T', str(T),
            '-dev', device
        ]
        model_name = 'IF (原始)'
    elif test_type == 'signed':
        cmd = [
            'python', 'main_test_signed.py',
            '-data', dataset,
            '-arch', 'vgg16_signed',
            '-id', identifier,
            '-T', str(T),
            '-dev', device
        ]
        model_name = 'SignedIF (SNM)'
    else:
        raise ValueError(f"Unknown test_type: {test_type}")
    
    print(f"\n{'='*60}")
    print(f"测试 {model_name} - T={T}")
    print(f"{'='*60}")
    print(f"命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            cwd='/root/autodl-tmp/QCFS',
            capture_output=True,
            text=True,
            check=True,
            timeout=600  # 10分钟超时
        )
        
        # 从输出中提取精度
        output = result.stdout + result.stderr
        print(output)
        
        # 提取精度值
        accuracy = extract_accuracy(output)
        
        return accuracy, output
    except subprocess.TimeoutExpired:
        print(f"测试超时 (T={T})")
        return None, "测试超时"
    except subprocess.CalledProcessError as e:
        print(f"错误: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return None, e.stdout + e.stderr

def extract_accuracy(output):
    """从输出中提取精度值"""
    accuracy = None
    
    # main_test.py 和 main_test_signed.py 的输出格式通常是单独一行数字
    # 查找所有浮点数，取最后一个在合理范围内的值（0-100）
    numbers = re.findall(r'\d+\.\d+', output)
    if numbers:
        # 从后往前查找，取第一个在合理范围内的值
        for num_str in reversed(numbers):
            num = float(num_str)
            if 0 <= num <= 100:
                accuracy = num
                break
    
    # 如果还是没找到，尝试查找带%的值
    if accuracy is None:
        percent_pattern = r'(\d+\.?\d*)%'
        matches = re.findall(percent_pattern, output)
        if matches:
            for match in reversed(matches):
                num = float(match)
                if 0 <= num <= 100:
                    accuracy = num
                    break
    
    # 如果还是没找到，尝试查找 "Test Accuracy: XX.XX%" 这样的格式
    if accuracy is None:
        acc_pattern = r'Test Accuracy:\s*(\d+\.?\d*)%'
        match = re.search(acc_pattern, output, re.IGNORECASE)
        if match:
            accuracy = float(match.group(1))
    
    return accuracy

def main():
    """主函数：测试多个T值并生成对比报告"""
    T_values = [1, 2, 4, 8, 16, 32]
    dataset = 'cifar100'
    arch = 'vgg16'
    identifier = 'cifar100-vgg16-l8-example'
    device = '0'
    
    results_if = {}
    results_signed = {}
    outputs_if = {}
    outputs_signed = {}
    
    print(f"\n{'='*80}")
    print(f"开始测试 CIFAR-100 权重在不同时间步下的性能对比")
    print(f"{'='*80}")
    print(f"模型: {arch} / vgg16_signed")
    print(f"权重文件: {identifier}")
    print(f"测试时间步: {T_values}")
    print(f"对比: IF (原始) vs SignedIF (SNM)")
    print(f"\n时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    # 运行所有IF测试
    print("\n" + "="*80)
    print("第一阶段: 测试原始IF模型")
    print("="*80)
    for T in T_values:
        accuracy, output = run_test(T, 'if', dataset, arch, identifier, device)
        results_if[T] = accuracy
        outputs_if[T] = output
        
        if accuracy is not None:
            print(f"\n✓ IF T={T}: {accuracy:.2f}%")
        else:
            print(f"\n✗ IF T={T}: 无法提取精度值")
    
    # 运行所有SignedIF测试
    print("\n" + "="*80)
    print("第二阶段: 测试SignedIF (SNM)模型")
    print("="*80)
    for T in T_values:
        accuracy, output = run_test(T, 'signed', dataset, arch, identifier, device)
        results_signed[T] = accuracy
        outputs_signed[T] = output
        
        if accuracy is not None:
            print(f"\n✓ SignedIF T={T}: {accuracy:.2f}%")
        else:
            print(f"\n✗ SignedIF T={T}: 无法提取精度值")
    
    # 生成Markdown报告
    md_content = generate_comparison_report(
        results_if, results_signed,
        outputs_if, outputs_signed,
        dataset, arch, identifier
    )
    
    # 保存到文件
    output_file = f'/root/autodl-tmp/QCFS/CIFAR100_COMPARISON_RESULTS.md'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    print(f"\n{'='*80}")
    print(f"测试完成！结果已保存到: {output_file}")
    print(f"{'='*80}\n")
    
    # 打印对比摘要
    print("\n测试结果对比摘要:")
    print("-" * 80)
    print(f"{'T':<6} {'IF (原始)':<15} {'SignedIF (SNM)':<18} {'提升':<12} {'提升率':<10}")
    print("-" * 80)
    for T in T_values:
        acc_if = results_if[T]
        acc_signed = results_signed[T]
        
        if acc_if is not None and acc_signed is not None:
            diff = acc_signed - acc_if
            rate = (diff / acc_if * 100) if acc_if > 0 else 0
            print(f"T={T:<3} {acc_if:>6.2f}%      {acc_signed:>6.2f}%        {diff:>+6.2f}%    {rate:>+6.2f}%")
        elif acc_if is not None:
            print(f"T={T:<3} {acc_if:>6.2f}%      {'失败':<18} {'-':<12} {'-':<10}")
        elif acc_signed is not None:
            print(f"T={T:<3} {'失败':<15} {acc_signed:>6.2f}%        {'-':<12} {'-':<10}")
        else:
            print(f"T={T:<3} {'失败':<15} {'失败':<18} {'-':<12} {'-':<10}")
    print("-" * 80)

def generate_comparison_report(results_if, results_signed, outputs_if, outputs_signed, 
                               dataset, arch, identifier):
    """生成对比Markdown格式的报告"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    md = f"""# CIFAR-100 IF vs SignedIF (SNM) 性能对比测试结果

## 测试信息

- **数据集**: {dataset.upper()}
- **模型架构**: {arch} / vgg16_signed
- **权重文件**: {identifier}
- **测试时间**: {timestamp}
- **测试脚本**: 
  - `main_test.py` (原始IF模型)
  - `main_test_signed.py` (SignedIF/SNM模型)

---

## 测试结果对比表

| 时间步 (T) | IF (原始) | SignedIF (SNM) | 提升 | 提升率 | 最佳模型 |
|-----------|----------|---------------|------|--------|---------|
"""
    
    for T in sorted(set(list(results_if.keys()) + list(results_signed.keys()))):
        acc_if = results_if.get(T)
        acc_signed = results_signed.get(T)
        
        if acc_if is not None and acc_signed is not None:
            diff = acc_signed - acc_if
            rate = (diff / acc_if * 100) if acc_if > 0 else 0
            best = "SignedIF 🏆" if diff > 0 else "IF" if diff < 0 else "平局"
            md += f"| T={T} | **{acc_if:.2f}%** | **{acc_signed:.2f}%** | {diff:+.2f}% | {rate:+.2f}% | {best} |\n"
        elif acc_if is not None:
            md += f"| T={T} | **{acc_if:.2f}%** | ❌ 失败 | - | - | IF |\n"
        elif acc_signed is not None:
            md += f"| T={T} | ❌ 失败 | **{acc_signed:.2f}%** | - | - | SignedIF |\n"
        else:
            md += f"| T={T} | ❌ 失败 | ❌ 失败 | - | - | - |\n"
    
    md += "\n---\n\n## 性能趋势分析\n\n"
    
    # IF模型趋势
    valid_if = {T: acc for T, acc in results_if.items() if acc is not None}
    if len(valid_if) > 1:
        md += "### IF (原始) 模型性能趋势\n\n"
        md += "| 时间步对比 | 精度提升 | 提升率 |\n"
        md += "|-----------|---------|--------|\n"
        
        prev_T = None
        prev_acc = None
        for T in sorted(valid_if.keys()):
            acc = valid_if[T]
            if prev_acc is not None:
                diff = acc - prev_acc
                rate = (diff / prev_acc * 100) if prev_acc > 0 else 0
                md += f"| T={prev_T} → T={T} | {diff:+.2f}% | {rate:+.2f}% |\n"
            prev_T = T
            prev_acc = acc
        md += "\n"
    
    # SignedIF模型趋势
    valid_signed = {T: acc for T, acc in results_signed.items() if acc is not None}
    if len(valid_signed) > 1:
        md += "### SignedIF (SNM) 模型性能趋势\n\n"
        md += "| 时间步对比 | 精度提升 | 提升率 |\n"
        md += "|-----------|---------|--------|\n"
        
        prev_T = None
        prev_acc = None
        for T in sorted(valid_signed.keys()):
            acc = valid_signed[T]
            if prev_acc is not None:
                diff = acc - prev_acc
                rate = (diff / prev_acc * 100) if prev_acc > 0 else 0
                md += f"| T={prev_T} → T={T} | {diff:+.2f}% | {rate:+.2f}% |\n"
            prev_T = T
            prev_acc = acc
        md += "\n"
    
    md += "---\n\n## 详细输出\n\n"
    
    # IF模型详细输出
    md += "### IF (原始) 模型详细输出\n\n"
    for T in sorted(results_if.keys()):
        md += f"#### T={T}\n\n"
        md += "```\n"
        md += outputs_if[T]
        md += "\n```\n\n"
    
    # SignedIF模型详细输出
    md += "### SignedIF (SNM) 模型详细输出\n\n"
    for T in sorted(results_signed.keys()):
        md += f"#### T={T}\n\n"
        md += "```\n"
        md += outputs_signed[T]
        md += "\n```\n\n"
    
    md += "---\n\n## 关键发现\n\n"
    
    # 统计信息
    if valid_if and valid_signed:
        # 找出最佳性能
        best_T_if = max(valid_if.keys(), key=lambda k: valid_if[k])
        best_acc_if = valid_if[best_T_if]
        
        best_T_signed = max(valid_signed.keys(), key=lambda k: valid_signed[k])
        best_acc_signed = valid_signed[best_T_signed]
        
        md += f"### 最佳性能\n\n"
        md += f"- **IF (原始)**: T={best_T_if} 时达到 **{best_acc_if:.2f}%**\n"
        md += f"- **SignedIF (SNM)**: T={best_T_signed} 时达到 **{best_acc_signed:.2f}%**\n"
        
        if best_acc_signed > best_acc_if:
            md += f"- **整体最佳**: SignedIF (SNM) 在 T={best_T_signed} 时达到 **{best_acc_signed:.2f}%**，比IF提升 **{best_acc_signed - best_acc_if:.2f}%**\n"
        elif best_acc_if > best_acc_signed:
            md += f"- **整体最佳**: IF (原始) 在 T={best_T_if} 时达到 **{best_acc_if:.2f}%**，比SignedIF提升 **{best_acc_if - best_acc_signed:.2f}%**\n"
        else:
            md += f"- **整体最佳**: 两种模型性能相同\n"
        
        md += "\n### 各时间步对比分析\n\n"
        
        # 统计SignedIF优于IF的时间步
        signed_wins = []
        if_wins = []
        ties = []
        
        for T in sorted(set(valid_if.keys()) & set(valid_signed.keys())):
            acc_if = valid_if[T]
            acc_signed = valid_signed[T]
            if acc_signed > acc_if:
                signed_wins.append(T)
            elif acc_if > acc_signed:
                if_wins.append(T)
            else:
                ties.append(T)
        
        if signed_wins:
            wins_str = ', '.join(map(str, signed_wins))
            md += f"- **SignedIF优势时间步** ({len(signed_wins)}个): T={wins_str}\n"
        if if_wins:
            wins_str = ', '.join(map(str, if_wins))
            md += f"- **IF优势时间步** ({len(if_wins)}个): T={wins_str}\n"
        if ties:
            ties_str = ', '.join(map(str, ties))
            md += f"- **性能相同时间步** ({len(ties)}个): T={ties_str}\n"
        
        md += "\n### 平均性能提升\n\n"
        
        # 计算平均提升
        diffs = []
        for T in sorted(set(valid_if.keys()) & set(valid_signed.keys())):
            diff = valid_signed[T] - valid_if[T]
            diffs.append(diff)
        
        if diffs:
            avg_diff = sum(diffs) / len(diffs)
            md += f"- **平均精度提升**: {avg_diff:+.2f}% (SignedIF相比IF)\n"
            md += f"- **最大提升**: {max(diffs):+.2f}% (在T={sorted(set(valid_if.keys()) & set(valid_signed.keys()))[diffs.index(max(diffs))]})\n"
            md += f"- **最小提升**: {min(diffs):+.2f}% (在T={sorted(set(valid_if.keys()) & set(valid_signed.keys()))[diffs.index(min(diffs))]})\n"
    
    md += "\n---\n\n## 结论\n\n"
    
    if valid_if and valid_signed:
        md += "### 性能总结\n\n"
        
        # 低时间步对比
        low_T = [T for T in [1, 2, 4] if T in valid_if and T in valid_signed]
        if low_T:
            md += "**低时间步 (T≤4)**:\n"
            for T in low_T:
                acc_if = valid_if[T]
                acc_signed = valid_signed[T]
                diff = acc_signed - acc_if
                md += f"- T={T}: IF={acc_if:.2f}%, SignedIF={acc_signed:.2f}% ({diff:+.2f}%)\n"
            md += "\n"
        
        # 中时间步对比
        mid_T = [T for T in [8, 16] if T in valid_if and T in valid_signed]
        if mid_T:
            md += "**中时间步 (T=8-16)**:\n"
            for T in mid_T:
                acc_if = valid_if[T]
                acc_signed = valid_signed[T]
                diff = acc_signed - acc_if
                md += f"- T={T}: IF={acc_if:.2f}%, SignedIF={acc_signed:.2f}% ({diff:+.2f}%)\n"
            md += "\n"
        
        # 高时间步对比
        high_T = [T for T in [32] if T in valid_if and T in valid_signed]
        if high_T:
            md += "**高时间步 (T≥32)**:\n"
            for T in high_T:
                acc_if = valid_if[T]
                acc_signed = valid_signed[T]
                diff = acc_signed - acc_if
                md += f"- T={T}: IF={acc_if:.2f}%, SignedIF={acc_signed:.2f}% ({diff:+.2f}%)\n"
            md += "\n"
    
    md += f"\n*报告生成时间: {timestamp}*\n"
    
    return md

if __name__ == '__main__':
    main()
