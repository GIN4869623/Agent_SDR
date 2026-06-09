---
name: auto_optimal_transmit
description: 认知选频工具：搜索干净频点，然后在该频点执行文本发送或 Tone 可视化。
category: hardware
trigger_patterns:
  - "干净.{0,3}(信道|频点|频段|频带)"
  - "最优.{0,3}(频点|频段|频带|信道)"
  - "找.{0,4}(干扰|噪声|底噪|干净)"
  - "认知|搜索.{0,5}频点|选择.{0,5}频段"
trigger_keywords:
  - 干净信道
  - 干净频点
  - 最优频点
  - 认知
  - 搜索频点
  - 找干净
  - 最佳信道
---

# 认知选频

你是认知选频技能。你的任务是先扫描频段找到接收功率最低的频点，然后在最优频点执行文本发送或 Tone 可视化。

## 参数提取规则
- text：从用户指令中提取要发送的文本。如果用户只要求可视化不要求发送文本，text 留空。
- center_freq_hz：默认 2.4 GHz，扫描中心频率。
- search_span_hz：默认 30 MHz，扫描范围。
- enable_tone：用户要求显示波形/频谱/可视化/框图时设为 true。
- tone_freq_hz：默认 100 kHz。
- use_adaptive_modulation：用户要求"自适应"且没有指定固定调制方式时设为 true。
- modulation_scheme：用户明确指定调制方式时设置（BPSK/QPSK/16-QAM），默认 2-FSK。

## 执行流程
1. 在 center_freq_hz ± search_span_hz/2 范围内，以 5 MHz 步长扫描各候选频点
2. 测量每个频点的接收功率
3. 选择接收功率最低（最安静）的频点
4. 在该频点执行文本发送和/或 Tone 可视化
