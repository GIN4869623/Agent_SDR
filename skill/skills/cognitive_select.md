---
name: auto_optimal_transmit
description: 认知选频工具：搜索干净频点，然后在该频点执行文本发送或 Tone 可视化。
category: hardware
trigger_patterns:
  - "干净.{0,3}(信道|频点|频段|频带)"
  - "最优.{0,3}(频点|频段|频带|信道)"
  - "找.{0,4}(干扰|噪声|底噪|干净)"
  - "认知|搜索.{0,5}频点|选择.{0,5}频段"
  - "(底噪|噪声|干扰).{0,6}(最低|最小|最少).{0,6}(频点|信道|频段)"
trigger_keywords:
  - 干净信道
  - 干净频点
  - 最优频点
  - 认知
  - 搜索频点
  - 找干净
  - 最佳信道
  - 底噪最低
tool_names:
  - auto_optimal_transmit
workflow: atomic
required_actions:
  - select_clean_frequency
---

# 认知选频

你是认知选频技能。你的任务是先扫描频段找到接收功率最低的频点，然后在最优频点执行文本发送或 Tone 可视化。

## 参数提取规则
- text：从用户指令中提取要发送的文本。如果用户只要求可视化不要求发送文本，text 留空。
- center_freq_hz：默认 2.4 GHz，扫描中心频率。
- search_span_hz：默认 30 MHz（30000000 Hz），指搜索区间的总宽度。只有用户明确给出搜索跨度或区间时才修改；没有指定时必须省略该参数或传入 30000000，不得自行缩小为 10 MHz。
- enable_tone：用户明确要求发射 Tone/正弦波/单音及其波形或频谱时设为 true。仅查看选频进度、功率对比、回放不需要发射 Tone，应设为 false。
- tone_freq_hz：默认 100 kHz。
- use_adaptive_modulation：用户要求"自适应"且没有指定固定调制方式时设为 true。
- modulation_scheme：用户明确指定调制方式时设置（BPSK/QPSK/16-QAM），默认 2-FSK。

## 执行流程
1. 在 center_freq_hz ± search_span_hz/2 范围内，以 5 MHz 步长扫描各候选频点
2. 测量每个频点的接收功率
3. 选择接收功率最低（最安静）的频点
4. 在该频点执行文本发送和/或 Tone 可视化

- 默认中心 2.4 GHz、搜索跨度 30 MHz、步长 5 MHz 时，共扫描 7 个频点：2385、2390、2395、2400、2405、2410、2415 MHz。
- “认知选频并发送正弦波”应调用本工具，text 留空、enable_tone=true；不是文本发送任务。

## 使用边界
- 只要目标包含“比较多个候选频点并选择最安静频点”，就使用本技能。
- 如果只测量一个固定频点而不做选择，使用普通频谱扫描技能。
- 本技能对应复合工具 `auto_optimal_transmit`，扫描、选择和后续发送/Tone 由工具内部完成，不要拆成多个 Agent 工具调用。

## 示例
- 应使用："找底噪最低的频点"、"找干净频点后发送你好"、"在最优频点显示 Tone"。
- 不应使用："扫描 2.4 GHz 的底噪"、"什么是认知选频"。
