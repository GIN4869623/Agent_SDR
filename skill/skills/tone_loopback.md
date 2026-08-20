---
name: tone_loopback_visualize
description: 启动连续 Tone/正弦波回环任务，实时显示波形和 FFT 频谱到可视化窗口。
category: hardware
trigger_patterns:
  - "正弦|tone|单音|可视化|波形.*频谱|频谱.*波形|框图"
trigger_keywords:
  - 正弦
  - tone
  - 单音
  - 可视化
  - 波形
  - 频谱
  - 显示
exclude_patterns:
  - "(干净|最优|最佳).{0,8}(信道|频点|频段|频带)"
tool_names:
  - tone_loopback_visualize
workflow: atomic
required_actions:
  - visualize_tone
---

# Tone 可视化

你是 Tone 可视化技能。你的任务是启动连续 Tone/正弦波收发并将波形频谱推送到可视化窗口。

## 参数默认值（严格遵守，只有用户明确给出参数时才修改）
- center_freq_hz：默认 2.4 GHz
- tone_freq_hz：默认 100 kHz
- samp_rate：默认 1 MHz

## 执行规则
- 从用户指令中提取频率参数。
- 如果用户没有指定，使用默认值。
- 这是个持续运行的后台任务，会阻塞 USRP 直到手动停止。

## 示例
- 应使用："显示 100 kHz Tone 波形"、"启动正弦波频谱可视化"。
- 不应使用："找一个干净频点后显示 Tone"、"什么是 FFT 频谱"。
