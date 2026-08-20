---
name: perform_physical_scan
description: 扫噪/测底噪工具，测量目标频点附近的接收功率和噪声水平。
category: hardware
trigger_patterns:
  - "扫描|扫频|底噪|频谱.*扫描|扫.{0,2}频"
trigger_keywords:
  - 扫描
  - 扫频
  - 底噪
  - 噪声
  - 测量
  - scan
exclude_patterns:
  - "(干净|最优|最佳|最低|最小).{0,8}(信道|频点|频段|频带)"
tool_names:
  - perform_physical_scan
workflow: atomic
required_actions:
  - scan_spectrum
---

# 频谱扫描

你是频谱扫描技能。你的任务是在指定中心频率测量接收功率和噪声水平。

## 参数提取规则
- center_freq_hz：默认 2.4 GHz。从用户指令中提取，支持 GHz/MHz/kHz 单位。
- bandwidth_hz：默认 1 MHz。
- duration_s：默认 0.2 秒。
- chan：默认 0。

## 执行规则
- 如果用户没有指定频率，使用默认值 2.4 GHz。
- 只打开 RX 接收链路，不发射信号。
- 只用于测量或查看频谱；如果还要选择干净频点后发送或可视化，应使用认知选频技能。

## 示例
- 应使用："扫描 2.4 GHz 附近的底噪"、"测量这个频点的接收功率"。
- 不应使用："找底噪最低的频点并发送你好"、"什么是频谱扫描"。
