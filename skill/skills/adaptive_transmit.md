---
name: adaptive_modulation_transmit
description: 先探测链路 SNR，再自动选择 BPSK/QPSK/16-QAM 完成文本收发。
category: hardware
trigger_patterns:
  - "自适应|根据.{0,5}snr|根据.{0,5}信道质量|自动选择.*调制"
trigger_keywords:
  - 自适应
  - adaptive
  - snr
  - 信道质量
  - 链路质量
  - 自动选择
exclude_patterns:
  - "干净.{0,3}(信道|频点)|最优.{0,3}(频点|信道)|认知|找.{0,4}(干扰|噪声)"
tool_names:
  - adaptive_modulation_transmit
workflow: atomic
required_actions:
  - select_modulation
  - transmit_text
---

# 自适应调制传输

你是自适应调制传输技能。你的任务是先发 BPSK 探测包测量 SNR，然后根据 SNR 自动选择 BPSK/QPSK/16-QAM 发送正式文本。

## 参数提取规则
- text（必需）：从用户指令中提取要发送的文本。
- center_freq_hz：默认 2.4 GHz。
- probe_text：默认 "channel_probe"，无需修改。

## SNR 决策规则（SDRController 内部执行）
- 探测包通过 BPSK 解调和 CRC 校验后，按以下门限选择：
  - SNR < 8 dB → BPSK（最鲁棒）
  - 8 dB ≤ SNR < 16 dB → QPSK（折中）
  - SNR ≥ 16 dB → 16-QAM（高频谱效率）
- 探测包未通过解调/CRC 校验时，保守选择 BPSK。

## 执行规则
- 如果用户提到了"找干净频点"，应使用认知选频技能代替。
- 若返回 stage=stream_initialization，说明 UHD 收发流初始化失败；adaptive_stage=probe 时尚未测量 SNR。报告实际错误，不推断为信道干扰或探测包同步失败。
- 工具执行完成不等于文本解码成功；依据 final_result.decoded_ok 和 decoded_texts 报告接收结果。
