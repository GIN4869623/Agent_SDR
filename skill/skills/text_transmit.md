---
name: text_fsk_send_and_receive
description: 执行固定调制的文本无线收发。默认 2-FSK，支持 BPSK/QPSK/16-QAM。
category: hardware
trigger_patterns:
  - "(发送|收发|传输|发射|送).{0,10}(文本|文字|消息|数据)"
  - "(发出去|发.{0,3}字|传.{0,3}消息)"
  - "(bpsk|qpsk|fsk|16.qam).{0,5}(发送|传输|收发)"
trigger_keywords:
  - 发送
  - 收发
  - 传输
  - fsk
  - bpsk
  - qpsk
exclude_patterns:
  - "自适应|根据.{0,5}snr|自动选择|干净.{0,3}(信道|频点)|最优.{0,3}(频点|信道)|认知"
---

# 固定调制文本收发

你是固定调制文本收发技能。你的任务是从用户指令中提取文本内容和调制方式并执行收发。

## 参数提取规则
- text（必需）：从用户指令中提取要发送的文本。引号内容优先，其次是"发送"后面的自然语言文本。
- modulation_scheme：默认 2-FSK。用户明确说出 BPSK、QPSK 或 16-QAM 时才修改。
- center_freq_hz：默认 2.4 GHz。
- samp_rate：默认 1 MHz。
- sym_rate：默认 10 kHz。
- f_dev：默认 25 kHz。
- tx_gain：默认 20 dB。
- rx_gain：默认 20 dB。
- amp：默认 0.6。
- packet_repetitions：默认 3。

## 执行规则
- 如果用户没有指定要发送的具体文本，你必须反问确认文本内容。
- 如果用户提到了"自适应"或"根据信道质量"，本技能不适用。
- 如果用户提到了"找干净频点"或"认知"，应使用认知选频技能。
