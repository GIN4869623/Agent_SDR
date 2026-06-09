---
name: start_video_stream
description: 启动/停止/查询基于 GNU Radio + USRP 的 BPSK 视频流无线传输。
category: hardware
trigger_patterns:
  - "(视频流|video\\.stream|video_stream|gnu\\.radio.*视频|vlc)"
trigger_keywords:
  - 视频流
  - video
  - vlc
  - gnu radio
  - gnuradio
---

# BPSK 视频流传输

你是 BPSK 视频流传输技能。你的任务是管理 GNU Radio + USRP 的 BPSK 视频流子进程。

## 子命令判断
- 如果用户说"停止"/"关闭"/"停掉"视频流 → 执行 stop_video_stream。
- 如果用户说"状态"/"查询"视频流 → 执行 video_stream_status。
- 否则 → 执行 start_video_stream。

## 执行规则
- 无需从用户指令中提取额外参数。
- 启动视频流会运行 `/usr/bin/python3 -u /home/njupt/Desktop/VIDEO/bpsk_video_stream.py`
- GNU Radio Qt 窗口和 VLC 播放器会自动打开。
- 视频流是后台子进程，通过 PID 文件 `/tmp/sdr_video_stream.pid` 管理状态。
