from __future__ import annotations

import threading
import time
import zlib
from copy import deepcopy
from typing import Any

import numpy as np

try:
    import uhd
except ImportError:  # pragma: no cover
    uhd = None  # type: ignore[assignment]


PREAMBLE_BYTE = 0xAA
PREAMBLE_LEN_BYTES = 32
SYNC_WORD = b"\x2D\xD4"
DEFAULT_CENTER_FREQ_HZ = 2.4e9
DEFAULT_TONE_FREQ_HZ = 100e3
DEFAULT_TONE_SAMPLE_RATE_HZ = 1e6
DEFAULT_TEXT_PACKET_REPETITIONS = 3
DEFAULT_TEXT_GUARD_INTERVAL_S = 0.02
DEFAULT_TEXT_RX_LEAD_S = 0.15
DEFAULT_TEXT_RX_TAIL_S = 0.8
DEFAULT_AMC_SAMPLE_RATE_HZ = 10e6
DEFAULT_AMC_SYMBOL_SPS = 10
DEFAULT_AMC_PACKET_REPETITIONS = 3
DEFAULT_AMC_GUARD_INTERVAL_S = 0.002


class SDRController:
    def __init__(self, ip: str) -> None:
        self.ip = ip
        self.hw_lock = threading.Lock()
        self.bg_stop_event = threading.Event()
        self.bg_thread: threading.Thread | None = None
        self.bg_runtime_lock = threading.Lock()
        self.bg_tx_stream = None
        self.bg_rx_stream = None
        self.bg_tx_thread: threading.Thread | None = None
        self.bg_rx_thread: threading.Thread | None = None
        self.visualization_lock = threading.Lock()
        self.driver_available = uhd is not None
        self.usrp = None
        self.hw_state = self._build_default_state()
        self.visualization_state = self._build_default_visualization_state()
        # 延迟连接：避免 uvicorn --reload 模式下 reloader 进程抢占 USRP
        # 导致实际处理请求的 server 进程无法连接
        if self.driver_available:
            self._connect()

    def _build_default_state(self) -> dict[str, dict[str, str]]:
        return {
            "USRP-01": {
                "center": "Offline",
                "sample": "Offline",
                "clock": "Offline",
                "gain": "Offline",
                "status": "offline",
                "modulation": "BPSK (Default)",
            },
            "USRP-02": {
                "center": "Offline",
                "sample": "Offline",
                "clock": "Offline",
                "gain": "Offline",
                "status": "offline",
                "modulation": "N/A",
            },
        }

    def _build_default_visualization_state(self) -> dict[str, Any]:
        return {
            "active": False,
            "task": None,
            "activation_id": 0,
            "frame_id": 0,
            "updated_at": None,
            "center_freq_hz": None,
            "tone_freq_hz": None,
            "sample_rate_hz": None,
            "iq_inphase": [],
            "iq_quadrature": [],
            "fft_freq_khz": [],
            "fft_magnitude_db": [],
            "rx_power_db": None,
            "peak_freq_khz": None,
            "tx_sends": 0,
            "tx_samples": 0,
            "rx_frames": 0,
            "error": None,
        }

    def _connect(self) -> None:
        if not self.driver_available:
            self.hw_state["USRP-01"]["center"] = "UHD not installed"
            self.hw_state["USRP-01"]["clock"] = "Driver missing"
            return

        last_error = ""
        for retry in range(6):
            try:
                self.usrp = uhd.usrp.MultiUSRP(f"addr={self.ip}")
                self.hw_state["USRP-01"].update(
                    {
                        "status": "online",
                        "center": "2.400 GHz (Idle)",
                        "sample": "1.0 MS/s",
                        "clock": "200 MHz",
                        "gain": "20 / 20 dB",
                        "modulation": "BPSK (Default)",
                    }
                )
                return
            except Exception as exc:  # pragma: no cover
                last_error = str(exc)
                self.usrp = None
                self.hw_state["USRP-01"]["center"] = (
                    f"Connect retry {retry + 1}/6: {last_error}"
                )
                if retry < 5:
                    time.sleep(1.5)
        self.hw_state["USRP-01"]["center"] = f"Connect failed: {last_error}"

    def disconnect_usrp(self) -> None:
        """释放 USRP 设备的 UHD 句柄，使其他进程（如 GNU Radio 视频流）可以独占访问。"""
        if self.usrp is not None:
            self.usrp = None
        self.hw_state["USRP-01"].update(
            {
                "status": "released",
                "center": "Released for external process",
                "sample": "Released",
                "clock": "Released",
                "gain": "Released",
            }
        )

    def reconnect_usrp(self) -> None:
        """重新打开 USRP 设备连接（外部进程释放后调用）。"""
        for attempt in range(5):
            self._connect()
            if self.usrp is not None:
                return
            time.sleep(0.5)
        # 最后一次尝试，即使失败也不再等待

    def get_hardware_snapshot(self) -> dict[str, dict[str, str]]:
        return deepcopy(self.hw_state)

    def get_visualization_snapshot(self) -> dict[str, Any]:
        with self.visualization_lock:
            return deepcopy(self.visualization_state)

    def _ensure_available(self) -> dict[str, Any] | None:
        if not self.driver_available:
            return self._error("UHD Python 驱动未安装，当前只能使用知识检索和前端界面。")
        if self.usrp is None:
            # 延迟重连：可能在 uvicorn --reload 模式下 reloader 进程抢占了 USRP
            self._connect()
            if self.usrp is None:
                return self._error("USRP 未连接，请检查设备 IP、网线和 UHD 驱动。")
        return None

    def _set_state(
        self,
        status: str,
        center: str = "2.400 GHz (Idle)",
        sample: str = "1.0 MS/s",
        gain: str = "20 / 20 dB",
        modulation: str | None = None,
    ) -> None:
        self.hw_state["USRP-01"]["status"] = status
        self.hw_state["USRP-01"]["center"] = center
        self.hw_state["USRP-01"]["sample"] = sample
        self.hw_state["USRP-01"]["gain"] = gain
        if modulation is not None:
            self.hw_state["USRP-01"]["modulation"] = modulation

    def _release_hw_lock(self) -> None:
        try:
            self.hw_lock.release()
        except RuntimeError:
            pass

    def _stop_rx_stream(self, rx_stream: Any) -> None:
        if rx_stream is None or uhd is None:
            return
        try:
            rx_stream.issue_stream_cmd(uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont))
        except Exception:
            pass

    def _finalize_tx_stream(self, tx_stream: Any) -> None:
        if tx_stream is None or uhd is None:
            return
        try:
            metadata = uhd.types.TXMetadata()
            metadata.has_time_spec = False
            metadata.start_of_burst = False
            metadata.end_of_burst = True
            tx_stream.send(np.zeros(1, dtype=np.complex64), metadata)
        except Exception:
            pass

    def _join_thread(self, thread: threading.Thread | None, timeout: float = 2.5) -> bool:
        if thread is None:
            return True
        thread.join(timeout=timeout)
        return not thread.is_alive()

    def _capture_background_runtime(self) -> tuple[Any, Any, threading.Thread | None, threading.Thread | None]:
        with self.bg_runtime_lock:
            return (
                self.bg_tx_stream,
                self.bg_rx_stream,
                self.bg_tx_thread,
                self.bg_rx_thread,
            )

    def _clear_background_runtime(self) -> None:
        with self.bg_runtime_lock:
            self.bg_tx_stream = None
            self.bg_rx_stream = None
            self.bg_tx_thread = None
            self.bg_rx_thread = None

    @staticmethod
    def _is_transient_transport_error(exc: Exception) -> bool:
        message = str(exc).lower()
        return "xport timed out" in message or "strs packet" in message

    def _set_hardware_runtime(
        self,
        *,
        tx_stream: Any = None,
        rx_stream: Any = None,
        tx_thread: threading.Thread | None = None,
        rx_thread: threading.Thread | None = None,
    ) -> None:
        with self.bg_runtime_lock:
            self.bg_tx_stream = tx_stream
            self.bg_rx_stream = rx_stream
            self.bg_tx_thread = tx_thread
            self.bg_rx_thread = rx_thread

    def _wait_for_hw_lock_release(self, timeout: float = 3.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not self.hw_lock.locked():
                return True
            time.sleep(0.05)
        return not self.hw_lock.locked()

    def _set_visualization_active(
        self,
        *,
        task: str,
        center_freq_hz: float,
        tone_freq_hz: float | None,
        sample_rate_hz: float,
    ) -> None:
        with self.visualization_lock:
            next_activation_id = int(self.visualization_state.get("activation_id", 0)) + 1
            self.visualization_state.update(
                {
                    "active": True,
                    "task": task,
                    "activation_id": next_activation_id,
                    "frame_id": 0,
                    "updated_at": time.time(),
                    "center_freq_hz": float(center_freq_hz),
                    "tone_freq_hz": None if tone_freq_hz is None else float(tone_freq_hz),
                    "sample_rate_hz": float(sample_rate_hz),
                    "iq_inphase": [],
                    "iq_quadrature": [],
                    "fft_freq_khz": [],
                    "fft_magnitude_db": [],
                    "rx_power_db": None,
                    "peak_freq_khz": None,
                    "tx_sends": 0,
                    "tx_samples": 0,
                    "rx_frames": 0,
                    "error": None,
                }
            )

    def _set_visualization_stopped(self, *, error: str | None = None) -> None:
        with self.visualization_lock:
            self.visualization_state.update(
                {
                    "active": False,
                    "task": None,
                    "frame_id": 0,
                    "updated_at": time.time(),
                    "iq_inphase": [],
                    "iq_quadrature": [],
                    "fft_freq_khz": [],
                    "fft_magnitude_db": [],
                    "rx_power_db": None,
                    "peak_freq_khz": None,
                    "tx_sends": 0,
                    "tx_samples": 0,
                    "rx_frames": 0,
                    "error": error,
                }
            )

    def _update_visualization_frame(self, iq: np.ndarray, samp_rate: float) -> None:
        iq = np.asarray(iq, dtype=np.complex64)
        if iq.size < 16:
            return

        display_iq = iq[: min(iq.size, 1024)]
        spectrum_iq = iq[: min(iq.size, 2048)]
        window = np.hanning(spectrum_iq.size).astype(np.float32)
        spectrum = np.fft.fftshift(np.fft.fft(spectrum_iq * window))
        magnitude = 20 * np.log10(np.abs(spectrum) + 1e-10)
        freqs = np.fft.fftshift(np.fft.fftfreq(spectrum_iq.size, d=1.0 / float(samp_rate))) / 1e3
        peak_index = int(np.argmax(magnitude))
        power_db = 10 * np.log10(np.mean(np.abs(iq) ** 2) + 1e-12)

        with self.visualization_lock:
            self.visualization_state["frame_id"] = int(self.visualization_state["frame_id"]) + 1
            self.visualization_state["updated_at"] = time.time()
            self.visualization_state["iq_inphase"] = _downsample_for_plot(np.real(display_iq), 400)
            self.visualization_state["iq_quadrature"] = _downsample_for_plot(np.imag(display_iq), 400)
            self.visualization_state["fft_freq_khz"] = _downsample_for_plot(freqs, 256)
            self.visualization_state["fft_magnitude_db"] = _downsample_for_plot(magnitude, 256)
            self.visualization_state["rx_power_db"] = round(float(power_db), 2)
            self.visualization_state["peak_freq_khz"] = round(float(freqs[peak_index]), 2)
            self.visualization_state["rx_frames"] = int(self.visualization_state.get("rx_frames", 0)) + 1
            self.visualization_state["error"] = None

    def _success(self, message: str, **payload: Any) -> dict[str, Any]:
        return {"status": "success", "message": message, **payload}

    def _error(self, message: str, **payload: Any) -> dict[str, Any]:
        return {"status": "error", "error": message, **payload}

    def has_background_task(self) -> bool:
        """Return True if the USRP has an active background hardware task."""
        thread = self.bg_thread
        tx_stream, rx_stream, tx_thread, rx_thread = self._capture_background_runtime()
        visualization_active = bool(
            self.get_visualization_snapshot().get("active", False)
        )
        return (
            thread is not None
            or tx_stream is not None
            or rx_stream is not None
            or tx_thread is not None
            or rx_thread is not None
            or visualization_active
            or self.hw_lock.locked()
        )

    def stop_hardware_task(self) -> dict[str, Any]:
        thread = self.bg_thread
        tx_stream, rx_stream, tx_thread, rx_thread = self._capture_background_runtime()
        visualization_active = bool(self.get_visualization_snapshot().get("active", False))
        has_hardware_task = (
            thread is not None
            or tx_stream is not None
            or rx_stream is not None
            or tx_thread is not None
            or rx_thread is not None
            or visualization_active
            or self.hw_lock.locked()
        )
        if not has_hardware_task:
            self.bg_stop_event.clear()
            self._set_visualization_stopped()
            return self._success("当前没有后台硬件任务在运行。")

        self.bg_stop_event.set()
        self._stop_rx_stream(rx_stream)
        self._finalize_tx_stream(tx_stream)
        self._join_thread(tx_thread, timeout=1.5)
        self._join_thread(rx_thread, timeout=1.5)

        if thread is None:
            if not self._wait_for_hw_lock_release(timeout=3.0):
                return self._error("已发送停止指令，但当前硬件任务仍未完全退出，请稍后重试。")
            self._clear_background_runtime()
            self._set_state("online")
            self._set_visualization_stopped()
            self.bg_stop_event.clear()
            return self._success("当前硬件任务已停止，USRP 已释放。")

        thread.join(timeout=3.0)
        if thread.is_alive():
            return self._error("已发送停止指令，但后台任务仍未完全退出，请稍后重试。")

        self.bg_thread = None
        self._clear_background_runtime()
        self._set_state("online")
        self._set_visualization_stopped()
        self.bg_stop_event.clear()
        self._release_hw_lock()
        return self._success("后台硬件任务已停止，USRP 已释放。")

    def perform_physical_scan(
        self,
        center_freq_hz: float = 2.4e9,
        bandwidth_hz: float = 1e6,
        duration_s: float = 0.2,
        chan: int = 0,
    ) -> dict[str, Any]:
        if error := self._ensure_available():
            return error
        if not self.hw_lock.acquire(blocking=False):
            return self._error("硬件正忙，请先停止当前任务。")

        self.bg_stop_event.clear()
        rx_stream = None
        try:
            self._set_state(
                "active",
                f"{center_freq_hz / 1e9:.3f} GHz (Scan)",
                f"{bandwidth_hz / 1e6:.1f} MS/s",
                "Rx Only",
            )
            self.usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(center_freq_hz), chan)
            self.usrp.set_rx_rate(bandwidth_hz, chan)
            time.sleep(0.05)

            stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
            stream_args.channels = [chan]
            rx_stream = self.usrp.get_rx_stream(stream_args)
            self._set_hardware_runtime(rx_stream=rx_stream)
            duration_s = max(0.02, min(10.0, float(duration_s)))
            buffer = np.zeros((1, 4096), dtype=np.complex64)
            metadata = uhd.types.RXMetadata()
            captured: list[np.ndarray] = []
            sample_goal = max(1024, int(float(bandwidth_hz) * duration_s))
            deadline = time.time() + duration_s + 0.5

            if self.bg_stop_event.is_set():
                return self._success("扫描任务已停止。", stopped=True)
            cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
            cmd.stream_now = True
            rx_stream.issue_stream_cmd(cmd)
            while (
                not self.bg_stop_event.is_set()
                and sum(chunk.size for chunk in captured) < sample_goal
                and time.time() < deadline
            ):
                received = rx_stream.recv(buffer, metadata, timeout=0.25)
                if received > 0:
                    captured.append(buffer[0, :received].copy())
            if self.bg_stop_event.is_set():
                return self._success("扫描任务已停止。", stopped=True)

            captured_iq = np.concatenate(captured) if captured else np.array([], dtype=np.complex64)
            if captured_iq.size == 0:
                return self._error(
                    "扫噪没有接收到有效 IQ 样本，请检查 RX 通道、天线、频率和 UHD 状态。",
                    center_freq_hz=float(center_freq_hz),
                    bandwidth_hz=float(bandwidth_hz),
                    duration_s=float(duration_s),
                    chan=int(chan),
                    scan_mode="rx_only",
                    tx_active=False,
                )

            power_db = 10 * np.log10(np.mean(np.abs(captured_iq) ** 2) + 1e-12)
            return self._success(
                "扫描完成。",
                measured_power_dB=round(float(power_db), 2),
                center_freq_hz=float(center_freq_hz),
                bandwidth_hz=float(bandwidth_hz),
                duration_s=float(duration_s),
                captured_samples=int(captured_iq.size),
                chan=int(chan),
                scan_mode="rx_only",
                tx_active=False,
            )
        finally:
            self._stop_rx_stream(rx_stream)
            rx_stream = None
            self._clear_background_runtime()
            self._set_state("online")
            self._release_hw_lock()

    def text_fsk_send_and_receive(
        self,
        text: str,
        modulation_scheme: str = "2-FSK",
        center_freq_hz: float = 2.4e9,
        samp_rate: float = 1e6,
        sym_rate: float = 10e3,
        f_dev: float = 25e3,
        tx_gain: float = 20.0,
        rx_gain: float = 20.0,
        amp: float = 0.6,
        packet_repetitions: int = DEFAULT_TEXT_PACKET_REPETITIONS,
    ) -> dict[str, Any]:
        if error := self._ensure_available():
            return error
        if not self.hw_lock.acquire(blocking=False):
            return self._error("硬件正忙，请先停止当前任务。")

        self.bg_stop_event.clear()
        tx_stream = None
        rx_stream = None
        rx_thread: threading.Thread | None = None
        stop_rx = threading.Event()
        try:
            try:
                modulation_scheme = normalize_text_modulation_scheme(modulation_scheme)
            except ValueError as exc:
                return self._error(str(exc))
            if modulation_scheme != "2-FSK":
                result = self._run_modulation_once(
                    text=text,
                    center_freq_hz=center_freq_hz,
                    mod_scheme=modulation_scheme,
                    samp_rate=samp_rate,
                    tx_gain=tx_gain,
                    rx_gain=rx_gain,
                    amp=amp,
                    packet_repetitions=packet_repetitions,
                    state_label="Fixed",
                )
                if result.get("status") == "success":
                    result["message"] = "固定调制文本链路执行完成。"
                    result["modulation_scheme"] = modulation_scheme
                    result["sym_rate"] = float(samp_rate) / max(
                        int(result.get("samples_per_symbol") or DEFAULT_AMC_SYMBOL_SPS),
                        1,
                    )
                    result["tx_gain"] = float(tx_gain)
                    result["rx_gain"] = float(rx_gain)
                return result

            self._set_state(
                "active",
                f"{center_freq_hz / 1e9:.3f} GHz (FSK)",
                f"{samp_rate / 1e6:.1f} MS/s",
                f"{tx_gain:.0f} / {rx_gain:.0f} dB",
                modulation="2-FSK",
            )

            packet_repetitions = max(1, min(int(packet_repetitions or 1), 8))
            sps = max(2, int(round(samp_rate / sym_rate)))
            sym_rate_eff = samp_rate / max(sps, 1)
            packet_iq = modulate_2fsk(
                hamming_7_4_encode(bytes_to_bits(build_packet(text))),
                samp_rate=samp_rate,
                sym_rate=sym_rate_eff,
                f_dev=f_dev,
                amp=amp,
            )
            guard_samples = max(int(round(DEFAULT_TEXT_GUARD_INTERVAL_S * samp_rate)), sps * 8)
            guard_iq = np.zeros(guard_samples, dtype=np.complex64)
            tx_parts: list[np.ndarray] = [guard_iq]
            for _ in range(packet_repetitions):
                tx_parts.append(packet_iq)
                tx_parts.append(guard_iq)
            tx_iq = np.concatenate(tx_parts)
            tx_time = len(tx_iq) / float(samp_rate)

            self.usrp.set_tx_rate(samp_rate, 0)
            self.usrp.set_tx_freq(uhd.libpyuhd.types.tune_request(center_freq_hz), 0)
            self.usrp.set_tx_gain(tx_gain, 0)
            self.usrp.set_rx_rate(samp_rate, 0)
            self.usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(center_freq_hz), 0)
            self.usrp.set_rx_gain(rx_gain, 0)

            tx_stream = self.usrp.get_tx_stream(uhd.usrp.StreamArgs("fc32", "sc16"))
            rx_stream = self.usrp.get_rx_stream(uhd.usrp.StreamArgs("fc32", "sc16"))
            self._set_hardware_runtime(tx_stream=tx_stream, rx_stream=rx_stream)

            rx_buffer = np.zeros((1, 8192), dtype=np.complex64)
            rx_metadata = uhd.types.RXMetadata()
            captured: list[np.ndarray] = []
            rx_metadata_errors: list[str] = []
            receive_started_at = time.time()
            receive_deadline = [receive_started_at + DEFAULT_TEXT_RX_LEAD_S + tx_time + DEFAULT_TEXT_RX_TAIL_S]

            def rx_worker() -> None:
                cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
                cmd.stream_now = True
                timeout_code = getattr(uhd.types.RXMetadataErrorCode, "timeout", None)
                none_code = getattr(uhd.types.RXMetadataErrorCode, "none", None)
                try:
                    rx_stream.issue_stream_cmd(cmd)
                    while (
                        not stop_rx.is_set()
                        and not self.bg_stop_event.is_set()
                        and time.time() < receive_deadline[0]
                    ):
                        received = rx_stream.recv(rx_buffer, rx_metadata, timeout=0.25)
                        error_code = getattr(rx_metadata, "error_code", None)
                        if received > 0 and error_code == none_code:
                            captured.append(rx_buffer[0, :received].copy())
                            continue
                        if error_code not in {None, timeout_code, none_code}:
                            rx_metadata_errors.append(str(error_code))
                            stop_rx.set()
                finally:
                    self._stop_rx_stream(rx_stream)

            rx_thread = threading.Thread(target=rx_worker, name="sdr-fsk-rx", daemon=True)
            self._set_hardware_runtime(tx_stream=tx_stream, rx_stream=rx_stream, rx_thread=rx_thread)
            rx_thread.start()
            time.sleep(DEFAULT_TEXT_RX_LEAD_S)

            tx_metadata = uhd.types.TXMetadata()
            tx_metadata.has_time_spec = False
            tx_metadata.start_of_burst = True
            tx_metadata.end_of_burst = False

            cursor = 0
            tx_samples_sent = 0
            tx_no_progress = 0
            tx_started_at = time.time()
            while cursor < len(tx_iq) and not self.bg_stop_event.is_set():
                chunk = tx_iq[cursor : cursor + 4096]
                tx_metadata.end_of_burst = cursor + len(chunk) >= len(tx_iq)
                sent = int(tx_stream.send(chunk, tx_metadata) or 0)
                tx_metadata.start_of_burst = False
                if sent <= 0:
                    tx_no_progress += 1
                    if tx_no_progress >= 5:
                        break
                    time.sleep(0.01)
                    continue
                tx_no_progress = 0
                accepted = min(sent, len(chunk))
                cursor += accepted
                tx_samples_sent += accepted

            tx_finished_at = time.time()
            receive_deadline[0] = max(receive_deadline[0], tx_finished_at + DEFAULT_TEXT_RX_TAIL_S)
            while (
                time.time() < receive_deadline[0]
                and not stop_rx.is_set()
                and not self.bg_stop_event.is_set()
            ):
                time.sleep(0.02)

            stop_rx.set()
            self._stop_rx_stream(rx_stream)
            self._join_thread(rx_thread, timeout=2.5)
            captured_iq = np.concatenate(captured) if captured else np.array([], dtype=np.complex64)
            if self.bg_stop_event.is_set():
                return self._success(
                    "文本链路任务已停止。",
                    stopped=True,
                    sent_text=text,
                    captured_samples=int(captured_iq.size),
                    center_freq_hz=float(center_freq_hz),
                    tx_samples_requested=int(tx_iq.size),
                    tx_samples_sent=int(tx_samples_sent),
                )
            rx_diagnostics = estimate_fsk_rx_diagnostics(captured_iq, samp_rate=float(samp_rate))
            decoded, decode_diag = (
                decode_texts_from_iq(
                    captured_iq,
                    samp_rate=float(samp_rate),
                    sym_rate=float(sym_rate_eff),
                    threshold_hz=0.0,
                )
                if captured
                else ([], {"failures": ["未捕获到任何IQ样本"]})
            )
            return self._success(
                "文本链路执行完成。",
                sent_text=text,
                modulation_scheme="2-FSK",
                decoded_texts=decoded,
                decoded_ok=bool(decoded),
                decode_diagnostics=decode_diag,
                captured_samples=int(captured_iq.size),
                center_freq_hz=float(center_freq_hz),
                packet_repetitions=int(packet_repetitions),
                tx_samples_requested=int(tx_iq.size),
                tx_samples_sent=int(tx_samples_sent),
                tx_complete=bool(tx_samples_sent >= int(tx_iq.size)),
                tx_time_s=round(float(tx_finished_at - tx_started_at), 4),
                receive_time_s=round(float(time.time() - receive_started_at), 4),
                rx_metadata_errors=rx_metadata_errors[:5],
                **rx_diagnostics,
            )
        finally:
            stop_rx.set()
            self._stop_rx_stream(rx_stream)
            self._finalize_tx_stream(tx_stream)
            self._join_thread(rx_thread, timeout=2.5)
            tx_stream = None
            rx_stream = None
            self._clear_background_runtime()
            self._set_state("online")
            self._release_hw_lock()

    def adaptive_modulation_transmit(
        self,
        text: str,
        center_freq_hz: float = 2.4e9,
        probe_text: str = "channel_probe",
    ) -> dict[str, Any]:
        if error := self._ensure_available():
            return error
        if not self.hw_lock.acquire(blocking=False):
            return self._error("硬件正忙，请先停止当前任务。")

        self.bg_stop_event.clear()
        try:
            probe_result = self._run_modulation_once(
                text=probe_text,
                center_freq_hz=center_freq_hz,
                mod_scheme="BPSK",
            )
            if probe_result.get("stopped"):
                return probe_result
            if probe_result.get("status") != "success":
                return probe_result

            estimated_snr_db = float(probe_result["estimated_snr_db"])
            probe_decoded_ok = bool(probe_result.get("decoded_ok"))
            selected_modulation = (
                self._decide_modulation_by_snr(estimated_snr_db)
                if probe_decoded_ok
                else "BPSK"
            )
            selection_guard = (
                "probe 包已通过 BPSK 解调和 CRC 校验，允许按 SNR 升阶。"
                if probe_decoded_ok
                else "probe 包未通过 BPSK 解调/CRC 校验，保守锁定 BPSK。"
            )
            final_result = self._run_modulation_once(
                text=text,
                center_freq_hz=center_freq_hz,
                mod_scheme=selected_modulation,
            )
            if final_result.get("stopped"):
                return final_result
            if final_result.get("status") != "success":
                return final_result

            return self._success(
                f"已根据链路质量自动选择 {selected_modulation} 完成发送，并按该调制方式完成接收解调。",
                center_freq_hz=float(center_freq_hz),
                probe_snr_db=round(estimated_snr_db, 2),
                selected_modulation=selected_modulation,
                probe_decoded_ok=probe_decoded_ok,
                selection_guard=selection_guard,
                probe_result=probe_result,
                final_result=final_result,
            )
        finally:
            self._set_state("online")
            self._release_hw_lock()

    def tone_loopback_visualize(
        self,
        center_freq_hz: float = DEFAULT_CENTER_FREQ_HZ,
        tone_freq_hz: float = DEFAULT_TONE_FREQ_HZ,
        samp_rate: float = DEFAULT_TONE_SAMPLE_RATE_HZ,
    ) -> dict[str, Any]:
        center_freq_hz = float(center_freq_hz or DEFAULT_CENTER_FREQ_HZ)
        tone_freq_hz = float(tone_freq_hz or DEFAULT_TONE_FREQ_HZ)
        samp_rate = float(samp_rate or DEFAULT_TONE_SAMPLE_RATE_HZ)
        if center_freq_hz <= 10e6:
            center_freq_hz = DEFAULT_CENTER_FREQ_HZ
        if tone_freq_hz <= 0:
            tone_freq_hz = DEFAULT_TONE_FREQ_HZ
        if samp_rate <= 0:
            samp_rate = DEFAULT_TONE_SAMPLE_RATE_HZ
        if abs(tone_freq_hz) >= samp_rate / 2:
            return self._error(
                "Tone 频率必须小于采样率的一半。",
                center_freq_hz=float(center_freq_hz),
                tone_freq_hz=float(tone_freq_hz),
                samp_rate=float(samp_rate),
            )
        if error := self._ensure_available():
            return error
        if not self.hw_lock.acquire(blocking=False):
            return self._error("硬件正忙，请先停止当前任务。")

        self.bg_stop_event.clear()
        self._set_state(
            "active",
            f"{center_freq_hz / 1e9:.3f} GHz (Tone)",
            f"{samp_rate / 1e6:.1f} MS/s",
            "20 / 20 dB",
            modulation="Tone",
        )
        self._set_visualization_active(
            task="tone_loopback",
            center_freq_hz=center_freq_hz,
            tone_freq_hz=tone_freq_hz,
            sample_rate_hz=samp_rate,
        )

        activity_lock = threading.Lock()
        stream_activity = {
            "tx_sends": 0,
            "tx_samples": 0,
            "rx_frames": 0,
        }

        def record_stream_activity(key: str, amount: int = 1) -> None:
            with activity_lock:
                stream_activity[key] = int(stream_activity.get(key, 0)) + int(amount)
            with self.visualization_lock:
                if key in {"tx_sends", "tx_samples"}:
                    self.visualization_state[key] = (
                        int(self.visualization_state.get(key, 0)) + int(amount)
                    )

        def get_stream_activity() -> dict[str, int]:
            with activity_lock:
                return dict(stream_activity)

        def daemon() -> None:
            tx_stream = None
            rx_stream = None
            tx_thread = None
            rx_thread = None
            worker_errors: list[str] = []

            try:
                if self.bg_stop_event.is_set():
                    return
                for setup_attempt in range(2):
                    try:
                        self.usrp.set_tx_freq(uhd.libpyuhd.types.tune_request(center_freq_hz), 0)
                        self.usrp.set_tx_rate(samp_rate, 0)
                        self.usrp.set_tx_gain(20.0, 0)
                        self.usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(center_freq_hz), 0)
                        self.usrp.set_rx_rate(samp_rate, 0)
                        self.usrp.set_rx_gain(20.0, 0)
                        time.sleep(0.3 if setup_attempt == 0 else 0.7)
                        if self.bg_stop_event.is_set():
                            return

                        rx_stream = self.usrp.get_rx_stream(uhd.usrp.StreamArgs("fc32", "sc16"))
                        tx_stream = self.usrp.get_tx_stream(uhd.usrp.StreamArgs("fc32", "sc16"))
                        self._set_hardware_runtime(tx_stream=tx_stream, rx_stream=rx_stream)
                        break
                    except Exception as exc:
                        self._stop_rx_stream(rx_stream)
                        self._finalize_tx_stream(tx_stream)
                        tx_stream = None
                        rx_stream = None
                        self._clear_background_runtime()
                        if setup_attempt == 0 and self._is_transient_transport_error(exc):
                            self.usrp = None
                            time.sleep(0.5)
                            self._connect()
                            if self.usrp is None:
                                raise
                            self._set_state(
                                "active",
                                f"{center_freq_hz / 1e9:.3f} GHz (Tone)",
                                f"{samp_rate / 1e6:.1f} MS/s",
                                "20 / 20 dB",
                                modulation="Tone",
                            )
                            continue
                        raise

                def tx_worker() -> None:
                    signal = gen_complex_tone(
                        tone_hz=tone_freq_hz,
                        samp_rate=samp_rate,
                        n_samps=10000,
                        amp=0.5,
                    )
                    metadata = uhd.types.TXMetadata()
                    metadata.has_time_spec = False
                    metadata.start_of_burst = True
                    try:
                        while not self.bg_stop_event.is_set():
                            if tx_stream is not None:
                                sent = tx_stream.send(signal, metadata)
                                if sent:
                                    record_stream_activity("tx_sends")
                                    record_stream_activity("tx_samples", int(sent))
                            metadata.start_of_burst = False
                    except Exception as exc:
                        worker_errors.append(f"TX worker failed: {exc}")
                        self.bg_stop_event.set()

                def rx_worker() -> None:
                    buffer = np.zeros((1, 4096), dtype=np.complex64)
                    metadata = uhd.types.RXMetadata()
                    start_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
                    start_cmd.stream_now = True
                    timeout_code = getattr(uhd.types.RXMetadataErrorCode, "timeout", None)
                    none_code = getattr(uhd.types.RXMetadataErrorCode, "none", None)

                    try:
                        rx_stream.issue_stream_cmd(start_cmd)
                        while not self.bg_stop_event.is_set():
                            received = rx_stream.recv(buffer, metadata, timeout=0.25)
                            error_code = getattr(metadata, "error_code", None)

                            if received > 0 and error_code == none_code:
                                self._update_visualization_frame(
                                    buffer[0][:received].copy(),
                                    samp_rate=samp_rate,
                                )
                                record_stream_activity("rx_frames")
                                continue

                            if error_code not in {None, timeout_code, none_code}:
                                worker_errors.append(f"RX metadata error: {error_code}")
                                self.bg_stop_event.set()
                    except Exception as exc:
                        worker_errors.append(f"RX worker failed: {exc}")
                        self.bg_stop_event.set()
                    finally:
                        try:
                            rx_stream.issue_stream_cmd(
                                uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
                            )
                        except Exception:
                            pass

                tx_thread = threading.Thread(target=tx_worker, name="sdr-tone-tx", daemon=True)
                rx_thread = threading.Thread(target=rx_worker, name="sdr-tone-rx", daemon=True)
                self._set_hardware_runtime(
                    tx_stream=tx_stream,
                    rx_stream=rx_stream,
                    tx_thread=tx_thread,
                    rx_thread=rx_thread,
                )
                if self.bg_stop_event.is_set():
                    return
                rx_thread.start()
                time.sleep(0.05)
                if self.bg_stop_event.is_set():
                    return
                tx_thread.start()

                while not self.bg_stop_event.is_set():
                    time.sleep(0.1)
            except Exception as exc:
                worker_errors.append(f"Tone task failed: {exc}")
            finally:
                self.bg_stop_event.set()
                self._stop_rx_stream(rx_stream)
                self._finalize_tx_stream(tx_stream)
                self._join_thread(tx_thread, timeout=2.5)
                self._join_thread(rx_thread, timeout=2.5)
                tx_stream = None
                rx_stream = None
                self._clear_background_runtime()
                self._set_state("online")
                self._set_visualization_stopped(
                    error=" | ".join(worker_errors) if worker_errors else None
                )
                self.bg_thread = None
                self._release_hw_lock()

        thread = threading.Thread(target=daemon, name="sdr-tone-loopback", daemon=True)
        self.bg_thread = thread
        try:
            thread.start()
        except Exception as exc:
            self.bg_thread = None
            self.bg_stop_event.set()
            self._set_state("online")
            self._set_visualization_stopped(error=str(exc))
            self._clear_background_runtime()
            self._release_hw_lock()
            return self._error(f"无法启动后台 Tone 可视化任务: {exc}")

        ready_deadline = time.time() + 6.0
        while time.time() < ready_deadline:
            snapshot = self.get_visualization_snapshot()
            activity = get_stream_activity()
            if int(snapshot.get("frame_id") or 0) > 0 and activity.get("tx_samples", 0) > 0:
                return self._success(
                    "Tone 已切到后台连续执行，波形和频谱会显示在独立可视化窗口中。",
                    center_freq_hz=float(center_freq_hz),
                    tone_freq_hz=float(tone_freq_hz),
                    visualization_ready=True,
                    first_frame_id=int(snapshot.get("frame_id") or 0),
                    tx_samples=int(activity.get("tx_samples", 0)),
                    rx_frames=int(activity.get("rx_frames", 0)),
                )
            if snapshot.get("error"):
                self.bg_stop_event.set()
                self._join_thread(thread, timeout=2.5)
                return self._error(
                    f"Tone 可视化启动失败: {snapshot.get('error')}",
                    center_freq_hz=float(center_freq_hz),
                    tone_freq_hz=float(tone_freq_hz),
                    visualization_ready=False,
                )
            if not thread.is_alive():
                snapshot = self.get_visualization_snapshot()
                return self._error(
                    f"Tone 可视化任务提前退出: {snapshot.get('error') or '未产生可视化数据'}",
                    center_freq_hz=float(center_freq_hz),
                    tone_freq_hz=float(tone_freq_hz),
                    visualization_ready=False,
                )
            time.sleep(0.05)

        self.bg_stop_event.set()
        self._join_thread(thread, timeout=2.5)
        activity = get_stream_activity()
        if activity.get("tx_samples", 0) <= 0:
            reason = "没有检测到 TX 实际送出样本"
        elif activity.get("rx_frames", 0) <= 0:
            reason = "没有检测到 RX 波形/频谱帧"
        else:
            reason = "可视化数据未在限定时间内稳定发布"
        return self._error(
            f"Tone 可视化启动失败：{reason}，已停止本次任务。请检查 USRP 连接、通道、频率、增益和回环路径。",
            center_freq_hz=float(center_freq_hz),
            tone_freq_hz=float(tone_freq_hz),
            visualization_ready=False,
            tx_samples=int(activity.get("tx_samples", 0)),
            rx_frames=int(activity.get("rx_frames", 0)),
        )

    def auto_optimal_transmit(
        self,
        text: str = "",
        modulation_scheme: str = "2-FSK",
        center_freq_hz: float = 2.4e9,
        search_span_hz: float = 30e6,
        enable_tone: bool = False,
        tone_freq_hz: float = 100e3,
        use_adaptive_modulation: bool = False,
    ) -> dict[str, Any]:
        if error := self._ensure_available():
            return error
        if not self.hw_lock.acquire(blocking=False):
            return self._error("硬件正忙，请先停止当前任务。")

        self.bg_stop_event.clear()
        best_freq = center_freq_hz
        min_power = float("inf")
        scan_logs: dict[str, float] = {}
        rx_stream = None
        stopped_by_user = False

        try:
            step_hz = 5e6
            start_hz = center_freq_hz - (search_span_hz / 2)
            end_hz = center_freq_hz + (search_span_hz / 2)
            freqs = np.arange(start_hz, end_hz + 1, step_hz)

            self._set_state(
                "active",
                f"{start_hz / 1e9:.3f}-{end_hz / 1e9:.3f} GHz (Scan)",
                "1.0 MS/s",
                "Rx Only",
            )
            self.usrp.set_rx_rate(1e6, 0)
            rx_stream = self.usrp.get_rx_stream(uhd.usrp.StreamArgs("fc32", "sc16"))
            self._set_hardware_runtime(rx_stream=rx_stream)
            buffer = np.zeros((1, 1024), dtype=np.complex64)
            metadata = uhd.types.RXMetadata()

            for freq in freqs:
                if self.bg_stop_event.is_set():
                    stopped_by_user = True
                    break
                self.usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(freq), 0)
                time.sleep(0.05)
                if self.bg_stop_event.is_set():
                    stopped_by_user = True
                    break
                cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
                cmd.num_samps = 1024
                cmd.stream_now = True
                rx_stream.issue_stream_cmd(cmd)
                rx_stream.recv(buffer, metadata)
                if self.bg_stop_event.is_set():
                    stopped_by_user = True
                    break
                power_db = 10 * np.log10(np.mean(np.abs(buffer[0]) ** 2) + 1e-12)
                scan_logs[f"{freq / 1e6:.1f} MHz"] = round(float(power_db), 2)
                if power_db < min_power:
                    min_power = float(power_db)
                    best_freq = float(freq)
        finally:
            self._stop_rx_stream(rx_stream)
            rx_stream = None
            self._clear_background_runtime()
            self._set_state("online")
            self._release_hw_lock()

        if stopped_by_user:
            return self._success(
                "认知选频任务已停止，未继续发送或启动可视化。",
                stopped=True,
                best_freq_hz=float(best_freq),
                scan_details=scan_logs,
            )

        result = self._success(
            f"已锁定较干净的信道 {best_freq / 1e6:.1f} MHz。",
            best_freq_hz=float(best_freq),
            min_power_db=round(min_power, 2),
            scan_details=scan_logs,
        )

        if text:
            if use_adaptive_modulation:
                result["tx_result"] = self.adaptive_modulation_transmit(
                    text=text,
                    center_freq_hz=best_freq,
                )
            else:
                result["tx_result"] = self.text_fsk_send_and_receive(
                    text=text,
                    modulation_scheme=modulation_scheme,
                    center_freq_hz=best_freq,
                )

        if enable_tone:
            result["tone_result"] = self.tone_loopback_visualize(
                center_freq_hz=best_freq,
                tone_freq_hz=tone_freq_hz,
            )

        return result

    def _decide_modulation_by_snr(self, snr_db: float) -> str:
        if snr_db < 8:
            return "BPSK"
        if snr_db < 16:
            return "QPSK"
        return "16-QAM"

    def _run_modulation_once(
        self,
        text: str,
        center_freq_hz: float,
        mod_scheme: str = "BPSK",
        samp_rate: float = DEFAULT_AMC_SAMPLE_RATE_HZ,
        tx_gain: float = 20.0,
        rx_gain: float = 20.0,
        amp: float = 0.5,
        packet_repetitions: int = DEFAULT_AMC_PACKET_REPETITIONS,
        state_label: str = "AMC",
    ) -> dict[str, Any]:
        tx_stream = None
        rx_stream = None
        captured: list[np.ndarray] = []
        thread: threading.Thread | None = None
        stop_rx = threading.Event()
        self._set_state(
            "active",
            f"{center_freq_hz / 1e9:.3f} GHz ({state_label})",
            f"{samp_rate / 1e6:.1f} MS/s",
            f"{tx_gain:.0f} / {rx_gain:.0f} dB",
            modulation=mod_scheme,
        )
        try:
            sps = DEFAULT_AMC_SYMBOL_SPS
            packet_repetitions = max(1, min(int(packet_repetitions or 1), 8))
            symbols = map_bits_to_linear_symbols(hamming_7_4_encode(bytes_to_bits(build_packet(text))), mod_scheme)
            packet_iq = np.repeat(symbols, sps).astype(np.complex64) * float(amp)
            guard_samples = max(int(round(DEFAULT_AMC_GUARD_INTERVAL_S * samp_rate)), sps * 16)
            guard_iq = np.zeros(guard_samples, dtype=np.complex64)
            tx_parts: list[np.ndarray] = [guard_iq]
            for _ in range(packet_repetitions):
                tx_parts.append(packet_iq)
                tx_parts.append(guard_iq)
            tx_iq = np.concatenate(tx_parts)
            tx_time = len(tx_iq) / float(samp_rate)

            self.usrp.set_tx_freq(uhd.libpyuhd.types.tune_request(center_freq_hz), 0)
            self.usrp.set_tx_rate(samp_rate, 0)
            self.usrp.set_tx_gain(tx_gain, 0)
            self.usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(center_freq_hz), 0)
            self.usrp.set_rx_rate(samp_rate, 0)
            self.usrp.set_rx_gain(rx_gain, 0)

            tx_stream = self.usrp.get_tx_stream(uhd.usrp.StreamArgs("fc32", "sc16"))
            rx_stream = self.usrp.get_rx_stream(uhd.usrp.StreamArgs("fc32", "sc16"))
            self._set_hardware_runtime(tx_stream=tx_stream, rx_stream=rx_stream)

            def rx_worker() -> None:
                cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
                cmd.stream_now = True
                buffer = np.zeros((1, 4096), dtype=np.complex64)
                metadata = uhd.types.RXMetadata()
                timeout_code = getattr(uhd.types.RXMetadataErrorCode, "timeout", None)
                none_code = getattr(uhd.types.RXMetadataErrorCode, "none", None)
                try:
                    rx_stream.issue_stream_cmd(cmd)
                    while not stop_rx.is_set() and not self.bg_stop_event.is_set():
                        received = rx_stream.recv(buffer, metadata, timeout=0.1)
                        error_code = getattr(metadata, "error_code", None)
                        if received > 0 and error_code in {None, none_code}:
                            captured.append(buffer[0][:received].copy())
                            continue
                        if error_code not in {None, timeout_code, none_code}:
                            stop_rx.set()
                finally:
                    self._stop_rx_stream(rx_stream)

            thread = threading.Thread(target=rx_worker, name="sdr-amc-rx", daemon=True)
            self._set_hardware_runtime(tx_stream=tx_stream, rx_stream=rx_stream, rx_thread=thread)
            thread.start()
            time.sleep(0.05)

            metadata = uhd.types.TXMetadata()
            metadata.has_time_spec = False
            metadata.start_of_burst = True
            metadata.end_of_burst = False

            chunk_size = 2048
            tx_samples_sent = 0
            for idx in range(0, len(tx_iq), chunk_size):
                if self.bg_stop_event.is_set():
                    break
                chunk = tx_iq[idx : idx + chunk_size]
                if idx + chunk_size >= len(tx_iq):
                    metadata.end_of_burst = True
                sent = int(tx_stream.send(chunk, metadata) or 0)
                tx_samples_sent += max(0, min(sent, len(chunk)))
                metadata.start_of_burst = False

            if not self.bg_stop_event.is_set():
                time.sleep(max(0.1, min(0.5, tx_time + 0.05)))
            stop_rx.set()
            self._stop_rx_stream(rx_stream)
            self._join_thread(thread, timeout=2.5)

            rx_iq = np.concatenate(captured) if captured else np.array([], dtype=np.complex64)
            if self.bg_stop_event.is_set():
                return self._success(
                    f"{mod_scheme} 测试已停止。",
                    stopped=True,
                    center_freq_hz=float(center_freq_hz),
                    tested_modulation=mod_scheme,
                    captured_samples=int(rx_iq.size),
                    tx_samples_requested=int(tx_iq.size),
                    tx_samples_sent=int(tx_samples_sent),
                )

            if rx_iq.size == 0:
                return self._error("未接收到回环信号。")

            m2 = np.mean(np.abs(rx_iq) ** 2)
            m4 = np.mean(np.abs(rx_iq) ** 4)
            estimated_linear = max(
                (
                    np.sqrt(max(2 * m2**2 - m4, 1e-10))
                    / max((m2 - np.sqrt(max(2 * m2**2 - m4, 1e-10))), 1e-10)
                ),
                1e-10,
            )
            snr_db = 10 * np.log10(estimated_linear + 1e-5)
            decoded, decode_diag = decode_linear_modulated_texts_from_iq(
                rx_iq,
                mod_scheme=mod_scheme,
                sps=sps,
            )

            return self._success(
                f"{mod_scheme} 测试完成。",
                center_freq_hz=float(center_freq_hz),
                tested_modulation=mod_scheme,
                modulation_scheme=mod_scheme,
                estimated_snr_db=round(float(snr_db), 2),
                sent_text=text,
                decoded_texts=decoded,
                decoded_ok=bool(decoded),
                decode_diagnostics=decode_diag,
                captured_samples=int(rx_iq.size),
                packet_repetitions=int(packet_repetitions),
                samp_rate=float(samp_rate),
                sym_rate=float(samp_rate) / max(int(sps), 1),
                tx_gain=float(tx_gain),
                rx_gain=float(rx_gain),
                tx_samples_requested=int(tx_iq.size),
                tx_samples_sent=int(tx_samples_sent),
                tx_complete=bool(tx_samples_sent >= int(tx_iq.size)),
                samples_per_symbol=int(sps),
            )
        finally:
            stop_rx.set()
            self._stop_rx_stream(rx_stream)
            self._finalize_tx_stream(tx_stream)
            self._join_thread(thread, timeout=2.5)
            tx_stream = None
            rx_stream = None
            self._clear_background_runtime()


def bytes_to_bits(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder="big").astype(np.uint8)


def bits_to_bytes(bits: np.ndarray) -> bytes:
    bits = np.asarray(bits, dtype=np.uint8)
    if len(bits) % 8 != 0:
        bits = np.pad(bits, (0, 8 - (len(bits) % 8)), mode="constant")
    return np.packbits(bits, bitorder="big").tobytes()


_H74_G = np.array([
    [1, 0, 0, 0, 1, 1, 1],
    [0, 1, 0, 0, 1, 1, 0],
    [0, 0, 1, 0, 1, 0, 1],
    [0, 0, 0, 1, 0, 1, 1],
], dtype=np.uint8)

_H74_H = np.array([
    [1, 1, 1, 0, 1, 0, 0],
    [1, 1, 0, 1, 0, 1, 0],
    [1, 0, 1, 1, 0, 0, 1],
], dtype=np.uint8)

_H74_SYNDROME_MAP: dict[int, int] = {}
for _col in range(7):
    _syndrome = int(np.dot(_H74_H, [1 if j == _col else 0 for j in range(7)]) % 2 @ (1 << np.arange(3)))
    _H74_SYNDROME_MAP[_syndrome] = _col


def hamming_7_4_encode(bits: np.ndarray) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.uint8)
    if bits.size == 0:
        return bits
    pad = (4 - (bits.size % 4)) % 4
    if pad:
        bits = np.pad(bits, (0, pad), mode="constant")
    return ((bits.reshape(-1, 4) @ _H74_G) % 2).ravel()


def hamming_7_4_decode(bits: np.ndarray) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.uint8)
    if bits.size == 0:
        return bits
    pad = (7 - (bits.size % 7)) % 7
    if pad:
        bits = np.pad(bits, (0, pad), mode="constant")
    codewords = bits.reshape(-1, 7)
    syndromes = (codewords @ _H74_H.T) % 2
    syndrome_int = np.dot(syndromes, 1 << np.arange(3))
    corrected = codewords.copy()
    for idx in range(len(corrected)):
        si = int(syndrome_int[idx])
        if si != 0:
            col = _H74_SYNDROME_MAP.get(si)
            if col is not None:
                corrected[idx, col] ^= 1
    return corrected[:, :4].ravel()


def build_packet(payload_text: str) -> bytes:
    payload = payload_text.encode("utf-8", errors="replace")
    return (
        bytes([PREAMBLE_BYTE]) * PREAMBLE_LEN_BYTES
        + SYNC_WORD
        + len(payload).to_bytes(2, "big")
        + payload
        + (zlib.crc32(payload) & 0xFFFFFFFF).to_bytes(4, "big")
    )


def normalize_text_modulation_scheme(modulation_scheme: str | None) -> str:
    raw = str(modulation_scheme or "2-FSK").strip().upper().replace("_", "-")
    compact = raw.replace(" ", "").replace("-", "")
    if compact in {"", "2FSK", "FSK"}:
        return "2-FSK"
    if compact == "BPSK":
        return "BPSK"
    if compact == "QPSK":
        return "QPSK"
    if compact in {"16QAM", "QAM16"}:
        return "16-QAM"
    raise ValueError(f"不支持的文本调制方式: {modulation_scheme}")


def map_bits_to_linear_symbols(bits: np.ndarray, mod_scheme: str) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.uint8)
    scheme = mod_scheme.upper()
    if scheme == "BPSK":
        symbols = (bits.astype(np.int16) * 2 - 1).astype(np.float32)
        return symbols.astype(np.complex64)

    if scheme == "QPSK":
        if len(bits) % 2 != 0:
            bits = np.append(bits, 0)
        bit_pairs = bits.reshape(-1, 2).astype(np.int16)
        sym_i = bit_pairs[:, 0] * 2 - 1
        sym_q = bit_pairs[:, 1] * 2 - 1
        return ((sym_i + 1j * sym_q) / np.sqrt(2.0)).astype(np.complex64)

    if scheme == "16-QAM":
        if len(bits) % 4 != 0:
            bits = np.append(bits, np.zeros(4 - len(bits) % 4, dtype=np.uint8))
        mapping = {(0, 0): -3, (0, 1): -1, (1, 1): 1, (1, 0): 3}
        values = []
        for idx in range(0, len(bits), 4):
            i_val = mapping[tuple(int(bit) for bit in bits[idx : idx + 2])]
            q_val = mapping[tuple(int(bit) for bit in bits[idx + 2 : idx + 4])]
            values.append((i_val + 1j * q_val) / np.sqrt(10.0))
        return np.asarray(values, dtype=np.complex64)

    raise ValueError(f"不支持的调制方式: {mod_scheme}")


def _extract_active_iq_window(iq: np.ndarray, sps: int) -> np.ndarray:
    iq = np.asarray(iq, dtype=np.complex64)
    if iq.size == 0:
        return iq

    max_samples = 600_000
    block = max(int(sps) * 8, 64)
    n_blocks = iq.size // block
    if n_blocks < 4:
        return iq[:max_samples]

    trimmed = iq[: n_blocks * block]
    block_power = np.mean(np.abs(trimmed.reshape(n_blocks, block)) ** 2, axis=1)
    finite = block_power[np.isfinite(block_power)]
    if finite.size == 0:
        return iq[:max_samples]

    floor = float(np.median(finite))
    peak = float(np.max(finite))
    if peak <= floor * 1.5:
        return iq[:max_samples]

    threshold = max(floor * 4.0, peak * 0.08)
    active = np.flatnonzero(block_power >= threshold)
    if active.size == 0:
        active = np.flatnonzero(block_power >= peak * 0.2)
    if active.size == 0:
        return iq[:max_samples]

    margin_blocks = 6
    start = max(0, int(active[0] - margin_blocks) * block)
    end = min(iq.size, int(active[-1] + margin_blocks + 1) * block)
    if end - start > max_samples:
        end = start + max_samples
    return iq[start:end]


def _symbols_from_oversampled_iq(iq: np.ndarray, sps: int, offset: int) -> np.ndarray:
    if offset >= len(iq):
        return np.array([], dtype=np.complex64)
    aligned = np.asarray(iq[offset:], dtype=np.complex64)
    n_symbols = len(aligned) // sps
    if n_symbols <= 0:
        return np.array([], dtype=np.complex64)
    symbols = aligned[: n_symbols * sps].reshape(n_symbols, sps).mean(axis=1)
    if symbols.size > 50_000:
        symbols = symbols[:50_000]
    power = float(np.mean(np.abs(symbols) ** 2))
    if not np.isfinite(power) or power <= 1e-12:
        return np.array([], dtype=np.complex64)
    return (symbols / np.sqrt(power)).astype(np.complex64)


def _remove_guard_dc(iq: np.ndarray) -> np.ndarray:
    iq = np.asarray(iq, dtype=np.complex64)
    if iq.size == 0:
        return iq
    power = np.abs(iq) ** 2
    finite_power = power[np.isfinite(power)]
    if finite_power.size == 0:
        return iq
    threshold = float(np.percentile(finite_power, 20))
    guard_like = iq[power <= threshold]
    if guard_like.size >= max(16, iq.size // 20):
        dc = np.mean(guard_like)
        return (iq - dc).astype(np.complex64)
    return iq


def _demap_linear_symbols_to_bits(symbols: np.ndarray, mod_scheme: str) -> np.ndarray:
    scheme = mod_scheme.upper()
    symbols = np.asarray(symbols, dtype=np.complex64)
    if scheme == "BPSK":
        return (symbols.real > 0).astype(np.uint8)

    if scheme == "QPSK":
        bits = np.empty(symbols.size * 2, dtype=np.uint8)
        bits[0::2] = (symbols.real > 0).astype(np.uint8)
        bits[1::2] = (symbols.imag > 0).astype(np.uint8)
        return bits

    if scheme == "16-QAM":
        threshold = 2.0 / np.sqrt(10.0)

        def axis_bits(values: np.ndarray) -> np.ndarray:
            out = np.empty((values.size, 2), dtype=np.uint8)
            out[values < -threshold] = (0, 0)
            out[(values >= -threshold) & (values < 0)] = (0, 1)
            out[(values >= 0) & (values < threshold)] = (1, 1)
            out[values >= threshold] = (1, 0)
            return out

        i_bits = axis_bits(symbols.real)
        q_bits = axis_bits(symbols.imag)
        bits = np.empty(symbols.size * 4, dtype=np.uint8)
        bits[0::4] = i_bits[:, 0]
        bits[1::4] = i_bits[:, 1]
        bits[2::4] = q_bits[:, 0]
        bits[3::4] = q_bits[:, 1]
        return bits

    raise ValueError(f"不支持的调制方式: {mod_scheme}")


def decode_linear_modulated_texts_from_iq(
    iq: np.ndarray,
    mod_scheme: str,
    sps: int,
) -> tuple[list[str], dict[str, Any]]:
    diag: dict[str, Any] = {
        "samples": int(iq.size),
        "sps": int(sps),
        "mod_scheme": mod_scheme,
        "failures": [],
    }
    iq = np.asarray(iq, dtype=np.complex64)
    if iq.size == 0 or sps <= 0:
        diag["failures"].append(f"IQ样本为空或sps({sps})无效")
        return [], diag

    window = _extract_active_iq_window(iq, sps=sps)
    diag["window_samples"] = int(window.size)
    if window.size == 0:
        diag["failures"].append("未提取到活跃信号窗口(功率太低或无信号)")
        return [], diag
    window = _remove_guard_dc(window)
    diag["window_after_dc"] = int(window.size)

    scheme = mod_scheme.upper()
    phase_steps = 32 if scheme == "BPSK" else 64
    diag["phase_steps"] = phase_steps
    rotations = np.exp(-1j * np.linspace(0.0, 2.0 * np.pi, phase_steps, endpoint=False))

    diag["offsets_tried"] = 0
    diag["combinations_tried"] = 0
    for offset in range(sps):
        symbols = _symbols_from_oversampled_iq(window, sps=sps, offset=offset)
        if symbols.size == 0:
            continue
        diag["offsets_tried"] += 1
        for use_conjugate in (False, True):
            oriented = np.conj(symbols) if use_conjugate else symbols
            for rotation in rotations:
                diag["combinations_tried"] += 1
                bits = _demap_linear_symbols_to_bits(oriented * rotation, scheme)
                for cw_off in range(7):
                    aligned = bits[cw_off:]
                    if aligned.size < 7:
                        continue
                    corrected = hamming_7_4_decode(aligned)
                    texts, pkt_errors = extract_all_packets_from_bits(corrected)
                    if texts:
                        return _dedupe_texts(texts), diag
                    for err in pkt_errors:
                        if err not in diag["failures"]:
                            diag["failures"].append(err)
    diag["failures"] = diag["failures"][:10]
    return [], diag


def modulate_2fsk(
    bits: np.ndarray,
    samp_rate: float,
    sym_rate: float,
    f_dev: float,
    amp: float,
) -> np.ndarray:
    sps = int(round(samp_rate / sym_rate))
    freqs_u = np.repeat(
        np.where(np.asarray(bits, dtype=np.uint8) == 1, +f_dev, -f_dev).astype(np.float64),
        sps,
    )
    phase = np.cumsum(2.0 * np.pi * freqs_u / samp_rate)
    return (amp * np.exp(1j * phase)).astype(np.complex64)


def fsk_freq_discriminator(iq: np.ndarray, samp_rate: float) -> np.ndarray:
    iq = np.asarray(iq, dtype=np.complex64)
    if len(iq) < 2:
        return np.array([], dtype=np.float32)
    return (
        np.angle(iq[1:] * np.conj(iq[:-1])).astype(np.float32)
        * samp_rate
        / (2.0 * np.pi)
    ).astype(np.float32)


def slicer_bits_from_freq(freq: np.ndarray, sps: int, threshold_hz: float) -> np.ndarray:
    freq = np.asarray(freq, dtype=np.float32)
    if len(freq) < sps:
        return np.array([], dtype=np.uint8)
    n_sym = len(freq) // sps
    sym_freq = freq[: n_sym * sps].reshape(n_sym, sps).mean(axis=1)
    return slicer_bits_from_symbol_freqs(sym_freq, threshold_hz)


def slicer_bits_from_symbol_freqs(
    sym_freq: np.ndarray,
    threshold_hz: float,
    *,
    invert: bool = False,
) -> np.ndarray:
    bits = (np.asarray(sym_freq, dtype=np.float32) > threshold_hz).astype(np.uint8)
    if invert:
        bits = 1 - bits
    return bits


def extract_all_packets_from_bits(bitbuf: np.ndarray) -> tuple[list[str], list[str]]:
    texts: list[str] = []
    errors: list[str] = []
    bitbuf = np.asarray(bitbuf, dtype=np.uint8)
    attempts = 0
    while True:
        if len(bitbuf) < (len(SYNC_WORD) + 2 + 4) * 8:
            if attempts == 0:
                errors.append(f"比特缓冲区太小({len(bitbuf)}bit)，不足以容纳最小数据包")
            break
        found_any = False
        for bit_offset in range(8):
            if len(bitbuf) <= bit_offset:
                continue
            data = bits_to_bytes(bitbuf[bit_offset:])
            idx = data.find(SYNC_WORD)
            if idx < 0:
                continue
            header_pos = idx + len(SYNC_WORD)
            if len(data) < header_pos + 2:
                continue
            payload_len = int.from_bytes(data[header_pos : header_pos + 2], "big")
            if payload_len == 0 or payload_len > 4096:
                errors.append(f"数据包长度异常({payload_len})，已跳过")
                bitbuf = bitbuf[bit_offset + (idx + 1) * 8:]
                found_any = True
                break
            payload_pos = header_pos + 2
            crc_pos = payload_pos + payload_len
            end_pos = crc_pos + 4
            if end_pos > len(data):
                errors.append(f"数据包长度{payload_len}超出剩余数据({len(data) - header_pos - 2}字节可用)")
                bitbuf = bitbuf[bit_offset + (idx + 1) * 8:]
                found_any = True
                break
            payload = data[payload_pos : payload_pos + payload_len]
            crc = int.from_bytes(data[crc_pos:end_pos], "big")
            if crc != (zlib.crc32(payload) & 0xFFFFFFFF):
                errors.append(f"CRC校验失败(计算值=0x{zlib.crc32(payload) & 0xFFFFFFFF:08X}, 期望值=0x{crc:08X})")
                bitbuf = bitbuf[bit_offset + (idx + 1) * 8:]
                found_any = True
                break
            texts.append(payload.decode("utf-8", errors="replace"))
            bitbuf = bitbuf[bit_offset + end_pos * 8 :]
            found_any = True
            break
        attempts += 1
        if not found_any:
            if not errors:
                data = bits_to_bytes(bitbuf)
                errors.append(f"未在{len(data)}字节数据中找到同步字0x{SYNC_WORD.hex().upper()}")
            break
    return texts, errors


def _fsk_threshold_candidates(sym_freq: np.ndarray, fallback_hz: float) -> list[float]:
    sym_freq = np.asarray(sym_freq, dtype=np.float32)
    finite = sym_freq[np.isfinite(sym_freq)]
    if finite.size == 0:
        return [float(fallback_hz)]

    candidates = [
        float(fallback_hz),
        0.0,
        float(np.mean(finite)),
        float(np.median(finite)),
        float((np.percentile(finite, 10) + np.percentile(finite, 90)) / 2.0),
    ]

    low = float(np.percentile(finite, 25))
    high = float(np.percentile(finite, 75))
    if high > low:
        threshold = (low + high) / 2.0
        for _ in range(8):
            lower_cluster = finite[finite <= threshold]
            upper_cluster = finite[finite > threshold]
            if lower_cluster.size == 0 or upper_cluster.size == 0:
                break
            next_threshold = (float(np.mean(lower_cluster)) + float(np.mean(upper_cluster))) / 2.0
            if abs(next_threshold - threshold) < 1e-3:
                break
            threshold = next_threshold
        candidates.append(float(threshold))

    deduped: list[float] = []
    for candidate in candidates:
        if not np.isfinite(candidate):
            continue
        if any(abs(candidate - existing) < 1e-3 for existing in deduped):
            continue
        deduped.append(float(candidate))
    return deduped


def _dedupe_texts(texts: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for text in texts:
        if text in seen:
            continue
        seen.add(text)
        unique.append(text)
    return unique


def estimate_fsk_rx_diagnostics(iq: np.ndarray, samp_rate: float) -> dict[str, Any]:
    iq = np.asarray(iq, dtype=np.complex64)
    if iq.size == 0:
        return {
            "rx_power_db": None,
            "rx_peak_power_db": None,
            "fsk_freq_low_hz": None,
            "fsk_freq_high_hz": None,
            "fsk_freq_span_hz": None,
        }

    power = np.abs(iq) ** 2
    rx_power_db = 10 * np.log10(float(np.mean(power)) + 1e-12)
    rx_peak_power_db = 10 * np.log10(float(np.max(power)) + 1e-12)

    freq = fsk_freq_discriminator(iq, samp_rate)
    finite = freq[np.isfinite(freq)]
    if finite.size == 0:
        return {
            "rx_power_db": round(float(rx_power_db), 2),
            "rx_peak_power_db": round(float(rx_peak_power_db), 2),
            "fsk_freq_low_hz": None,
            "fsk_freq_high_hz": None,
            "fsk_freq_span_hz": None,
        }

    low = float(np.percentile(finite, 10))
    high = float(np.percentile(finite, 90))
    return {
        "rx_power_db": round(float(rx_power_db), 2),
        "rx_peak_power_db": round(float(rx_peak_power_db), 2),
        "fsk_freq_low_hz": round(low, 2),
        "fsk_freq_high_hz": round(high, 2),
        "fsk_freq_span_hz": round(high - low, 2),
    }


def decode_texts_from_iq(
    iq: np.ndarray,
    samp_rate: float,
    sym_rate: float,
    threshold_hz: float,
) -> tuple[list[str], dict[str, Any]]:
    diag: dict[str, Any] = {
        "samples": int(iq.size),
        "sps": int(round(samp_rate / sym_rate)),
        "offsets_tried": 0,
        "combinations_tried": 0,
        "failures": [],
    }
    sps = diag["sps"]
    freq = fsk_freq_discriminator(iq, samp_rate)
    diag["freq_samples"] = int(freq.size)
    if freq.size == 0:
        diag["failures"].append("IQ样本太少，无法进行频率鉴频")
        return [], diag

    for sym_off in range(max(sps, 1)):
        offset_freq = freq[sym_off:]
        if len(offset_freq) < sps:
            continue
        diag["offsets_tried"] += 1
        n_sym = len(offset_freq) // sps
        sym_freq = offset_freq[: n_sym * sps].reshape(n_sym, sps).mean(axis=1)
        candidates = _fsk_threshold_candidates(sym_freq, threshold_hz)
        for candidate in candidates:
            for invert in (False, True):
                diag["combinations_tried"] += 1
                bits = slicer_bits_from_symbol_freqs(sym_freq, candidate, invert=invert)
                if len(bits) == 0:
                    continue
                for cw_off in range(7):
                    aligned = bits[cw_off:]
                    if aligned.size < 7:
                        continue
                    corrected = hamming_7_4_decode(aligned)
                    texts, pkt_errors = extract_all_packets_from_bits(corrected)
                    if texts:
                        return _dedupe_texts(texts), diag
                    for err in pkt_errors:
                        if err not in diag["failures"]:
                            diag["failures"].append(err)
    diag["failures"] = diag["failures"][:10]
    return [], diag


def gen_complex_tone(tone_hz: float, samp_rate: float, n_samps: int, amp: float) -> np.ndarray:
    phases = 2.0 * np.pi * float(tone_hz) * (
        np.arange(n_samps, dtype=np.float64) / float(samp_rate)
    )
    return (amp * np.exp(1j * phases)).astype(np.complex64)


def _downsample_for_plot(values: np.ndarray, max_points: int) -> list[float]:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return []
    if arr.size <= max_points:
        return [round(float(item), 4) for item in arr]

    indices = np.linspace(0, arr.size - 1, num=max_points, dtype=np.int32)
    return [round(float(arr[index]), 4) for index in indices]
