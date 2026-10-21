from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from hardware import sdr_controller as sdr


class CognitiveSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        # Never connect to physical equipment in these tests.
        with patch.object(sdr, "uhd", None):
            self.controller = sdr.SDRController("test-device")
        self.controller.driver_available = True
        self.stream = Mock()
        self.controller.usrp = Mock()
        self.controller.usrp.get_rx_stream.return_value = self.stream
        self.uhd = SimpleNamespace(
            usrp=SimpleNamespace(StreamArgs=Mock()),
            types=SimpleNamespace(
                RXMetadata=lambda: SimpleNamespace(error_code="none"),
                RXMetadataErrorCode=SimpleNamespace(none="none"),
                StreamMode=SimpleNamespace(num_done="num_done", stop_cont="stop", start_cont="start_cont"),
                StreamCMD=lambda mode: SimpleNamespace(mode=mode),
                TXMetadata=SimpleNamespace,
            ),
            libpyuhd=SimpleNamespace(types=SimpleNamespace(tune_request=lambda freq: freq)),
        )
        self.addCleanup(patch.stopall)
        patch.object(sdr, "uhd", self.uhd).start()
        patch.object(sdr.time, "sleep", return_value=None).start()

    def receive(self, amplitudes, *, samples=1024):
        values = iter(amplitudes)

        def recv(buffer, metadata):
            amplitude = next(values)
            buffer[:] = 0
            if amplitude is None:
                return 0
            buffer[0, :samples] = amplitude
            metadata.error_code = "none"
            return samples

        self.stream.recv.side_effect = recv

    def test_running_snapshot_final_minimum_and_tie(self) -> None:
        self.receive([0.2, 0.1, 0.1])
        snapshots = []
        self.controller.usrp.set_rx_freq.side_effect = lambda *_: snapshots.append(
            self.controller.get_visualization_snapshot()["cognitive_scan"]
        )
        result = self.controller.auto_optimal_transmit(search_span_hz=10e6)
        scan = self.controller.get_visualization_snapshot()["cognitive_scan"]
        self.assertEqual(result["best_freq_hz"], 2.4e9)
        self.assertEqual(scan["status"], "completed")
        self.assertEqual(scan["best_freq_hz"], result["best_freq_hz"])
        self.assertAlmostEqual(scan["min_power_db"], -20, places=4)
        self.assertEqual(len(snapshots), 3)
        self.assertEqual(snapshots[0]["candidates"][0]["status"], "measuring")
        self.assertEqual(snapshots[1]["candidates"][0]["status"], "measured")
        self.assertEqual(snapshots[2]["best_freq_hz"], 2.4e9)
        # Snapshots are independent from mutable internal state.
        scan["candidates"].clear()
        self.assertEqual(len(self.controller.get_visualization_snapshot()["cognitive_scan"]["candidates"]), 3)

    def test_default_span_produces_seven_points_and_keeps_power_precision(self) -> None:
        self.receive([0.1234567] * 7)
        result = self.controller.auto_optimal_transmit()
        scan = self.controller.get_visualization_snapshot()["cognitive_scan"]
        self.assertEqual(result["search_span_hz"], 30e6)
        self.assertEqual(result["candidate_count"], 7)
        self.assertEqual([p["freq_hz"] for p in scan["candidates"]], [2.385e9 + i * 5e6 for i in range(7)])
        self.assertEqual(result["min_power_db"], scan["min_power_db"])
        self.assertNotEqual(result["min_power_db"], round(result["min_power_db"], 2))

    def test_fixed_text_preserves_packet_snapshot_and_scan_history(self) -> None:
        self.receive([0.1, 0.2])
        self.controller.auto_optimal_transmit(search_span_hz=5e6)
        history = self.controller.get_visualization_snapshot()["cognitive_scan"]
        iq = np.concatenate([np.zeros(2048), np.ones(2048) * 0.3, np.zeros(2048)]).astype(np.complex64)
        def run(**kwargs):
            self.controller._publish_packet_capture(iq, kwargs["samp_rate"])
            return {"status": "success", "decoded_ok": True, "tx_samples_sent": 1234}
        self.controller._run_modulation_once = Mock(side_effect=run)
        self.controller.text_fsk_send_and_receive(text="test", modulation_scheme="BPSK", center_freq_hz=2.45e9)
        snapshot = self.controller.get_visualization_snapshot()
        self.assertEqual(snapshot["task"], "text_transmit")
        self.assertEqual(snapshot["task_status"], "completed")
        self.assertFalse(snapshot["active"])
        self.assertTrue(snapshot["decoded_ok"])
        self.assertEqual(snapshot["modulation_scheme"], "BPSK")
        self.assertEqual(snapshot["center_freq_hz"], 2.45e9)
        self.assertEqual(snapshot["captured_samples"], iq.size)
        self.assertEqual(snapshot["capture_window_start_s"], 2048 / 1e6)
        self.assertGreater(len(snapshot["fft_magnitude_db"]), 0)
        self.assertEqual(snapshot["cognitive_scan"], history)
        self.assertFalse(self.controller.has_background_task())

    def test_adaptive_snapshot_keeps_final_modulation_and_decode_result(self) -> None:
        iq = np.ones(2048, dtype=np.complex64) * 0.3
        def run(**kwargs):
            self.controller._publish_packet_capture(iq, 10e6)
            with self.controller.visualization_lock:
                self.controller.visualization_state["modulation_scheme"] = kwargs["mod_scheme"]
            return {"status": "success", "decoded_ok": True, "estimated_snr_db": 12.0}
        self.controller._run_modulation_once = Mock(side_effect=run)
        self.controller.adaptive_modulation_transmit(text="test")
        snapshot = self.controller.get_visualization_snapshot()
        self.assertEqual(snapshot["task"], "adaptive_transmit")
        self.assertEqual(snapshot["task_status"], "completed")
        self.assertEqual(snapshot["modulation_scheme"], "QPSK")
        self.assertTrue(snapshot["decoded_ok"])
        self.assertFalse(self.controller.has_background_task())

    def test_fsk_stopped_capture_remains_available(self) -> None:
        def recv(buffer, metadata, **_):
            buffer[0, :1024] = 0.3
            self.controller.bg_stop_event.set()
            return 1024
        self.stream.recv.side_effect = recv
        self.controller.usrp.get_tx_stream.return_value.send.side_effect = lambda iq, *_: len(iq)
        result = self.controller.text_fsk_send_and_receive(text="test")
        snapshot = self.controller.get_visualization_snapshot()
        self.assertTrue(result["stopped"])
        self.assertEqual(snapshot["task_status"], "stopped")
        self.assertEqual(snapshot["captured_samples"], 1024)
        self.assertGreater(len(snapshot["iq_inphase"]), 0)

    def test_partial_receive_uses_actual_samples_and_empty_receive_is_excluded(self) -> None:
        self.receive([None, 0.1, 0.2], samples=256)
        result = self.controller.auto_optimal_transmit(search_span_hz=10e6)
        scan = self.controller.get_visualization_snapshot()["cognitive_scan"]
        self.assertEqual(scan["candidates"][0]["status"], "invalid")
        self.assertEqual(scan["candidates"][1]["samples"], 256)
        self.assertEqual(result["best_freq_hz"], 2.4e9)
        self.assertAlmostEqual(result["min_power_db"], -20, places=2)

    def test_no_valid_samples_never_transmits(self) -> None:
        self.receive([None] * 3)
        self.controller.text_fsk_send_and_receive = Mock()
        result = self.controller.auto_optimal_transmit(text="test", search_span_hz=10e6)
        self.assertEqual(result["status"], "error")
        self.assertEqual(self.controller.get_visualization_snapshot()["cognitive_scan"]["status"], "error")
        self.controller.text_fsk_send_and_receive.assert_not_called()
        self.assertFalse(self.controller.hw_lock.locked())

    def test_stop_retains_partial_results_and_never_transmits(self) -> None:
        self.receive([0.1])
        def tune(freq, _channel):
            if freq == 2.4e9:
                self.controller.bg_stop_event.set()
        self.controller.usrp.set_rx_freq.side_effect = tune
        self.controller.text_fsk_send_and_receive = Mock()
        result = self.controller.auto_optimal_transmit(text="test", search_span_hz=10e6)
        scan = self.controller.get_visualization_snapshot()["cognitive_scan"]
        self.assertTrue(result["stopped"])
        self.assertEqual(scan["status"], "stopped")
        self.assertEqual(scan["best_freq_hz"], 2.395e9)
        self.assertEqual(scan["candidates"][1]["status"], "pending")
        self.controller.text_fsk_send_and_receive.assert_not_called()

    def test_metadata_error_and_exception_are_visible(self) -> None:
        def invalid_recv(buffer, metadata):
            metadata.error_code = "overflow"
            return 1024
        self.stream.recv.side_effect = invalid_recv
        self.assertEqual(self.controller.auto_optimal_transmit(search_span_hz=5e6)["status"], "error")
        self.stream.recv.side_effect = RuntimeError("receive failed")
        with self.assertRaisesRegex(RuntimeError, "receive failed"):
            self.controller.auto_optimal_transmit(search_span_hz=5e6)
        scan = self.controller.get_visualization_snapshot()["cognitive_scan"]
        self.assertEqual(scan["status"], "error")
        self.assertEqual(scan["error"], "receive failed")
        self.assertFalse(self.controller.hw_lock.locked())

    def test_results_survive_tone_and_next_scan_replaces_them(self) -> None:
        self.receive([0.1, 0.2, 0.3])
        self.controller.auto_optimal_transmit(search_span_hz=10e6)
        scan = self.controller.get_visualization_snapshot()["cognitive_scan"]
        self.controller._set_visualization_active(task="tone_loopback", center_freq_hz=2.395e9, tone_freq_hz=1e5, sample_rate_hz=1e6)
        self.controller._set_visualization_stopped()
        self.assertEqual(self.controller.get_visualization_snapshot()["cognitive_scan"], scan)
        self.receive([0.3, 0.1])
        self.controller.auto_optimal_transmit(search_span_hz=5e6)
        new = self.controller.get_visualization_snapshot()["cognitive_scan"]
        self.assertEqual(new["run_id"], scan["run_id"] + 1)
        self.assertEqual(len(new["candidates"]), 2)

    def test_receive_exception_during_stop_is_not_reported_as_failure(self) -> None:
        def interrupted_recv(*_):
            self.controller.bg_stop_event.set()
            raise RuntimeError("stream stopped")
        self.stream.recv.side_effect = interrupted_recv
        result = self.controller.auto_optimal_transmit(search_span_hz=5e6)
        self.assertTrue(result["stopped"])
        scan = self.controller.get_visualization_snapshot()["cognitive_scan"]
        self.assertEqual(scan["status"], "stopped")
        self.assertIsNone(scan["error"])
        self.assertFalse(self.controller.hw_lock.locked())


if __name__ == "__main__":
    unittest.main()
