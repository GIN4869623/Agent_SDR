from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from hardware import sdr_controller as sdr


TRANSPORT_TIMEOUT = "tx xport timed out wating for a strs packet during initialization"


class AdaptiveTransmitTests(unittest.TestCase):
    def setUp(self) -> None:
        with patch.object(sdr, "uhd", None):
            self.controller = sdr.SDRController("test-device")
        self.controller.driver_available = True
        self.device = Mock()
        self.controller.usrp = self.device
        fake_uhd = SimpleNamespace(
            usrp=SimpleNamespace(StreamArgs=Mock()),
            types=SimpleNamespace(
                StreamMode=SimpleNamespace(stop_cont="stop"),
                StreamCMD=lambda mode: SimpleNamespace(mode=mode),
                TXMetadata=SimpleNamespace,
            ),
            libpyuhd=SimpleNamespace(types=SimpleNamespace(tune_request=lambda freq: freq)),
        )
        self.driver_patch = patch.object(sdr, "uhd", fake_uhd)
        self.driver_patch.start()
        self.addCleanup(self.driver_patch.stop)
        # Setup delays are irrelevant to fake transport tests.
        self.wait_patch = patch.object(self.controller.bg_stop_event, "wait", return_value=False)
        self.wait_patch.start()
        self.addCleanup(self.wait_patch.stop)

    def test_transport_timeout_reconnects_before_any_send(self) -> None:
        new_device = Mock()
        first_rx = self.device.get_rx_stream.return_value
        self.device.get_tx_stream.side_effect = RuntimeError(TRANSPORT_TIMEOUT)
        self.controller._connect = Mock(side_effect=lambda: setattr(self.controller, "usrp", new_device))

        tx, rx = self.controller._open_linear_modulation_streams(2.4e9, 10e6, 20, 20)

        self.controller._connect.assert_called_once()
        self.assertIs(tx, new_device.get_tx_stream.return_value)
        self.assertIs(rx, new_device.get_rx_stream.return_value)
        first_rx.issue_stream_cmd.assert_called_once()
        new_device.set_tx_rate.assert_called_once_with(10e6, 0)
        tx.send.assert_not_called()
        self.assertEqual(self.controller.bg_stop_event.wait.call_args_list[0].args, (0.3,))

    def test_persistent_timeout_preserves_real_error_and_never_sends_payload(self) -> None:
        self.device.get_tx_stream.side_effect = RuntimeError(TRANSPORT_TIMEOUT)
        self.controller._connect = Mock(side_effect=lambda: setattr(self.controller, "usrp", self.device))
        with patch.object(self.controller, "_run_modulation_once", wraps=self.controller._run_modulation_once) as run:
            result = self.controller.adaptive_modulation_transmit("test")

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["stage"], "stream_initialization")
        self.assertEqual(result["adaptive_stage"], "probe")
        self.assertFalse(result["tx_started"])
        self.assertIn(TRANSPORT_TIMEOUT, result["error"])
        self.assertEqual(self.device.get_tx_stream.call_count, 2)
        self.controller._connect.assert_called_once()
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args.kwargs["text"], "channel_probe")
        self.assertEqual(self.controller.get_visualization_snapshot()["error"], result["error"])
        self.assertFalse(self.controller.has_background_task())

    def test_unrelated_setup_error_is_not_retried(self) -> None:
        self.device.get_tx_stream.side_effect = RuntimeError("invalid stream configuration")
        self.controller._connect = Mock()
        result = self.controller.adaptive_modulation_transmit("test")
        self.assertEqual(result["status"], "error")
        self.assertEqual(self.device.get_tx_stream.call_count, 1)
        self.controller._connect.assert_not_called()
        self.assertFalse(self.controller.hw_lock.locked())

    def test_stop_during_settling_does_not_open_or_send(self) -> None:
        def stop(_timeout):
            self.controller.bg_stop_event.set()
            return True
        self.controller.bg_stop_event.wait.side_effect = stop
        result = self.controller.adaptive_modulation_transmit("test")
        self.assertTrue(result["stopped"])
        self.device.get_rx_stream.assert_not_called()
        self.device.get_tx_stream.assert_not_called()
        self.assertFalse(self.controller.has_background_task())

    def test_formal_send_error_never_replays_probe_or_payload(self) -> None:
        self.controller._run_modulation_once = Mock(side_effect=[
            {"status": "success", "decoded_ok": True, "estimated_snr_db": 12},
            RuntimeError("send timeout after partial transmission"),
        ])
        result = self.controller.adaptive_modulation_transmit("test")
        self.assertEqual(result["adaptive_stage"], "payload")
        self.assertEqual(result["status"], "error")
        self.assertEqual(self.controller._run_modulation_once.call_count, 2)
        self.assertFalse(self.controller.has_background_task())

    def test_failed_decode_is_explicit_and_keeps_conservative_modulation(self) -> None:
        self.controller._run_modulation_once = Mock(return_value={
            "status": "success", "decoded_ok": False, "estimated_snr_db": 25,
        })
        result = self.controller.adaptive_modulation_transmit("test")
        self.assertEqual(result["selected_modulation"], "BPSK")
        self.assertFalse(result["decoded_ok"])
        self.assertIn("未通过", result["message"])


if __name__ == "__main__":
    unittest.main()
