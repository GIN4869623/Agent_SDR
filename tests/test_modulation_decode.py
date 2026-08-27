from __future__ import annotations

import unittest

import numpy as np

from hardware.sdr_controller import (
    build_packet,
    bytes_to_bits,
    decode_linear_modulated_texts_from_iq,
    hamming_7_4_encode,
    map_bits_to_linear_symbols,
)


class LinearModulationDecodeTests(unittest.TestCase):
    SAMPLE_RATE = 1e6
    SPS = 10

    def make_capture(
        self,
        scheme: str,
        text: str,
        *,
        cfo_hz: float,
        phase_rad: float,
    ) -> np.ndarray:
        bits = hamming_7_4_encode(bytes_to_bits(build_packet(text)))
        symbols = map_bits_to_linear_symbols(bits, scheme)
        packet = np.repeat(symbols, self.SPS).astype(np.complex64) * 0.6
        guard = np.zeros(20_000, dtype=np.complex64)
        capture = np.concatenate([guard, packet, guard, packet, guard])
        sample_index = np.arange(capture.size, dtype=np.float64)
        rotation = np.exp(
            1j
            * (
                phase_rad
                + 2.0 * np.pi * float(cfo_hz) * sample_index / self.SAMPLE_RATE
            )
        )
        rng = np.random.default_rng(20260827)
        noise = (
            rng.normal(size=capture.size) + 1j * rng.normal(size=capture.size)
        ) * 0.005
        return (capture * rotation + noise).astype(np.complex64)

    def test_decodes_repeated_linear_packets_with_phase_and_cfo(self) -> None:
        for scheme in ("BPSK", "QPSK", "16-QAM"):
            with self.subTest(scheme=scheme):
                text = f"{scheme}测试"
                capture = self.make_capture(
                    scheme,
                    text,
                    cfo_hz=1_000.0,
                    phase_rad=0.7,
                )
                decoded, diagnostics = decode_linear_modulated_texts_from_iq(
                    capture,
                    mod_scheme=scheme,
                    sps=self.SPS,
                    samp_rate=self.SAMPLE_RATE,
                )

                self.assertIn(text, decoded)
                self.assertIn("estimated_cfo_hz", diagnostics)
                self.assertLess(
                    abs(float(diagnostics["estimated_cfo_hz"]) - 1_000.0),
                    100.0,
                )


if __name__ == "__main__":
    unittest.main()
