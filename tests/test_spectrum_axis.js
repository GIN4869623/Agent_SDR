const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const source = fs.readFileSync(path.join(__dirname, "../static/js/visualization-window.js"), "utf8");
const elements = new Map([["fft-frequency-reference", { textContent: "" }]]);
const context = {
    window: { devicePixelRatio: 1 },
    document: { getElementById: (id) => elements.get(id) },
    visualizationState: {
        center_freq_hz: 2.385e9,
        sample_rate_hz: 1e6,
        frame_id: 1,
        iq_inphase: [1],
        fft_magnitude_db: [0],
        peak_freq_khz: 100,
    },
};
vm.createContext(context);
vm.runInContext(source.slice(
    source.indexOf("function normalizeVisualizationState("),
    source.indexOf("async function fetchVisualizationState("),
), context);

// The real even-length FFT frequency range used to produce a -0.24 kHz
// midpoint. Odd-length FFTs and other rates must have the same exact zero.
for (const samples of [1024, 2048, 2049]) {
    for (const sampleRate of [100e3, 1e6, 10e6]) {
        const freqs = Array.from({ length: samples }, (_, index) =>
            (index - Math.floor(samples / 2)) * sampleRate / samples / 1000);
        const axis = context.makeSpectrumFrequencyAxis(freqs, sampleRate);
        assert.equal(axis.min, -sampleRate / 2000);
        assert.equal(axis.max, sampleRate / 2000);
        assert.equal(axis.ticks[2].value, 0);
        assert.equal(axis.ticks[2].label, "0 (中心)");
        assert.equal(axis.ticks[2].emphasized, true);
    }
}
const missingRate = context.makeSpectrumFrequencyAxis([-500, 499.51171875], null);
assert.equal(missingRate.min, -500);
assert.equal(missingRate.max, 500);
assert.equal(context.makeSpectrumFrequencyAxis([], null).ticks[2].label, "0 (中心)");

// Run the real Canvas renderer and verify the center label's actual position
// at desktop and embedded/mobile sizes, rather than only testing tick data.
for (const width of [360, 900]) {
    const labels = [];
    const ctx = {
        setTransform() {}, clearRect() {}, fillRect() {}, beginPath() {},
        moveTo() {}, lineTo() {}, stroke() {}, save() {}, restore() {},
        translate() {}, rotate() {},
        fillText(label, x, y) { labels.push({ label, x, y }); },
    };
    const canvas = { getContext: () => ctx, getBoundingClientRect: () => ({ width, height: 280 }) };
    context.drawSpectrumCanvas(canvas, [-500, 0, 100, 499.51171875], [-80, -60, 0, -80]);
    const margin = context.getChartMargins(width);
    const center = labels.find((label) => label.label === "0 (中心)");
    assert.equal(center.x, margin.left + (width - margin.left - margin.right) / 2);
    assert.ok(labels.some((label) => label.label === "相对中心频率的偏移 (kHz)"));
    assert.ok(labels.some((label) => label.label === "+250"));
    assert.ok(!labels.some((label) => label.label === "-0.24"));
}

context.updateSpectrumFrequencyReference();
const reference = elements.get("fft-frequency-reference");
assert.ok(reference.textContent.includes("0 kHz 对应射频中心 2.3850000 GHz"));
assert.ok(reference.textContent.includes("+100.0000 kHz（射频 2.3851000 GHz）"));
// Genuine negative peak offsets remain measurements, separate from the axis.
context.visualizationState.peak_freq_khz = -0.244140625;
context.updateSpectrumFrequencyReference();
assert.ok(reference.textContent.includes("峰值偏移：-0.2441 kHz"));
context.visualizationState.center_freq_hz = null;
context.visualizationState.frame_id = 0;
context.updateSpectrumFrequencyReference();
assert.ok(reference.textContent.includes("待获取射频中心频率"));
assert.ok(!reference.textContent.includes("峰值"));

console.log("FFT axis and frequency-reference regression checks passed.");
