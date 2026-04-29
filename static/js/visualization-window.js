const VISUALIZATION_URL = "/api/visualization";
const STOP_HARDWARE_URL = "/api/hardware/stop";
const HARDWARE_STATUS_URL = "/api/hardware_status";
const EMBEDDED_MODE = new URLSearchParams(window.location.search).get("embedded") === "1";

let visualizationState = normalizeVisualizationState();
let hardwareBusy = false;
let lastRenderedFrameId = -1;

const iqCanvas = document.getElementById("iq-canvas");
const fftCanvas = document.getElementById("fft-canvas");
const statusText = document.getElementById("viz-status-text");
const stopBtn = document.getElementById("viz-stop-btn");
const refreshBtn = document.getElementById("viz-refresh-btn");

function normalizeVisualizationState(raw = {}) {
    return {
        active: Boolean(raw.active),
        task: raw.task ?? null,
        activation_id: Number(raw.activation_id || 0),
        frame_id: Number(raw.frame_id || 0),
        center_freq_hz: raw.center_freq_hz == null ? null : Number(raw.center_freq_hz),
        tone_freq_hz: raw.tone_freq_hz == null ? null : Number(raw.tone_freq_hz),
        sample_rate_hz: raw.sample_rate_hz == null ? null : Number(raw.sample_rate_hz),
        iq_inphase: Array.isArray(raw.iq_inphase) ? raw.iq_inphase : [],
        iq_quadrature: Array.isArray(raw.iq_quadrature) ? raw.iq_quadrature : [],
        fft_freq_khz: Array.isArray(raw.fft_freq_khz) ? raw.fft_freq_khz : [],
        fft_magnitude_db: Array.isArray(raw.fft_magnitude_db) ? raw.fft_magnitude_db : [],
        rx_power_db: raw.rx_power_db == null ? null : Number(raw.rx_power_db),
        peak_freq_khz: raw.peak_freq_khz == null ? null : Number(raw.peak_freq_khz),
        error: raw.error ?? null,
    };
}

function formatHz(value) {
    if (value == null || Number.isNaN(Number(value))) return "--";
    const numeric = Number(value);
    if (Math.abs(numeric) >= 1e9) return `${(numeric / 1e9).toFixed(3)} GHz`;
    if (Math.abs(numeric) >= 1e6) return `${(numeric / 1e6).toFixed(3)} MHz`;
    if (Math.abs(numeric) >= 1e3) return `${(numeric / 1e3).toFixed(3)} kHz`;
    return `${numeric.toFixed(2)} Hz`;
}

function formatDb(value) {
    if (value == null || Number.isNaN(Number(value))) return "--";
    return `${Number(value).toFixed(2)} dB`;
}

function formatTaskName(task) {
    if (!task) return "--";
    const known = {
        tone_loopback: "Tone 回环",
    };
    return known[task] || String(task).replace(/_/g, " ");
}

function setText(id, value) {
    const el = document.getElementById(id);
    if (el) el.textContent = value;
}

function hasFrameData() {
    return (
        Number(visualizationState.frame_id || 0) > 0 &&
        Array.isArray(visualizationState.iq_inphase) &&
        visualizationState.iq_inphase.length > 0 &&
        Array.isArray(visualizationState.fft_magnitude_db) &&
        visualizationState.fft_magnitude_db.length > 0
    );
}

function ensureCanvasSurface(canvas) {
    if (!canvas) return null;
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;

    const ratio = Math.max(window.devicePixelRatio || 1, 1);
    const rect = canvas.getBoundingClientRect();
    const cssWidth = Math.max(Math.round(rect.width || 1), 1);
    const cssHeight = Math.max(Math.round(rect.height || 1), 1);
    const pixelWidth = cssWidth * ratio;
    const pixelHeight = cssHeight * ratio;

    if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
        canvas.width = pixelWidth;
        canvas.height = pixelHeight;
    }

    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    return { ctx, width: cssWidth, height: cssHeight };
}

function getChartMargins(width) {
    if (width < 520) {
        return { top: 22, right: 16, bottom: 42, left: 48 };
    }
    return { top: 24, right: 24, bottom: 46, left: 58 };
}

function makeLinearTicks(minValue, maxValue, count, formatter) {
    const min = Number(minValue);
    const max = Number(maxValue);
    if (!Number.isFinite(min) || !Number.isFinite(max)) return [];
    if (Math.abs(max - min) < 1e-9) {
        return [{ value: min, label: formatter(min) }];
    }

    const ticks = [];
    for (let index = 0; index < count; index += 1) {
        const ratio = index / Math.max(count - 1, 1);
        const value = min + (max - min) * ratio;
        ticks.push({ value, label: formatter(value) });
    }
    return ticks;
}

function makeIndexTicks(length) {
    const maxIndex = Math.max(Number(length || 0) - 1, 0);
    return makeLinearTicks(0, maxIndex, maxIndex > 1 ? 4 : 2, (value) => `${Math.round(value)}`);
}

function formatAxisValue(value) {
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) return "--";
    if (Math.abs(numeric) >= 100) return numeric.toFixed(0);
    if (Math.abs(numeric) >= 10) return numeric.toFixed(1);
    return numeric.toFixed(2);
}

function defaultSpectrumHalfSpanKHz() {
    const sampleRate = Number(visualizationState.sample_rate_hz);
    if (Number.isFinite(sampleRate) && sampleRate > 0) {
        return Math.max(sampleRate / 2000, 0.5);
    }
    return 500;
}

function drawChartChrome(ctx, width, height, options = {}) {
    const {
        xLabel = "",
        yLabel = "",
        xTicks = [],
        yTicks = [],
        minX = 0,
        maxX = 1,
        minY = -1,
        maxY = 1,
    } = options;
    const margin = getChartMargins(width);
    const plot = {
        x: margin.left,
        y: margin.top,
        width: Math.max(width - margin.left - margin.right, 1),
        height: Math.max(height - margin.top - margin.bottom, 1),
    };

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#020617";
    ctx.fillRect(0, 0, width, height);

    ctx.strokeStyle = "rgba(148, 163, 184, 0.16)";
    ctx.lineWidth = 1;
    xTicks.forEach((tick) => {
        const ratio = (Number(tick.value) - minX) / Math.max(maxX - minX, 1e-9);
        const pos = plot.x + ratio * plot.width + 0.5;
        ctx.beginPath();
        ctx.moveTo(pos, plot.y);
        ctx.lineTo(pos, plot.y + plot.height);
        ctx.stroke();
    });
    yTicks.forEach((tick) => {
        const ratio = (Number(tick.value) - minY) / Math.max(maxY - minY, 1e-9);
        const pos = plot.y + plot.height - ratio * plot.height + 0.5;
        ctx.beginPath();
        ctx.moveTo(plot.x, pos);
        ctx.lineTo(plot.x + plot.width, pos);
        ctx.stroke();
    });

    ctx.strokeStyle = "rgba(226, 232, 240, 0.62)";
    ctx.beginPath();
    ctx.moveTo(plot.x, plot.y);
    ctx.lineTo(plot.x, plot.y + plot.height);
    ctx.lineTo(plot.x + plot.width, plot.y + plot.height);
    ctx.stroke();

    ctx.fillStyle = "rgba(203, 213, 225, 0.92)";
    ctx.font = "11px Consolas, monospace";
    ctx.textBaseline = "top";
    xTicks.forEach((tick) => {
        const ratio = (Number(tick.value) - minX) / Math.max(maxX - minX, 1e-9);
        const pos = plot.x + ratio * plot.width;
        ctx.textAlign = ratio <= 0.05 ? "left" : ratio >= 0.95 ? "right" : "center";
        ctx.fillText(tick.label, pos, plot.y + plot.height + 8);
    });

    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    yTicks.forEach((tick) => {
        const ratio = (Number(tick.value) - minY) / Math.max(maxY - minY, 1e-9);
        const pos = plot.y + plot.height - ratio * plot.height;
        ctx.fillText(tick.label, plot.x - 8, pos);
    });

    ctx.fillStyle = "rgba(148, 163, 184, 0.95)";
    ctx.textAlign = "center";
    ctx.textBaseline = "bottom";
    ctx.fillText(xLabel, plot.x + plot.width / 2, height - 8);

    ctx.save();
    ctx.translate(14, plot.y + plot.height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(yLabel, 0, 0);
    ctx.restore();

    return plot;
}

function drawPlaceholder(ctx, plot, text) {
    ctx.fillStyle = "rgba(148, 163, 184, 0.88)";
    ctx.font = "13px Consolas, monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(text, plot.x + plot.width / 2, plot.y + plot.height / 2);
}

function drawPolyline(ctx, plot, xValues, yValues, color, minX, maxX, minY, maxY) {
    if (!Array.isArray(yValues) || yValues.length === 0) return;

    const safeXValues = Array.isArray(xValues) && xValues.length === yValues.length
        ? xValues
        : yValues.map((_, index) => index);

    const xSpan = Math.max(maxX - minX, 1e-9);
    const ySpan = Math.max(maxY - minY, 1e-9);
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.6;
    ctx.beginPath();

    let started = false;
    yValues.forEach((rawValue, index) => {
        const xValue = Number(safeXValues[index]);
        const yValue = Number(rawValue);
        if (!Number.isFinite(xValue) || !Number.isFinite(yValue)) return;

        const x = plot.x + ((xValue - minX) / xSpan) * plot.width;
        const y = plot.y + plot.height - ((yValue - minY) / ySpan) * plot.height;
        if (!started) {
            ctx.moveTo(x, y);
            started = true;
        } else {
            ctx.lineTo(x, y);
        }
    });

    ctx.stroke();
}

function drawWaveformCanvas(canvas, iSeries, qSeries) {
    const surface = ensureCanvasSurface(canvas);
    if (!surface) return;

    const { ctx, width, height } = surface;
    const maxIndex = Math.max(Math.max(iSeries.length, qSeries.length) - 1, 1);
    const yMin = -1.2;
    const yMax = 1.2;
    const plot = drawChartChrome(ctx, width, height, {
        xLabel: "采样点",
        yLabel: "幅度",
        xTicks: makeIndexTicks(Math.max(iSeries.length, qSeries.length)),
        yTicks: makeLinearTicks(yMin, yMax, 5, formatAxisValue),
        minX: 0,
        maxX: maxIndex,
        minY: yMin,
        maxY: yMax,
    });

    const hasData = Array.isArray(iSeries) && iSeries.length > 0 && Array.isArray(qSeries) && qSeries.length > 0;
    if (!hasData) {
        drawPlaceholder(ctx, plot, "等待波形采样数据");
        return;
    }

    drawPolyline(ctx, plot, null, iSeries, "#38bdf8", 0, maxIndex, yMin, yMax);
    drawPolyline(ctx, plot, null, qSeries, "#34d399", 0, maxIndex, yMin, yMax);

    ctx.fillStyle = "#38bdf8";
    ctx.font = "12px Consolas, monospace";
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    ctx.fillText("I", plot.x + 8, plot.y + 8);
    ctx.fillStyle = "#34d399";
    ctx.fillText("Q", plot.x + 24, plot.y + 8);
}

function drawSpectrumCanvas(canvas, freqs, magnitudes) {
    const surface = ensureCanvasSurface(canvas);
    if (!surface) return;

    const { ctx, width, height } = surface;
    const numericFreqs = Array.isArray(freqs) ? freqs.map(Number).filter(Number.isFinite) : [];
    const numericMagnitudes = Array.isArray(magnitudes) ? magnitudes.map(Number).filter(Number.isFinite) : [];
    const defaultHalfSpan = defaultSpectrumHalfSpanKHz();
    const minFreq = numericFreqs.length ? Math.min(...numericFreqs) : -defaultHalfSpan;
    const maxFreq = numericFreqs.length ? Math.max(...numericFreqs) : defaultHalfSpan;
    const minMag = numericMagnitudes.length ? Math.min(...numericMagnitudes) : -120;
    const maxMag = numericMagnitudes.length ? Math.max(...numericMagnitudes) : 0;
    const paddedMinMag = minMag === maxMag ? minMag - 1 : minMag;
    const paddedMaxMag = minMag === maxMag ? maxMag + 1 : maxMag;
    const plot = drawChartChrome(ctx, width, height, {
        xLabel: "频率偏移 (kHz)",
        yLabel: "幅度 (dB)",
        xTicks: makeLinearTicks(minFreq, maxFreq, 5, formatAxisValue),
        yTicks: makeLinearTicks(paddedMinMag, paddedMaxMag, 5, formatAxisValue),
        minX: minFreq,
        maxX: maxFreq,
        minY: paddedMinMag,
        maxY: paddedMaxMag,
    });

    const hasData = Array.isArray(freqs) && freqs.length > 0 && Array.isArray(magnitudes) && magnitudes.length > 0;
    if (!hasData) {
        drawPlaceholder(ctx, plot, "等待频谱采样数据");
        return;
    }

    drawPolyline(ctx, plot, freqs, magnitudes, "#f59e0b", minFreq, maxFreq, paddedMinMag, paddedMaxMag);

    ctx.fillStyle = "#f59e0b";
    ctx.font = "12px Consolas, monospace";
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    ctx.fillText("FFT", plot.x + 8, plot.y + 8);
}

function updateWindowSummary() {
    const taskName = formatTaskName(visualizationState.task);

    setText("viz-task-name", taskName);
    setText("viz-center-freq", formatHz(visualizationState.center_freq_hz));
    setText("viz-tone-freq", formatHz(visualizationState.tone_freq_hz));
    setText("viz-sample-rate", formatHz(visualizationState.sample_rate_hz));
    setText("viz-rx-power", formatDb(visualizationState.rx_power_db));
    setText(
        "viz-peak-freq",
        visualizationState.peak_freq_khz == null
            ? "--"
            : `${Number(visualizationState.peak_freq_khz).toFixed(2)} kHz`,
    );

    if (stopBtn) {
        const canStop = visualizationState.active || hardwareBusy;
        stopBtn.disabled = !canStop;
        stopBtn.textContent = canStop ? "停止任务" : "当前无任务";
    }

    if (statusText) {
        if (visualizationState.active) {
            statusText.textContent = hasFrameData()
                ? `任务 ${taskName} 正在推送实时波形与频谱数据。`
                : `任务 ${taskName} 已启动，正在等待第一帧 IQ / FFT 数据。`;
        } else if (hardwareBusy) {
            statusText.textContent = "硬件任务正在运行或启动中，可点击停止任务释放 USRP。";
        } else if (visualizationState.error) {
            statusText.textContent = `上一次可视化任务异常结束：${visualizationState.error}`;
        } else {
            statusText.textContent = "空闲中。从主控台发起可视化任务后，波形和频谱会显示在这里。";
        }
    }

    document.title = visualizationState.active
        ? `${taskName} | SDR 可视化监视窗`
        : "SDR 可视化监视窗";
}

function renderPlots(force = false) {
    const frameId = Number(visualizationState.frame_id || 0);
    if (!force && frameId === lastRenderedFrameId) return;

    drawWaveformCanvas(iqCanvas, visualizationState.iq_inphase || [], visualizationState.iq_quadrature || []);
    drawSpectrumCanvas(fftCanvas, visualizationState.fft_freq_khz || [], visualizationState.fft_magnitude_db || []);
    lastRenderedFrameId = frameId;
}

async function fetchVisualizationState() {
    try {
        const res = await fetch(VISUALIZATION_URL, { cache: "no-store" });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        visualizationState = normalizeVisualizationState(await res.json());
    } catch (error) {
        console.error("Failed to fetch visualization state:", error);
        visualizationState = normalizeVisualizationState({
            error: "Visualization feed unavailable",
        });
    } finally {
        updateWindowSummary();
        renderPlots();
    }
}

async function fetchHardwareBusyState() {
    try {
        const res = await fetch(HARDWARE_STATUS_URL, { cache: "no-store" });
        if (!res.ok) return;
        const data = await res.json();
        hardwareBusy = Boolean(
            data.diagnostics &&
            data.diagnostics["USRP-01"] &&
            data.diagnostics["USRP-01"].status === "active",
        );
    } catch (error) {
        console.error("Failed to fetch hardware status:", error);
    } finally {
        updateWindowSummary();
    }
}

async function stopHardwareTask() {
    if (stopBtn) {
        stopBtn.disabled = true;
        stopBtn.textContent = "停止中...";
    }

    try {
        await fetch(STOP_HARDWARE_URL, { method: "POST" });
    } catch (error) {
        console.error("Failed to stop hardware task:", error);
    } finally {
        await fetchHardwareBusyState();
        await fetchVisualizationState();
    }
}

if (refreshBtn) {
    refreshBtn.addEventListener("click", () => {
        fetchVisualizationState();
    });
}

if (stopBtn) {
    stopBtn.addEventListener("click", () => {
        stopHardwareTask();
    });
}

window.addEventListener("resize", () => {
    renderPlots(true);
});

window.addEventListener("beforeunload", () => {
    if (EMBEDDED_MODE || (!visualizationState.active && !hardwareBusy) || !navigator.sendBeacon) return;
    navigator.sendBeacon(
        STOP_HARDWARE_URL,
        new Blob(["{}"], { type: "application/json" }),
    );
});

window.onload = () => {
    if (EMBEDDED_MODE) {
        document.body.classList.add("embedded-mode");
    }
    updateWindowSummary();
    renderPlots(true);
    fetchVisualizationState();
    fetchHardwareBusyState();
    setInterval(fetchVisualizationState, 250);
    setInterval(fetchHardwareBusyState, 1000);
};
