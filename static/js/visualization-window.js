const VISUALIZATION_URL = "/api/visualization";
const STOP_HARDWARE_URL = "/api/hardware/stop";
const HARDWARE_STATUS_URL = "/api/hardware_status";
const EMBEDDED_MODE = new URLSearchParams(window.location.search).get("embedded") === "1";

let visualizationState = normalizeVisualizationState();
let hardwareBusy = false;
let lastRenderedPlotKey = "";

const iqCanvas = document.getElementById("iq-canvas");
const fftCanvas = document.getElementById("fft-canvas");
const statusText = document.getElementById("viz-status-text");
const stopBtn = document.getElementById("viz-stop-btn");
const refreshBtn = document.getElementById("viz-refresh-btn");
const cognitiveView = new CognitiveSelectionView();

function normalizeVisualizationState(raw = {}) {
    return {
        active: Boolean(raw.active),
        task: raw.task ?? null,
        task_status: raw.task_status ?? null,
        modulation_scheme: raw.modulation_scheme ?? null,
        decoded_ok: raw.decoded_ok ?? null,
        captured_samples: Number(raw.captured_samples || 0),
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
        cognitive_scan: raw.cognitive_scan ?? null,
    };
}

function formatHz(value) {
    if (value == null || Number.isNaN(Number(value))) return "--";
    const numeric = Number(value);
    if (Math.abs(numeric) >= 1e9) return `${(numeric / 1e9).toFixed(7)} GHz`;
    if (Math.abs(numeric) >= 1e6) return `${(numeric / 1e6).toFixed(4)} MHz`;
    if (Math.abs(numeric) >= 1e3) return `${(numeric / 1e3).toFixed(4)} kHz`;
    return `${numeric.toFixed(4)} Hz`;
}

function formatDb(value) {
    if (value == null || Number.isNaN(Number(value))) return "--";
    return `${Number(value).toFixed(4)} dB`;
}

function formatTaskName(task) {
    if (!task) return "--";
    const known = {
        tone_loopback: "Tone 回环",
        text_transmit: "固定调制文本收发",
        adaptive_transmit: "自适应文本收发",
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

function makeSpectrumFrequencyAxis(freqs, sampleRateHz) {
    const sampleRate = Number(sampleRateHz);
    const numericFreqs = Array.isArray(freqs) ? freqs.map(Number).filter(Number.isFinite) : [];
    let halfSpan = 500;
    if (Number.isFinite(sampleRate) && sampleRate > 0) {
        halfSpan = sampleRate / 2000;
    } else if (numericFreqs.some((value) => value !== 0)) {
        halfSpan = Math.max(...numericFreqs.map(Math.abs));
    }
    // FFT bins exclude the positive Nyquist endpoint for even-sized FFTs.
    // Use the symmetric sampling band, so its midpoint is exactly DC.
    const ticks = [-1, -0.5, 0, 0.5, 1].map((fraction) => {
        const value = fraction * halfSpan;
        return {
            value,
            label: value === 0 ? "0 (中心)" : `${value > 0 ? "+" : "-"}${formatAxisValue(Math.abs(value))}`,
            emphasized: value === 0,
        };
    });
    return { min: -halfSpan, max: halfSpan, ticks };
}

function updateSpectrumFrequencyReference() {
    const centerHz = visualizationState.center_freq_hz;
    const center = centerHz == null ? "待获取射频中心频率" : `0 kHz 对应射频中心 ${formatHz(centerHz)}`;
    const peakKHz = visualizationState.peak_freq_khz;
    let peak = "";
    if (hasFrameData() && peakKHz != null) {
        const offset = `${peakKHz > 0 ? "+" : ""}${Number(peakKHz).toFixed(4)} kHz`;
        peak = ` 当前峰值偏移：${offset}`;
        if (centerHz != null) peak += `（射频 ${formatHz(centerHz + peakKHz * 1000)}）`;
        peak += "。";
    }
    setText("fft-frequency-reference", `${center}；负值表示低于中心，正值表示高于中心。${peak}`);
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
        ctx.strokeStyle = tick.emphasized ? "rgba(56, 189, 248, 0.55)" : "rgba(148, 163, 184, 0.16)";
        const ratio = (Number(tick.value) - minX) / Math.max(maxX - minX, 1e-9);
        const pos = plot.x + ratio * plot.width + 0.5;
        ctx.beginPath();
        ctx.moveTo(pos, plot.y);
        ctx.lineTo(pos, plot.y + plot.height);
        ctx.stroke();
    });
    ctx.strokeStyle = "rgba(148, 163, 184, 0.16)";
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
        ctx.fillStyle = tick.emphasized ? "#7dd3fc" : "rgba(203, 213, 225, 0.92)";
        ctx.font = tick.emphasized ? "bold 11px Consolas, monospace" : "11px Consolas, monospace";
        const ratio = (Number(tick.value) - minX) / Math.max(maxX - minX, 1e-9);
        const pos = plot.x + ratio * plot.width;
        ctx.textAlign = ratio <= 0.05 ? "left" : ratio >= 0.95 ? "right" : "center";
        ctx.fillText(tick.label, pos, plot.y + plot.height + 8);
    });

    ctx.fillStyle = "rgba(203, 213, 225, 0.92)";
    ctx.font = "11px Consolas, monospace";
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
    const numericMagnitudes = Array.isArray(magnitudes) ? magnitudes.map(Number).filter(Number.isFinite) : [];
    const axis = makeSpectrumFrequencyAxis(freqs, visualizationState.sample_rate_hz);
    const minFreq = axis.min;
    const maxFreq = axis.max;
    const minMag = numericMagnitudes.length ? Math.min(...numericMagnitudes) : -120;
    const maxMag = numericMagnitudes.length ? Math.max(...numericMagnitudes) : 0;
    const paddedMinMag = minMag === maxMag ? minMag - 1 : minMag;
    const paddedMaxMag = minMag === maxMag ? maxMag + 1 : maxMag;
    const plot = drawChartChrome(ctx, width, height, {
        xLabel: "相对中心频率的偏移 (kHz)",
        yLabel: "幅度 (dB)",
        xTicks: axis.ticks,
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
    updateSpectrumFrequencyReference();
    const packet = ["text_transmit", "adaptive_transmit"].includes(visualizationState.task);
    const scan = !visualizationState.active && !packet && visualizationState.cognitive_scan?.run_id ? visualizationState.cognitive_scan : null;
    const taskName = scan ? (scan.status === "scanning" ? "认知选频" : "上次认知选频") : formatTaskName(visualizationState.task);

    setText("viz-task-name", taskName);
    setText("viz-center-freq", formatHz(scan ? scan.current_freq_hz ?? scan.best_freq_hz ?? scan.center_freq_hz : visualizationState.center_freq_hz));
    setText("viz-tone-freq", formatHz(scan ? null : visualizationState.tone_freq_hz));
    setText("viz-sample-rate", formatHz(scan ? scan.sample_rate_hz : visualizationState.sample_rate_hz));
    setText("viz-rx-power", formatDb(scan ? scan.min_power_db : visualizationState.rx_power_db));
    setText("viz-rx-power-label", scan ? "最低接收功率" : packet ? "片段接收功率" : "接收功率");
    setText(
        "viz-peak-freq",
        scan || visualizationState.peak_freq_khz == null
            ? "--"
            : `${Number(visualizationState.peak_freq_khz).toFixed(4)} kHz`,
    );

    if (stopBtn) {
        const canStop = visualizationState.active || scan?.status === "scanning" || hardwareBusy;
        stopBtn.disabled = !canStop;
        stopBtn.textContent = canStop ? "停止任务" : "当前无任务";
    }

    if (statusText) {
        if (visualizationState.cognitive_scan?.status === "scanning") {
            statusText.textContent = "正在扫描并比较候选频点，认知选频过程显示在下方。";
        } else if (packet) {
            const modulation = visualizationState.modulation_scheme || "--";
            const phase = visualizationState.task_status === "stopped" ? "已停止" : "已完成";
            const decode = visualizationState.decoded_ok == null ? "" : visualizationState.decoded_ok ? "解码成功。" : "未成功解码。";
            statusText.textContent = visualizationState.active
                ? `${taskName}（${modulation}）正在发送和接收，完成后显示本次接收采样快照。`
                : visualizationState.task_status === "error"
                    ? `${taskName}异常结束：${visualizationState.error || "未取得有效结果"}`
                    : `${taskName}（${modulation}）${phase}。${decode}${hasFrameData() ? "IQ / FFT 显示本次接收数据中能量最高片段的快照。" : "本次没有可显示的有效 IQ / FFT 采样。"}`;
        } else if (visualizationState.active) {
            statusText.textContent = hasFrameData()
                ? `任务 ${taskName} 正在推送实时波形与频谱数据。`
                : `任务 ${taskName} 已启动，正在等待第一帧 IQ / FFT 数据。`;
        } else if (hardwareBusy) {
            statusText.textContent = "硬件任务正在运行或启动中，可点击停止任务释放 USRP。";
        } else if (visualizationState.error) {
            statusText.textContent = `上一次可视化任务异常结束：${visualizationState.error}`;
        } else if (scan) {
            statusText.textContent = `这里保留上次认知选频的${scan.status === "completed" ? "完整" : "部分"}结果，可回放比较过程。`;
        } else {
            statusText.textContent = "空闲中。从主控台发起可视化任务后，波形和频谱会显示在这里。";
        }
    }

    document.title = visualizationState.active
        ? `${taskName} | SDR 可视化监视窗`
        : "SDR 可视化监视窗";
}

function renderPlots(force = false) {
    const plotKey = `${visualizationState.activation_id}:${visualizationState.frame_id}:${visualizationState.sample_rate_hz}:${visualizationState.iq_inphase.length}:${visualizationState.fft_magnitude_db.length}`;
    if (!force && plotKey === lastRenderedPlotKey) return;

    drawWaveformCanvas(iqCanvas, visualizationState.iq_inphase || [], visualizationState.iq_quadrature || []);
    drawSpectrumCanvas(fftCanvas, visualizationState.fft_freq_khz || [], visualizationState.fft_magnitude_db || []);
    lastRenderedPlotKey = plotKey;
}

async function fetchVisualizationState() {
    try {
        const res = await fetch(VISUALIZATION_URL, { cache: "no-store" });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        visualizationState = normalizeVisualizationState(await res.json());
        cognitiveView.update(visualizationState.cognitive_scan, {
            historical: ["text_transmit", "adaptive_transmit", "tone_loopback"].includes(visualizationState.task),
        });
    } catch (error) {
        console.error("Failed to fetch visualization state:", error);
        cognitiveView.markUnavailable();
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
