const API_STREAM_URL = "/api/chat/stream";
const HEALTH_URL = "/api/health";
const RESET_URL = "/api/session/reset";
const HARDWARE_STATUS_URL = "/api/hardware_status";
const VISUALIZATION_URL = "/api/visualization";
const STOP_HARDWARE_URL = "/api/hardware/stop";
const VISUALIZATION_WINDOW_URL = "/visualization.html";
const VISUALIZATION_EMBED_URL = "/visualization.html?embedded=1";
const VISUALIZATION_WINDOW_NAME = "sdr-visualization-window";
const SESSION_KEY = "ai_sdr_session_id";
const TEMPERATURE_KEY = "ai_sdr_temperature";
const DEFAULT_TEMPERATURE = 0.20;
const MIN_TEMPERATURE = 0.0;
const MAX_TEMPERATURE = 2.0;

let isSending = false;
let conversationHistory = [];
let currentMode = "think";
let hardwareDiagnostics = {};
let currentPopoverDevice = null;
let sessionId = getOrCreateSessionId();
let currentTemperature = getStoredTemperature();
let configuredModelMap = {};
let currentActiveModel = "";
let visualizationState = normalizeVisualizationState();
let visualizationWindowRef = null;
let pendingVisualizationAutoOpen = false;
let lastVisualizationActivationId = 0;
let visualizationStateInitialized = false;

const chatBox = document.getElementById("chat-box");
const userInput = document.getElementById("user-input");
const sendBtn = document.getElementById("send-btn");
const vizStatusText = document.getElementById("viz-status-text");
const vizStopBtn = document.getElementById("viz-stop-btn");
const vizOpenBtn = document.getElementById("viz-open-btn");
const vizModal = document.getElementById("viz-modal");
const vizModalStatus = document.getElementById("viz-modal-status");
const vizModalFrame = document.getElementById("viz-modal-frame");
const vizModalStopBtn = document.getElementById("viz-modal-stop-btn");
const temperatureSlider = document.getElementById("temperature-slider");
const temperatureValue = document.getElementById("temperature-value");
const activeModelName = document.getElementById("active-model-name");
const activeModelHint = document.getElementById("active-model-hint");

const USER_AVATAR = "assets/njupt.png";
const USER_AVATAR_FALLBACK = "https://api.iconify.design/mdi/account-circle.svg?color=%230ea5e9";
const AI_AVATAR = "https://api.iconify.design/fluent-emoji/robot.svg";
const THINK_TAG_PAIRS = [
    ["<redacted_thinking>", "</redacted_thinking>"],
    ["<think>", "</think>"],
];

function getOrCreateSessionId() {
    const cached = localStorage.getItem(SESSION_KEY);
    if (cached) return cached;

    const nextId = typeof crypto !== "undefined" && crypto.randomUUID
        ? crypto.randomUUID()
        : `sess-${Date.now().toString(36)}`;
    localStorage.setItem(SESSION_KEY, nextId);
    return nextId;
}

function clampTemperature(value) {
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) return DEFAULT_TEMPERATURE;
    return Math.max(MIN_TEMPERATURE, Math.min(MAX_TEMPERATURE, numeric));
}

function formatTemperature(value) {
    return clampTemperature(value).toFixed(2);
}

function getStoredTemperature() {
    const cached = localStorage.getItem(TEMPERATURE_KEY);
    if (!cached) return DEFAULT_TEMPERATURE;
    return clampTemperature(cached);
}

function updateTemperatureUI() {
    if (temperatureSlider) temperatureSlider.value = formatTemperature(currentTemperature);
    if (temperatureValue) temperatureValue.textContent = formatTemperature(currentTemperature);
}

function setTemperature(value, { persist = true } = {}) {
    currentTemperature = clampTemperature(value);
    if (persist) {
        localStorage.setItem(TEMPERATURE_KEY, formatTemperature(currentTemperature));
    }
    updateTemperatureUI();
}

function updateActiveModelUI({ modelName = null, hint = null } = {}) {
    if (modelName) currentActiveModel = String(modelName);
    const configuredModel = configuredModelMap[currentMode] || "";
    const displayName = modelName || currentActiveModel || configuredModel || "--";

    if (activeModelName) activeModelName.textContent = displayName;
    if (!activeModelHint) return;

    if (hint) {
        activeModelHint.textContent = hint;
        return;
    }

    if (modelName) {
        activeModelHint.textContent = `实际运行: ${currentMode}`;
    } else if (configuredModel) {
        activeModelHint.textContent = `当前模式 ${currentMode} 的配置模型`;
    } else {
        activeModelHint.textContent = "等待后端配置";
    }
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
        tx_samples: Number(raw.tx_samples || 0),
        rx_frames: Number(raw.rx_frames || 0),
        error: raw.error ?? null,
    };
}

function escapeHtml(str) {
    return String(str)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
}

function parseThinkTags(full) {
    const text = String(full);
    for (const [open, close] of THINK_TAG_PAIRS) {
        const openIndex = text.indexOf(open);
        if (openIndex === -1) continue;

        const afterOpen = text.slice(openIndex + open.length);
        const closeIndex = afterOpen.indexOf(close);
        if (closeIndex === -1) {
            return {
                matched: true,
                phase: "thinking",
                preamble: text.slice(0, openIndex),
                think: afterOpen,
                answer: "",
            };
        }

        return {
            matched: true,
            phase: "done",
            preamble: text.slice(0, openIndex),
            think: afterOpen.slice(0, closeIndex),
            answer: afterOpen.slice(closeIndex + close.length),
        };
    }

    return { matched: false, phase: "plain", preamble: "", think: "", answer: text };
}

function formatAiBubbleHtml(raw) {
    const text = String(raw);
    if (!text.includes("<")) {
        return escapeHtml(text).replace(/\n/g, "<br>");
    }

    const parsed = parseThinkTags(text);
    const esc = (value) => escapeHtml(value).replace(/\n/g, "<br>");

    if (!parsed.matched || parsed.phase === "plain") {
        return esc(parsed.answer);
    }

    const preambleHtml = parsed.preamble.trim()
        ? `<div class="think-preamble">${esc(parsed.preamble)}</div>`
        : "";

    if (parsed.phase === "thinking") {
        return (
            preambleHtml +
            `<div class="think-block"><div class="think-label">推理过程</div><div class="think-body">${esc(parsed.think)}</div></div>` +
            `<div class="answer-block"><div class="think-pending-hint">推理结束后将显示正式回答。</div></div>`
        );
    }

    const thinkHtml = parsed.think.trim()
        ? `<div class="think-block"><div class="think-label">推理过程</div><div class="think-body">${esc(parsed.think)}</div></div>`
        : "";
    const answerHtml = parsed.answer.trim()
        ? `<div class="answer-block"><div class="answer-label">回答</div><div class="answer-body">${esc(parsed.answer)}</div></div>`
        : "";

    return preambleHtml + thinkHtml + answerHtml;
}

function appendStreamingAIMessage() {
    const wrapperDiv = document.createElement("div");
    wrapperDiv.className = "message ai-wrapper";
    wrapperDiv.innerHTML = `<img class="avatar ai-avatar" src="${AI_AVATAR}" alt="AI">
        <div class="msg-content">
            <div class="msg-bubble streaming-bubble"></div>
            <div class="msg-timing" hidden></div>
            <div class="hw-terminal streaming-hw-log" style="display:none"></div>
        </div>`;
    chatBox.appendChild(wrapperDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
    return wrapperDiv;
}

function formatDurationMs(value) {
    const numeric = Number(value);
    if (!Number.isFinite(numeric) || numeric < 0) return null;
    if (numeric >= 1000) return `${(numeric / 1000).toFixed(2)} s`;
    return `${Math.round(numeric)} ms`;
}

function buildTimingSummary(timing, fallbackMs, statusLabel) {
    const parts = [];
    const totalLabel = formatDurationMs(
        timing && typeof timing.total_ms === "number" ? timing.total_ms : fallbackMs,
    );
    if (totalLabel) parts.push(`总耗时 ${totalLabel}`);

    if (timing && timing.route) {
        parts.push(`路由 ${timing.route}`);
    }
    if (timing && Number(timing.knowledge_lookup_ms) > 0) {
        parts.push(`检索 ${formatDurationMs(timing.knowledge_lookup_ms)}`);
    }
    if (timing && Number(timing.tool_decision_ms) > 0) {
        parts.push(`规划 ${formatDurationMs(timing.tool_decision_ms)}`);
    }
    if (timing && Number(timing.tool_execution_ms) > 0) {
        parts.push(`技能 ${formatDurationMs(timing.tool_execution_ms)}`);
    }
    if (timing && Number(timing.final_reply_ms) > 0) {
        parts.push(`回复 ${formatDurationMs(timing.final_reply_ms)}`);
    }
    if (timing && Number(timing.first_token_ms) > 0) {
        parts.push(`首字 ${formatDurationMs(timing.first_token_ms)}`);
    }
    if (timing && Number(timing.tool_count) > 0) {
        parts.push(`工具 ${Number(timing.tool_count)} 次`);
    }
    if (statusLabel) {
        parts.push(statusLabel === "failed" ? "失败" : statusLabel);
    }

    return parts.join(" | ");
}

function showRoundTiming(streamingRoot, startMark, statusLabel, timing = null) {
    const el = streamingRoot.querySelector(".msg-timing");
    if (!el) return;
    const totalMs = performance.now() - startMark;
    el.textContent = buildTimingSummary(timing, totalMs, statusLabel);
    el.hidden = false;
    chatBox.scrollTop = chatBox.scrollHeight;
}

function appendMessage(sender, text, hwLogs = []) {
    const wrapperDiv = document.createElement("div");
    wrapperDiv.className = `message ${sender === "user" ? "user-wrapper" : "ai-wrapper"}`;

    const bubbleInner = sender === "ai"
        ? formatAiBubbleHtml(text)
        : escapeHtml(text).replace(/\n/g, "<br>");

    let contentHtml = `<div class="msg-content"><div class="msg-bubble">${bubbleInner}</div>`;
    if (hwLogs.length > 0) {
        contentHtml += "<div class=\"hw-terminal\">";
        hwLogs.forEach((log) => {
            contentHtml += `<div class="log-line">-> ${escapeHtml(log)}</div>`;
        });
        contentHtml += "</div>";
    }
    contentHtml += "</div>";

    if (sender === "user") {
        wrapperDiv.innerHTML = `${contentHtml}<img class="avatar user-avatar" src="${USER_AVATAR}" alt="User" onerror="this.onerror=null;this.src='${USER_AVATAR_FALLBACK}'">`;
    } else {
        wrapperDiv.innerHTML = `<img class="avatar ai-avatar" src="${AI_AVATAR}" alt="AI">${contentHtml}`;
    }

    chatBox.appendChild(wrapperDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
}

function applyStreamingHwLog(container, line) {
    const term = container.querySelector(".streaming-hw-log");
    if (!term || !line) return;
    term.style.display = "block";
    const row = document.createElement("div");
    row.className = "log-line";
    row.textContent = `-> ${line}`;
    term.appendChild(row);
    chatBox.scrollTop = chatBox.scrollHeight;
}

function setModelMode(mode) {
    currentMode = mode === "fast" ? "fast" : "think";
    updateModeUI();
    updateActiveModelUI();
}

function updateModeUI() {
    const thinkBtn = document.getElementById("think-mode-btn");
    const fastBtn = document.getElementById("fast-mode-btn");

    if (thinkBtn) {
        thinkBtn.classList.toggle("active", currentMode === "think");
        thinkBtn.setAttribute("aria-pressed", currentMode === "think" ? "true" : "false");
    }
    if (fastBtn) {
        fastBtn.classList.toggle("active", currentMode === "fast");
        fastBtn.setAttribute("aria-pressed", currentMode === "fast" ? "true" : "false");
    }
}

async function fetchHealthConfig() {
    try {
        const res = await fetch(HEALTH_URL, { cache: "no-store" });
        if (!res.ok) return;
        const data = await res.json();
        configuredModelMap = data.models || {};
        updateActiveModelUI();
    } catch (error) {
        console.error("Failed to fetch health config:", error);
        updateActiveModelUI({ hint: "无法读取后端模型配置" });
    }
}

function updatePopoverUI() {
    if (!currentPopoverDevice) return;
    const pop = document.getElementById("cyber-popover");
    if (!pop || pop.classList.contains("hidden")) return;

    const diag = hardwareDiagnostics[currentPopoverDevice];
    if (!diag) return;

    document.getElementById("popover-title").textContent = `${currentPopoverDevice} DIAGNOSTICS`;
    document.getElementById("popover-center-freq").textContent = diag.center;
    document.getElementById("popover-sample-rate").textContent = diag.sample;
    document.getElementById("popover-master-clock").textContent = diag.clock;
    document.getElementById("popover-gain").textContent = diag.gain;
}

function openPopoverForDevice(deviceName) {
    currentPopoverDevice = deviceName;
    updatePopoverUI();
    const pop = document.getElementById("cyber-popover");
    if (pop) {
        pop.classList.remove("hidden");
        pop.setAttribute("aria-hidden", "false");
    }
}

function closePopover() {
    const pop = document.getElementById("cyber-popover");
    if (!pop) return;
    pop.classList.add("hidden");
    pop.setAttribute("aria-hidden", "true");
    currentPopoverDevice = null;
}

function handleUSRPClick(deviceName) {
    openPopoverForDevice(deviceName);
}

function setTextContent(id, value) {
    const el = document.getElementById(id);
    if (el) el.textContent = value;
}

function isVisualizationModalVisible() {
    return Boolean(vizModal) && !vizModal.classList.contains("hidden");
}

function ensureVisualizationFrameLoaded() {
    if (!vizModalFrame) return;
    if (!vizModalFrame.getAttribute("src")) {
        vizModalFrame.setAttribute("src", VISUALIZATION_EMBED_URL);
    }
}

function describeVisualizationState() {
    const taskName = formatTaskName(visualizationState.task);
    if (visualizationState.active && visualizationState.frame_id > 0) {
        return `任务 ${taskName} 正在推送实时波形与频谱数据。`;
    }
    if (visualizationState.active) {
        return `任务 ${taskName} 已启动，正在等待第一帧 IQ / FFT 数据。`;
    }
    if (pendingVisualizationAutoOpen) {
        return "认知选频/可视化任务正在启动，可在这里停止当前硬件任务。";
    }
    if (visualizationState.error) {
        return `上一次可视化任务异常结束：${visualizationState.error}`;
    }
    return "空闲中。下达可视化命令后，前端会自动弹出监视窗口。";
}

function isVisualizationTriggerLog(line) {
    const text = String(line || "");
    return (
        text.includes("tone_loopback_visualize") ||
        (
            text.includes("auto_optimal_transmit") &&
            (
                text.includes("enable_tone") ||
                text.includes("tone_freq_hz") ||
                /Tone|tone|正弦|单音/.test(text)
            )
        ) ||
        (
            text.includes("auto_optimal_transmit") &&
            /可视化|框图|认知|干净信道|最优信道|干净频点|最优频点/.test(text)
        )
    );
}

function hasVisualizationFrameData() {
    const requiresTx = visualizationState.task === "tone_loopback";
    return (
        Number(visualizationState.frame_id || 0) > 0 &&
        (!requiresTx || Number(visualizationState.tx_samples || 0) > 0) &&
        Array.isArray(visualizationState.iq_inphase) &&
        visualizationState.iq_inphase.length > 0 &&
        Array.isArray(visualizationState.fft_magnitude_db) &&
        visualizationState.fft_magnitude_db.length > 0
    );
}

function hasVisualizationWindow() {
    return visualizationWindowRef && !visualizationWindowRef.closed;
}

function showVisualizationModal() {
    if (!vizModal) return;
    ensureVisualizationFrameLoaded();
    vizModal.classList.remove("hidden");
    vizModal.setAttribute("aria-hidden", "false");
    updateVisualizationModalStatus();
    updateVisualizationPanel();
}

function hideVisualizationModal() {
    if (!vizModal) return;
    vizModal.classList.add("hidden");
    vizModal.setAttribute("aria-hidden", "true");
    updateVisualizationPanel();
}

function openVisualizationWindow() {
    const features = [
        "width=1320",
        "height=900",
        "resizable=yes",
        "scrollbars=yes",
        "noopener=no",
    ].join(",");

    if (!hasVisualizationWindow()) {
        visualizationWindowRef = window.open(
            VISUALIZATION_WINDOW_URL,
            VISUALIZATION_WINDOW_NAME,
            features,
        );
    }

    if (hasVisualizationWindow()) {
        visualizationWindowRef.focus();
    }

    showVisualizationModal();
    updateVisualizationPanel();
}

function updateVisualizationModalStatus() {
    if (!vizModalStatus) return;
    vizModalStatus.textContent = describeVisualizationState();
}

function updateVisualizationStopButtons({ disabled, label }) {
    [vizStopBtn, vizModalStopBtn].forEach((button) => {
        if (!button) return;
        button.disabled = disabled;
        button.textContent = label;
    });
}

function updateVisualizationPanel() {
    const active = Boolean(visualizationState.active);
    const hardwareBusy = Boolean(
        hardwareDiagnostics &&
        hardwareDiagnostics["USRP-01"] &&
        hardwareDiagnostics["USRP-01"].status === "active",
    );
    const canStop = active || pendingVisualizationAutoOpen || hardwareBusy;
    const taskName = formatTaskName(visualizationState.task);
    const hasDetachedWindow = hasVisualizationWindow();
    const hasVisibleModal = isVisualizationModalVisible();

    setTextContent("viz-task-name", taskName);
    setTextContent("viz-center-freq", formatHz(visualizationState.center_freq_hz));
    setTextContent("viz-tone-freq", formatHz(visualizationState.tone_freq_hz));
    setTextContent("viz-sample-rate", formatHz(visualizationState.sample_rate_hz));
    setTextContent("viz-rx-power", formatDb(visualizationState.rx_power_db));
    setTextContent(
        "viz-peak-freq",
        visualizationState.peak_freq_khz == null
            ? "--"
            : `${Number(visualizationState.peak_freq_khz).toFixed(2)} kHz`,
    );

    updateVisualizationStopButtons({
        disabled: !canStop,
        label: canStop ? "停止任务" : "当前无任务",
    });

    if (vizOpenBtn) {
        if (active && !hasVisibleModal) {
            vizOpenBtn.textContent = "查看窗口";
        } else if (hasVisibleModal) {
            vizOpenBtn.textContent = hasDetachedWindow ? "窗口已显示" : "重新打开";
        } else {
            vizOpenBtn.textContent = "弹出窗口";
        }
    }

    if (vizStatusText) {
        vizStatusText.textContent = describeVisualizationState();
    }

    updateVisualizationModalStatus();
}

async function fetchHardwareStatus() {
    try {
        const res = await fetch(HARDWARE_STATUS_URL, { cache: "no-store" });
        if (!res.ok) return;

        const data = await res.json();
        setTextContent("deploy-server-ip", `IP: ${data.server_ip}`);
        setTextContent("deploy-usrp1-ip", `IP: ${data.usrp_01_ip}`);
        setTextContent("deploy-usrp2-ip", `IP: ${data.usrp_02_ip}`);

        if (!data.diagnostics) return;
        hardwareDiagnostics = data.diagnostics;
        updatePopoverUI();

        const usrp1 = hardwareDiagnostics["USRP-01"];
        if (!usrp1) return;

        const cardDot = document.getElementById("usrp1-card-dot");
        const cardText = document.getElementById("usrp1-card-text");
        const sceneNode = document.getElementById("usrp1-scene-node");
        if (usrp1.status === "active") {
            if (cardDot) cardDot.className = "status-dot status-active";
            if (cardText) cardText.textContent = "收发中";
            if (sceneNode) sceneNode.className = "scene-node scene-usrp active-txrx";
        } else if (usrp1.status === "offline") {
            if (cardDot) cardDot.className = "status-dot status-offline";
            if (cardText) cardText.textContent = "离线";
            if (sceneNode) sceneNode.className = "scene-node scene-usrp offline-node";
        } else {
            if (cardDot) cardDot.className = "status-dot status-online";
            if (cardText) cardText.textContent = "在线";
            if (sceneNode) sceneNode.className = "scene-node scene-usrp online-node";
        }

        const modBadge = document.getElementById("usrp1-mod-badge");
        if (modBadge && usrp1.modulation) {
            modBadge.textContent = `MODE: ${usrp1.modulation}`;
            modBadge.className = "mod-badge";
            if (usrp1.modulation.includes("BPSK")) modBadge.classList.add("mod-bpsk");
            else if (usrp1.modulation.includes("QPSK")) modBadge.classList.add("mod-qpsk");
            else if (usrp1.modulation.includes("16-QAM")) modBadge.classList.add("mod-16qam");
            else modBadge.classList.add("mod-bpsk");
        }
    } catch (error) {
        console.error("Failed to fetch hardware status:", error);
        setTextContent("deploy-server-ip", "IP: unavailable");
    }
}

async function fetchVisualizationState() {
    const previousActivationId = Number(visualizationState.activation_id || 0);
    const wasActive = Boolean(visualizationState.active);
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
        const activationId = Number(visualizationState.activation_id || 0);
        const hasFrame = hasVisualizationFrameData();
        if (!visualizationStateInitialized) {
            visualizationStateInitialized = true;
            lastVisualizationActivationId = activationId;
            updateVisualizationPanel();
            return;
        }
        const activationChanged =
            activationId !== previousActivationId || activationId !== lastVisualizationActivationId;
        if (activationChanged) {
            lastVisualizationActivationId = activationId;
        }
        const shouldAutoOpenOnReady =
            pendingVisualizationAutoOpen &&
            visualizationState.active &&
            hasFrame &&
            (activationChanged || !wasActive || !isVisualizationModalVisible());
        if (shouldAutoOpenOnReady && !isVisualizationModalVisible()) {
            showVisualizationModal();
            pendingVisualizationAutoOpen = false;
        }
        if (!visualizationState.active) {
            pendingVisualizationAutoOpen = false;
        }
        updateVisualizationPanel();
    }
}

async function stopHardwareTask() {
    updateVisualizationStopButtons({ disabled: true, label: "停止中..." });

    try {
        await fetch(STOP_HARDWARE_URL, { method: "POST" });
    } catch (error) {
        console.error("Failed to stop hardware task:", error);
    } finally {
        await fetchVisualizationState();
        await fetchHardwareStatus();
    }
}

async function clearContext() {
    conversationHistory = [];

    try {
        await fetch(RESET_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ session_id: sessionId }),
        });
    } catch (error) {
        console.error("Failed to clear session:", error);
    }

    appendMessage("ai", "会话上下文已清除，现在可以重新开始新的任务。");
}

async function sendMessage() {
    const text = userInput.value.trim();
    if (!text || isSending) return;

    appendMessage("user", text);
    userInput.value = "";
    isSending = true;
    sendBtn.disabled = true;
    sendBtn.textContent = "发送中...";

    const streamingRoot = appendStreamingAIMessage();
    const bubbleEl = streamingRoot.querySelector(".streaming-bubble");
    const roundStart = performance.now();
    let buffer = "";
    let assembled = "";
    let bubbleRaf = 0;

    const cancelBubbleRaf = () => {
        if (bubbleRaf) {
            clearTimeout(bubbleRaf);
            bubbleRaf = 0;
        }
    };

    // Render in ~30ms batches so fast models still show visible streaming.
    // requestAnimationFrame coalesces too aggressively when tokens arrive
    // in the same 16ms frame (e.g. qwen3.5:122b outputs 50+ tok/s).
    const scheduleBubbleUpdate = () => {
        if (bubbleRaf) return;
        bubbleRaf = setTimeout(() => {
            bubbleRaf = 0;
            bubbleEl.innerHTML = formatAiBubbleHtml(assembled);
            chatBox.scrollTop = chatBox.scrollHeight;
        }, 30);
    };

    try {
        const response = await fetch(API_STREAM_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                instruction: text,
                session_id: sessionId,
                mode: currentMode,
                temperature: currentTemperature,
            }),
        });
        if (!response.ok) throw new Error(`HTTP Error: ${response.status}`);

        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });

            let separatorIndex;
            while ((separatorIndex = buffer.indexOf("\n\n")) !== -1) {
                const block = buffer.slice(0, separatorIndex).trim();
                buffer = buffer.slice(separatorIndex + 2);

                for (const rawLine of block.split("\n")) {
                    const line = rawLine.trim();
                    if (!line.startsWith("data:")) continue;

                    const payload = line.slice(5).trim();
                    if (!payload || payload === "[DONE]") continue;

                    let data;
                    try {
                        data = JSON.parse(payload);
                    } catch (error) {
                        console.warn("SSE JSON parse failed:", payload);
                        continue;
                    }

                    if (data.type === "delta" && data.text) {
                        assembled += data.text;
                        scheduleBubbleUpdate();
                    } else if (data.type === "meta") {
                        if (data.active_mode === "fast" || data.active_mode === "think") {
                            currentMode = data.active_mode;
                            updateModeUI();
                        }
                        if (typeof data.active_temperature === "number") {
                            setTemperature(data.active_temperature);
                        }
                        if (data.active_model) {
                            updateActiveModelUI({ modelName: data.active_model });
                        }
                    } else if (data.type === "tool_log" && data.line) {
                        applyStreamingHwLog(streamingRoot, data.line);
                        if (isVisualizationTriggerLog(data.line)) {
                            pendingVisualizationAutoOpen = true;
                            fetchVisualizationState();
                        }
                    } else if (data.type === "visualization" && data.action === "open") {
                        pendingVisualizationAutoOpen = true;
                        fetchVisualizationState();
                    } else if (data.type === "done" && data.status === "success") {
                        if (data.active_mode === "fast" || data.active_mode === "think") {
                            currentMode = data.active_mode;
                            updateModeUI();
                        }
                        if (typeof data.active_temperature === "number") {
                            setTemperature(data.active_temperature);
                        }
                        if (data.active_model) {
                            updateActiveModelUI({ modelName: data.active_model });
                        }

                        conversationHistory = data.updated_history || [];
                        const reply = data.reply != null ? data.reply : assembled;
                        cancelBubbleRaf();
                        bubbleEl.innerHTML = formatAiBubbleHtml(reply);
                        chatBox.scrollTop = chatBox.scrollHeight;

                        const hwLogs = data.hardware_logs || [];
                        if (hwLogs.length > 0) {
                            const term = streamingRoot.querySelector(".streaming-hw-log");
                            if (term) {
                                term.innerHTML = "";
                                term.style.display = "block";
                                hwLogs.forEach((log) => {
                                    const row = document.createElement("div");
                                    row.className = "log-line";
                                    row.textContent = `-> ${log}`;
                                    term.appendChild(row);
                                });
                            }
                        }

                        showRoundTiming(streamingRoot, roundStart, null, data.timing || null);
                    } else if (data.type === "error") {
                        cancelBubbleRaf();
                        bubbleEl.innerHTML = escapeHtml(`Error: ${data.message || "Unknown error"}`).replace(/\n/g, "<br>");
                        showRoundTiming(streamingRoot, roundStart, "failed");
                    }
                }
            }
        }
    } catch (error) {
        cancelBubbleRaf();
        bubbleEl.innerHTML = escapeHtml(`Request failed: ${error.message}`).replace(/\n/g, "<br>");
        showRoundTiming(streamingRoot, roundStart, "failed");
    } finally {
        isSending = false;
        sendBtn.disabled = false;
        sendBtn.textContent = "发送消息";
        userInput.focus();
        fetchHardwareStatus();
        fetchVisualizationState();
    }
}

document.addEventListener("keydown", (event) => {
    if (event.key !== "Escape") return;
    if (isVisualizationModalVisible()) {
        hideVisualizationModal();
        return;
    }
    closePopover();
});

document.addEventListener("click", (event) => {
    const pop = document.getElementById("cyber-popover");
    if (!pop || pop.classList.contains("hidden")) return;

    const clickedInside = pop.contains(event.target);
    const clickedTrigger = event.target && event.target.closest && event.target.closest(".device-card");
    if (!clickedInside && !clickedTrigger) closePopover();
});

if (userInput) {
    userInput.addEventListener("keypress", (event) => {
        if (event.key === "Enter") sendMessage();
    });
}

if (temperatureSlider) {
    temperatureSlider.addEventListener("input", (event) => {
        setTemperature(event.target.value);
    });
}

window.addEventListener("resize", () => {
    updateVisualizationPanel();
});

window.onload = () => {
    updateTemperatureUI();
    updateModeUI();
    updateActiveModelUI();
    updateVisualizationPanel();
    updateVisualizationModalStatus();
    if (userInput) userInput.focus();
    fetchHealthConfig();
    fetchHardwareStatus();
    fetchVisualizationState();
    setInterval(fetchHardwareStatus, 1000);
    setInterval(fetchVisualizationState, 1000);
};
