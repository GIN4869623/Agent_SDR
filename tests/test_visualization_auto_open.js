const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const source = fs.readFileSync(path.join(__dirname, "../static/js/app.js"), "utf8");
// Execute the actual log and polling handlers with UI/network dependencies stubbed.
const handlers = source.slice(
    source.indexOf("function isVisualizationTriggerLog("),
    source.indexOf("function hasVisualizationFrameData("),
) + source.slice(
    source.indexOf("async function fetchVisualizationState("),
    source.indexOf("async function stopHardwareTask("),
);

async function poll(state, { initialized = true, pending = false, previous = {}, blocked = false } = {}) {
    let opened = 0;
    const context = {
        console,
        VISUALIZATION_URL: "/api/visualization",
        visualizationState: previous,
        visualizationStateInitialized: initialized,
        lastVisualizationActivationId: previous.activation_id || 0,
        pendingVisualizationAutoOpen: pending,
        visualizationAutoOpenBlockedByText: blocked,
        normalizeVisualizationState: (raw) => raw,
        fetch: async () => ({ ok: true, json: async () => state }),
        isPacketVisualization: () => ["text_transmit", "adaptive_transmit"].includes(context.visualizationState.task),
        hasVisualizationFrameData: () => context.visualizationState.frame_id > 0,
        isVisualizationModalVisible: () => opened > 0,
        showVisualizationModal: () => { opened += 1; },
        updateVisualizationPanel: () => {},
    };
    vm.createContext(context);
    vm.runInContext(handlers, context);
    await context.fetchVisualizationState();
    return { opened, context };
}

async function main() {
    for (const task of ["text_transmit", "adaptive_transmit"]) {
        const running = { task, active: true, activation_id: 1, frame_id: 1 };
        assert.equal((await poll(running)).opened, 0, `${task}: new activation`);
        assert.equal((await poll(running, { initialized: false })).opened, 0, `${task}: page reload`);
        assert.equal((await poll(running, { pending: true })).opened, 0, `${task}: pending open event`);
        assert.equal((await poll({ ...running, active: false, task_status: "completed" })).opened, 0, `${task}: completed snapshot`);
        assert.equal((await poll({ ...running, active: false, frame_id: 0, task_status: "error" })).opened, 0, `${task}: failed setup`);
        for (const status of ["scanning", "completed", "error"]) {
            const withHistory = { ...running, cognitive_scan: { run_id: 5, status } };
            assert.equal((await poll(withHistory, { pending: true })).opened, 0, `${task}: scan ${status}`);
            assert.equal((await poll(withHistory, { initialized: false })).opened, 0, `${task}: reload with scan ${status}`);
            assert.equal((await poll({ ...withHistory, active: false })).opened, 0, `${task}: completed with scan ${status}`);
        }
    }
    const { context } = await poll({});
    for (const tool of ["text_fsk_send_and_receive", "adaptive_modulation_transmit"]) {
        assert.equal(context.isVisualizationTriggerLog(`正在调用 ${tool}`), false);
    }
    assert.equal(context.isVisualizationTriggerLog("正在调用 auto_optimal_transmit"), true);
    assert.equal(context.isVisualizationTriggerLog("正在调用 tone_loopback_visualize"), true);
    const scan = { cognitive_scan: { run_id: 1, status: "scanning" } };
    assert.equal((await poll(scan, { blocked: true })).opened, 0, "text call before snapshot updates");
    assert.equal((await poll(scan, { blocked: true, initialized: false })).opened, 0, "text call while initial poll is pending");
    const tone = { task: "tone_loopback", active: true, activation_id: 1, frame_id: 1 };
    assert.equal((await poll(tone, { blocked: true, pending: true })).opened, 0, "delayed Tone response during text call");
    context.handleVisualizationToolCall("adaptive_modulation_transmit");
    assert.equal(context.visualizationAutoOpenBlockedByText, true);
    assert.equal(context.pendingVisualizationAutoOpen, false);
    context.handleVisualizationToolCall("search_sdr_knowledge");
    assert.equal(context.visualizationAutoOpenBlockedByText, true);
    context.handleVisualizationToolCall("tone_loopback_visualize");
    assert.equal(context.visualizationAutoOpenBlockedByText, false);
    assert.equal(context.pendingVisualizationAutoOpen, true);
    context.handleVisualizationToolCall("auto_optimal_transmit", { text: "test" });
    assert.equal(context.visualizationAutoOpenBlockedByText, true);
    context.handleVisualizationToolCall("auto_optimal_transmit", { text: "test", enable_tone: true });
    assert.equal(context.visualizationAutoOpenBlockedByText, false);
    context.handleVisualizationToolCall("auto_optimal_transmit");
    assert.equal(context.visualizationAutoOpenBlockedByText, false);
    assert.equal((await poll(scan)).opened, 1, "new cognitive scan");
    assert.equal((await poll(scan, { initialized: false })).opened, 1, "reload during scan");
    assert.equal((await poll({ task: "tone_loopback", active: true, activation_id: 1, frame_id: 1 }, { pending: true })).opened, 1, "Tone frame ready");
    console.log("Visualization auto-open regression checks passed.");
}

main().catch((error) => { console.error(error); process.exitCode = 1; });
