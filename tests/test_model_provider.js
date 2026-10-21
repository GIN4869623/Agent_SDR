const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const path = require("node:path");
const source = fs.readFileSync(path.join(__dirname, "../static/js/app.js"), "utf8");
const elements = new Map();
for (const id of ["model-toolbar", "local-provider-btn", "deepseek-provider-btn", "local-model-modes", "think-mode-btn", "fast-mode-btn", "temperature-control"]) {
    elements.set(id, { classList: { toggle() {} }, setAttribute(name, value) { this[name] = value; } });
}
const storage = new Map();
const messages = [];
const requests = [];
const context = {
    console, isSending: false, currentMode: "think", localModelMode: "think", configuredModelProvider: "local",
    configuredModelProviders: {
        local: { configured: true, models: { think: "local-think", fast: "local-fast" } },
        deepseek: { configured: false, models: { think: "cloud-think", fast: "cloud-fast" }, message: "请补全 .env 中的 DEEPSEEK_API_KEY" },
    },
    configuredModelMap: {}, currentActiveModel: "", currentTemperature: 0.2,
    temperatureSlider: {}, activeModelName: {}, activeModelHint: {},
    MODEL_PROVIDER_KEY: "provider", HEALTH_URL: "/api/health", API_STREAM_URL: "/api/chat/stream",
    sessionId: "test", userInput: { value: "你好", focus() {} }, sendBtn: {},
    localStorage: { getItem: (key) => storage.get(key), setItem: (key, value) => storage.set(key, value) },
    document: { getElementById: (id) => elements.get(id) },
    appendMessage: (role, message) => messages.push(message),
    appendStreamingAIMessage: () => ({ querySelector: () => ({ innerHTML: "" }) }),
    performance: { now: () => 0 }, TextDecoder, clearTimeout, setTimeout,
    fetchHardwareStatus() {}, fetchVisualizationState() {},
    fetch: async (url, options) => {
        requests.push(JSON.parse(options.body));
        return { ok: true, body: { getReader: () => ({ read: async () => ({ done: true }) }) } };
    },
};
vm.createContext(context);
vm.runInContext(
    source.slice(source.indexOf("function updateActiveModelUI("), source.indexOf("function formatHz(")) +
    source.slice(source.indexOf("function setModelMode("), source.indexOf("function updatePopoverUI(")) +
    source.slice(source.indexOf("async function sendMessage("), source.indexOf('document.addEventListener("keydown"')),
    context,
);

async function main() {
    context.setModelProvider("deepseek");
    assert.equal(elements.get("model-toolbar")["data-provider"], "deepseek");
    assert.equal(storage.get("provider"), "deepseek");
    assert.equal(context.activeModelName.textContent, "cloud-think");
    assert.equal(context.temperatureSlider.disabled, true);
    assert.equal(elements.get("local-model-modes").hidden, true);
    assert.equal(elements.get("temperature-control").hidden, true);
    assert.equal(elements.get("local-provider-btn").textContent, "本地");
    assert.equal(elements.get("deepseek-provider-btn").textContent, "云端");
    assert.equal(elements.get("deepseek-provider-btn").title, "");
    assert.equal(context.activeModelHint.textContent, "");
    assert.equal(context.activeModelHint.hidden, true);
    await context.sendMessage();
    assert.equal(requests.length, 0);
    assert.equal(context.userInput.value, "你好");
    assert.equal(messages[0], "云端模型暂不可用。");
    context.configuredModelProviders.deepseek.configured = true;
    await context.sendMessage();
    assert.equal(requests[0].provider, "deepseek");
    context.setModelProvider("local");
    assert.equal(elements.get("model-toolbar")["data-provider"], "local");
    assert.equal(context.activeModelName.textContent, "local-think");
    assert.equal(elements.get("local-model-modes").hidden, false);
    assert.equal(elements.get("temperature-control").hidden, false);
    assert.equal(context.temperatureSlider.disabled, false);
    context.setModelMode("fast");
    context.setModelProvider("deepseek");
    assert.equal(elements.get("local-model-modes").hidden, true);
    assert.equal(elements.get("temperature-control").hidden, true);
    assert.equal(context.currentMode, "think");
    context.setModelMode("fast");
    assert.equal(context.currentMode, "think", "cloud has a fixed mode");
    assert.equal(context.activeModelHint.hidden, false);
    assert.equal(context.activeModelHint.textContent, "云端");
    context.setModelProvider("local");
    assert.equal(context.currentMode, "fast", "local mode is restored");
    assert.equal(context.activeModelName.textContent, "local-fast");
    context.userInput.value = "再试一次";
    await context.sendMessage();
    assert.equal(requests[1].provider, "local");
    context.isSending = true;
    context.setModelProvider("deepseek");
    assert.equal(context.configuredModelProvider, "local");
    context.isSending = false;
    context.fetch = async () => ({ ok: true, json: async () => ({ llm_provider: "deepseek", providers: context.configuredModelProviders }) });
    await context.fetchHealthConfig();
    assert.equal(context.configuredModelProvider, "local", "refresh preserves browser selection");
    console.log("本地/云端文案、模式显隐、模式恢复、配置提示移除和请求路由检查通过。");
}
main().catch((error) => { console.error(error); process.exitCode = 1; });
