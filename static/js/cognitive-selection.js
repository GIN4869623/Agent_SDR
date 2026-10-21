/* Uses measured snapshots only. Demo and replay never issue hardware commands. */
class CognitiveSelectionView {
    constructor() {
        this.real = { run_id: 0, status: "idle", candidates: [] };
        this.demo = null;
        this.limit = null;
        this.timer = null;
        this.signature = "";
        this.historical = false;
        this.element = (id) => document.getElementById(`cognitive-${id}`);
        this.element("demo").addEventListener("click", () => this.showDemo());
        this.element("live").addEventListener("click", () => {
            this.stopReplay();
            this.demo = null;
            this.limit = null;
            this.render();
        });
        this.element("replay").addEventListener("click", () => {
            if (this.timer) {
                this.stopReplay();
                this.render();
            } else {
                this.startReplay();
            }
        });
        this.element("step").addEventListener("input", (event) => {
            this.stopReplay();
            this.limit = Number(event.target.value);
            this.render();
        });
        this.render();
        if (new URLSearchParams(window.location.search).get("demo") === "1") this.showDemo();
    }

    update(scan, { historical = false } = {}) {
        if (!scan) return;
        const signature = JSON.stringify({ scan, historical });
        if (signature === this.signature) return;
        this.signature = signature;
        this.historical = historical;
        if (scan.run_id !== this.real.run_id || scan.status === "scanning") {
            this.stopReplay();
            this.demo = null;
            this.limit = null;
        }
        this.real = scan;
        this.render();
    }

    markUnavailable() {
        this.signature = "";
        this.element("status").textContent = "实测数据连接中断，下面保留上一次收到的结果。回放和示例演示仍可使用。";
        this.element("status").hidden = false;
    }

    stopReplay() {
        clearInterval(this.timer);
        this.timer = null;
    }

    showDemo() {
        if (this.real.status === "scanning") return;
        this.demo = {
            status: "completed", center_freq_hz: 2.4e9, search_span_hz: 30e6,
            step_hz: 5e6, sample_rate_hz: 1e6, samples_per_candidate: 1024,
            candidates: [-46.2, -58.5, -52.1, -71.8, -63.4, -49.7, -55.3].map((power, i) => ({
                freq_hz: 2.385e9 + i * 5e6, power_db: power, status: "measured",
            })),
        };
        this.startReplay();
    }

    data() { return this.demo || this.real; }

    attempted() {
        return this.data().candidates.filter((point) => ["measured", "invalid"].includes(point.status)).length;
    }

    startReplay() {
        if (this.data().status === "scanning" || !this.attempted()) return;
        this.stopReplay();
        this.limit = 0;
        this.render();
        this.timer = setInterval(() => {
            this.limit += 1;
            if (this.limit >= this.attempted()) this.stopReplay();
            this.render();
        }, 650);
        this.render();
    }

    visibleCandidates() {
        let attempted = 0;
        return this.data().candidates.map((point) => {
            if (["measured", "invalid"].includes(point.status)) attempted += 1;
            return this.limit !== null && attempted > this.limit
                ? { ...point, power_db: null, status: "pending" } : point;
        });
    }

    render() {
        const scan = this.data();
        const points = this.visibleCandidates();
        const measured = points.filter((point) => point.status === "measured" && Number.isFinite(point.power_db));
        const best = measured.reduce((minimum, point) => !minimum || point.power_db < minimum.power_db ? point : minimum, null);
        const count = points.filter((point) => ["measured", "invalid"].includes(point.status)).length;
        const invalid = points.filter((point) => point.status === "invalid").length;
        const replaying = this.limit !== null && this.limit < this.attempted();
        const complete = scan.status === "completed" && !replaying;
        const mhz = (freq) => (freq / 1e6).toFixed(4);
        this.element("best-label").textContent = complete ? "最终选中的频点" : "当前最低功率频点";
        this.element("best").textContent = best ? `${mhz(best.freq_hz)} MHz` : "--";
        this.element("power").textContent = best ? `${best.power_db.toFixed(4)} dB` : "--";
        this.element("count").textContent = `${count} / ${points.length}`;
        this.element("progress").max = Math.max(points.length, 1);
        this.element("progress").value = count;
        let status = "";
        if (scan.status === "scanning") status = `正在扫描${scan.current_freq_hz ? ` ${mhz(scan.current_freq_hz)} MHz` : "候选频点"}；每获得一项有效测量，就与当前最低功率比较。`;
        else if (replaying) status = `比较过程回放：已查看 ${count} 个频点。${best ? `当前保留 ${mhz(best.freq_hz)} MHz（${best.power_db.toFixed(4)} dB）。` : "等待第一项有效测量。"}`;
        else if (complete && best) status = `扫描完成：${mhz(best.freq_hz)} MHz 的平均接收功率最低，因此选中该频点。${invalid ? `其中 ${invalid} 项无效采样已排除。` : ""}`;
        else if (scan.status === "stopped") status = "扫描已停止。这里保留已测量的结果，未继续执行发送；当前最低值只代表已扫描部分。";
        else if (scan.status === "error") status = `扫描失败：${scan.error || "没有有效测量"}。已测量的部分结果仅供查看。`;
        this.element("status").textContent = (this.demo ? "示例演示，不代表实际环境。" : "") + status;
        this.element("status").hidden = !this.element("status").textContent;
        this.element("replay").disabled = scan.status === "scanning" || this.attempted() === 0;
        this.element("replay").textContent = this.timer ? "暂停回放" : "回放比较过程";
        this.element("demo").disabled = this.real.status === "scanning";
        this.element("live").hidden = !this.demo;
        this.element("step").max = this.attempted();
        this.element("step").value = this.limit === null ? this.attempted() : this.limit;
        this.element("step").disabled = scan.status === "scanning" || this.attempted() === 0;
        this.element("step-value").textContent = this.element("step").value;
        this.drawChart(points, best);
        this.element("table").replaceChildren(...points.map((point) => {
            const row = document.createElement("tr");
            const isBest = point === best;
            const label = isBest ? (complete ? "最终选中" : "当前最低") : point.status === "measured" ? "功率较高或相同" : point.status === "invalid" ? "无效采样，已排除" : point.status === "measuring" ? "正在测量" : "尚未测量";
            [mhz(point.freq_hz), point.power_db == null ? "--" : point.power_db.toFixed(4), label].forEach((text) => {
                const cell = document.createElement("td"); cell.textContent = text; row.appendChild(cell);
            });
            if (isBest) row.classList.add("cognitive-selected");
            return row;
        }));
    }

    drawChart(points, best) {
        const svg = this.element("chart");
        if (!points.length) {
            svg.innerHTML = '<text x="380" y="150" text-anchor="middle" fill="#94a3b8">等待扫描结果，或点击“示例演示”</text>';
            return;
        }
        const values = points.filter((point) => point.power_db != null).map((point) => point.power_db);
        const low = values.length ? Math.min(...values) : -100;
        const high = values.length ? Math.max(...values) : -20;
        const padding = Math.max(high - low, 0.001) * 0.2;
        const min = low - padding;
        const max = high + padding;
        const x = (i) => 75 + (i + 0.5) * 650 / points.length;
        const y = (power) => 245 - (power - min) / (max - min) * 205;
        const labelsEvery = Math.max(1, Math.ceil(points.length / 10));
        let content = '<text x="15" y="20" fill="#94a3b8" font-size="12">平均接收功率</text>';
        for (let i = 0; i <= 4; i += 1) {
            const power = min + (max - min) * i / 4;
            content += `<line x1="75" y1="${y(power)}" x2="725" y2="${y(power)}" stroke="#243247"/><text x="65" y="${y(power) + 4}" text-anchor="end" fill="#94a3b8" font-size="11">${power.toFixed(4)}</text>`;
        }
        points.forEach((point, i) => {
            const measured = point.power_db != null;
            const color = point === best ? "#34d399" : point.status === "invalid" ? "#f87171" : point.status === "measuring" ? "#f59e0b" : "#38bdf8";
            const cy = measured ? y(point.power_db) : 245;
            const title = `${(point.freq_hz / 1e6).toFixed(4)} MHz：${measured ? `${point.power_db.toFixed(4)} dB` : point.status === "invalid" ? "无效采样" : "待测量"}`;
            content += `<circle cx="${x(i)}" cy="${cy}" r="${point === best ? 8 : 5}" fill="${measured ? color : "#0f172a"}" stroke="${color}" stroke-width="2"><title>${title}</title></circle>`;
            if (measured && (points.length <= 10 || point === best)) content += `<text x="${x(i)}" y="${cy - 15}" text-anchor="middle" fill="${color}" font-size="12">${point.power_db.toFixed(4)}</text>`;
            if (i % labelsEvery === 0 || i === points.length - 1) content += `<text x="${x(i)}" y="268" text-anchor="middle" fill="#94a3b8" font-size="12">${(point.freq_hz / 1e6).toFixed(1)}</text>`;
        });
        content += '<text x="400" y="293" text-anchor="middle" fill="#94a3b8" font-size="12">候选中心频率（MHz）</text>';
        svg.innerHTML = content;
    }
}
