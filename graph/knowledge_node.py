from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import ToolMessage

from config.settings import Settings
from core.rag import LocalKnowledgeBase
from graph.state import SDRState
from prompts.system_prompts import build_knowledge_agent_system_prompt


CONSULT_KNOWLEDGE_NAME = "consult_knowledge_agent"
BAILIAN_COMPATIBLE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def _analyze_hardware_results(
    client: Any,
    results: list[dict[str, Any]],
    model: str,
) -> str:
    results_json = json.dumps(results, ensure_ascii=False, indent=2)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": build_knowledge_agent_system_prompt()},
                {
                    "role": "user",
                    "content": f"请分析以下 SDR 硬件执行结果并给出建议：\n{results_json}",
                },
            ],
            temperature=0.3,
            stream=False,
        )
        return response.choices[0].message.content or ""
    except Exception:
        return ""


def _rule_based_suggestions(results: list[dict[str, Any]]) -> str:
    suggestions: list[str] = []
    for item in results:
        result = item.get("result", {})
        if result.get("status") == "error":
            error = result.get("error", "")
            if "硬件正忙" in error:
                suggestions.append("USRP 正忙，已自动尝试停止后台任务并重试。如仍失败，请手动停止当前任务。")
            elif "连接" in error or "timeout" in error.lower():
                suggestions.append("USRP 连接异常，请检查网线连接、设备 IP 和 UHD 驱动状态。")
            elif "UHD" in error:
                suggestions.append("UHD 驱动异常，请确认 USRP 驱动安装正确且设备已上电。")

        snr = result.get("estimated_snr_db") or result.get("probe_snr_db")
        if snr is not None:
            try:
                snr_val = float(snr)
                if snr_val < -18:
                    suggestions.append(f"SNR 仅为 {snr_val:.2f} dB，信道质量很差，建议尝试更换频点或降低符号率。")
                elif snr_val < -10:
                    suggestions.append(f"SNR 为 {snr_val:.2f} dB，信道质量一般，可尝试适当提高增益或更换频点。")
            except (TypeError, ValueError):
                pass

        decoded_ok = result.get("decoded_ok")
        if decoded_ok is False:
            suggestions.append(
                "接收端解码失败，可能原因：符号同步偏差、频率偏移过大、增益不足或信道噪声过高。"
                "建议提高符号率、增加 preamble 长度或调整 RX 增益。"
            )

    if not suggestions:
        return "当前硬件结果无明显异常，无需特殊建议。"
    return "\n".join(f"- {s}" for s in suggestions)


def _extract_query_and_flag(tool_calls: list[Any]) -> tuple[str, bool]:
    for tc in tool_calls:
        # Normalise: LangGraph-converted format {"name": ..., "args": {...}}
        # vs OpenAI format {"function": {"name": ..., "arguments": "..."}}
        if isinstance(tc, dict):
            if "function" in tc:
                fn = tc["function"]
                name = fn.get("name", "")
                raw_args = fn.get("arguments", "{}")
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except json.JSONDecodeError:
                    args = {}
            else:
                name = tc.get("name", "")
                args = tc.get("args", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
        else:
            name = getattr(tc, "name", "")
            raw_args = getattr(tc, "args", "{}") if hasattr(tc, "args") else "{}"
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except json.JSONDecodeError:
                args = {}

        if name == CONSULT_KNOWLEDGE_NAME:
            return args.get("query", ""), bool(args.get("analyze_results", False))
    return "", False


def create_knowledge_node(
    knowledge_base: LocalKnowledgeBase,
    bailian_api_key: str,
    settings: Settings,
):
    bailian_client = None
    try:
        from openai import OpenAI
        if bailian_api_key:
            bailian_client = OpenAI(
                api_key=bailian_api_key,
                base_url=BAILIAN_COMPATIBLE_URL,
            )
    except Exception:
        pass

    def knowledge_node(state: SDRState) -> dict[str, Any]:
        messages = state.get("messages", [])
        logs: list[str] = list(state.get("tool_logs", []))
        last_message = messages[-1] if messages else None
        if last_message is None:
            return {"next_action": "respond"}

        raw_tool_calls = getattr(last_message, "tool_calls", None) or []
        if isinstance(last_message, dict):
            raw_tool_calls = last_message.get("tool_calls", [])

        query, analyze_results = _extract_query_and_flag(raw_tool_calls)

        knowledge_text = ""

        # Mode 1: RAG retrieval
        if query:
            rag_result = knowledge_base.query(query)
            if rag_result:
                knowledge_text += rag_result + "\n\n"
                logs.append("知识代理已完成知识库检索。")

        # Mode 2: Hardware result analysis via Bailian chat
        hw_results = state.get("hardware_results", [])
        if analyze_results and hw_results and bailian_client is not None:
            analysis_model = getattr(settings, "bailian_embedding_model", "qwen-plus") or "qwen-plus"
            analysis = _analyze_hardware_results(bailian_client, hw_results, analysis_model)
            if analysis:
                knowledge_text += "【硬件结果分析】\n" + analysis + "\n\n"
                logs.append("知识代理已完成硬件结果分析。")

        # Rule-based suggestions (only when there are hardware results to analyze)
        if hw_results:
            suggestions = _rule_based_suggestions(hw_results)
            if suggestions:
                knowledge_text += "【建议】\n" + suggestions + "\n"

        if not knowledge_text and query:
            knowledge_text = f"知识库未检索到与「{query}」直接匹配的内容，请基于已有知识回答。"
        elif not knowledge_text:
            knowledge_text = "知识代理本次未检索到相关内容。"

        tool_call_id = ""
        for tc in raw_tool_calls:
            if isinstance(tc, dict):
                name = tc.get("function", {}).get("name", "") if "function" in tc else tc.get("name", "")
                if name == CONSULT_KNOWLEDGE_NAME:
                    tool_call_id = tc.get("id", "")
                    break
            elif hasattr(tc, "name") and getattr(tc, "name", "") == CONSULT_KNOWLEDGE_NAME:
                tool_call_id = getattr(tc, "id", "")
                break

        tool_msg = ToolMessage(
            content=knowledge_text.strip(),
            tool_call_id=tool_call_id,
            name=CONSULT_KNOWLEDGE_NAME,
        )

        return {
            "messages": [tool_msg],
            "knowledge_context": knowledge_text.strip(),
            "tool_logs": logs,
            "next_action": "respond",
        }

    return knowledge_node
