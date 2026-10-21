"""Semantic routing and task planning for the Agent SDR loop.

The router decides which *kind* of workflow should handle a request.  The
skill selector then narrows a hardware workflow to the tools that are valid
for its current stage.  Keeping those decisions separate prevents knowledge
and hardware tools from competing in one unconstrained model call.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

from skill.skill_spec import SkillRegistry, SkillSpec


RouteName = Literal["chat", "knowledge", "hardware"]


@dataclass(frozen=True)
class RouteDecision:
    route: RouteName
    confidence: float
    reason: str
    requires_hardware_action: bool = False
    requires_knowledge: bool = False
    needs_post_analysis: bool = False


@dataclass(frozen=True)
class TaskPlan:
    route: RouteName
    skill: SkillSpec | None
    skill_score: float
    skill_margin: float
    high_confidence: bool
    workflow: str
    initial_allowed_tools: tuple[str, ...]
    required_actions: frozenset[str]
    needs_clarification: bool
    skill_candidates: tuple[tuple[str, float], ...] = ()


_HARDWARE_OBJECT = re.compile(
    r"(usrp|硬件|设备|物理层|tone|正弦|单音|回环|频谱|底噪|噪声|频点|频段|"
    r"信道|无线|文本|消息|调制|fsk|bpsk|qpsk|qam|snr|增益|采样率|视频流|"
    r"video|vlc|gnu\s*radio|波形|可视化)",
    re.I,
)
_SDR_KNOWLEDGE = re.compile(
    r"(sdr|软件无线电|usrp|x300|物理层|频谱|频点|频段|信道|调制|解调|"
    r"fsk|bpsk|qpsk|qam|ofdm|snr|增益|采样率|符号率|带宽|天线|噪声|底噪|"
    r"链路|tone|正弦|视频流|gnu\s*radio|扫描|扫频|认知选频|认知无线)",
    re.I,
)
_EXECUTION = re.compile(
    r"(启动|停止|关闭|终止|释放|执行|发送|发射|传输|收发|扫描|扫频|扫噪|"
    r"测量|测试|运行|打开|显示|切换|找.{0,12}(干净|最优|最佳|最低|最小)|"
    r"选择.{0,8}(频点|信道|频段)|查询.{0,6}(状态|是否运行)|"
    r"查看.{0,6}(状态|频谱|波形|可视化))",
    re.I,
)
_COMMAND_SHAPE = re.compile(
    r"^\s*(?:请|帮我|给我|麻烦|现在|立即|直接)?\s*(?:启动|停止|关闭|终止|释放|"
    r"执行|发送|发射|传输|收发|扫描|扫频|扫噪|测量|测试|运行|打开|显示|切换|"
    r"找|选择|查询|查看)",
    re.I,
)
_NO_EXECUTION = re.compile(
    r"(^\s*(不要|不用|无需|不需要|别).{0,6}(执行|操作|调用|扫描|发送|启动|停止|测量|显示))|"
    r"((只|仅).{0,6}(解释|说明|告诉|介绍))|"
    r"((不要|不用|无需|不需要).{0,4}(执行|操作|调用).{0,6}(硬件|工具))",
    re.I,
)
_EXPLANATION = re.compile(
    r"(什么是|是什么意思|为什么|为何|原理|区别|比较|解释|说明|介绍|怎么理解|"
    r"如何理解|怎么用|如何使用|怎么做|如何实现|操作步骤|文档|手册|资料|"
    r"分析|诊断|原因|建议)",
    re.I,
)
_QUESTION_SHAPE = re.compile(r"^\s*(什么|为什么|为何|怎么|如何|能不能|是否可以|可不可以)", re.I)
_CAPABILITY_QUERY = re.compile(
    r"(你能|能不能|是否可以|可不可以|支持不支持|有什么能力|你会|你可以).{0,20}"
    r"(扫描|发送|硬件|频谱|视频|调制|usrp)?",
    re.I,
)
_SOCIAL_OR_IDENTITY = re.compile(
    r"^(你好|您好|嗨|hello|hi|谢谢|感谢|再见|你是谁|你是什么|你的身份|"
    r"你能做什么|有什么能力)[！!。.?？\s]*$",
    re.I,
)
_REPEAT_ACTION = re.compile(r"^(再来一次|再做一次|重复一次|再发一次|重新执行|继续刚才的|还是刚才那个)", re.I)
_ANALYZE_AFTER_ACTION = re.compile(r"(并|然后|之后|完成后).{0,12}(解释|分析|诊断|说明)", re.I)
_ACTION_AFTER_EXPLANATION = re.compile(
    r"(解释|说明|介绍|分析).{0,30}(并|然后|同时|之后).{0,12}"
    r"(发送|扫描|启动|停止|显示|测量|执行)",
    re.I,
)

_VIDEO_STOP = re.compile(r"(停止|关闭|停掉|终止).{0,8}(视频流|video|vlc)|(视频流|video|vlc).{0,8}(停止|关闭|停掉|终止)", re.I)
_VIDEO_STATUS = re.compile(r"(状态|是否运行|查询).{0,8}(视频流|video|vlc)|(视频流|video|vlc).{0,8}(状态|是否运行)", re.I)
_TEXT_TRANSMIT = re.compile(r"(发送|发射|传输|收发|发出去)|(文本|文字|消息).{0,12}(发送|发射|传输)", re.I)
_NEGATED_TRANSMIT = re.compile(r"(不|不要|无需|不用).{0,6}(发送|发射|传输|收发)", re.I)
_TONE_SIGNAL = re.compile(r"(tone|正弦|单音|连续波)", re.I)
_TONE_TRANSMISSION = re.compile(
    r"(发送|发射|传输|收发|发出去)\s*(?:一个|一段|连续)?\s*"
    r"(?:\d+(?:\.\d+)?\s*(?:hz|khz|mhz)\s*)?"
    r"(?:tone|正弦(?:波)?|单音|连续波)(?:信号)?", re.I,
)
_TEXT_PAYLOAD = re.compile(r"(文本|文字|消息|数据|[\"'“‘])", re.I)
_ADAPTIVE_REQUEST = re.compile(r"(自适应|adaptive|自动选择|自动切换|根据信道|根据链路|根据\s*snr|按\s*snr)", re.I)


def _history_text(history: list[dict[str, str]] | None) -> str:
    recent = history[-4:] if history else []
    return "\n".join(str(item.get("content", "")) for item in recent)


def route_instruction(
    instruction: str,
    history: list[dict[str, str]] | None = None,
) -> RouteDecision:
    """Classify a request conservatively before any tool is exposed."""
    text = instruction.strip()
    if not text or _SOCIAL_OR_IDENTITY.search(text):
        return RouteDecision("chat", 0.98, "普通交流或身份/能力询问")

    if _CAPABILITY_QUERY.search(text):
        return RouteDecision("chat", 0.9, "询问系统能力，没有要求立即执行")

    mentions_hardware = bool(_HARDWARE_OBJECT.search(text))
    asks_explanation = bool(_EXPLANATION.search(text) or _QUESTION_SHAPE.search(text))
    forbids_execution = bool(_NO_EXECUTION.search(text))
    command_shaped = bool(_COMMAND_SHAPE.search(text))
    action_after_explanation = bool(_ACTION_AFTER_EXPLANATION.search(text))
    explicit_execution = bool(
        _EXECUTION.search(text)
        and (command_shaped or not asks_explanation or action_after_explanation)
    )

    if _REPEAT_ACTION.search(text) and _EXECUTION.search(_history_text(history)):
        return RouteDecision(
            "hardware", 0.82, "结合短时记忆，用户要求重复上一项硬件操作",
            requires_hardware_action=True,
        )

    if forbids_execution:
        route: RouteName = "knowledge" if mentions_hardware or _SDR_KNOWLEDGE.search(text) else "chat"
        return RouteDecision(
            route, 0.96, "用户明确禁止执行，只需要说明",
            requires_knowledge=route == "knowledge",
        )

    if explicit_execution and mentions_hardware:
        needs_analysis = bool(
            _ANALYZE_AFTER_ACTION.search(text) or action_after_explanation
        )
        return RouteDecision(
            "hardware", 0.94, "检测到明确的硬件动作和 SDR/设备对象",
            requires_hardware_action=True,
            requires_knowledge=needs_analysis,
            needs_post_analysis=needs_analysis,
        )

    if asks_explanation and (_SDR_KNOWLEDGE.search(text) or mentions_hardware):
        return RouteDecision(
            "knowledge", 0.93, "请求 SDR 概念、方法或结果分析，没有明确执行命令",
            requires_knowledge=True,
        )

    if _SDR_KNOWLEDGE.search(text):
        return RouteDecision(
            "knowledge", 0.76, "包含 SDR 专业对象但没有明确硬件执行动作",
            requires_knowledge=True,
        )

    return RouteDecision("chat", 0.72, "无需硬件副作用或专业资料检索")


def _video_subcommand(instruction: str) -> tuple[tuple[str, ...], frozenset[str]]:
    if _VIDEO_STOP.search(instruction):
        return ("stop_video_stream",), frozenset({"stop_video_stream"})
    if _VIDEO_STATUS.search(instruction):
        return ("video_stream_status",), frozenset({"query_video_status"})
    return ("start_video_stream",), frozenset({"start_video_stream"})


def _dynamic_required_actions(
    skill: SkillSpec,
    instruction: str,
) -> frozenset[str]:
    actions = set(skill.required_actions)
    if skill.name == "auto_optimal_transmit":
        tone_signal_only = (
            bool(_TONE_TRANSMISSION.search(instruction))
            and not _TEXT_PAYLOAD.search(instruction)
            and not _TEXT_TRANSMIT.search(_TONE_TRANSMISSION.sub("", instruction))
        )
        if _TEXT_TRANSMIT.search(instruction) and not _NEGATED_TRANSMIT.search(instruction) and not tone_signal_only:
            actions.add("transmit_text")
        if _TONE_SIGNAL.search(instruction):
            actions.add("visualize_tone")
        if _ADAPTIVE_REQUEST.search(instruction):
            actions.add("select_modulation")
    return frozenset(actions)


def build_task_plan(
    instruction: str,
    route: RouteDecision,
    skill_registry: SkillRegistry | None,
    history: list[dict[str, str]] | None = None,
) -> TaskPlan:
    """Build the initial workflow plan for one user request."""
    planning_instruction = instruction
    if _REPEAT_ACTION.search(instruction):
        for item in reversed(history or []):
            if item.get("role") == "user" and str(item.get("content", "")).strip():
                planning_instruction = (
                    f"{item['content']}\n用户要求对上述任务再执行一次。"
                )
                break

    if route.route == "chat":
        return TaskPlan(
            route="chat", skill=None, skill_score=0.0, skill_margin=0.0,
            high_confidence=True, workflow="direct", initial_allowed_tools=(),
            required_actions=frozenset(), needs_clarification=False,
        )

    if route.route == "knowledge":
        skill = skill_registry.get("search_sdr_knowledge") if skill_registry else None
        return TaskPlan(
            route="knowledge", skill=skill, skill_score=1.0, skill_margin=1.0,
            high_confidence=True, workflow="fallback",
            initial_allowed_tools=("search_sdr_knowledge",),
            required_actions=frozenset({"answer_with_knowledge"}),
            needs_clarification=False,
            skill_candidates=(("search_sdr_knowledge", 1.0),),
        )

    matches = (
        skill_registry.match(planning_instruction, category="hardware", min_confidence=0.4)
        if skill_registry else []
    )
    if not matches:
        return TaskPlan(
            route="hardware", skill=None, skill_score=0.0, skill_margin=0.0,
            high_confidence=False, workflow="clarify", initial_allowed_tools=(),
            required_actions=frozenset(), needs_clarification=True,
        )

    skill, score = matches[0]
    runner_up = matches[1][1] if len(matches) > 1 else 0.0
    margin = max(0.0, score - runner_up)
    high_confidence = score >= 0.6 and margin >= 0.15
    if len(matches) == 1 and score >= 0.5:
        high_confidence = True

    if skill.workflow == "subcommand" and skill.name == "start_video_stream":
        allowed_tools, required_actions = _video_subcommand(planning_instruction)
    else:
        allowed_tools = tuple(skill.tool_names or [skill.name])
        required_actions = _dynamic_required_actions(skill, planning_instruction)

    # Medium confidence exposes only the close candidates from the same route,
    # never the entire system tool catalogue.
    if not high_confidence:
        candidate_tools: list[str] = []
        for candidate, candidate_score in matches[:3]:
            if score - candidate_score > 0.2:
                continue
            candidate_tools.extend(candidate.tool_names or [candidate.name])
        allowed_tools = tuple(dict.fromkeys(candidate_tools)) or allowed_tools

    if route.needs_post_analysis:
        required_actions = frozenset(set(required_actions) | {"answer_with_knowledge"})

    return TaskPlan(
        route="hardware",
        skill=skill,
        skill_score=score,
        skill_margin=margin,
        high_confidence=high_confidence,
        workflow=skill.workflow,
        initial_allowed_tools=allowed_tools,
        required_actions=required_actions,
        needs_clarification=False,
        skill_candidates=tuple(
            (candidate.name, round(candidate_score, 4))
            for candidate, candidate_score in matches[:3]
        ),
    )


def completed_actions_for_result(
    tool_name: str,
    args: dict[str, Any],
    result: dict[str, Any],
) -> set[str]:
    """Translate a successful tool result into objective business effects."""
    status = str(result.get("status", "")).lower()
    if status != "success":
        return set()

    if tool_name == "perform_physical_scan":
        return set() if result.get("stopped") else {"scan_spectrum"}
    if tool_name == "tone_loopback_visualize":
        return {"visualize_tone"}
    if tool_name == "stop_hardware_task":
        return {"stop_hardware"}
    if tool_name == "text_fsk_send_and_receive":
        return set() if result.get("stopped") else {"transmit_text"}
    if tool_name == "adaptive_modulation_transmit":
        return {"probe_link", "select_modulation", "transmit_text"}
    if tool_name == "auto_optimal_transmit":
        actions = {"scan_spectrum"}
        if result.get("stopped"):
            return actions
        actions.add("select_clean_frequency")
        tx_result = result.get("tx_result")
        tone_result = result.get("tone_result")
        if (
            str(args.get("text", "")).strip()
            and isinstance(tx_result, dict)
            and tx_result.get("status") == "success"
            and not tx_result.get("stopped")
        ):
            actions.add("transmit_text")
        if (
            bool(args.get("enable_tone"))
            and isinstance(tone_result, dict)
            and tone_result.get("status") == "success"
        ):
            actions.add("visualize_tone")
        if bool(args.get("use_adaptive_modulation")) and "transmit_text" in actions:
            actions.add("select_modulation")
        return actions
    if tool_name == "start_video_stream":
        return {"start_video_stream"}
    if tool_name == "stop_video_stream":
        return {"stop_video_stream"}
    if tool_name == "video_stream_status":
        return {"query_video_status"}
    if tool_name in {"search_sdr_knowledge", "web_search"} and result.get("results"):
        return {"answer_with_knowledge"}
    return set()


def route_system_note(route: RouteDecision, plan: TaskPlan) -> str:
    if route.route == "chat":
        return "当前请求属于普通交流。不要调用任何工具，直接用中文回答。"
    if plan.needs_clarification:
        return (
            "当前请求可能涉及硬件操作，但动作或目标不够明确。不要调用工具，"
            "请只提出一个简短澄清问题。"
        )
    allowed = "、".join(plan.initial_allowed_tools)
    if route.route == "knowledge":
        return (
            "当前请求属于 SDR 知识问答。第一阶段只允许调用本地知识检索工具；"
            "只有本地结果为空时，系统才会开放联网搜索。"
        )
    return (
        f"当前请求属于硬件执行，当前工作流允许的工具为：{allowed}。"
        "禁止调用列表之外的工具；工具完成业务目标后停止调用并生成最终回答。"
    )
