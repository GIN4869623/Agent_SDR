from __future__ import annotations

import unittest

from skill.skill_spec import SkillRegistry
from skill.skills import _load_skills
from src.agent.router import build_task_plan, route_instruction


class RouterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.skills = SkillRegistry(list(_load_skills().values()))

    def plan(self, instruction: str):
        route = route_instruction(instruction, [])
        return route, build_task_plan(instruction, route, self.skills)

    def test_chat_has_no_tools(self) -> None:
        route, plan = self.plan("你好")
        self.assertEqual(route.route, "chat")
        self.assertEqual(plan.initial_allowed_tools, ())

    def test_theory_question_uses_knowledge_route(self) -> None:
        route, plan = self.plan("什么是 BPSK")
        self.assertEqual(route.route, "knowledge")
        self.assertEqual(plan.initial_allowed_tools, ("search_sdr_knowledge",))

    def test_how_to_scan_does_not_execute_hardware(self) -> None:
        route, _ = self.plan("如何扫描 2.4 GHz 的频谱")
        self.assertEqual(route.route, "knowledge")

    def test_scan_command_only_exposes_scan_tool(self) -> None:
        route, plan = self.plan("帮我扫描 2.4 GHz 附近的底噪")
        self.assertEqual(route.route, "hardware")
        self.assertEqual(plan.initial_allowed_tools, ("perform_physical_scan",))
        self.assertEqual(plan.required_actions, frozenset({"scan_spectrum"}))

    def test_lowest_noise_selects_cognitive_skill(self) -> None:
        route, plan = self.plan("帮我找底噪最低的频点")
        self.assertEqual(route.route, "hardware")
        self.assertEqual(plan.skill.name, "auto_optimal_transmit")
        self.assertEqual(plan.initial_allowed_tools, ("auto_optimal_transmit",))

    def test_cognitive_transmit_requires_both_actions(self) -> None:
        _, plan = self.plan("找干净频点并发送你好")
        self.assertEqual(
            plan.required_actions,
            frozenset({"select_clean_frequency", "transmit_text"}),
        )

    def test_cognitive_sine_request_uses_cognitive_defaults_and_no_text_action(self) -> None:
        _, plan = self.plan("在 2.4 GHz 附近进行认知选频并发送正弦波信号")
        self.assertEqual(plan.skill.name, "auto_optimal_transmit")
        self.assertEqual(plan.initial_allowed_tools, ("auto_optimal_transmit",))
        self.assertEqual(plan.required_actions, frozenset({"select_clean_frequency", "visualize_tone"}))
        self.assertIn("30000000", plan.skill.system_prompt)

    def test_cognitive_scan_with_sine_still_excludes_fixed_scan_and_tone_skills(self) -> None:
        _, plan = self.plan("扫描 2.4 GHz 附近并进行认知选频，然后发射正弦波")
        self.assertEqual(plan.initial_allowed_tools, ("auto_optimal_transmit",))

    def test_cognitive_progress_visualization_does_not_require_tone(self) -> None:
        _, plan = self.plan("找干净频点并展示选频功率对比可视化，不发送")
        self.assertEqual(plan.required_actions, frozenset({"select_clean_frequency"}))

    def test_cognitive_text_and_tone_requires_both(self) -> None:
        _, plan = self.plan("认知选频后发送文本“你好”，再显示 Tone 波形")
        self.assertEqual(plan.required_actions, frozenset({"select_clean_frequency", "transmit_text", "visualize_tone"}))

    def test_cognitive_unquoted_text_followed_by_tone_keeps_text_action(self) -> None:
        _, plan = self.plan("认知选频后发送你好，然后显示 Tone 波形")
        self.assertEqual(plan.required_actions, frozenset({"select_clean_frequency", "transmit_text", "visualize_tone"}))

    def test_explanation_then_execution_is_a_mixed_hardware_task(self) -> None:
        route, plan = self.plan("解释 BPSK，然后用 BPSK 发送你好")
        self.assertEqual(route.route, "hardware")
        self.assertTrue(route.needs_post_analysis)
        self.assertIn("transmit_text", plan.required_actions)
        self.assertIn("answer_with_knowledge", plan.required_actions)

    def test_video_skill_selects_one_subcommand_tool(self) -> None:
        _, stop_plan = self.plan("停止视频流")
        status_route, status_plan = self.plan("查看视频流状态")
        self.assertEqual(stop_plan.initial_allowed_tools, ("stop_video_stream",))
        self.assertEqual(status_route.route, "hardware")
        self.assertEqual(status_plan.initial_allowed_tools, ("video_stream_status",))

    def test_hardware_with_analysis_marks_post_knowledge_action(self) -> None:
        route, plan = self.plan("扫描 2.4 GHz 并解释为什么噪声高")
        self.assertTrue(route.needs_post_analysis)
        self.assertEqual(
            plan.required_actions,
            frozenset({"scan_spectrum", "answer_with_knowledge"}),
        )

    def test_negating_transmit_does_not_cancel_frequency_selection(self) -> None:
        route, plan = self.plan("找干净频点，但不要发送")
        self.assertEqual(route.route, "hardware")
        self.assertEqual(plan.required_actions, frozenset({"select_clean_frequency"}))

    def test_repeat_request_reuses_previous_user_intent(self) -> None:
        history = [
            {"role": "user", "content": "用 QPSK 发送你好"},
            {"role": "assistant", "content": "发送完成"},
        ]
        route = route_instruction("再来一次", history)
        plan = build_task_plan("再来一次", route, self.skills, history)
        self.assertEqual(route.route, "hardware")
        self.assertEqual(plan.initial_allowed_tools, ("text_fsk_send_and_receive",))


if __name__ == "__main__":
    unittest.main()
