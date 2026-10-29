# -*- coding: utf-8 -*-
"""智能艾特门禁核心单元测试（含规则状态机与 JEV 语义模型分支）"""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from astrbot_plugin_at_gate.gatekeeper import (
    AtGatekeeper,
    Decision,
    is_at_bot,
    extract_content,
)


class MockAt:
    def __init__(self, qq: str | int) -> None:
        self.qq = qq


class MockPlain:
    def __init__(self, text: str) -> None:
        self.text = text


class MockImage:
    pass


class TestAtGatekeeper(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.config = {
            "enabled": True,
            "use_jev": False,  # 基础测试默认使用规则兜底
            "at_threshold": 2,
            "window_seconds": 30,
            "suppress_empty_at": True,
            "suppress_during_active_run": True,
            "cooldown_after_reply_sec": 5,
            "allow_distinct_questions": True,
            "min_distinct_length": 3,
            "rush_keywords": "?,？,在吗,人呢,快点",
            "log_intercept": False,
        }
        self.gk = AtGatekeeper(self.config)
        self.umo = "TestPlatform:GroupMessage:123456"
        self.sender = "999888"

    async def test_not_at_bot(self):
        d = await self.gk.evaluate(self.umo, self.sender, is_at_bot=False, raw_text="你好", now=100.0)
        self.assertFalse(d.should_block)
        self.assertEqual(d.reason, "not_at_bot")

    async def test_plugin_disabled(self):
        self.gk.enabled = False
        d = await self.gk.evaluate(self.umo, self.sender, is_at_bot=True, raw_text="", now=100.0)
        self.assertFalse(d.should_block)
        self.assertEqual(d.reason, "plugin_disabled")

    async def test_single_normal_question(self):
        d = await self.gk.evaluate(self.umo, self.sender, is_at_bot=True, raw_text="你是什么生图模型", now=100.0)
        self.assertFalse(d.should_block)
        self.assertEqual(d.reason, "normal")
        self.assertEqual(d.at_count, 1)

    async def test_chat_log_scenario_active_run_empty_at(self):
        """复现聊天记录：第1条触发生成中，第2、3条连点空@，全部被静默拦截"""
        now = 1000.0
        d1 = await self.gk.evaluate(self.umo, self.sender, is_at_bot=True, raw_text="你是什么生图模型", now=now)
        self.assertFalse(d1.should_block)

        self.gk.mark_run_begin(self.umo, self.sender, question_text="你是什么生图模型", now=now + 0.1)

        # 38秒后用户连发第 1 个空 @（等不耐烦）
        d2 = await self.gk.evaluate(self.umo, self.sender, is_at_bot=True, raw_text="[At:Bot]", now=now + 38.0)
        self.assertTrue(d2.should_block)
        self.assertEqual(d2.reason, "empty_at_during_active_run")

        # 14毫秒后用户连发第 2 个空 @
        d3 = await self.gk.evaluate(self.umo, self.sender, is_at_bot=True, raw_text="", now=now + 38.014)
        self.assertTrue(d3.should_block)
        self.assertEqual(d3.reason, "empty_at_during_active_run")

        self.gk.mark_run_done(self.umo, self.sender, now=now + 58.0)
        self.assertEqual(self.gk.stats_intercepted_total, 2)

    async def test_active_run_rush_keyword(self):
        """生成期间发送催促词，应被拦截"""
        now = 2000.0
        self.gk.mark_run_begin(self.umo, self.sender, question_text="在写代码", now=now)
        d = await self.gk.evaluate(self.umo, self.sender, is_at_bot=True, raw_text="在吗", now=now + 5.0)
        self.assertTrue(d.should_block)
        self.assertEqual(d.reason, "rush_during_active_run")

    async def test_cooldown_after_reply(self):
        """刚回复完毕的保护期内，积压的空@应被拦截"""
        now = 3000.0
        self.gk.mark_run_begin(self.umo, self.sender, now=now)
        self.gk.mark_run_done(self.umo, self.sender, now=now + 10.0)

        # 刚回复完第 2 秒（<= 5s）
        d = await self.gk.evaluate(self.umo, self.sender, is_at_bot=True, raw_text="", now=now + 12.0)
        self.assertTrue(d.should_block)
        self.assertEqual(d.reason, "empty_at_within_reply_cooldown")

        # 超过 5 秒（第 6 秒）
        d2 = await self.gk.evaluate(self.umo, self.sender, is_at_bot=True, raw_text="今天吃什么", now=now + 16.0)
        self.assertFalse(d2.should_block)

    async def test_window_threshold_multiple_at(self):
        """窗口内多次@超过阈值（>2次）：空@与催促拦截，有效新问题放行"""
        now = 4000.0
        d1 = await self.gk.evaluate(self.umo, self.sender, is_at_bot=True, raw_text="第一问", now=now)
        self.assertFalse(d1.should_block)

        d2 = await self.gk.evaluate(self.umo, self.sender, is_at_bot=True, raw_text="第二问", now=now + 5.0)
        self.assertFalse(d2.should_block)

        d3 = await self.gk.evaluate(self.umo, self.sender, is_at_bot=True, raw_text="", now=now + 10.0)
        self.assertTrue(d3.should_block)
        self.assertIn("empty_at_threshold_exceeded", d3.reason)

        d4 = await self.gk.evaluate(self.umo, self.sender, is_at_bot=True, raw_text="快点？？", now=now + 12.0)
        self.assertTrue(d4.should_block)
        self.assertIn("rush_threshold_exceeded", d4.reason)

        d5 = await self.gk.evaluate(self.umo, self.sender, is_at_bot=True, raw_text="明天深圳天气怎么样", now=now + 15.0)
        self.assertFalse(d5.should_block)
        self.assertIn("distinct_question_allowed", d5.reason)

        d6 = await self.gk.evaluate(self.umo, self.sender, is_at_bot=True, raw_text="明天深圳天气怎么样", now=now + 16.0)
        self.assertTrue(d6.should_block)
        self.assertIn("duplicate_text_threshold_exceeded", d6.reason)

    async def test_window_expiration(self):
        """超过时间窗口后，计数重置"""
        now = 5000.0
        d1 = await self.gk.evaluate(self.umo, self.sender, is_at_bot=True, raw_text="问题一", now=now)
        self.assertEqual(d1.at_count, 1)

        d2 = await self.gk.evaluate(self.umo, self.sender, is_at_bot=True, raw_text="问题二", now=now + 35.0)
        self.assertFalse(d2.should_block)
        self.assertEqual(d2.at_count, 1)

    async def test_media_content_not_empty(self):
        """包含图片的消息不应被当作纯空 @ 处理"""
        now = 6000.0
        self.gk.mark_run_begin(self.umo, self.sender, now=now)
        d = await self.gk.evaluate(self.umo, self.sender, is_at_bot=True, raw_text="", has_media=True, now=now + 5.0)
        self.assertFalse(d.should_block)

    async def test_user_isolation(self):
        """不同用户之间的计数与状态完全隔离"""
        now = 7000.0
        user_a = "111"
        user_b = "222"

        self.gk.mark_run_begin(self.umo, user_a, now=now)
        d_a = await self.gk.evaluate(self.umo, user_a, is_at_bot=True, raw_text="", now=now + 1.0)
        self.assertTrue(d_a.should_block)

        d_b = await self.gk.evaluate(self.umo, user_b, is_at_bot=True, raw_text="你好", now=now + 1.0)
        self.assertFalse(d_b.should_block)
        self.assertEqual(d_b.at_count, 1)

    async def test_cleanup_expired(self):
        """清理过期历史与孤儿活跃任务"""
        now = 8000.0
        await self.gk.evaluate(self.umo, self.sender, is_at_bot=True, raw_text="测试", now=now)
        self.gk.mark_run_begin(self.umo, self.sender, now=now)

        self.assertIn((self.umo, self.sender), self.gk._history)
        self.assertIn((self.umo, self.sender), self.gk._active_runs)

        self.gk.cleanup_expired(now=now + 200.0)
        self.assertNotIn((self.umo, self.sender), self.gk._history)
        self.assertNotIn((self.umo, self.sender), self.gk._active_runs)

    async def test_event_helpers(self):
        """测试 is_at_bot 与 extract_content 辅助函数"""
        event_at = SimpleNamespace(
            is_at_or_wake_command=False,
            get_self_id=lambda: "2538085748",
            message_obj=SimpleNamespace(
                message=[MockAt(qq="2538085748"), MockPlain(text=" 你好")]
            ),
            message_str="你好",
        )
        self.assertTrue(is_at_bot(event_at))
        text, has_media = extract_content(event_at)
        self.assertEqual(text, " 你好")
        self.assertFalse(has_media)

        event_other = SimpleNamespace(
            is_at_or_wake_command=False,
            get_self_id=lambda: "2538085748",
            message_obj=SimpleNamespace(
                message=[MockAt(qq="123456789"), MockPlain(text=" 看看这个")]
            ),
            message_str="看看这个",
        )
        self.assertFalse(is_at_bot(event_other))

        event_img = SimpleNamespace(
            is_at_or_wake_command=True,
            get_self_id=lambda: "2538085748",
            message_obj=SimpleNamespace(
                message=[MockAt(qq="2538085748"), MockImage()]
            ),
            message_str="",
        )
        self.assertTrue(is_at_bot(event_img))
        _, has_media = extract_content(event_img)
        self.assertTrue(has_media)

        # 私聊事件即使带有 is_at_or_wake_command 也必须返回 False
        event_private = SimpleNamespace(
            is_at_or_wake_command=True,
            is_private_chat=lambda: True,
            get_self_id=lambda: "2538085748",
            message_obj=SimpleNamespace(message=[MockPlain(text="好哦")]),
            message_str="好哦",
        )
        self.assertFalse(is_at_bot(event_private))

        # 私聊 UMO 兜底也必须返回 False
        event_friend_umo = SimpleNamespace(
            is_at_or_wake_command=True,
            unified_msg_origin="Atri:FriendMessage:3217243301",
            get_self_id=lambda: "2538085748",
            message_obj=SimpleNamespace(message=[MockPlain(text="好哦")]),
            message_str="好哦",
        )
        self.assertFalse(is_at_bot(event_friend_umo))
    # ==================== JEV 语义意图模型专项测试 ====================

    async def test_jev_judge_impatient_rush(self):
        """JEV 识别等待期不耐烦催促：Jev 给出低概率 -> 判定拦截"""
        self.gk.update_config({
            "use_jev": True,
            "jev_api_key": "test_typesafe_key",
            "jev_threshold": 0.5,
        })
        now = 9000.0
        self.gk.mark_run_begin(self.umo, self.sender, question_text="画一只赛博朋克猫", now=now)

        # 模拟 JEV 判断 "怎么还没出来" 属于催促 (noul = 0.08 < 0.5)
        with patch.object(
            self.gk.jev_client,
            "evaluate_should_respond",
            new_callable=AsyncMock,
            return_value=(False, 0.08, "jev_rejected(noul=0.080,th=0.5)"),
        ) as mock_jev:
            d = await self.gk.evaluate(
                self.umo, self.sender, is_at_bot=True, raw_text="怎么还没出来", now=now + 10.0
            )
            mock_jev.assert_called_once()
            self.assertTrue(d.should_block)
            self.assertIn("jev_rejected", d.reason)
            self.assertEqual(d.score, 0.08)

    async def test_jev_judge_new_substantive_requirement(self):
        """JEV 识别等待期用户提出的实质性新需求：Jev 给出高概率 -> 判定放行"""
        self.gk.update_config({
            "use_jev": True,
            "jev_api_key": "test_typesafe_key",
            "jev_threshold": 0.5,
        })
        now = 9100.0
        self.gk.mark_run_begin(self.umo, self.sender, question_text="画一只赛博朋克猫", now=now)

        # 模拟用户在等待期间补充 "对了加上霓虹雨伞"，JEV 判定是具有实质新意图 (noul = 0.92 >= 0.5)
        with patch.object(
            self.gk.jev_client,
            "evaluate_should_respond",
            new_callable=AsyncMock,
            return_value=(True, 0.92, "jev_approved(noul=0.920,th=0.5)"),
        ) as mock_jev:
            d = await self.gk.evaluate(
                self.umo, self.sender, is_at_bot=True, raw_text="对了加上霓虹雨伞", now=now + 10.0
            )
            mock_jev.assert_called_once()
            self.assertFalse(d.should_block)
            self.assertIn("jev_approved", d.reason)
            self.assertEqual(d.score, 0.92)

    async def test_jev_graceful_fallback_on_error(self):
        """JEV 网络异常/超时时，自动降级为规则兜底"""
        self.gk.update_config({
            "use_jev": True,
            "jev_api_key": "test_typesafe_key",
            "rush_keywords": "催催催",
        })
        now = 9200.0
        self.gk.mark_run_begin(self.umo, self.sender, now=now)

        # 模拟 JEV 超时异常返回 score = -1.0
        with patch.object(
            self.gk.jev_client,
            "evaluate_should_respond",
            new_callable=AsyncMock,
            return_value=(True, -1.0, "jev_timeout_5.0s"),
        ):
            # 催促短语命中了规则兜底 -> 规则拦截生效，不会因 JEV 挂掉而崩溃或漏判
            d = await self.gk.evaluate(
                self.umo, self.sender, is_at_bot=True, raw_text="催催催", now=now + 10.0
            )
            self.assertTrue(d.should_block)
            self.assertEqual(d.reason, "rush_during_active_run")

    def test_normalize_endpoint(self):
        """测试各类第三方 JEV 地址与 OpenRouter 端点归一化"""
        from astrbot_plugin_at_gate.jev_client import normalize_endpoint

        # 官方默认
        self.assertEqual(normalize_endpoint(""), "https://api.typesafe.ai/v1/systemone")
        self.assertEqual(normalize_endpoint("https://api.typesafe.ai/v1"), "https://api.typesafe.ai/v1/systemone")
        # OpenRouter 各种输入形态
        self.assertEqual(normalize_endpoint("https://openrouter.ai/api"), "https://openrouter.ai/api/v1/systemone")
        self.assertEqual(normalize_endpoint("https://openrouter.ai/api/v1"), "https://openrouter.ai/api/v1/systemone")
        self.assertEqual(normalize_endpoint("https://openrouter.ai/api/v1/systemone"), "https://openrouter.ai/api/v1/systemone")
        # 第三方反代 / New-API
        self.assertEqual(normalize_endpoint("https://my-proxy.com/v1"), "https://my-proxy.com/v1/systemone")
        self.assertEqual(normalize_endpoint("https://my-proxy.com/v1/chat/completions"), "https://my-proxy.com/v1/chat/completions")

    async def test_third_party_chat_completions_protocol(self):
        """测试第三方 OpenAI-compatible /chat/completions 兼容协议"""
        self.gk.update_config({
            "use_jev": True,
            "jev_api_key": "test_third_party_key",
            "jev_base_url": "https://custom-proxy.com/v1/chat/completions",
            "jev_model": "deepseek-ai/deepseek-v3",
        })
        self.assertTrue(self.gk.jev_client.is_chat_completions)
        now = 9500.0
        self.gk.mark_run_begin(self.umo, self.sender, question_text="画图", now=now)

        with patch.object(
            self.gk.jev_client,
            "_eval_should_respond_via_chat",
            new_callable=AsyncMock,
            return_value=(False, 0.1, "third_party_rejected(conf=0.90)"),
        ) as mock_chat:
            d = await self.gk.evaluate(
                self.umo, self.sender, is_at_bot=True, raw_text="快点啊", now=now + 5.0
            )
            mock_chat.assert_called_once()
            self.assertTrue(d.should_block)
            self.assertIn("jev_rejected", d.reason)

    async def test_native_provider_reuse(self):
        """测试零配置模式：自动复用 AstrBot 原生配置的聊天大模型"""
        mock_provider = SimpleNamespace(
            text_chat=AsyncMock(
                return_value=SimpleNamespace(
                    completion_text='{"should_respond": false, "confidence": 0.85, "reason": "等待期催促"}'
                )
            )
        )
        self.gk.jev_client.provider_getter = lambda: mock_provider
        self.gk.update_config({
            "use_jev": True,
            "jev_api_key": "",  # 未配置独立 Key
            "reuse_astrbot_provider": True,
        })
        self.assertTrue(self.gk.jev_client.is_available)

        now = 9600.0
        self.gk.mark_run_begin(self.umo, self.sender, question_text="写论文", now=now)

        d = await self.gk.evaluate(
            self.umo, self.sender, is_at_bot=True, raw_text="好了没", now=now + 5.0
        )
        mock_provider.text_chat.assert_called_once()
        self.assertTrue(d.should_block)
        self.assertIn("jev_rejected", d.reason)

    # ==================== 防抖功能测试 ====================

    def test_message_buffer(self):
        """测试消息缓冲区的追加、合并与图片去重"""
        from astrbot_plugin_at_gate.debounce_buffer import MessageBuffer

        buf = MessageBuffer()
        buf.add("今天下午")
        buf.add("有空吗")
        self.assertEqual(buf.get_full_text(), "今天下午 有空吗")

        # 测试图片追加
        img1 = MockImage()
        img1.url = "https://example.com/1.jpg"
        event1 = SimpleNamespace(message_obj=SimpleNamespace(message=[img1]))
        buf.add("", event=event1)
        self.assertEqual(len(buf.images), 1)

        buf.clear()
        self.assertEqual(buf.get_full_text(), "")
        self.assertEqual(len(buf.images), 0)

    async def test_sentence_completeness_heuristic(self):
        """测试句子完整性启发式规则兜底"""
        # 未完结的短语
        incomplete_1, _, _ = await self.gk.jev_client.evaluate_sentence_completeness("如果明天")
        self.assertFalse(incomplete_1)

        incomplete_2, _, _ = await self.gk.jev_client.evaluate_sentence_completeness("那个")
        self.assertFalse(incomplete_2)

        # 完结的句子（带句末标点）
        complete_1, _, _ = await self.gk.jev_client.evaluate_sentence_completeness("今天吃什么？")
        self.assertTrue(complete_1)

        complete_2, _, _ = await self.gk.jev_client.evaluate_sentence_completeness("帮我写一段Python代码。")
        self.assertTrue(complete_2)

        # 带图消息直接判定完整
        complete_img, _, _ = await self.gk.jev_client.evaluate_sentence_completeness("", has_media=True)
        self.assertTrue(complete_img)

    async def test_sentence_completeness_model(self):
        """测试模型判定句子完整性（JEV / OpenAI 协议）"""
        self.gk.update_config({
            "use_jev": True,
            "jev_api_key": "test_key",
            "debounce_threshold": 0.8,
        })

        # 模拟模型判定完整
        with patch.object(
            self.gk.jev_client,
            "_eval_completeness_via_systemone",
            new_callable=AsyncMock,
            return_value=(True, 0.95, "jev_complete"),
        ):
            res, score, _ = await self.gk.jev_client.evaluate_sentence_completeness("这是一段完整的话")
            self.assertTrue(res)
            self.assertEqual(score, 0.95)

        # 模拟模型判定未完结
        with patch.object(
            self.gk.jev_client,
            "_eval_completeness_via_systemone",
            new_callable=AsyncMock,
            return_value=(False, 0.3, "jev_incomplete"),
        ):
            res, score, _ = await self.gk.jev_client.evaluate_sentence_completeness("话说一半")
            self.assertFalse(res)
            self.assertEqual(score, 0.3)


if __name__ == "__main__":
    unittest.main()
