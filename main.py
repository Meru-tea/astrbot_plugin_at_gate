# -*- coding: utf-8 -*-
"""智能艾特门禁与消息防抖一体化插件（astrbot_plugin_at_gate）

融合两大核心能力：
1. 前置消息防抖（参考并融合 astrbot_plugin_debounce）：
   - 智能识别用户未说完的碎句与条件句，自动进入滑动等待窗口；
   - 连发多条文字与图片自动合并为同一次 LLM 请求，降低调用频次与碎句抢答。
2. 后置生成期与多艾特门禁（AtGate 独家核心）：
   - 解决用户在模型思考生成期间不耐烦连续连点@、纯空@、催促刷屏导致的后续排队连发多条机械式回复；
   - 在规则状态机与 JEV 语义模型判定下，实现 0 冗余回复，只回正题一次。

模型与兼容性：
- 支持 TypeSafe / OpenRouter JEV 原生 System One 协议（Noul 概率计算）
- 支持通用第三方大模型（OpenAI Chat Completions 协议）
- 支持 AstrBot 原生配置大模型零配置自动复用
- 离线 0 Token 状态机兜底
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, Optional, Set

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import LLMResponse, ProviderRequest
from astrbot.api.star import Context, Star, register

from .debounce_buffer import MessageBuffer, get_image_components, image_source
from .gatekeeper import AtGatekeeper, Decision, extract_content, is_at_bot

PLUGIN_NAME = "astrbot_plugin_at_gate"
VERSION = "1.0.2"
DISPLAY_NAME = "智能艾特门禁与消息防抖"


@register(
    PLUGIN_NAME,
    "Meru",
    "智能消息防抖合并与多次@生成期催促门禁，支持 JEV/通用大模型/原生模型零配置",
    VERSION,
)
class AtGatePlugin(Star):
    def __init__(self, context: Context, config: dict[str, Any] | None = None) -> None:
        super().__init__(context)
        self.config: dict[str, Any] = config or {}
        self.gatekeeper = AtGatekeeper(self.config, provider_getter=self._get_provider)

        # 防抖相关状态
        self.enable_debounce: bool = bool(self.config.get("enable_debounce", True))
        self.debounce_timeout: float = float(self.config.get("debounce_timeout", 8.0))
        self.usage_scope: str = str(self.config.get("usage_scope", "both"))

        self.buffers: Dict[str, MessageBuffer] = {}
        self.waiting_sessions: Set[str] = set()
        self.monitor_tasks: Dict[str, asyncio.Task] = {}
        self.skip_debounce_msg_ids: Set[str] = set()
        self.pending_llm_sessions: Set[str] = set()
        self.discard_next_response: Set[str] = set()
        self.should_cancel_msg_ids: Set[str] = set()
        self.waiting_msg_ids: Dict[str, str] = {}

        self._cleanup_task: asyncio.Task[None] | None = None
        self._start_cleanup_loop()

        jev = self.gatekeeper.jev_client
        if jev.api_key:
            status_desc = f"独立第三方模型 ({jev.model})"
        elif jev.is_available:
            chosen = self.config.get("astrbot_provider_id") or "AstrBot默认主模型"
            status_desc = f"AstrBot内部模型 ({chosen})"
        else:
            status_desc = "规则兜底模式"

        logger.info(
            f"[{DISPLAY_NAME}] 插件已加载 v{VERSION} (门禁={self.gatekeeper.enabled}, "
            f"防抖={self.enable_debounce}, 决策模式={status_desc})"
        )

    def _get_provider(self) -> Any:
        try:
            pid = str(self.config.get("astrbot_provider_id", "") or "").strip()
            if pid:
                # 1. 尝试直接通过 ID 获取
                prov = self.context.get_provider_by_id(pid)
                if prov is not None:
                    return prov
                # 2. 遍历已加载的所有文本生成提供者进行匹配
                for p in self.context.get_all_providers():
                    meta_id = getattr(getattr(p, "meta", None), "id", "")
                    curr_model = p.get_model() if hasattr(p, "get_model") else ""
                    if pid in (meta_id, curr_model):
                        return p
            return self.context.get_using_provider()
        except Exception:
            return None
    def _start_cleanup_loop(self) -> None:
        async def _loop() -> None:
            while True:
                try:
                    await asyncio.sleep(60.0)
                    self.gatekeeper.cleanup_expired()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.warning(f"[{DISPLAY_NAME}] cleanup loop error: {e}")

        self._cleanup_task = asyncio.create_task(_loop())

    async def terminate(self) -> None:
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
        for task in list(self.monitor_tasks.values()):
            if not task.done():
                task.cancel()
        self.monitor_tasks.clear()
        logger.info(f"[{DISPLAY_NAME}] 插件已卸载")

    def _get_buffer(self, session_id: str) -> MessageBuffer:
        if session_id not in self.buffers:
            self.buffers[session_id] = MessageBuffer()
        return self.buffers[session_id]

    def _match_scope(self, event: AstrMessageEvent) -> bool:
        if self.usage_scope == "both":
            return True
        is_private = event.is_private_chat()
        if self.usage_scope == "group" and is_private:
            return False
        if self.usage_scope == "private" and not is_private:
            return False
        return True

    # ==================== 1. 后置生成期与多艾特门禁 (Priority 999) ====================

    @filter.on_agent_begin()
    async def on_agent_begin(self, event: AstrMessageEvent, run_context: Any) -> None:
        """感知会话 Agent 开始执行，登记为活跃状态并记录正在回答的提问文本"""
        try:
            if getattr(event, "is_private_chat", lambda: False)():
                return
            umo = getattr(event, "unified_msg_origin", "")
            sender_id = str(event.get_sender_id() or "")
            raw_text, _ = extract_content(event)
            if umo:
                self.gatekeeper.mark_run_begin(umo, sender_id, question_text=raw_text)
        except Exception as e:
            logger.debug(f"[{DISPLAY_NAME}] on_agent_begin failed: {e}")

    @filter.on_agent_done()
    async def on_agent_done(
        self, event: AstrMessageEvent, run_context: Any, response: Any
    ) -> None:
        """感知会话 Agent 结束执行，解除活跃状态并启动冷却期"""
        try:
            if getattr(event, "is_private_chat", lambda: False)():
                return
            umo = getattr(event, "unified_msg_origin", "")
            sender_id = str(event.get_sender_id() or "")
            if umo:
                self.gatekeeper.mark_run_done(umo, sender_id)
        except Exception as e:
            logger.debug(f"[{DISPLAY_NAME}] on_agent_done failed: {e}")

    @filter.event_message_type(filter.EventMessageType.ALL, priority=999)
    async def on_message_gate(self, event: AstrMessageEvent) -> None:
        """最高优先级事件门禁：在进入 LLM 或下游插件前完成冗余 @ 识别与拦截（仅群聊生效）"""
        if not self.gatekeeper.enabled:
            return

        # 私聊直接放行，艾特门禁仅针对群聊
        if getattr(event, "is_private_chat", lambda: False)():
            return

        is_at = is_at_bot(event)
        if not is_at:
            return

        umo = getattr(event, "unified_msg_origin", "")
        sender_id = str(event.get_sender_id() or "")
        raw_text, has_media = extract_content(event)

        decision: Decision = await self.gatekeeper.evaluate(
            umo=umo,
            sender_id=sender_id,
            is_at_bot=is_at,
            raw_text=raw_text,
            has_media=has_media,
        )

        if decision.should_block:
            if self.gatekeeper.log_intercept:
                logger.info(
                    f"[{DISPLAY_NAME}] [门禁拦截] 已拦截冗余/催促艾特: reason={decision.reason} "
                    f"sender={sender_id} umo={umo} at_count={decision.at_count} text={raw_text!r}"
                )
            event.stop_event()

    # ==================== 2. 前置消息防抖合并 (Debounce) ====================

    @filter.on_waiting_llm_request()
    async def on_waiting_llm_request(self, event: AstrMessageEvent) -> None:
        """即将调用 LLM 时的通知（在 session lock 之前）- 用于检测新消息到达并重置防抖"""
        if not self.enable_debounce or not self._match_scope(event):
            return

        session_id = event.session_id
        msg_id = str(getattr(getattr(event, "message_obj", None), "message_id", "") or "")

        if msg_id in self.skip_debounce_msg_ids:
            return

        if session_id in self.waiting_msg_ids:
            old_msg_id = self.waiting_msg_ids[session_id]
            self.should_cancel_msg_ids.add(old_msg_id)

        self.waiting_msg_ids[session_id] = msg_id

        if session_id in self.monitor_tasks:
            self.monitor_tasks[session_id].cancel()
            self.monitor_tasks.pop(session_id, None)

        if session_id in self.pending_llm_sessions:
            self.discard_next_response.add(session_id)

    @filter.on_llm_request(priority=100)
    async def on_llm_request(self, event: AstrMessageEvent, req: ProviderRequest) -> None:
        """核心防抖逻辑：检查句子完整度并按需合并连发碎句"""
        if not self.enable_debounce or not self._match_scope(event):
            return

        session_id = event.session_id
        msg_id = str(getattr(getattr(event, "message_obj", None), "message_id", "") or "")
        message_text = (getattr(event, "message_str", "") or "").strip()
        if not message_text and not req.image_urls:
            return
        if self.waiting_msg_ids.get(session_id) == msg_id:
            self.waiting_msg_ids.pop(session_id, None)

        # 优先检查取消标记
        if msg_id in self.should_cancel_msg_ids:
            self.should_cancel_msg_ids.discard(msg_id)
            buffer = self._get_buffer(session_id)
            buffer.add(message_text, event, req.image_urls)
            self.waiting_sessions.add(session_id)
            event.stop_event()
            return

        # 跳过重放的完整消息
        if msg_id in self.skip_debounce_msg_ids:
            self.skip_debounce_msg_ids.discard(msg_id)
            self.pending_llm_sessions.add(session_id)
            return

        buffer = self._get_buffer(session_id)
        should_merge = (session_id in self.waiting_sessions) or (session_id in self.pending_llm_sessions)

        if should_merge:
            buffer.add(message_text, event, req.image_urls)
            req.image_urls = list(buffer.image_urls)

            full_text = buffer.get_full_text()
            has_images = bool(buffer.images or buffer.image_urls)
            is_complete, score, reason = await self.gatekeeper.jev_client.evaluate_sentence_completeness(
                full_text, has_media=has_images
            )

            if is_complete and session_id not in self.waiting_msg_ids:
                if session_id in self.monitor_tasks:
                    self.monitor_tasks[session_id].cancel()
                    self.monitor_tasks.pop(session_id, None)

                self.waiting_sessions.discard(session_id)
                req.prompt = full_text
                self.discard_next_response.discard(session_id)
                self.pending_llm_sessions.add(session_id)
                logger.debug(f"[{DISPLAY_NAME}] [防抖合并发送]: {full_text} (score={score:.2f}, reason={reason})")
                return
            else:
                # 仍不完整，阻止当前单独响应，继续监控等待
                event.stop_event()
                if self.debounce_timeout > 0 and session_id not in self.monitor_tasks:
                    self.monitor_tasks[session_id] = asyncio.create_task(
                        self._monitor_session(session_id, self.debounce_timeout)
                    )
                return

        # 首次到达的消息，加入 buffer
        buffer.add(message_text, event, req.image_urls)
        req.image_urls = list(buffer.image_urls)
        full_text = buffer.get_full_text()
        has_images = bool(buffer.images or buffer.image_urls)

        is_complete, score, reason = await self.gatekeeper.jev_client.evaluate_sentence_completeness(
            full_text, has_media=has_images
        )

        if is_complete and session_id not in self.waiting_msg_ids:
            # 完整，直接通过
            self.discard_next_response.discard(session_id)
            self.pending_llm_sessions.add(session_id)
            return
        else:
            # 句子不完整，拦截并开启防抖等待窗口
            event.stop_event()
            self.waiting_sessions.add(session_id)
            logger.debug(f"[{DISPLAY_NAME}] [防抖拦截等待]: {full_text} (score={score:.2f}, reason={reason})")

            if self.debounce_timeout > 0 and session_id not in self.monitor_tasks:
                self.monitor_tasks[session_id] = asyncio.create_task(
                    self._monitor_session(session_id, self.debounce_timeout)
                )

    @filter.on_llm_response()
    async def on_llm_response(self, event: AstrMessageEvent, resp: LLMResponse) -> None:
        """模型响应完成回调：清理已发送完成的缓冲区"""
        session_id = event.session_id
        self.pending_llm_sessions.discard(session_id)

        if session_id in self.discard_next_response:
            self.discard_next_response.discard(session_id)
            event.stop_event()
            return

        if session_id in self.buffers:
            self.buffers[session_id].clear()
        self.waiting_sessions.discard(session_id)

    async def _monitor_session(self, session_id: str, timeout_seconds: float) -> None:
        """监控会话超时，超时后自动重放合并后的完整消息"""
        try:
            await asyncio.sleep(timeout_seconds)

            if session_id not in self.waiting_sessions:
                return

            buffer = self.buffers.get(session_id)
            if not buffer:
                return

            full_text = buffer.get_full_text()
            saved_event = buffer.event
            saved_images = list(buffer.images)

            if (not full_text and not saved_images) or not saved_event:
                return

            logger.debug(f"[{DISPLAY_NAME}] [防抖等待超时，合并触发]: {session_id} -> {full_text}")

            if await self._send_fake_event(saved_event, full_text, images=saved_images):
                self.waiting_sessions.discard(session_id)
                buffer.clear()

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"[{DISPLAY_NAME}] 防抖超时发送失败: {e}")
        finally:
            if self.monitor_tasks.get(session_id) is asyncio.current_task():
                self.monitor_tasks.pop(session_id, None)

    async def _send_fake_event(
        self, original_event: AstrMessageEvent, message_text: str, *, images=None
    ) -> bool:
        """重放合并后的消息事件到 EventBus 唤醒模型"""
        replay_id = None
        try:
            from astrbot.core.star.star_tools import StarTools
            from astrbot.core.message.components import Plain

            original_message = getattr(getattr(original_event, "message_obj", None), "message", []) or []
            new_message_components = []

            for component in original_message:
                if not isinstance(component, Plain) and type(component).__name__ not in ("Plain", "PlainComp"):
                    if not (images is not None and getattr(component, "type", None) == "Image"):
                        new_message_components.append(component)

            if images:
                new_message_components.extend(images)

            new_message_components.insert(0, Plain(message_text))

            msg_type = getattr(original_event.message_obj, "type", "")
            msg_type_str = getattr(msg_type, "value", str(msg_type)) if msg_type else "GroupMessage"
            if isinstance(msg_type_str, str) and "." in msg_type_str:
                # 兜底防止 "MessageType.GROUP_MESSAGE" 传入
                msg_type_str = msg_type_str.split(".")[-1]
                if msg_type_str == "GROUP_MESSAGE":
                    msg_type_str = "GroupMessage"
                elif msg_type_str == "FRIEND_MESSAGE":
                    msg_type_str = "FriendMessage"

            self_id = str(
                original_event.get_self_id()
                if callable(getattr(original_event, "get_self_id", None))
                else getattr(getattr(original_event, "message_obj", None), "self_id", "") or ""
            )

            new_message = await StarTools.create_message(
                type=msg_type_str,
                self_id=self_id,
                session_id=original_event.session_id,
                sender=original_event.message_obj.sender,
                message=new_message_components,
                message_str=message_text,
                group_id=original_event.get_group_id() or "",
                message_id=original_event.message_obj.message_id,
            )

            self.skip_debounce_msg_ids.add(new_message.message_id)
            replay_id = new_message.message_id

            await StarTools.create_event(
                abm=new_message,
                platform=original_event.get_platform_name(),
                is_wake=True,
            )
            return True

        except Exception as e:
            if replay_id is not None:
                self.skip_debounce_msg_ids.discard(replay_id)
            logger.error(f"[{DISPLAY_NAME}] 重放事件失败: {e}")
            return False

    # ==================== 3. 管理与状态面板 ====================

    @filter.command("atgate")
    async def atgate_cmd(self, event: AstrMessageEvent) -> None:
        """查看门禁与防抖状态总览（仅管理员）"""
        if not event.is_admin():
            await event.send(event.plain_result("权限不足，仅管理员可查看门禁状态。"))
            return

        jev = self.gatekeeper.jev_client
        if jev.api_key:
            proto = "SystemOne" if (jev.is_jev_model and not jev.is_chat_completions) else "ChatCompletions"
            jev_info = f"独立第三方API ({jev.model}, 协议={proto}, 决策调用: {self.gatekeeper.stats_jev_evaluations}次)"
        elif jev.is_available:
            chosen = self.config.get("astrbot_provider_id") or "默认主模型"
            jev_info = f"AstrBot内部模型 ({chosen}, 决策调用: {self.gatekeeper.stats_jev_evaluations}次)"
        else:
            jev_info = "纯规则兜底模式"

        stats = (
            f"🛡️ 【{DISPLAY_NAME} v{VERSION} 运行状态】\n"
            f"• 门禁开关: {'开启' if self.gatekeeper.enabled else '关闭'}\n"
            f"• 消息防抖: {'开启' if self.enable_debounce else '关闭'} (超时: {self.debounce_timeout:.0f}s, 阈值: {jev.debounce_threshold})\n"
            f"• 语义决策: {jev_info}\n"
            f"• 触发阈值: {self.gatekeeper.at_threshold} 次\n"
            f"• 统计窗口: {self.gatekeeper.window_seconds:.0f} 秒\n"
            f"• 纯空@拦截: {'开启' if self.gatekeeper.suppress_empty_at else '关闭'}\n"
            f"• 生成期抑制: {'开启' if self.gatekeeper.suppress_during_active_run else '关闭'}\n"
            f"• 回复保护期: {self.gatekeeper.cooldown_after_reply_sec:.0f} 秒\n"
            f"• 门禁拦截总计: {self.gatekeeper.stats_intercepted_total} 次\n"
        )
        if self.gatekeeper.stats_intercepted_by_reason:
            stats += "• 拦截原因细分:\n"
            for r, count in sorted(
                self.gatekeeper.stats_intercepted_by_reason.items(),
                key=lambda x: -x[1],
            ):
                stats += f"  - {r}: {count} 次\n"
        else:
            stats += "• 拦截原因细分: 暂无拦截记录\n"

        await event.send(event.plain_result(stats.strip()))
