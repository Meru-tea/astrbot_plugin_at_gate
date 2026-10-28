# -*- coding: utf-8 -*-
"""智能艾特门禁核心判定器（AtGatekeeper）

支持 JEV 语义意图模型决策与纯规则状态机兜底。
"""
from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import Any

from .jev_client import JevClient


@dataclass
class AtRecord:
    timestamp: float
    text: str
    has_media: bool


@dataclass
class Decision:
    should_block: bool
    reason: str
    at_count: int
    is_active_run: bool
    score: float | None = None


def is_at_bot(event: Any) -> bool:
    """检查事件是否 @ 了当前机器人（纯鸭子类型，零依赖）"""
    if getattr(event, "is_at_or_wake_command", False):
        return True
    get_self_id = getattr(event, "get_self_id", None)
    self_id = str(get_self_id() if callable(get_self_id) else getattr(event, "self_id", "") or "")
    message_obj = getattr(event, "message_obj", None)
    message = getattr(message_obj, "message", []) or []
    for comp in message:
        cname = type(comp).__name__
        if cname in ("At", "AtComp", "AtAll") or "At" in cname or hasattr(comp, "qq"):
            qq = str(getattr(comp, "qq", ""))
            if qq in (self_id, "all") or getattr(comp, "all", False):
                return True
    return False


def extract_content(event: Any) -> tuple[str, bool]:
    """提取消息中的纯文本与媒体标识（纯鸭子类型，零依赖）"""
    plain_parts: list[str] = []
    has_media = False
    message_obj = getattr(event, "message_obj", None)
    message = getattr(message_obj, "message", []) or []
    for comp in message:
        cname = type(comp).__name__
        if cname in ("Plain", "PlainComp") or "Plain" in cname or (hasattr(comp, "text") and not hasattr(comp, "qq")):
            text = getattr(comp, "text", "") or ""
            if text:
                plain_parts.append(text)
        elif cname in ("Image", "ImageComp", "Record", "Video") or any(m in cname for m in ("Image", "Record", "Video", "Audio")):
            has_media = True

    full_text = "".join(plain_parts)
    if not full_text:
        full_text = getattr(event, "message_str", "") or ""
    return full_text, has_media


class AtGatekeeper:
    def __init__(
        self,
        config: dict[str, Any] | None = None,
        provider_getter: Any = None,
    ) -> None:
        self.jev_client = JevClient(config or {}, provider_getter=provider_getter)
        self.update_config(config or {})
        # key: (umo, sender_id) -> list[AtRecord]
        self._history: dict[tuple[str, str], list[AtRecord]] = {}
        # key: (umo, sender_id) -> start_timestamp
        self._active_runs: dict[tuple[str, str], float] = {}
        # key: (umo, sender_id) -> text of current in-flight question
        self._active_run_questions: dict[tuple[str, str], str] = {}
        # key: umo -> set(sender_id) currently active in this umo
        self._active_umo_runs: dict[str, set[str]] = {}
        # key: (umo, sender_id) -> last_reply_timestamp
        self._last_reply_times: dict[tuple[str, str], float] = {}
        # 拦截统计
        self.stats_intercepted_total: int = 0
        self.stats_intercepted_by_reason: dict[str, int] = {}
        self.stats_jev_evaluations: int = 0

    def update_config(self, config: dict[str, Any]) -> None:
        self.enabled: bool = bool(config.get("enabled", True))
        self.at_threshold: int = int(config.get("at_threshold", 2))
        self.window_seconds: float = float(config.get("window_seconds", 30))
        self.suppress_empty_at: bool = bool(config.get("suppress_empty_at", True))
        self.suppress_during_active_run: bool = bool(
            config.get("suppress_during_active_run", True)
        )
        self.cooldown_after_reply_sec: float = float(
            config.get("cooldown_after_reply_sec", 5)
        )
        self.allow_distinct_questions: bool = bool(
            config.get("allow_distinct_questions", True)
        )
        self.min_distinct_length: int = int(config.get("min_distinct_length", 3))
        self.log_intercept: bool = bool(config.get("log_intercept", True))

        raw_keywords = config.get(
            "rush_keywords",
            "?,？,??,？？,在吗,在么,人呢,说话,快点,出来,喂,hi,hello,ciallo",
        )
        if isinstance(raw_keywords, str):
            parts = [p.strip().lower() for p in raw_keywords.replace("，", ",").split(",") if p.strip()]
            self.rush_keywords: set[str] = set(parts)
        elif isinstance(raw_keywords, (list, set)):
            self.rush_keywords = {str(k).strip().lower() for k in raw_keywords if str(k).strip()}
        else:
            self.rush_keywords = set()

        self.jev_client.update_config(config)

    def mark_run_begin(
        self,
        umo: str,
        sender_id: str,
        question_text: str = "",
        now: float | None = None,
    ) -> None:
        """记录某个 UMO / 发送者的 Agent 开始执行"""
        ts = now if now is not None else time.time()
        key = (umo, str(sender_id))
        self._active_runs[key] = ts
        if question_text:
            self._active_run_questions[key] = self.clean_text(question_text)
        if umo not in self._active_umo_runs:
            self._active_umo_runs[umo] = set()
        self._active_umo_runs[umo].add(str(sender_id))

    def mark_run_done(self, umo: str, sender_id: str, now: float | None = None) -> None:
        """记录某个 UMO / 发送者的 Agent 执行结束，记录回复完成时间戳"""
        ts = now if now is not None else time.time()
        key = (umo, str(sender_id))
        self._active_runs.pop(key, None)
        self._active_run_questions.pop(key, None)
        if umo in self._active_umo_runs:
            self._active_umo_runs[umo].discard(str(sender_id))
            if not self._active_umo_runs[umo]:
                self._active_umo_runs.pop(umo, None)
        self._last_reply_times[key] = ts

    def clean_text(self, text: str) -> str:
        """清理纯文本（去除首尾空白、常见 At 标记残留）"""
        if not text:
            return ""
        t = re.sub(r"\[At:[^\]]+\]", "", text)
        t = re.sub(r"@\S+", "", t)
        return t.strip()

    def is_rush_phrase(self, text: str) -> bool:
        """判断是否为常见催促词或无意义短语（规则兜底用）"""
        cleaned = self.clean_text(text).lower()
        if not cleaned:
            return True
        if cleaned in self.rush_keywords:
            return True
        stripped_punct = re.sub(r"[\s\?？!！\.。…~～、，,\-_/]+", "", cleaned)
        if not stripped_punct:
            return True
        if stripped_punct in self.rush_keywords:
            return True
        return False

    async def evaluate(
        self,
        umo: str,
        sender_id: str,
        is_at_bot: bool,
        raw_text: str,
        has_media: bool = False,
        now: float | None = None,
    ) -> Decision:
        """评估一条到达的消息是否应当被智能门禁拦截（支持 JEV 语义模型决策与规则兜底）"""
        ts = now if now is not None else time.time()
        key = (umo, str(sender_id))

        if not self.enabled:
            return Decision(should_block=False, reason="plugin_disabled", at_count=0, is_active_run=False)

        if not is_at_bot:
            return Decision(should_block=False, reason="not_at_bot", at_count=0, is_active_run=False)

        text = self.clean_text(raw_text)
        is_empty = (not text) and (not has_media)

        # 检查当前是否处于活跃生成中（超时超过 120s 视作孤儿丢弃）
        is_active = False
        in_flight_question = ""
        if key in self._active_runs:
            run_start = self._active_runs[key]
            if ts - run_start < 120.0:
                is_active = True
                in_flight_question = self._active_run_questions.get(key, "")
            else:
                self._active_runs.pop(key, None)
                self._active_run_questions.pop(key, None)
        else:
            # 双重感知：检查 AstrBot 核心是否有活跃的 Agent Runner（无缝兼容各类陪伴插件的长耗时生成）
            try:
                from astrbot.core.pipeline.process_stage.follow_up import _ACTIVE_AGENT_RUNNERS
                if umo in _ACTIVE_AGENT_RUNNERS:
                    is_active = True
            except Exception:
                pass
        # 统计滑动时间窗口历史
        records = self._history.setdefault(key, [])
        cutoff = ts - self.window_seconds
        records = [r for r in records if r.timestamp >= cutoff]
        records.append(AtRecord(timestamp=ts, text=text, has_media=has_media))
        self._history[key] = records
        count = len(records)

        # 1. 纯空 @（无任何文本且无媒体）：手抖连发或无意义点按
        # 除非配置了强制将纯空@也交给Jev（jev_judge_all），否则快速规则拦截，0 Token 0 延迟
        if is_empty and self.suppress_empty_at and not self.jev_client.judge_all:
            if is_active and self.suppress_during_active_run:
                return self._intercept("empty_at_during_active_run", key, ts, text, has_media, is_active)
            last_reply = self._last_reply_times.get(key, 0.0)
            if (ts - last_reply) < self.cooldown_after_reply_sec:
                return self._intercept("empty_at_within_reply_cooldown", key, ts, text, has_media, is_active)
            if count > self.at_threshold:
                return self._intercept(f"empty_at_threshold_exceeded(count={count})", key, ts, text, has_media, is_active)

        # 2. 如果启用了 JEV 语义模型，当满足触发条件时优先交由 JEV 模型做出概率判定
        needs_jev_judgment = False
        if self.jev_client.is_available:
            if is_active and self.suppress_during_active_run:
                needs_jev_judgment = True
            elif count > self.at_threshold:
                needs_jev_judgment = True
            elif self.jev_client.judge_all:
                needs_jev_judgment = True

        if needs_jev_judgment:
            self.stats_jev_evaluations += 1
            context = {
                "bot_status": "generating_reply" if is_active else "idle",
                "at_count": count,
                "previous_text": in_flight_question or (records[-2].text if len(records) >= 2 else ""),
                "dialogue_context": f"用户在最近{self.window_seconds:.0f}秒内连续@了{count}次，上一轮状态为{'正在生成回答' if is_active else '空闲'}。",
            }
            should_respond, score, reason = await self.jev_client.evaluate_should_respond(
                text=raw_text or "（纯艾特，无文字）",
                context=context,
            )

            # JEV 给出有效概率分数（score >= 0）
            if score >= 0.0:
                if not should_respond:
                    # Jev 判定不需要回复（催促/多余） -> 拦截
                    return self._intercept(f"jev_rejected(p={score:.3f},count={count})", key, ts, text, has_media, is_active, score=score)
                else:
                    # Jev 判定需要回复（有新需求/新问题） -> 放行
                    return Decision(
                        should_block=False,
                        reason=f"jev_approved(p={score:.3f},count={count})",
                        at_count=count,
                        is_active_run=is_active,
                        score=score,
                    )
            # JEV 网络异常/超时等导致 score == -1 -> 降级走下面的规则兜底

        # 3. 规则兜底逻辑（JEV未配置或调用降级）
        is_rush = self.is_rush_phrase(raw_text) and (not has_media)

        # 3.1 活跃生成中
        if is_active and self.suppress_during_active_run:
            if is_empty and self.suppress_empty_at:
                return self._intercept("empty_at_during_active_run", key, ts, text, has_media, is_active)
            if is_rush:
                return self._intercept("rush_during_active_run", key, ts, text, has_media, is_active)

        # 3.2 回复冷却期内
        last_reply = self._last_reply_times.get(key, 0.0)
        is_within_cooldown = (ts - last_reply) < self.cooldown_after_reply_sec
        if is_within_cooldown:
            if is_empty and self.suppress_empty_at:
                return self._intercept("empty_at_within_reply_cooldown", key, ts, text, has_media, is_active)
            if is_rush:
                return self._intercept("rush_within_reply_cooldown", key, ts, text, has_media, is_active)

        # 3.3 窗口内频次超标
        if count > self.at_threshold:
            if is_empty and self.suppress_empty_at:
                return self._intercept(f"empty_at_threshold_exceeded(count={count})", key, ts, text, has_media, is_active)
            if is_rush:
                return self._intercept(f"rush_threshold_exceeded(count={count})", key, ts, text, has_media, is_active)

            if self.allow_distinct_questions and len(text) >= self.min_distinct_length and not is_rush:
                prev_texts = [r.text for r in records[:-1] if r.text]
                if prev_texts and text == prev_texts[-1]:
                    return self._intercept(f"duplicate_text_threshold_exceeded(count={count})", key, ts, text, has_media, is_active)
                return Decision(should_block=False, reason=f"distinct_question_allowed(count={count})", at_count=count, is_active_run=is_active)

            return self._intercept(f"multiple_at_rate_limited(count={count})", key, ts, text, has_media, is_active)

        return Decision(should_block=False, reason="normal", at_count=count, is_active_run=is_active)

    def _intercept(
        self,
        reason: str,
        key: tuple[str, str],
        ts: float,
        text: str,
        has_media: bool,
        is_active: bool,
        score: float | None = None,
    ) -> Decision:
        self.stats_intercepted_total += 1
        self.stats_intercepted_by_reason[reason] = (
            self.stats_intercepted_by_reason.get(reason, 0) + 1
        )
        count = len(self._history.get(key, []))
        return Decision(should_block=True, reason=reason, at_count=count, is_active_run=is_active, score=score)

    def cleanup_expired(self, now: float | None = None) -> None:
        """定期清理过期历史记录与超时状态"""
        ts = now if now is not None else time.time()
        cutoff = ts - self.window_seconds

        expired_keys = []
        for key, records in self._history.items():
            valid_records = [r for r in records if r.timestamp >= cutoff]
            if valid_records:
                self._history[key] = valid_records
            else:
                expired_keys.append(key)

        for key in expired_keys:
            self._history.pop(key, None)

        expired_runs = [k for k, start_ts in self._active_runs.items() if ts - start_ts > 180.0]
        for k in expired_runs:
            self._active_runs.pop(k, None)
            self._active_run_questions.pop(k, None)
            umo, sender = k
            if umo in self._active_umo_runs:
                self._active_umo_runs[umo].discard(sender)
                if not self._active_umo_runs[umo]:
                    self._active_umo_runs.pop(umo, None)
