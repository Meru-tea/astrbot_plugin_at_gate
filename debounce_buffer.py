# -*- coding: utf-8 -*-
"""消息防抖缓冲区管理（MessageBuffer）

用于前置收集用户分多次发送的碎片消息与图片，并在合适时机合并为完整请求。
参考并兼容 astrbot_plugin_debounce 的缓冲区架构。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


def image_source(image: Any) -> str:
    """提取图片组件的可识别源地址或文件路径"""
    if isinstance(image, str):
        return image
    return getattr(image, "url", None) or getattr(image, "file", None) or ""


def get_image_components(event: Any) -> list[Any]:
    """提取消息中的真实图片组件（纯鸭子类型）"""
    message = getattr(getattr(event, "message_obj", None), "message", None) or []
    return [
        part for part in message
        if getattr(part, "type", None) == "Image"
        or "Image" in type(part).__name__
        or hasattr(part, "url")
        or hasattr(part, "file")
    ]


@dataclass
class MessageBuffer:
    """会话消息缓冲区，按 session 暂存用户尚未说完的碎片消息"""

    messages: list[str] = field(default_factory=list)
    images: list[Any] = field(default_factory=list)
    image_urls: list[str] = field(default_factory=list)
    last_update: float = field(default_factory=time.time)
    event: Any = None

    def add(
        self,
        message: str,
        event: Any = None,
        image_urls: list[str] | None = None,
    ) -> None:
        if message:
            # 避免相邻重复完全相同的文本行无意义累加
            if not self.messages or self.messages[-1] != message:
                self.messages.append(message)
        self.last_update = time.time()

        if event:
            self.event = event
            known = {image_source(img) for img in self.images}
            for img in get_image_components(event):
                src = image_source(img)
                if src and src not in known:
                    self.images.append(img)
                    known.add(src)
                elif not src:
                    self.images.append(img)

        if image_urls:
            for url in image_urls:
                if url not in self.image_urls:
                    self.image_urls.append(url)

    def get_full_text(self) -> str:
        """获取合并后的完整文本（以空格拼接）"""
        return " ".join([m for m in self.messages if m.strip()]).strip()

    def clear(self) -> None:
        """清空缓冲区"""
        self.messages.clear()
        self.images.clear()
        self.image_urls.clear()
        self.event = None

    def is_timeout(self, timeout_seconds: float) -> bool:
        """判断是否超过防抖等待时长"""
        return (time.time() - self.last_update) >= timeout_seconds
