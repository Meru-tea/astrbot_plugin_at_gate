# -*- coding: utf-8 -*-
try:
    from .main import AtGatePlugin

    __all__ = ["AtGatePlugin"]
except ImportError:
    # 允许在未安装 AstrBot 的开发/自检环境中运行纯逻辑单元测试
    pass
