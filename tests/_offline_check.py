"""断网跑测试：证明测试套件不访问外部主机。

只拦截非本地地址 —— Starlette 的 TestClient 内部会用本地 socket 做 ASGI 传输，
那不是外部网络调用。

用法: uv run python tests/_offline_check.py
"""
import socket
import sys

import pytest

_LOCAL = {"127.0.0.1", "::1", "localhost", "0.0.0.0"}
_orig = socket.socket.connect


def _guard(self, addr):
    host = addr[0] if isinstance(addr, tuple) else addr
    if isinstance(host, str) and host not in _LOCAL:
        raise AssertionError(f"测试访问了外部主机: {addr!r}")
    return _orig(self, addr)


socket.socket.connect = _guard
_orig_gai = socket.getaddrinfo


def _gai(host, *a, **k):
    if isinstance(host, str) and host not in _LOCAL:
        raise AssertionError(f"测试解析了外部域名: {host!r}")
    return _orig_gai(host, *a, **k)


socket.getaddrinfo = _gai
sys.exit(pytest.main(["tests/", "-q", "--no-header"]))
