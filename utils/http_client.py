"""
统一的 HTTP 请求工具：对外部 API 的调用提供超时与自动重试能力。
所有需要访问第三方接口的工具都应通过这里的函数发起请求，避免散落各处的重复实现。
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from utils.logger_handler import logger

# 默认超时时间（秒）
DEFAULT_TIMEOUT = 10
# 默认重试次数
DEFAULT_RETRIES = 3
# 触发重试的 HTTP 状态码
_RETRY_STATUS_CODES = (429, 500, 502, 503, 504)


def _build_session(total_retries: int = DEFAULT_RETRIES, backoff: float = 0.5) -> requests.Session:
    """
    构建一个带自动重试的 requests.Session。
    :param total_retries: 最大重试次数
    :param backoff: 退避因子，重试间隔 = backoff * (2 ** 已重试次数)
    """
    session = requests.Session()
    retry = Retry(
        total=total_retries,
        connect=total_retries,
        read=total_retries,
        backoff_factor=backoff,
        status_forcelist=_RETRY_STATUS_CODES,
        allowed_methods=frozenset(["GET", "POST"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


# 模块级共享会话，复用连接
_session = _build_session()


def get_json(url: str, params: dict = None, timeout: int = DEFAULT_TIMEOUT,
             headers: dict = None) -> dict | None:
    """
    发起 GET 请求并解析 JSON 响应。
    任何网络/解析异常都会被捕获并记录日志，返回 None，由调用方做优雅降级。
    :param url: 请求地址
    :param params: query 参数
    :param timeout: 超时时间（秒）
    :param headers: 请求头
    :return: 解析后的 dict；失败返回 None
    """
    try:
        resp = _session.get(url, params=params, timeout=timeout, headers=headers)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"HTTP GET 请求失败: {url} params={params} err={e}")
        return None
    except ValueError as e:  # JSON 解析失败
        logger.error(f"HTTP 响应 JSON 解析失败: {url} err={e}")
        return None
