import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import datetime

from langchain_core.tools import tool
from rag.rag_service import RagSummarizeService
from utils.config_handler import agent_config
from utils.path_tool import get_abs_path
from utils.logger_handler import logger
from utils.http_client import get_json


rag_service = RagSummarizeService()

# 外部数据缓存（按文件修改时间自动重载）
external_data = {}
_external_data_mtime = None

# 报告生成的共享上下文，由 fill_context_for_report 准备，供报告流程使用
report_context = {"user_id": None, "month": None, "ready": False}

# 当前演示用户 ID（项目暂无登录体系，可通过环境变量 CURRENT_USER_ID 覆盖）
DEFAULT_USER_ID = "1001"

# Open-Meteo WMO 天气码 -> 中文描述
_WEATHER_CODE_MAP = {
    0: "晴", 1: "大部晴朗", 2: "局部多云", 3: "阴",
    45: "雾", 48: "冻雾",
    51: "毛毛雨", 53: "毛毛雨", 55: "毛毛雨",
    56: "冻毛毛雨", 57: "冻毛毛雨",
    61: "小雨", 63: "中雨", 65: "大雨",
    66: "冻雨", 67: "冻雨",
    71: "小雪", 73: "中雪", 75: "大雪", 77: "雪粒",
    80: "阵雨", 81: "强阵雨", 82: "暴雨",
    85: "阵雪", 86: "强阵雪",
    95: "雷暴", 96: "雷暴伴冰雹", 99: "雷暴伴冰雹",
}


def _resolve_user_id() -> str:
    """解析当前用户 ID。项目暂无登录体系，默认返回演示用户，可用环境变量覆盖。"""
    return os.getenv("CURRENT_USER_ID", DEFAULT_USER_ID)


def _resolve_current_month() -> str:
    """获取当前月份，格式 YYYY-MM。"""
    return datetime.now().strftime("%Y-%m")


@tool(description="根据查询问题，从知识库中提取相关文档并生成摘要")
def rag_summarize(query: str) -> str:
    return rag_service.rag_summarize(query)


@tool(description="根据城市名称，获取该城市的实时天气、温度、湿度、降雨概率，以消息字符串形式返回")
def get_weather(city: str) -> str:
    # 1. 城市名 -> 经纬度（Open-Meteo 地理编码，免 key）
    geo = get_json(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1, "language": "zh", "format": "json"},
    )
    if not geo or not geo.get("results"):
        logger.warning(f"未能解析城市[{city}]的地理位置")
        return f"暂未查询到{city}的天气信息"

    location = geo["results"][0]
    lat, lon = location.get("latitude"), location.get("longitude")

    # 2. 经纬度 -> 实时天气
    data = get_json(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,precipitation_probability,weather_code",
            "timezone": "auto",
        },
    )
    if not data or "current" not in data:
        logger.warning(f"未能获取城市[{city}]的实时天气")
        return f"暂未查询到{city}的天气信息"

    current = data["current"]
    desc = _WEATHER_CODE_MAP.get(current.get("weather_code"), "未知天气")
    temp = current.get("temperature_2m")
    humidity = current.get("relative_humidity_2m")
    rain_prob = current.get("precipitation_probability") or 0
    return (f"{city}当前天气：{desc}，气温{temp}℃，"
            f"相对湿度{humidity}%，降雨概率{rain_prob}%")


@tool(description="获取用户当前所在城市，以纯字符串形式返回")
def get_user_location() -> str:
    # 通过 IP 定位 API 自动获取用户所在城市（工具自定位，无需手动配置）
    data = get_json(
        "http://ip-api.com/json/",
        params={"lang": "zh-CN", "fields": "status,country,regionName,city"},
    )
    if not data or data.get("status") != "success":
        logger.warning("IP 定位失败，返回未知城市")
        return "未知"
    return data.get("city") or data.get("regionName") or "未知"


@tool(description="获取当前用户ID，以纯字符串形式返回，格式为数字字符串")
def get_user_id() -> str:
    return _resolve_user_id()


@tool(description="获取当前月份，以纯字符串形式返回，格式为YYYY-MM")
def get_current_month() -> str:
    return _resolve_current_month()


@tool(description="为报告生成准备上下文：解析并缓存当前用户ID与目标月份，供报告生成流程使用")
def fill_context_for_report() -> str:
    report_context["user_id"] = _resolve_user_id()
    report_context["month"] = _resolve_current_month()
    report_context["ready"] = True
    logger.info(f"报告上下文已准备: {report_context}")
    return (f"报告上下文已准备完成（用户 {report_context['user_id']}，"
            f"月份 {report_context['month']}）")


def generate_external_data() -> bool:
    """
    从 CSV 加载外部数据到内存，文件更新后自动重载。
    :return: 是否加载成功
    """
    global _external_data_mtime
    external_data_path = get_abs_path(agent_config["external_data_path"])

    if not os.path.exists(external_data_path):
        logger.error(f"外部数据文件不存在: {external_data_path}")
        return False

    mtime = os.path.getmtime(external_data_path)
    if external_data and _external_data_mtime == mtime:
        return True  # 已加载且文件未变更

    external_data.clear()
    try:
        with open(external_data_path, "r", encoding="utf-8") as f:
            lines = f.readlines()[1:]  # 跳过表头

        for line in lines:
            line = line.strip()
            if not line:
                continue
            arr = line.split(",")
            if len(arr) < 6:
                logger.warning(f"外部数据行格式不正确，已跳过: {line}")
                continue
            user_id = arr[0].replace('"', "").strip()
            feature = arr[1].replace('"', "").strip()
            efficiency = arr[2].replace('"', "").strip()
            consumables = arr[3].replace('"', "").strip()
            comparison = arr[4].replace('"', "").strip()
            time = arr[5].replace('"', "").strip()

            external_data.setdefault(user_id, {})[time] = {
                "特征": feature,
                "效率": efficiency,
                "耗材": consumables,
                "对比": comparison,
            }
        _external_data_mtime = mtime
        return True
    except Exception as e:
        logger.error(f"加载外部数据失败: {e}")
        return False


@tool(description="根据用户ID和月份(YYYY-MM)，获取该用户在该月份的扫地机器人使用记录，以消息字符串形式返回")
def fetch_external_data(user_id: str, month: str) -> str:
    if not generate_external_data():
        return "使用记录数据当前不可用，请稍后再试"

    def _lookup(uid: str, m: str):
        return external_data.get(uid, {}).get(m)

    record = _lookup(user_id, month)
    # 安全兜底：模型偶发未使用 get_current_month 返回的真实月份（如臆造日期）。
    # 报告场景本就针对"当前月份"，依次回退：报告上下文月份 → 系统当前月份。
    if not record:
        fallback_months = []
        if report_context.get("ready") and report_context.get("month"):
            fallback_months.append(report_context["month"])
        cur_month = datetime.now().strftime("%Y-%m")
        if cur_month != month:
            fallback_months.append(cur_month)
        for fm in fallback_months:
            rec = _lookup(user_id, fm)
            if rec:
                logger.warning(f"传入月份 {month} 无数据，回退到 {fm}")
                record = rec
                month = fm
                break

    if not record:
        logger.error(f"未能检索到用户ID {user_id} 在 {month} 的外部数据")
        return f"未查询到用户 {user_id} 在 {month} 的使用记录"

    return (f"用户 {user_id} 在 {month} 的使用记录："
            f"使用特征：{record['特征']}；"
            f"清洁效率：{record['效率']}；"
            f"耗材状态：{record['耗材']}；"
            f"对比：{record['对比']}")
