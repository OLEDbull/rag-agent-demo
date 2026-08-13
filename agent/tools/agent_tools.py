import sys
import os
import re
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import datetime

from langchain_core.tools import tool
from rag.rag_service import RagSummarizeService
from utils.config_handler import agent_config
from utils.path_tool import get_abs_path
from utils.logger_handler import logger
from utils.http_client import get_json, get_text


rag_service = RagSummarizeService()

# 外部数据缓存（按文件修改时间自动重载）
external_data = {}
_external_data_mtime = None

# 报告生成的共享上下文，由 fill_context_for_report 准备，供报告流程使用
report_context = {"user_id": None, "month": None, "ready": False}

# 当前演示用户 ID（项目暂无登录体系，可通过环境变量 CURRENT_USER_ID 覆盖）
DEFAULT_USER_ID = "1001"

# 客户端真实 IP 注入点：由 Streamlit 前端在每次请求时设置（set_client_ip）
# 不设置时为 None，get_user_location 将退化为按服务端 IP 定位（在远程部署下不准）
_client_ip = None


def set_client_ip(ip: str):
    """供 Streamlit 前端调用：把当前浏览器请求的真实客户端 IP 注入到工具层"""
    global _client_ip
    _client_ip = (ip or "").strip() or None

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


def _geocode_city(city: str) -> dict | None:
    """
    智能地理编码：将城市名解析为经纬度。
    解决 Open-Meteo 中文搜索时同名小地名排在大城市前面的问题：
    - 多取几个候选结果（count=5）
    - 优先选择有 population 字段的（大城市有人口数据，小村庄/岛屿没有）
    - 若首轮无人口结果，尝试追加"市"字重搜（如"青岛"→"青岛市"）
    - 多个人口结果中选人口最多的
    """
    def _search(name: str) -> list:
        g = get_json(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": name, "count": 5, "language": "zh", "format": "json"},
        )
        return g.get("results", []) if g else []

    def _pick_best(results: list) -> dict | None:
        """从候选结果中选最优：优先有人口的（选人口最多的），否则取第一个。"""
        if not results:
            return None
        with_pop = [r for r in results if r.get("population")]
        if with_pop:
            best = max(with_pop, key=lambda r: r.get("population", 0))
            logger.info(f"地理编码选中(人口优先): {best.get('name')}, {best.get('admin1')}, "
                        f"pop={best.get('population')}, lat={best.get('latitude')}, lon={best.get('longitude')}")
            return best
        logger.info(f"地理编码选中(首个): {results[0].get('name')}, {results[0].get('admin1')}")
        return results[0]

    # 首轮搜索
    results = _search(city)
    best = _pick_best(results)
    if best and best.get("population"):
        return best

    # 首轮无人口结果，尝试追加"市"字（中文城市常见行政后缀）
    if not city.endswith("市"):
        city_shi = city + "市"
        logger.info(f"首轮未找到大城市，尝试搜索: {city_shi}")
        results2 = _search(city_shi)
        best2 = _pick_best(results2)
        if best2:
            return best2

    return best


@tool(description="根据城市名称，获取该城市的实时天气、温度、湿度、降雨概率，以消息字符串形式返回")
def get_weather(city: str) -> str:
    # 1. 城市名 -> 经纬度（智能地理编码）
    location = _geocode_city(city)
    if not location:
        logger.warning(f"未能解析城市[{city}]的地理位置")
        return f"暂未查询到{city}的天气信息"

    lat, lon = location.get("latitude"), location.get("longitude")
    resolved_name = location.get("name", city)
    logger.info(f"天气查询: {city} -> {resolved_name} ({lat}, {lon})")

    # 2. 经纬度 -> 实时天气（增加云量和体感温度，描述更准确）
    data = get_json(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,precipitation_probability,weather_code,cloud_cover",
            "timezone": "auto",
        },
    )
    if not data or "current" not in data:
        logger.warning(f"未能获取城市[{city}]的实时天气")
        return f"暂未查询到{city}的天气信息"

    current = data["current"]
    wc = current.get("weather_code")
    cloud = current.get("cloud_cover", 0)

    # 天气描述：结合天气码和云量做更精确的判断
    if wc == 0:
        desc = "晴" if cloud < 20 else "晴间多云"
    elif wc == 1:
        desc = "大部晴朗" if cloud < 50 else "多云"
    elif wc == 2:
        desc = "局部多云" if cloud < 70 else "多云"
    elif wc == 3:
        desc = "阴"
    elif wc in (45, 48):
        desc = "雾"
    elif wc in (51, 56):
        desc = "小毛毛雨"
    elif wc in (53,):
        desc = "毛毛雨"
    elif wc in (55, 57):
        desc = "大毛毛雨"
    elif wc == 61:
        desc = "小雨"
    elif wc == 63:
        desc = "中雨"
    elif wc == 65:
        desc = "大雨"
    elif wc in (66, 67):
        desc = "冻雨"
    elif wc == 71:
        desc = "小雪"
    elif wc == 73:
        desc = "中雪"
    elif wc == 75:
        desc = "大雪"
    elif wc == 77:
        desc = "雪粒"
    elif wc == 80:
        desc = "阵雨"
    elif wc == 81:
        desc = "强阵雨"
    elif wc == 82:
        desc = "暴雨"
    elif wc in (85,):
        desc = "阵雪"
    elif wc == 86:
        desc = "强阵雪"
    elif wc in (95, 96, 99):
        desc = "雷暴"
    else:
        desc = _WEATHER_CODE_MAP.get(wc, "未知天气")

    temp = current.get("temperature_2m")
    humidity = current.get("relative_humidity_2m")
    rain_prob = current.get("precipitation_probability") or 0
    return (f"{resolved_name}当前天气：{desc}，气温{temp}℃，"
            f"相对湿度{humidity}%，降雨概率{rain_prob}%")


def _locate_by_cipcc(target_ip: str = None) -> str | None:
    """
    通过国内 IP 定位服务 cip.cc 获取城市名（中文，对国内 IP/教育网 IP 准确度高）。
    :param target_ip: 指定查询的 IP；None 表示查询调用方自身公网 IP
    :return: 城市名（如"青岛"）；失败返回 None
    """
    url = f"http://cip.cc/{target_ip}" if target_ip else "http://cip.cc/"
    # cip.cc 对浏览器 UA 返回 HTML 页面，对 curl/脚本类 UA 返回纯文本
    text = get_text(url, headers={"User-Agent": "curl/8.0.0"}, encoding="utf-8")
    if not text:
        return None
    # 解析格式示例：
    #   IP	: 222.206.18.169
    #   地址	: 中国 山东 青岛
    #   运营商	: 教育网
    m = re.search(r'地址\s*:\s*(.+)', text)
    if not m:
        logger.warning(f"cip.cc 返回格式异常，无法解析地址: {text[:200]}")
        return None
    addr_line = m.group(1).strip()
    parts = [p.strip() for p in addr_line.split() if p.strip()]
    # parts 形如 ["中国", "山东", "青岛"] 或 ["中国", "北京", "北京"]
    if len(parts) >= 3:
        city = parts[2]  # 最后一段为城市
    elif len(parts) == 2:
        city = parts[1]  # 直辖市
    else:
        city = addr_line
    # 去除"省""市"后缀的情况（cip.cc 通常不带，但做兼容）
    city = city.rstrip("市省")
    logger.info(f"cip.cc 定位结果: IP={target_ip or '(自身)'}, 地址行='{addr_line}', 城市='{city}'")
    return city or None


def _locate_by_ipapi(target_ip: str = None) -> str | None:
    """
    通过 ip-api.com（国外 IP 地理库）获取城市名，作为 cip.cc 的兜底。
    注意：该库对国内教育网/运营商 NAT IP 的城市定位可能不准确。
    """
    params = {"lang": "zh-CN", "fields": "status,country,regionName,city"}
    if target_ip:
        params["ip"] = target_ip
    url = f"http://ip-api.com/json/{target_ip}" if target_ip else "http://ip-api.com/json/"
    data = get_json(url, params=params)
    if not data or data.get("status") != "success":
        logger.warning(f"ip-api 定位失败: {data}")
        return None
    city = data.get("city") or data.get("regionName")
    if city:
        city = city.rstrip("市省")
    logger.info(f"ip-api 定位结果: IP={target_ip or '(自身)'}, 城市='{city}'")
    return city or None


@tool(description="获取用户当前所在城市，以纯字符串形式返回")
def get_user_location() -> str:
    """
    IP 定位策略：
    1. 优先使用前端注入的真实客户端 IP（生产/远程部署）；无注入则查询服务端公网 IP
    2. 主用国内服务 cip.cc（对中国 IP/教育网 IP 准确度高）
    3. cip.cc 失败时回退到 ip-api.com（国际库，国内准确度次之）
    4. 全部失败返回"未知"，由上层 prompt 引导用户手动提供城市
    """
    target_ip = _client_ip  # 可能为 None
    if target_ip:
        logger.info(f"使用前端注入的客户端 IP 定位: {target_ip}")
    else:
        logger.info("未注入客户端 IP，查询服务端自身公网 IP 定位")

    # 主链路：cip.cc（国内库）
    city = _locate_by_cipcc(target_ip)
    if city:
        return city

    # 兜底：ip-api（国际库）
    logger.warning("cip.cc 定位失败，回退到 ip-api.com")
    city = _locate_by_ipapi(target_ip)
    if city:
        return city

    logger.error("所有 IP 定位服务均失败，返回未知")
    return "未知"


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
