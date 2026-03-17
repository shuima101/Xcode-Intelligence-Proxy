"""
Xcode AI Proxy - Python 版本
使用 FastAPI 重写的 AI 代理服务，支持智谱 GLM-4.6、Kimi 和 DeepSeek 模型
根据 models.toml 配置文件动态加载可用模型
"""

import os
import sys
import asyncio
import logging
import time
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, Union
import json
import toml

import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# 加载模型配置文件
try:
    with open("models.toml", "r") as f:
        MODEL_CONFIG = toml.load(f)
except FileNotFoundError:
    logger.error("❌ 模型配置文件 models.toml 未找到")
    sys.exit(1)

# 服务器配置 - 从TOML文件读取
SERVER_CONFIG = MODEL_CONFIG.get("server", {})
PORT = int(SERVER_CONFIG.get("port", 3000))
HOST = SERVER_CONFIG.get("host", "0.0.0.0")

# 重试配置 - 从TOML文件读取
MAX_RETRIES = int(SERVER_CONFIG.get("max_retries", 3))
RETRY_DELAY = int(SERVER_CONFIG.get("retry_delay", 1000)) / 1000  # 转换为秒
REQUEST_TIMEOUT = int(SERVER_CONFIG.get("request_timeout", 60000)) / 1000  # 转换为秒

# 供应商配置 - 从配置文件加载供应商信息（不包含密钥和模型列表）
PROVIDER_CONFIGS = {}

# 遍历配置文件中的所有模型供应商
for provider, config in MODEL_CONFIG.items():
    if provider in ["server", "env_vars", "custom"]:  # 跳过非供应商配置
        continue

    # 获取供应商的基础配置
    provider_type = config.get("type")
    api_url = config.get("api_url")

    if not provider_type or not api_url:
        logger.warning(f"⚠️ 供应商 {provider} 配置不完整，跳过")
        continue

    # 支持通过环境变量覆盖API URL
    api_url_env_var = f"{provider.upper()}_BASE_URL"
    api_url = os.getenv(api_url_env_var, api_url)

    PROVIDER_CONFIGS[provider] = {
        "api_url": api_url,
        "type": provider_type,
    }
    logger.info(f"  ✅ 加载供应商: {provider} ({provider_type}) - {api_url}")

if not PROVIDER_CONFIGS:
    logger.error("❌ 未配置任何供应商")
    logger.error("请在 models.toml 文件中至少配置一个供应商")
    sys.exit(1)

# 模型路由缓存 - 按客户端 API 密钥缓存 model_id → provider_name 的映射
MODEL_ROUTE_CACHE: Dict[str, Dict[str, str]] = {}
MODEL_CACHE_TIMESTAMP: Dict[str, float] = {}
CACHE_TTL_SECONDS = 300  # 5分钟

logger.info(f"📋 已加载 {len(PROVIDER_CONFIGS)} 个供应商配置")

# FastAPI 应用初始化
app = FastAPI(
    title="Xcode AI Proxy",
    description="AI 代理服务，支持智谱 GLM-4.6、Kimi 和 DeepSeek 模型",
    version="1.0.0",
)

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 请求模型
class ChatCompletionRequest(BaseModel):
    model: str
    messages: list
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None


# 工具函数：提取客户端 API 密钥
def extract_client_key(request: Request) -> Optional[str]:
    """从 Authorization 头提取密钥（支持 Bearer 前缀或直接密钥）"""
    auth = request.headers.get("authorization", "")
    if not auth:
        return None

    # 支持 "Bearer <key>" 格式
    if auth.startswith("Bearer "):
        key = auth[7:].strip()
        return key if key else None

    # 支持直接传密钥（不带 Bearer 前缀）
    key = auth.strip()
    return key if key else None


# 工具函数：向单个供应商请求模型列表
async def fetch_provider_models(provider_name: str, provider_config: dict, api_key: str) -> list:
    """向单个供应商的 /v1/models 端点请求模型列表，失败静默返回空列表"""
    try:
        api_url = provider_config["api_url"]
        # 尝试标准的 /v1/models 端点
        models_url = f"{api_url.rstrip('/')}/v1/models"

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                models_url,
                headers={"Authorization": f"Bearer {api_key}"}
            )

            if response.status_code == 200:
                data = response.json()
                models = data.get("data", [])
                logger.info(f"  ✅ {provider_name}: 获取到 {len(models)} 个模型")
                return [(m.get("id"), provider_name) for m in models if m.get("id")]
            else:
                logger.debug(f"  ⚠️ {provider_name}: HTTP {response.status_code}")
                return []
    except Exception as e:
        logger.debug(f"  ⚠️ {provider_name}: {str(e)}")
        return []


# 工具函数：构建模型路由缓存
async def build_model_cache(api_key: str) -> Dict[str, str]:
    """并发查询所有供应商，构建 model_id → provider_name 的映射并缓存"""
    logger.info(f"🔄 开始构建模型缓存...")

    # 并发查询所有供应商
    tasks = [
        fetch_provider_models(provider_name, provider_config, api_key)
        for provider_name, provider_config in PROVIDER_CONFIGS.items()
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # 聚合结果
    model_to_provider = {}
    for result in results:
        if isinstance(result, list):
            for model_id, provider_name in result:
                if model_id not in model_to_provider:
                    model_to_provider[model_id] = provider_name

    # 缓存结果
    cache_key = api_key[:16] if len(api_key) > 16 else api_key  # 使用密钥前缀作为缓存键
    MODEL_ROUTE_CACHE[cache_key] = model_to_provider
    MODEL_CACHE_TIMESTAMP[cache_key] = time.time()

    logger.info(f"✅ 模型缓存构建完成，共 {len(model_to_provider)} 个模型")
    return model_to_provider


# 通用重试装饰器
async def with_retry(operation, max_retries=MAX_RETRIES, base_delay=RETRY_DELAY):
    """通用异步重试函数"""
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"🔄 第{attempt}次尝试")
            return await operation()
        except Exception as error:
            last_error = error
            logger.error(f"❌ 第{attempt}次尝试失败: {str(error)}")

            if attempt < max_retries:
                delay = base_delay * attempt  # 递增延迟
                logger.info(f"⏳ {delay}秒后重试...")
                await asyncio.sleep(delay)

    logger.error(f"❌ 所有{max_retries}次重试都失败了")
    # 如果没有捕获到具体异常，避免 raise None，提供一个明确的回退错误
    if last_error:
        raise last_error
    else:
        raise RuntimeError("Operation failed after retries with no exception captured")


# 中间件：请求日志
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = datetime.now()
    logger.info(f"{start_time.isoformat()} - {request.method} {request.url.path}")

    # 记录请求头（过滤敏感信息）
    safe_headers = {k: v for k, v in request.headers.items() if k.lower() != "authorization"}
    logger.info(f"请求头: {safe_headers}")

    response = await call_next(request)

    process_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"请求处理时间: {process_time:.3f}秒")
    logger.info(f"响应状态码: {response.status_code}")

    return response


# 健康检查
@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


# 调试端点
@app.get("/debug/config")
async def debug_config():
    """调试配置信息"""
    return {
        "available_providers": list(PROVIDER_CONFIGS.keys()),
        "provider_configs": {
            provider_name: {
                "type": config["type"],
                "api_url": config["api_url"],
            }
            for provider_name, config in PROVIDER_CONFIGS.items()
        },
        "cache_info": {
            "cached_keys": len(MODEL_ROUTE_CACHE),
            "cache_ttl_seconds": CACHE_TTL_SECONDS,
        }
    }


# 模型列表
@app.get("/v1/models")
async def list_models(request: Request):
    """返回支持的模型列表（从上游供应商动态获取）"""
    logger.info("📋 请求模型列表")

    # 提取客户端 API 密钥
    client_api_key = extract_client_key(request)
    if not client_api_key:
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "message": "未提供 API 密钥，请在请求头中设置 'Authorization: Bearer <your-api-key>'",
                    "type": "auth_error"
                }
            }
        )

    # 检查缓存
    cache_key = client_api_key[:16] if len(client_api_key) > 16 else client_api_key
    cache_time = MODEL_CACHE_TIMESTAMP.get(cache_key, 0)
    is_cache_valid = (time.time() - cache_time) < CACHE_TTL_SECONDS

    if not is_cache_valid or cache_key not in MODEL_ROUTE_CACHE:
        # 缓存过期或不存在，重新构建
        logger.info("🔄 缓存过期或不存在，重新获取模型列表")
        model_to_provider = await build_model_cache(client_api_key)
    else:
        logger.info("✅ 使用缓存的模型列表")
        model_to_provider = MODEL_ROUTE_CACHE[cache_key]

    # 构建返回的模型列表
    model_list = [
        {
            "id": model_id,
            "object": "model",
            "created": 1677610602,
            "owned_by": provider_name,
        }
        for model_id, provider_name in model_to_provider.items()
    ]

    logger.info(f"📋 返回 {len(model_list)} 个模型")
    return {"object": "list", "data": model_list}


# 智谱 API 处理
async def handle_zhipu_request(request_body: dict, client_api_key: str, provider_config: dict) -> Union[dict, StreamingResponse]:
    """处理智谱 API 请求"""
    logger.info("📡 路由到智谱API")

    async def make_request():
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            response = await client.post(
                f"{provider_config['api_url']}/chat/completions",
                json={**request_body, "model": "glm-4.6"},
                headers={
                    "Authorization": f"Bearer {client_api_key}",
                    "Content-Type": "application/json",
                },
            )
            # 非 2xx 状态会触发 raise_for_status() 抛出 HTTPStatusError
            response.raise_for_status()
            return response

    response = await with_retry(make_request)
    logger.info(f"✅ 智谱API响应状态: {response.status_code}")

    is_stream = bool(request_body.get("stream", False))
    if is_stream:
        logger.info("🔄 返回智谱流式响应")

        # 直接返回原始流式响应，不修改任何内容
        response_headers = dict(response.headers)
        # 移除可能引起问题的头部
        response_headers.pop("content-length", None)
        response_headers.pop("content-encoding", None)
        response_headers["content-type"] = "text/event-stream; charset=utf-8"

        async def generate():
            async for chunk in response.aiter_bytes(chunk_size=8192):
                yield chunk

        return StreamingResponse(
            generate(), status_code=response.status_code, headers=response_headers
        )
    else:
        logger.info("📦 返回智谱非流式响应")
        return response.json()


# Kimi API 处理
async def handle_kimi_request(request_body: dict, client_api_key: str, provider_config: dict) -> Union[dict, StreamingResponse]:
    """处理 Kimi API 请求"""
    logger.info("📡 路由到Kimi API")

    async def make_request():
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            response = await client.post(
                f"{provider_config['api_url']}/chat/completions",
                json={**request_body, "model": "kimi-k2-0905-preview"},
                headers={
                    "Authorization": f"Bearer {client_api_key}",
                    "Content-Type": "application/json",
                },
            )
            # 非 2xx 状态会触发 raise_for_status() 抛出 HTTPStatusError
            response.raise_for_status()
            return response

    response = await with_retry(make_request)
    logger.info(f"✅ Kimi API响应状态: {response.status_code}")

    if request_body.get("stream", False):
        logger.info("🔄 返回Kimi流式响应")

        # 直接返回原始流式响应，不修改任何内容
        response_headers = dict(response.headers)
        # 移除可能引起问题的头部
        response_headers.pop("content-length", None)
        response_headers.pop("content-encoding", None)
        response_headers["content-type"] = "text/event-stream; charset=utf-8"

        async def generate():
            async for chunk in response.aiter_bytes(chunk_size=8192):
                yield chunk

        return StreamingResponse(
            generate(), status_code=response.status_code, headers=response_headers
        )
    else:
        logger.info("📦 返回Kimi非流式响应")
        return response.json()


# 新增：清洗 messages，确保每条 message['content'] 为字符串
def sanitize_messages(messages):
    """
    确保 messages 是 list，每个 message 为 dict 且 message['content'] 为字符串。
    - 如果 message 是字符串 -> 转为 {'role':'user','content': str}
    - 如果 content 是 list -> 将元素 join（非字符串元素 json.dumps）
    - 其他非字符串 -> json.dumps
    """
    import json

    if not isinstance(messages, list):
        logger.warning("messages 不是列表，已尝试转换为单项列表")
        return [{"role": "user", "content": str(messages)}]

    sanitized = []
    for idx, m in enumerate(messages):
        # 字符串形式的 message，视为 user
        if isinstance(m, str):
            sanitized.append({"role": "user", "content": m})
            continue

        if not isinstance(m, dict):
            # 无法识别的类型，序列化为字符串
            sanitized.append(
                {"role": "user", "content": json.dumps(m, ensure_ascii=False)}
            )
            continue

        content = m.get("content", "")
        if isinstance(content, str):
            s = content
        elif isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                else:
                    parts.append(json.dumps(part, ensure_ascii=False))
            s = "\n".join(parts)
        else:
            s = json.dumps(content, ensure_ascii=False)

        new_m = {**m, "content": s}
        sanitized.append(new_m)

    return sanitized


async def parse_sse_stream(resp: httpx.Response) -> str:
    """解析 response 的 SSE 流，并且把解析的结果暂时存到本地字符串中"""
    buffer = ""
    fragments = []

    async for chunk in resp.aiter_text(chunk_size=8192):
        buffer += chunk

        while "\n\n" in buffer:
            event, buffer = buffer.split("\n\n", 1)
            if not event.strip():
                continue

            for line in event.splitlines():
                if not line.startswith("data:"):
                    continue

                data = line[5:].strip()
                if not data:
                    continue
                if data == "[DONE]":
                    return "".join(fragments)

                try:
                    payload = json.loads(data)
                except json.JSONDecodeError:
                    fragments.append(data)
                    continue

                if isinstance(payload, dict):
                    choices = payload.get("choices") or []
                    for choice in choices:
                        delta = choice.get("delta") or {}
                        message = choice.get("message") or {}
                        for block in (delta, message):
                            content_piece = block.get("content")
                            if content_piece:
                                fragments.append(content_piece)

                    if not choices and payload.get("content"):
                        content_value = payload["content"]
                        if isinstance(content_value, str):
                            fragments.append(content_value)
                        else:
                            fragments.append(json.dumps(content_value, ensure_ascii=False))
                else:
                    fragments.append(str(payload))

    return "".join(fragments)


def process_parsed_stream_cache(parsed_stream_cache: str) -> str:
    """对 parsed_stream_cache 进行处理"""
    try:
        payload = json.loads(parsed_stream_cache)
    except json.JSONDecodeError:
        return parsed_stream_cache

    try:
        json.loads(payload.get("text", ""))
        return process_parsed_stream_cache(payload.get("text", ""))
    except (json.JSONDecodeError, AttributeError):
        return payload.get("text", "")


# DeepSeek API 处理
async def handle_deepseek_request(request_body: dict, client_api_key: str, provider_config: dict) -> Union[dict, StreamingResponse]:
    """处理 DeepSeek API 请求"""
    logger.info("📡 路由到DeepSeek API")

    request_body['messages'] = sanitize_messages(request_body['messages'])
    logger.info('🧹 在 handle_proxy 中已清洗 messages')

    model = request_body.get("model", "deepseek-reasoner")
    logger.info(f"🔍 使用 DeepSeek 模型: {model}")

    async def make_request():
        # 过滤 DeepSeek API 支持的参数
        supported_params = {
            "model",
            "messages",
            "stream",
            "temperature",
            "max_tokens",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "stop",
        }

        # 构建清理后的请求数据
        request_data = {
            key: value for key, value in request_body.items() if key in supported_params
        }

        # 确保模型名称正确
        request_data["model"] = model

        # 移除空的数组参数
        if "tools" in request_body and not request_body["tools"]:
            logger.info("🧹 移除空的 tools 参数")

        # 记录过滤的参数
        filtered_params = set(request_body.keys()) - set(request_data.keys())
        if filtered_params:
            logger.info(f"🧹 已过滤不支持的参数: {filtered_params}")

        logger.info(f'📤 发送到 DeepSeek API: {provider_config["api_url"]}/chat/completions')
        logger.info(f"📋 请求参数: {list(request_data.keys())}")

        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            response = await client.post(
                f"{provider_config['api_url']}/chat/completions",
                json=request_data,
                headers={
                    "Authorization": f"Bearer {client_api_key}",
                    "Content-Type": "application/json",
                },
            )

            # 记录响应状态和错误信息
            logger.info(f"📥 DeepSeek API 响应状态: {response.status_code}")
            if response.status_code != 200:
                response_text = response.text
                logger.error(f"❌ DeepSeek API 错误响应: {response_text}")

            # 非 2xx 状态会触发 raise_for_status() 抛出 HTTPStatusError
            response.raise_for_status()
            return response

    response = await with_retry(make_request)
    logger.info(f"✅ DeepSeek API响应状态: {response.status_code}")

    if request_body.get("stream", False):
        logger.info("🔄 返回DeepSeek流式响应")

        # 直接返回原始流式响应，不修改任何内容
        response_headers = dict(response.headers)
        # 移除可能引起问题的头部
        response_headers.pop("content-length", None)
        response_headers.pop("content-encoding", None)
        response_headers["content-type"] = "text/event-stream; charset=utf-8"

        # 解析 response 的 SSE 流，并且把解析的结果暂时存到本地字符串中
        parsed_stream_cache = await parse_sse_stream(response)
        logger.info(f"🧩 DeepSeek流式缓存解析结果: {parsed_stream_cache!r}")

        # 对 parsed_stream_cache 进行处理。
        parsed_stream_cache = process_parsed_stream_cache(parsed_stream_cache)
        logger.info(f"🧩 DeepSeek流式缓存处理后结果: {parsed_stream_cache!r}")

        async def generate():
            # 将解析后的文本拆分为多个 SSE 块并逐个推送
            chunk_size = 1024
            text = parsed_stream_cache or ""
            stream_id = str(uuid.uuid4())
            system_fingerprint = "fp_proxy_stream"
            for index, start in enumerate(range(0, len(text), chunk_size)):
                segment = text[start : start + chunk_size]
                payload = {
                    "id": stream_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "system_fingerprint": system_fingerprint,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": segment},
                            "logprobs": None,
                            "finish_reason": None,
                        }
                    ],
                }
                sse_chunk = f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                logger.debug(f"🔀 发送SSE块(index={index}): {sse_chunk!r}")
                yield sse_chunk.encode("utf-8")
                await asyncio.sleep(0)

            # 发送结束块，指示完成
            finish_payload = {
                "id": stream_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "system_fingerprint": system_fingerprint,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "logprobs": None,
                        "finish_reason": "stop",
                    }
                ],
            }
            finish_chunk = f"data: {json.dumps(finish_payload, ensure_ascii=False)}\n\n"
            logger.debug(f"🏁 发送SSE结束块: {finish_chunk!r}")
            yield finish_chunk.encode("utf-8")
            yield b"data: [DONE]\n\n"

        return StreamingResponse(
            generate(), status_code=response.status_code, headers=response_headers
        )
    else:
        logger.info("📦 返回DeepSeek非流式响应")
        return response.json()  # 代理处理函数


async def handle_t8star_request(request_body: dict, client_api_key: str, provider_config: dict) -> Union[dict, StreamingResponse]:
    model = request_body.get("model")

    async def make_request():
        supported_params = {
            "model",
            "messages",
            "stream",
            "temperature",
            "max_tokens",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "stop",
            "tools",
            "tool_choice",
        }
        request_data = {k: v for k, v in request_body.items() if k in supported_params}
        request_data["model"] = model

        _path = os.getenv('T8STAR_PATH', '/v1/chat/completions')
        endpoint = provider_config['api_url'].rstrip('/') + _path
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            headers = {
                "Authorization": f"Bearer {client_api_key}",
                "Content-Type": "application/json",
            }
            if request_body.get("stream"):
                headers["Accept"] = "text/event-stream"
            else:
                headers["Accept"] = "application/json"
            response = await client.post(
                endpoint,
                json=request_data,
                headers=headers,
            )
            response.raise_for_status()
            return response

    response = await with_retry(make_request)
    logger.info(f"T8Star响应状态: {response.status_code}")
    logger.info(f"T8Star响应头content-type: {response.headers.get('content-type', '')}")

    if request_body.get("stream", False):
        response_headers = dict(response.headers)
        response_headers.pop("content-length", None)
        response_headers.pop("content-encoding", None)
        response_headers["content-type"] = "text/event-stream; charset=utf-8"

        content_type = response.headers.get('content-type', '')
        async def generate():
            first_logged = False
            buffer = b""
            async for chunk in response.aiter_bytes(chunk_size=8192):
                if not first_logged:
                    preview_text = chunk[:1024].decode('utf-8', errors='replace')
                    logger.info(f"🧩 T8Star首块预览: {preview_text!r}")
                    first_logged = True
                # 若响应是 text/html，直接透传但额外日志提示
                if 'text/html' in content_type:
                    logger.error("T8Star返回HTML，可能需要调整路径或模型名")
                buffer += chunk
                # 尝试将 HTML 转换为一条错误SSE，避免上游解析失败
                if 'text/html' in content_type and b'<!doctype html>' in buffer.lower():
                    payload = {
                        "id": str(uuid.uuid4()),
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [{"index":0,"delta":{"content":"[Provider HTML response]"},"logprobs":None,"finish_reason":None}],
                    }
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode('utf-8')
                    yield b"data: [DONE]\n\n"
                    return
                yield chunk

        return StreamingResponse(
            generate(), status_code=response.status_code, headers=response_headers
        )
    else:
        try:
            data = response.json()
            logger.info(f"🧩 T8Star非流式JSON预览: {json.dumps(data, ensure_ascii=False)[:1000]}")
            return data
        except Exception:
            text_preview = response.text[:1000]
            logger.error(f"🧩 T8Star非流式响应文本预览: {text_preview!r}")
            raise


# Claude Messages API 处理（用于 aicoding.sh 等使用 Claude API 格式的服务）
async def handle_claude_request(request_body: dict, client_api_key: str, provider_config: dict) -> Union[dict, StreamingResponse]:
    """处理 Claude Messages API 请求"""
    model = request_body.get("model")
    logger.info(f"📡 路由到 Claude Messages API: {model}")

    async def make_request():
        # Claude Messages API 使用 /v1/messages 端点
        endpoint = f"{provider_config['api_url'].rstrip('/')}/v1/messages"

        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            # Claude API 不需要 Bearer 前缀
            headers = {
                "Authorization": client_api_key,
                "Content-Type": "application/json",
            }

            response = await client.post(
                endpoint,
                json=request_body,
                headers=headers,
            )
            response.raise_for_status()
            return response

    response = await with_retry(make_request)
    logger.info(f"✅ Claude API 响应状态: {response.status_code}")

    if request_body.get("stream", False):
        logger.info("🔄 返回 Claude 流式响应")

        response_headers = dict(response.headers)
        response_headers.pop("content-length", None)
        response_headers.pop("content-encoding", None)
        response_headers["content-type"] = "text/event-stream; charset=utf-8"

        async def generate():
            async for chunk in response.aiter_bytes(chunk_size=8192):
                yield chunk

        return StreamingResponse(
            generate(), status_code=response.status_code, headers=response_headers
        )
    else:
        logger.info("📦 返回 Claude 非流式响应")
        return response.json()


async def handle_proxy(request_data: dict, client_api_key: str):
    """处理代理请求"""
    try:
        model = request_data.get("model")
        logger.info(f"🎯 请求模型: {model}")
        logger.info(f'🔍 是否流式: {request_data.get("stream", False)}')

        if not model:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": "缺少 model 参数",
                        "type": "invalid_request_error",
                    }
                },
            )

        # 检查模型路由缓存
        cache_key = client_api_key[:16] if len(client_api_key) > 16 else client_api_key
        cache_time = MODEL_CACHE_TIMESTAMP.get(cache_key, 0)
        is_cache_valid = (time.time() - cache_time) < CACHE_TTL_SECONDS

        # 如果缓存无效或不存在，重新构建
        if not is_cache_valid or cache_key not in MODEL_ROUTE_CACHE:
            logger.info("🔄 缓存过期或不存在，重新获取模型列表")
            model_to_provider = await build_model_cache(client_api_key)
        else:
            model_to_provider = MODEL_ROUTE_CACHE[cache_key]

        # 查找模型对应的供应商
        provider_name = model_to_provider.get(model)
        if not provider_name:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": f"不支持的模型: {model}。请先调用 /v1/models 查看可用模型列表",
                        "type": "invalid_request_error",
                    }
                },
            )

        # 获取供应商配置
        provider_config = PROVIDER_CONFIGS.get(provider_name)
        if not provider_config:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": {
                        "message": f"供应商配置不存在: {provider_name}",
                        "type": "internal_error",
                    }
                },
            )

        logger.info(f"📍 路由到供应商: {provider_name} ({provider_config['type']})")

        # 根据供应商类型调用对应的处理函数
        if provider_config["type"] == "zhipu":
            return await handle_zhipu_request(request_data, client_api_key, provider_config)
        elif provider_config["type"] == "kimi":
            return await handle_kimi_request(request_data, client_api_key, provider_config)
        elif provider_config["type"] == "deepseek":
            return await handle_deepseek_request(request_data, client_api_key, provider_config)
        elif provider_config["type"] == "claude":
            return await handle_claude_request(request_data, client_api_key, provider_config)
        elif provider_config["type"] in ("t8star", "openai"):
            return await handle_t8star_request(request_data, client_api_key, provider_config)
        else:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": {
                        "message": f"未知的供应商类型: {provider_config['type']}",
                        "type": "internal_error",
                    }
                },
            )

    except HTTPException:
        raise
    except httpx.HTTPStatusError as error:
        logger.error(
            f"❌ HTTP 状态错误: {error.response.status_code} - {error.response.text}"
        )
        raise HTTPException(
            status_code=error.response.status_code,
            detail={
                "error": {
                    "message": f"API 请求失败: {error.response.status_code} - {error.response.text}",
                    "type": "api_error",
                }
            },
        )
    except httpx.RequestError as error:
        logger.error(f"❌ 请求错误: {str(error)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": f"网络请求失败: {str(error)}",
                    "type": "network_error",
                }
            },
        )
    except Exception as error:
        logger.error(f"❌ 代理请求失败: {str(error)}")
        raise HTTPException(
            status_code=500,
            detail={"error": {"message": str(error), "type": "proxy_error"}},
        )


# Chat Completions 接口
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """OpenAI 兼容的聊天完成接口"""
    try:
        # 提取客户端 API 密钥
        client_api_key = extract_client_key(request)
        if not client_api_key:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": {
                        "message": "未提供 API 密钥，请在请求头中设置 'Authorization: Bearer <your-api-key>'",
                        "type": "auth_error"
                    }
                }
            )

        body = await request.json()
        logger.info(f"请求体: {body}")

        # 验证必需字段
        if "model" not in body:
            logger.error("请求体缺少 'model' 字段")
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": "Missing required field: 'model'",
                        "type": "invalid_request_error",
                    }
                },
            )

        if "messages" not in body:
            logger.error("请求体缺少 'messages' 字段")
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": "Missing required field: 'messages'",
                        "type": "invalid_request_error",
                    }
                },
            )

        return await handle_proxy(body, client_api_key)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"解析请求体失败: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": f"Invalid request body: {str(e)}",
                    "type": "invalid_request_error",
                }
            },
        )


@app.post("/api/v1/chat/completions")
async def api_chat_completions(request: Request):
    """备用聊天完成接口"""
    try:
        # 提取客户端 API 密钥
        client_api_key = extract_client_key(request)
        if not client_api_key:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": {
                        "message": "未提供 API 密钥，请在请求头中设置 'Authorization: Bearer <your-api-key>'",
                        "type": "auth_error"
                    }
                }
            )

        body = await request.json()
        logger.info(f"API接口请求体: {body}")
        return await handle_proxy(body, client_api_key)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"API接口解析请求体失败: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": f"Invalid request body: {str(e)}",
                    "type": "invalid_request_error",
                }
            },
        )


@app.post("/v1/messages")
async def messages(request: Request):
    """消息接口"""
    try:
        # 提取客户端 API 密钥
        client_api_key = extract_client_key(request)
        if not client_api_key:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": {
                        "message": "未提供 API 密钥，请在请求头中设置 'Authorization: Bearer <your-api-key>'",
                        "type": "auth_error"
                    }
                }
            )

        body = await request.json()
        logger.info(f"消息接口请求体: {body}")
        return await handle_proxy(body, client_api_key)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"消息接口解析请求体失败: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": f"Invalid request body: {str(e)}",
                    "type": "invalid_request_error",
                }
            },
        )


# 启动函数
def main():
    """启动服务器"""
    logger.info("🚀 Xcode AI 代理服务已启动（多租户模式）")
    logger.info(f"📡 监听地址: http://{HOST}:{PORT}")
    logger.info("🎯 已配置的供应商:")
    for provider, config in PROVIDER_CONFIGS.items():
        logger.info(f"   ✅ {provider} ({config['type']}) - {config['api_url']}")

    if not PROVIDER_CONFIGS:
        logger.error("❌ 没有可用的供应商，请检查 models.toml 配置")
        return

    logger.info("⚙️ 重试配置:")
    logger.info(f"   最大重试次数: {MAX_RETRIES}")
    logger.info(f"   重试延迟: {int(RETRY_DELAY * 1000)}ms (递增)")
    logger.info(f"   请求超时: {int(REQUEST_TIMEOUT * 1000)}ms")

    logger.info("📋 配置 Xcode:")
    logger.info(f"   ANTHROPIC_BASE_URL: http://localhost:{PORT}")
    logger.info("   ANTHROPIC_AUTH_TOKEN: <your-api-key>")
    logger.info("💡 提示: 客户端需要在 Authorization 头中传入自己的 API 密钥")
    logger.info("🔧 功能: 多租户支持，动态模型列表，流式响应，智能重试")

    uvicorn.run(
        "server:app", host=HOST, port=PORT, reload=False, log_level="info"
    )


if __name__ == "__main__":
    main()
