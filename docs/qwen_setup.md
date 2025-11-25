# 本地Qwen模型配置指南

## 概述

本系统支持使用本地部署的Qwen模型，通过OpenAI兼容API进行调用。这样可以：
- 节省API费用
- 保护数据隐私
- 获得更快的响应速度

## 前置要求

1. **部署Qwen模型服务**：需要有一个支持OpenAI兼容API的Qwen模型服务
2. **推荐方案**：使用vLLM、TGI（Text Generation Inference）或其他支持OpenAI兼容API的框架

## 使用vLLM部署Qwen（推荐）

### 1. 安装vLLM

```bash
pip install vllm
```

### 2. 启动Qwen服务

```bash
# 使用vLLM启动Qwen模型服务
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --port 8000 \
    --api-key not-needed
```

或者使用其他Qwen模型：
```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-14B-Instruct \
    --port 8000
```

### 3. 验证服务

```bash
curl http://localhost:8000/v1/models
```

应该返回模型列表。

### 4. 配置环境变量

创建 `.env` 文件：

```bash
USE_LLM=true
LLM_API_PROVIDER=qwen
QWEN_BASE_URL=http://localhost:8000/v1
QWEN_MODEL=qwen  # 或你的模型名称
QWEN_API_KEY=not-needed
```

### 5. 运行游戏

```bash
python -m src.main
```

## 使用其他框架

### TGI (Text Generation Inference)

```bash
# 安装TGI
pip install text-generation

# 启动服务
text-generation-launcher --model-id Qwen/Qwen2.5-7B-Instruct --port 8000
```

### Ollama

如果使用Ollama部署Qwen：

```bash
# 拉取Qwen模型
ollama pull qwen2.5:7b

# 启动Ollama服务（默认端口11434）
# 需要配置Ollama使用OpenAI兼容API
```

配置：
```bash
QWEN_BASE_URL=http://localhost:11434/v1
QWEN_MODEL=qwen2.5:7b
```

## 常见问题

### 1. 连接失败

- 检查服务是否正在运行：`curl http://localhost:8000/v1/models`
- 确认端口号正确
- 检查防火墙设置

### 2. 模型响应慢

- 使用更小的模型（如7B而不是14B）
- 调整vLLM的`--max-model-len`参数
- 使用GPU加速

### 3. 内存不足

- 使用量化模型
- 减少并发请求
- 使用更小的模型

## 性能优化建议

1. **使用量化模型**：可以显著减少显存占用
2. **调整batch size**：根据GPU显存调整
3. **使用Flash Attention**：vLLM默认启用，可以提升性能

## 示例配置

### 完整.env配置示例

```bash
# 启用LLM
USE_LLM=true

# 使用Qwen
LLM_API_PROVIDER=qwen

# Qwen服务地址（根据你的部署调整）
QWEN_BASE_URL=http://localhost:8000/v1

# 模型名称（根据你的模型调整）
QWEN_MODEL=qwen

# API密钥（本地部署通常不需要）
QWEN_API_KEY=not-needed

# 可选：启用LangGraph
USE_LANGGRAPH=false
```

## 测试连接

可以使用以下Python代码测试连接：

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="qwen",
    messages=[
        {"role": "user", "content": "你好"}
    ]
)

print(response.choices[0].message.content)
```

如果成功输出回复，说明连接正常。

