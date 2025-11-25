# AI阿瓦隆多智能体系统

一个基于多智能体的阿瓦隆游戏系统，每个智能体能够动态适应不同角色，展现出类似人类玩家的策略深度。

## 项目结构

```
aiAVALON/
├── src/
│   ├── game/
│   │   ├── __init__.py
│   │   ├── rules.py          # 游戏规则定义
│   │   ├── roles.py          # 角色定义
│   │   ├── game_engine.py    # 中央游戏引擎
│   │   └── langgraph_game.py # LangGraph游戏引擎（可选）
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── base_agent.py     # 智能体基类
│   │   ├── belief_system.py  # 动态信念系统
│   │   ├── strategy.py       # 策略相关类型定义（Personality, DecisionContext）
│   │   ├── llm_strategy.py   # LLM策略引擎（必需）
│   │   └── communication.py  # 沟通生成器（已废弃，现由LLM生成）
│   └── main.py               # 主程序入口
├── docs/
│   └── langgraph_integration.md  # LangGraph集成文档
├── requirements.txt
├── .env.example              # 环境变量示例
└── README.md
```

## 安装

```bash
pip install -r requirements.txt
```

## 运行

**重要：系统现在仅支持LLM策略引擎，必须配置LLM API才能运行。**

### 配置LLM API

1. 创建 `.env` 文件并配置API密钥（选择一种方式）：
   - **OpenAI**: `OPENAI_API_KEY=your_api_key_here`
   - **DeepSeek**: `DEEPSEEK_API_KEY=your_api_key_here` 并设置 `LLM_API_PROVIDER=deepseek`
   - **本地Qwen**: 设置 `LLM_API_PROVIDER=qwen` 和 `QWEN_BASE_URL=http://localhost:8000/v1`

2. 运行游戏：
```bash
python -m src.main
```

系统会自动使用配置的LLM进行所有决策。如果未配置API密钥，程序会提示错误并退出。

## 功能特性

- 多智能体架构，每个智能体可适应任何角色
- 动态信念系统，基于贝叶斯推理更新对其他玩家身份的判断
- **LLM策略引擎**：使用大语言模型进行智能决策（必需）
  - 支持OpenAI、DeepSeek、本地Qwen等多种模型
  - 所有决策（队伍提议、投票、任务投票、刺杀、发言）均由LLM完成
- **LangGraph集成**：可选的状态图管理，提供更清晰的状态流转（可选）
- 自然语言沟通生成，LLM生成更自然、更有策略性的发言
- 支持流局保护机制，避免因过度拒绝导致坏人直接获胜

## LLM配置

系统支持使用多种LLM进行智能决策：
- **OpenAI API**: GPT系列模型
- **DeepSeek API**: DeepSeek模型
- **本地Qwen模型**: 通过OpenAI兼容API使用本地部署的Qwen模型

LLM会分析游戏状态、信念系统和历史信息，做出更智能的决策。

### 配置步骤

#### 使用OpenAI API

1. 创建 `.env` 文件：
```bash
OPENAI_API_KEY=your_openai_api_key_here
LLM_API_PROVIDER=openai  # 可选，默认为openai
LLM_MODEL=gpt-4o-mini     # 可选，默认使用gpt-4o-mini
```

2. 运行游戏：
```bash
python -m src.main
```

#### 使用DeepSeek API

1. 创建 `.env` 文件：
```bash
DEEPSEEK_API_KEY=your_deepseek_api_key_here
LLM_API_PROVIDER=deepseek
LLM_MODEL=deepseek-chat
```

2. 运行游戏：
```bash
python -m src.main
```

#### 使用本地Qwen模型

1. 确保本地已部署Qwen模型（支持OpenAI兼容API），例如使用vLLM或类似框架

2. 创建 `.env` 文件：
```bash
LLM_API_PROVIDER=qwen
QWEN_BASE_URL=http://localhost:8000/v1
QWEN_MODEL=qwen
QWEN_API_KEY=not-needed  # 本地部署通常不需要真实key
```

3. 运行游戏：
```bash
python -m src.main
```

**注意**: 
- DeepSeek API与OpenAI API兼容
- Qwen需要本地部署并支持OpenAI兼容API（如vLLM）
- 默认Qwen地址为 `http://localhost:8000/v1`，可通过 `QWEN_BASE_URL` 修改

### LLM功能

- **队伍提议**：根据游戏状态和信念系统智能选择队伍成员
- **投票决策**：综合考虑流局风险、可疑玩家等因素
- **任务投票**：坏人阵营智能决定是否破坏任务
- **刺杀决策**：刺客根据游戏历史判断梅林身份
- **发言生成**：生成更自然、更有策略性的发言

