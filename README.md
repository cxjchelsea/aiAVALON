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
│   │   └── game_engine.py    # 中央游戏引擎
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── base_agent.py     # 智能体基类
│   │   ├── belief_system.py  # 动态信念系统
│   │   ├── strategy.py       # 策略决策引擎（规则引擎）
│   │   ├── llm_strategy.py   # LLM策略引擎
│   │   └── communication.py # 沟通生成器
│   └── main.py               # 主程序入口
├── requirements.txt
├── .env.example              # 环境变量示例
└── README.md
```

## 安装

```bash
pip install -r requirements.txt
```

## 运行

### 使用普通策略引擎（默认）

```bash
python -m src.main
```

### 使用LLM策略引擎

1. 创建 `.env` 文件并配置API密钥：
```bash
OPENAI_API_KEY=your_api_key_here
USE_LLM=true
LLM_MODEL=gpt-4o-mini  # 可选，默认使用gpt-4o-mini
```

2. 运行游戏：
```bash
python -m src.main
```

系统会自动检测环境变量，如果设置了 `USE_LLM=true` 且配置了 `OPENAI_API_KEY`，就会使用LLM进行决策。

## 功能特性

- 多智能体架构，每个智能体可适应任何角色
- 动态信念系统，基于贝叶斯推理更新对其他玩家身份的判断
- 双重策略引擎：
  - **规则引擎**：基于规则的快速决策（默认）
  - **LLM引擎**：使用大语言模型进行智能决策（可选）
- 自然语言沟通生成，生成有策略目的的发言
- 支持流局保护机制，避免因过度拒绝导致坏人直接获胜

## LLM配置

系统支持使用OpenAI API进行智能决策。LLM会分析游戏状态、信念系统和历史信息，做出更智能的决策。

### 配置步骤

#### 使用OpenAI API

1. 创建 `.env` 文件：
```bash
OPENAI_API_KEY=your_openai_api_key_here
USE_LLM=true
LLM_API_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
```

2. 运行游戏：
```bash
python -m src.main
```

#### 使用DeepSeek API

1. 创建 `.env` 文件：
```bash
DEEPSEEK_API_KEY=your_deepseek_api_key_here
USE_LLM=true
LLM_API_PROVIDER=deepseek
LLM_MODEL=deepseek-chat
```

2. 运行游戏：
```bash
python -m src.main
```

**注意**: DeepSeek API与OpenAI API兼容，如果同时设置了`DEEPSEEK_API_KEY`和`LLM_API_PROVIDER=deepseek`，系统会优先使用DeepSeek。

### LLM功能

- **队伍提议**：根据游戏状态和信念系统智能选择队伍成员
- **投票决策**：综合考虑流局风险、可疑玩家等因素
- **任务投票**：坏人阵营智能决定是否破坏任务
- **刺杀决策**：刺客根据游戏历史判断梅林身份
- **发言生成**：生成更自然、更有策略性的发言

