# LLM策略引擎优化说明

本文档说明了对LLM策略引擎的三项重要优化。

## 优化概览

### 1. 角色专业化的 Prompt Engineering ✅

**问题**：为所有角色使用通用的 Prompt，会导致 AI 行为模式单一，无法体现角色特性。

**解决方案**：
- 为每个角色（梅林、刺客、派西维尔、莫甘娜、忠臣、莫德雷德）设计了专属的 Prompt 模板
- 为每种关键行为（组队、投票、任务投票、刺杀、发言）设计了专属的 Prompt 模板
- 所有 Prompt 模板存放在 `/prompts` 目录下，便于管理和迭代

**目录结构**：
```
prompts/
├── roles/              # 角色专属Prompt
│   ├── merlin.md      # 梅林
│   ├── assassin.md    # 刺客
│   ├── percival.md    # 派西维尔
│   ├── morgana.md     # 莫甘娜
│   ├── servant.md     # 忠臣
│   └── mordred.md     # 莫德雷德
├── actions/            # 行为专属Prompt
│   ├── team_proposal.md    # 队伍提议
│   ├── vote.md             # 投票决策
│   ├── mission_vote.md     # 任务投票
│   ├── assassination.md    # 刺杀决策
│   └── speech.md           # 发言生成
└── README.md          # 说明文档
```

**特点**：
- 每个角色的 Prompt 强调其特殊能力和目标
- 明确约束和策略要点
- 提供示例发言风格（好的/坏的）

### 2. 基于记忆（Memory）的连续决策 ✅

**问题**：每次调用 LLM 都是独立的，AI 不记得之前的对话和决策，导致行为不一致。

**解决方案**：
- 在 `LLMStrategyEngine` 中添加了 `memory` 列表，存储对话历史和关键事件
- 实现了 `add_to_memory()` 方法，自动记录关键决策和行为
- 实现了 `get_memory_summary()` 方法，返回最近10条记忆
- 限制记忆长度为20条，防止Context窗口溢出

**记忆内容包括**：
- 队伍提议："第X轮：我提议了队伍 [玩家列表]"
- 投票决策："第X轮投票：我对队伍 [玩家列表] 投了同意/拒绝票"
- 任务投票："第X轮任务：我投了成功/失败票"
- 刺杀决策："刺杀阶段：我选择刺杀 [玩家名]"
- 发言："第X轮讨论：我说了\"[发言内容]...\""

**使用方式**：
- 记忆自动记录到每个决策中
- 在构建 Prompt 时，记忆会被包含在 System Prompt 中
- LLM 可以基于历史记忆做出更连贯的决策

### 3. 减少幻觉 (Hallucination) - 引入事实核查和思维链推理 ✅

**问题1**：LLM 可能会 "编造" 不存在的游戏信息或之前的发言。

**解决方案1 - 事实核查**：
- 实现了 `_build_fact_check_context()` 方法，将游戏的客观事实以结构化的 JSON 格式提供给 LLM
- 在 System Prompt 中明确要求："你的回答必须基于以下提供的游戏事实（JSON格式），不得编造信息"
- 在决策后验证结果的有效性（如队伍大小、玩家ID等）

**事实核查内容包括**：
- 当前轮次、阶段
- 成功/失败任务数
- 投票轮次
- 当前队长
- 任务配置
- 所有玩家列表
- 任务历史（结构化数据）

**问题2**：直接让 LLM 给出结论，过程不透明，且可能是错误的跳跃式思维。

**解决方案2 - 思维链推理 (Chain-of-Thought, CoT)**：
- 在所有 Prompt 模板中引入了 CoT 步骤
- 要求 LLM 按照明确的步骤进行思考
- 要求 LLM 在 JSON 输出中包含 `thinking_process` 字段，展示思考过程

**CoT 步骤示例（队伍提议）**：
1. 分析当前游戏状态
2. 分析任务历史（关键推理依据）
3. 评估每个玩家
4. 考虑角色目标
5. 做出最终决策

**好处**：
- **可解释性**：能看到 AI 是如何思考的，便于发现逻辑漏洞
- **更高质量的决策**：迫使模型进行更深层次的思考，从而做出更合理的决定
- **减少幻觉**：通过逐步推理，减少跳跃式思维导致的错误

## 实现细节

### 代码修改

**`src/agent/llm_strategy.py`**：

1. **新增方法**：
   - `add_to_memory(event: str)`: 添加事件到记忆
   - `get_memory_summary() -> str`: 获取记忆摘要
   - `_load_prompt_template(role_name: str, action_name: str) -> Optional[str]`: 加载Prompt模板
   - `_build_fact_check_context(...) -> Dict`: 构建事实核查上下文

2. **修改的方法**：
   - `decide_team_proposal()`: 集成Prompt模板、记忆、事实核查、CoT
   - `decide_vote()`: 集成Prompt模板、记忆、事实核查、CoT
   - `decide_mission_vote()`: 集成Prompt模板、记忆、事实核查、CoT
   - `decide_assassination()`: 集成Prompt模板、记忆、事实核查、CoT
   - `generate_speech()`: 集成Prompt模板、记忆、事实核查、CoT

3. **新增属性**：
   - `memory: List[str]`: 记忆列表
   - `max_memory_size: int = 20`: 最大记忆长度
   - `prompts_dir: str`: Prompt模板目录路径

### Prompt模板加载机制

- 系统会自动从 `/prompts/roles/{role_name}.md` 和 `/prompts/actions/{action_name}.md` 加载模板
- 如果模板文件不存在，会回退到默认的 Prompt（保持向后兼容）
- 模板加载失败时会打印警告，但不影响系统运行

### 记忆管理

- 记忆自动记录到每个决策中
- 记忆长度限制为20条，超过时自动删除最旧的记忆
- 在构建 Prompt 时，只包含最近10条记忆（避免Context过长）

### 事实核查流程

1. **构建事实**：将游戏状态转换为结构化的 JSON
2. **提供事实**：在 System Prompt 中明确提供事实数据
3. **要求核查**：明确要求 LLM 基于事实决策
4. **验证结果**：在决策后验证结果的有效性

### CoT推理流程

1. **定义步骤**：在 Prompt 中明确定义思考步骤
2. **要求展示**：要求 LLM 在 JSON 输出中包含 `thinking_process`
3. **逐步推理**：LLM 必须按照步骤进行思考
4. **最终决策**：基于思考过程做出最终决策

## 使用示例

### 记忆记录示例

```python
# 自动记录到记忆
self.add_to_memory(f"第{context.current_round}轮：我提议了队伍 {', '.join(team_names)}")
self.add_to_memory(f"第{context.current_round}轮投票：我对队伍 {', '.join(team_names)} 投了同意票")
```

### Prompt模板使用示例

```python
# 自动加载模板
template = self._load_prompt_template("merlin", "team_proposal")
# 如果模板存在，使用模板；否则使用默认Prompt
```

### 事实核查示例

```python
# 构建事实
facts = self._build_fact_check_context(context, all_players, mission_history)
facts_json = json.dumps(facts, ensure_ascii=False, indent=2)

# 在Prompt中提供事实
system_prompt = f"""
...
**重要：事实核查**
你的回答必须基于以下提供的游戏事实（JSON格式），不得编造信息：
{facts_json}
...
"""
```

### CoT推理示例

```python
user_prompt = f"""
...
**请按照以下步骤思考（Chain-of-Thought）**：
1. 分析当前游戏状态
2. 分析任务历史
3. 评估每个玩家
4. 考虑角色目标
5. 做出最终决策

请以JSON格式返回你的决策，格式如下：
{{
    "thinking_process": {{
        "step1_game_state": "你的分析...",
        "step2_mission_history": "你的分析...",
        ...
    }},
    "team": [玩家ID列表]
}}
"""
```

## 效果预期

1. **角色差异化**：不同角色的 AI 行为更加符合角色特性
2. **决策连贯性**：AI 能够记住之前的决策，行为更加一致
3. **减少幻觉**：AI 基于真实游戏事实决策，不会编造信息
4. **可解释性**：可以看到 AI 的思考过程，便于调试和优化
5. **更高质量决策**：通过 CoT 推理，决策更加合理

## 后续优化建议

1. **记忆压缩**：当记忆过长时，可以压缩为摘要
2. **记忆重要性评分**：根据重要性保留记忆，而不是简单的FIFO
3. **Prompt模板版本管理**：支持Prompt模板的版本控制和A/B测试
4. **事实核查增强**：可以添加更多的事实验证规则
5. **CoT步骤优化**：根据不同决策类型，优化CoT步骤

