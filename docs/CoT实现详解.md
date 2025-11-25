# Chain-of-Thought (CoT) 思维链推理实现详解

## 什么是CoT？

Chain-of-Thought (CoT) 是一种提示工程技术，要求LLM在给出最终答案之前，先展示其推理过程。这有助于：
- **提高决策质量**：迫使模型进行更深层次的思考
- **增强可解释性**：可以看到AI是如何思考的
- **减少错误**：通过逐步推理，减少跳跃式思维导致的错误

## CoT在本系统中的实现

### 1. Prompt层面的CoT实现

#### 1.1 在Prompt模板中定义思考步骤

每个行为（队伍提议、投票、任务投票等）的Prompt模板都包含明确的思考步骤：

**示例：队伍提议的CoT步骤**（`prompts/actions/team_proposal.md`）

```markdown
## 思考步骤（Chain-of-Thought）

请按照以下步骤进行思考，并在回答中展示你的思考过程：

### 步骤1：分析当前游戏状态
- 当前是第几轮任务？
- 成功/失败任务数是多少？
- 投票轮次是多少？

### 步骤2：分析任务历史（关键推理依据）
- 关键推理规则1：如果任务失败且失败票数等于队伍人数，说明队伍中所有人都是坏人！
- 关键推理规则2：如果任务失败但失败票数小于队伍人数，说明队伍中有部分坏人在破坏任务
- 关键推理规则3：如果任务成功，说明队伍中可能都是好人

### 步骤3：评估每个玩家
- 信念系统中的信任度评分
- 之前的投票行为
- 之前的任务投票行为
- 可见信息（如果你有特殊能力）

### 步骤4：考虑角色目标
- 如果你是好人：选择最可信的玩家
- 如果你是坏人：尽量包含坏人队友，但也要包含一些好人以隐藏身份

### 步骤5：做出决策
基于以上分析，选择最合适的队伍成员。
```

#### 1.2 在User Prompt中明确要求CoT

在代码中构建User Prompt时，会明确要求LLM按照步骤思考：

```python
user_prompt = f"""
{game_context}

**请按照以下步骤思考（Chain-of-Thought）**：
1. 分析当前游戏状态
2. 分析任务历史（关键推理依据）
3. 评估每个玩家
4. 考虑角色目标
5. 做出最终决策

请以JSON格式返回你的决策，格式如下：
{{
    "thinking_process": {{
        "step1_game_state": "你的分析...",
        "step2_mission_history": "你的分析...",
        "step3_player_evaluation": "你的分析...",
        "step4_role_objectives": "你的分析...",
        "step5_decision": "你的最终决策理由..."
    }},
    "team": [玩家ID列表]
}}
"""
```

### 2. 输出格式要求

#### 2.1 JSON格式的思考过程

要求LLM以JSON格式返回，包含：
- `thinking_process`：包含每个步骤的分析
- 最终决策：如 `team`, `vote`, `success`, `target` 等

**示例输出格式**：

```json
{
    "thinking_process": {
        "step1_game_state": "当前是第2轮任务，成功1次，失败0次。投票轮次是0，这是第一次投票。",
        "step2_mission_history": "第1轮任务成功，队伍包含玩家A和B，说明他们可能是好人。",
        "step3_player_evaluation": "玩家A信任度0.8，玩家B信任度0.7，玩家C信任度0.4（可疑）。",
        "step4_role_objectives": "作为好人，我应该选择最可信的玩家A和B。",
        "step5_decision": "基于以上分析，我选择玩家A和B组成队伍。"
    },
    "team": [0, 1]
}
```

### 3. 代码层面的CoT实现

#### 3.1 构建包含CoT要求的Prompt

在 `llm_strategy.py` 中，每个决策方法都会构建包含CoT要求的Prompt：

```python
def decide_team_proposal(self, context, belief_system, all_players, mission_history):
    # ... 构建游戏上下文 ...
    
    # 构建User Prompt（包含CoT要求）
    user_prompt = f"""
    {game_context}
    
    **请按照以下步骤思考（Chain-of-Thought）**：
    1. 分析当前游戏状态
    2. 分析任务历史（关键推理依据）
    3. 评估每个玩家
    4. 考虑角色目标
    5. 做出最终决策
    
    请以JSON格式返回你的决策，格式如下：
    {{
        "thinking_process": {{
            "step1_game_state": "你的分析...",
            "step2_mission_history": "你的分析...",
            "step3_player_evaluation": "你的分析...",
            "step4_role_objectives": "你的分析...",
            "step5_decision": "你的最终决策理由..."
        }},
        "team": [玩家ID列表]
    }}
    """
    
    response = self._call_llm(user_prompt, system_prompt)
    # ... 解析JSON ...
```

#### 3.2 解析CoT输出

虽然当前代码主要使用最终决策（如 `team`, `vote`），但 `thinking_process` 字段也被解析出来，可以用于：
- 调试和日志记录
- 分析AI的思考过程
- 未来可能的优化

```python
decision = json.loads(response)
team = decision.get("team", [])
thinking_process = decision.get("thinking_process", {})  # 虽然当前未使用，但已解析

# 可以记录思考过程用于调试
if thinking_process:
    print(f"思考过程: {thinking_process}")
```

### 4. 不同决策类型的CoT步骤

#### 4.1 队伍提议的CoT步骤

```python
1. 分析当前游戏状态
2. 分析任务历史（关键推理依据）
3. 评估每个玩家
4. 考虑角色目标
5. 做出最终决策
```

#### 4.2 投票决策的CoT步骤

```python
1. 检查流局风险（关键！）
2. 分析提议的队伍
3. 分析任务历史
4. 考虑角色目标
5. 做出最终决策
```

#### 4.3 任务投票的CoT步骤

```python
1. 检查阵营（好人必须投成功）
2. 分析当前局势
3. 分析任务队伍
4. 考虑隐藏身份
5. 做出最终决策
```

#### 4.4 刺杀决策的CoT步骤

```python
1. 回顾游戏历史
2. 分析梅林的特征
3. 排除不可能的人
4. 评估每个候选人
5. 做出最终决策
```

#### 4.5 发言生成的CoT步骤

```python
1. 分析当前局势
2. 确定发言目的
3. 分析最近发言
4. 考虑角色身份
5. 生成发言
```

### 5. CoT的优势

#### 5.1 提高决策质量

通过逐步推理，LLM能够：
- 更全面地考虑各种因素
- 避免跳跃式思维
- 做出更合理的决策

#### 5.2 增强可解释性

可以看到AI的思考过程：
- 为什么选择这个队伍？
- 为什么投票同意/拒绝？
- 如何识别梅林？

#### 5.3 便于调试和优化

通过分析 `thinking_process`，可以：
- 发现AI的逻辑漏洞
- 优化Prompt模板
- 改进推理步骤

### 6. CoT的实际效果

#### 6.1 示例：队伍提议的思考过程

**输入**：第2轮任务，需要3人队伍

**CoT输出**：
```json
{
    "thinking_process": {
        "step1_game_state": "当前是第2轮，成功1次失败0次。这是第一次投票。",
        "step2_mission_history": "第1轮任务成功，队伍[玩家A, 玩家B]都投了成功票，说明他们可能是好人。",
        "step3_player_evaluation": "玩家A信任度0.85，玩家B信任度0.75，玩家C信任度0.45（可疑），玩家D信任度0.70。",
        "step4_role_objectives": "作为好人，我应该选择最可信的玩家，避免可疑玩家。",
        "step5_decision": "基于分析，我选择玩家A、B、D，他们都是可信的玩家。"
    },
    "team": [0, 1, 3]
}
```

#### 6.2 示例：投票决策的思考过程

**输入**：提议队伍[玩家A, 玩家C]，第4次投票

**CoT输出**：
```json
{
    "thinking_process": {
        "step1_rejection_risk": "这是第4次投票（vote_round=3），如果拒绝可能导致流局，好人应该更倾向于同意。",
        "step2_team_analysis": "队伍包含玩家A（信任度0.8）和玩家C（信任度0.4，可疑）。",
        "step3_mission_history": "玩家C在第1轮任务中表现可疑，但第1轮任务成功了。",
        "step4_role_objectives": "作为好人，虽然玩家C可疑，但这是第4次投票，应该同意避免流局。",
        "step5_decision": "虽然队伍中有可疑玩家，但为了避免流局，我选择同意。"
    },
    "vote": true
}
```

### 7. CoT的局限性

#### 7.1 当前实现

- **思考过程未被使用**：虽然要求LLM输出思考过程，但代码中主要使用最终决策
- **无法验证思考过程**：没有验证思考过程是否合理
- **无法利用思考过程**：思考过程没有被用于改进决策

#### 7.2 未来优化方向

1. **思考过程验证**：
   - 检查思考过程是否符合逻辑
   - 验证思考过程是否基于提供的事实

2. **思考过程利用**：
   - 如果思考过程显示逻辑错误，可以要求LLM重新思考
   - 可以基于思考过程调整决策

3. **思考过程记录**：
   - 将思考过程记录到日志
   - 用于分析和优化

4. **思考过程展示**：
   - 在游戏输出中显示AI的思考过程
   - 帮助理解AI的决策

### 8. 代码位置

CoT实现的主要代码位置：

1. **Prompt模板**：`prompts/actions/*.md`
   - 定义思考步骤
   - 定义输出格式

2. **Prompt构建**：`src/agent/llm_strategy.py`
   - `decide_team_proposal()`: 第370-397行
   - `decide_vote()`: 第503-527行
   - `decide_mission_vote()`: 第626-650行
   - `decide_assassination()`: 第723-747行
   - `generate_speech()`: 第849-863行

3. **输出解析**：`src/agent/llm_strategy.py`
   - 第402-412行：解析JSON，提取 `thinking_process` 和最终决策

### 9. 总结

CoT在本系统中的实现方式：

1. **Prompt层面**：
   - 在Prompt模板中定义明确的思考步骤
   - 在User Prompt中明确要求按照步骤思考

2. **输出格式**：
   - 要求JSON格式，包含 `thinking_process` 字段
   - 每个步骤对应一个字段

3. **代码实现**：
   - 构建包含CoT要求的Prompt
   - 解析包含思考过程的JSON输出
   - 使用最终决策，思考过程可用于调试

4. **效果**：
   - 提高决策质量
   - 增强可解释性
   - 便于调试和优化

CoT通过引导LLM进行逐步推理，使AI的决策更加合理和可解释。

