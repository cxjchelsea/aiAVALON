# Prompt模板目录

本目录包含所有角色的专属Prompt模板和不同行为的Prompt模板。

## 目录结构

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
└── README.md          # 本文件
```

## 设计原则

1. **角色专业化**：每个角色有专属的Prompt，强调其特殊能力和目标
2. **行为专业化**：每种关键行为（组队、投票、发言等）有专属的Prompt模板
3. **思维链推理**：所有Prompt都包含Chain-of-Thought步骤，要求LLM展示思考过程
4. **事实核查**：所有Prompt都强调必须基于提供的游戏事实，不得编造信息

## 使用方法

在`LLMStrategyEngine`中，根据角色和行为类型加载对应的Prompt模板，然后填充游戏状态、信念系统等信息。

## 迭代建议

- 根据实际游戏表现，不断优化Prompt
- 可以添加更多角色（如奥伯伦、爪牙等）的Prompt
- 可以根据不同游戏阶段调整Prompt策略

