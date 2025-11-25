"""
基于LLM的策略决策引擎
使用大语言模型进行智能决策
"""
from typing import List, Dict, Optional
import json
import os
from dotenv import load_dotenv

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.rules import Team, GamePhase
from game.roles import RoleType
from agent.belief_system import BeliefSystem
from agent.strategy import DecisionContext, Personality

# 加载环境变量
load_dotenv()

try:
    from openai import OpenAI
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("警告: OpenAI库未安装，LLM功能将不可用")


class LLMStrategyEngine:
    """基于LLM的策略决策引擎"""
    
    def __init__(self, my_role: RoleType, my_team: Team, my_player_id: int, 
                 my_name: str, personality: Personality = Personality.ANALYTICAL,
                 api_key: Optional[str] = None, model: str = "gpt-4o-mini",
                 api_provider: str = "openai"):
        """
        api_provider: "openai" 或 "deepseek"
        """
        self.my_role = my_role
        self.my_team = my_team
        self.my_player_id = my_player_id
        self.my_name = my_name
        self.personality = personality
        self.model = model
        self.api_provider = api_provider.lower()
        
        # 初始化LLM客户端
        if LLM_AVAILABLE:
            # 根据提供商选择API密钥和base_url
            if self.api_provider == "deepseek":
                api_key = api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
                base_url = "https://api.deepseek.com"
            elif self.api_provider == "qwen":
                # 本地Qwen模型（通过OpenAI兼容API）
                api_key = api_key or os.getenv("QWEN_API_KEY", "not-needed")  # 本地部署通常不需要真实key
                base_url = os.getenv("QWEN_BASE_URL", "http://localhost:8000/v1")  # 默认本地地址
            elif self.api_provider == "openai":
                api_key = api_key or os.getenv("OPENAI_API_KEY")
                base_url = None  # 使用OpenAI默认URL
            else:
                # 自定义API提供商
                api_key = api_key or os.getenv(f"{self.api_provider.upper()}_API_KEY") or os.getenv("OPENAI_API_KEY")
                base_url = os.getenv(f"{self.api_provider.upper()}_BASE_URL")
            
            if api_key or base_url:
                if base_url:
                    # 需要指定base_url（DeepSeek、Qwen或自定义）
                    self.client = OpenAI(api_key=api_key, base_url=base_url)
                else:
                    # OpenAI使用默认配置
                    self.client = OpenAI(api_key=api_key)
            else:
                self.client = None
                provider_name = self.api_provider.upper()
                print(f"警告: 未设置{provider_name} API配置，LLM功能将不可用")
        else:
            self.client = None
    
    def _call_llm(self, prompt: str, system_prompt: str = None, max_retries: int = 2) -> str:
        """调用LLM，带重试机制"""
        if not self.client:
            raise RuntimeError("LLM客户端未初始化")
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=500,
                    timeout=30  # 30秒超时
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    # 重试前等待一小段时间
                    import time
                    time.sleep(0.5)
                    continue
                else:
                    # 最后一次尝试失败，抛出异常
                    if "timeout" in str(e).lower() or "timed out" in str(e).lower():
                        print(f"LLM调用超时 (尝试 {attempt + 1}/{max_retries + 1})")
                    else:
                        print(f"LLM调用错误: {e}")
                    raise
    
    def _build_game_context_description(self, context: DecisionContext, 
                                        belief_system: BeliefSystem,
                                        all_players: List[Dict],
                                        proposed_team: Optional[List[int]] = None,
                                        mission_history: Optional[List[Dict]] = None) -> str:
        """构建游戏上下文描述"""
        # 获取玩家名称映射
        player_names = {p["player_id"]: p["name"] for p in all_players}
        
        # 构建游戏状态描述
        game_state_desc = f"""
游戏状态：
- 当前阶段: {context.game_phase.value}
- 当前轮次: {context.current_round}
- 成功任务数: {context.successful_missions}
- 失败任务数: {context.failed_missions}
- 投票轮次: {context.vote_round} (最多5次，如果5次都未通过则坏人直接获胜)
- 当前队长: {player_names.get(context.current_leader, f"玩家{context.current_leader}")}
"""
        
        if proposed_team:
            team_names = [player_names.get(pid, f"玩家{pid}") for pid in proposed_team]
            game_state_desc += f"- 提议的队伍: {', '.join(team_names)}\n"
        
        # 构建任务历史描述（关键信息！）
        mission_history_desc = ""
        if mission_history:
            mission_history_desc = "\n任务历史（重要推理依据）：\n"
            for mission in mission_history:
                team_str = ", ".join(mission["team"])
                result_str = "成功" if mission["success"] else "失败"
                mission_history_desc += f"- 第{mission['round']}轮: 队伍 [{team_str}] - {result_str}"
                if not mission["success"]:
                    mission_history_desc += f" (失败票数: {mission['fail_count']}/{mission['team_size']})"
                    # 关键推理提示
                    if mission['fail_count'] == mission['team_size']:
                        mission_history_desc += f" ⚠️ 关键信息：失败票数等于队伍人数，说明队伍中所有人都是坏人！"
                    elif mission['fail_count'] > 0:
                        mission_history_desc += f" ⚠️ 关键信息：队伍中有{mission['fail_count']}个坏人投了失败票"
                mission_history_desc += "\n"
        
        # 构建信念系统描述
        belief_desc = "\n对其他玩家的判断：\n"
        for player_id, belief in belief_system.beliefs.items():
            if player_id == self.my_player_id:
                continue
            player_name = player_names.get(player_id, f"玩家{player_id}")
            good_prob = belief.team_probabilities.get(Team.GOOD, 0.5)
            evil_prob = belief.team_probabilities.get(Team.EVIL, 0.5)
            trust = belief.trust_score
            belief_desc += f"- {player_name}: 好人概率 {good_prob:.2f}, 坏人概率 {evil_prob:.2f}, 信任度 {trust:.2f}\n"
        
        # 构建任务配置
        mission_desc = ""
        if context.mission_config:
            mission_desc = f"\n当前任务配置：\n- 队伍人数: {context.mission_config.get('team_size', 2)}\n- 需要失败票数: {context.mission_config.get('fails_needed', 1)}\n"
        
        return game_state_desc + mission_history_desc + belief_desc + mission_desc
    
    def decide_team_proposal(self, context: DecisionContext, belief_system: BeliefSystem,
                            all_players: List[Dict], mission_history: Optional[List[Dict]] = None) -> List[int]:
        """使用LLM决定提议的队伍"""
        if not self.client:
            # 如果LLM不可用，回退到简单策略
            return self._fallback_team_proposal(context, belief_system)
        
        game_context = self._build_game_context_description(context, belief_system, all_players, 
                                                          mission_history=mission_history)
        player_names = {p["player_id"]: p["name"] for p in all_players}
        all_player_list = [f"{pid}: {name}" for pid, name in player_names.items() if pid != self.my_player_id]
        
        # 构建角色信息描述
        role_info = f"你是{self.my_role.value}（{self.my_team.value}阵营）"
        if self.my_role == RoleType.MERLIN:
            role_info += "。你能看到所有坏人（除了莫德雷德），但必须隐藏身份，避免被刺客发现。"
        elif self.my_role == RoleType.PERCIVAL:
            role_info += "。你能看到梅林和莫甘娜（但不知道哪个是梅林），需要保护真正的梅林。"
        elif self.my_role == RoleType.ASSASSIN:
            role_info += "。你能看到所有坏人队友，你的目标是在游戏结束时刺杀梅林。"
        elif self.my_role == RoleType.MORGANA:
            role_info += "。你能看到所有坏人队友，并且会出现在派西维尔的视野中（伪装成梅林）。"
        elif self.my_team == Team.GOOD:
            role_info += "。你是好人阵营，需要帮助好人完成任务。"
        else:
            role_info += "。你是坏人阵营，需要破坏任务或刺杀梅林。"
        
        system_prompt = f"""你是一个阿瓦隆游戏中的玩家。
{role_info}
你的名字是{self.my_name}。
你的人格特质是{self.personality.value}。

游戏规则：
- 好人阵营需要完成3个任务才能获胜
- 坏人阵营需要破坏3个任务，或者在好人完成3个任务后成功刺杀梅林
- 如果同一任务的投票连续5次都未通过，坏人直接获胜
- 作为队长，你需要提议一个队伍去执行任务（队伍人数由当前任务决定）
- 队长可以选择自己加入队伍

**关键推理规则**：
- 如果任务失败且失败票数等于队伍人数，说明队伍中所有人都是坏人！
- 如果任务失败但失败票数小于队伍人数，说明队伍中有部分坏人在破坏任务
- 如果任务成功，说明队伍中可能都是好人（或者坏人选择隐藏身份）

请根据你的角色信息、可见信息、任务历史、游戏状态和你的判断，选择最合适的队伍成员。"""
        
        # 构建可见信息描述
        visible_info_desc = ""
        if all_players:
            my_info = next((p for p in all_players if p.get("is_self")), None)
            if my_info:
                visible_players = [p for p in all_players if not p.get("is_self") and (p.get("team") or p.get("possible_merlin"))]
                if visible_players:
                    visible_info_desc = "\n你能看到的玩家信息：\n"
                    for p in visible_players:
                        if p.get("possible_merlin"):
                            visible_info_desc += f"- {p['name']} (ID:{p['player_id']}): 可能是梅林\n"
                        elif p.get("team"):
                            visible_info_desc += f"- {p['name']} (ID:{p['player_id']}): {p['team']}阵营\n"
        
        user_prompt = f"""{game_context}
{visible_info_desc}
所有玩家（你可以选择任意{context.mission_config.get('team_size', 2)}人，包括你自己）：
{chr(10).join([f"{pid}: {name}" for pid, name in player_names.items()])}

当前任务需要 {context.mission_config.get('team_size', 2)} 人。
注意：作为队长，你可以选择自己加入队伍。

请以JSON格式返回你的决策，格式如下：
{{
    "reasoning": "你的推理过程（简要说明为什么选择这些玩家）",
    "team": [玩家ID列表，例如 [0, 1, 2]]
}}

只返回JSON，不要其他内容。"""
        
        try:
            response = self._call_llm(user_prompt, system_prompt)
            # 尝试解析JSON
            # 移除可能的markdown代码块标记
            response = response.strip()
            if response.startswith("```"):
                # 移除代码块标记
                lines = response.split("\n")
                response = "\n".join(lines[1:-1])
            if response.startswith("```json"):
                lines = response.split("\n")
                response = "\n".join(lines[1:-1])
            
            decision = json.loads(response)
            team = decision.get("team", [])
            
            # 验证队伍大小
            required_size = context.mission_config.get("team_size", 2)
            if len(team) != required_size:
                # 如果LLM返回的队伍大小不对，回退
                return self._fallback_team_proposal(context, belief_system)
            
            return team
        except Exception as e:
            if "timeout" not in str(e).lower() and "timed out" not in str(e).lower():
                print(f"LLM决策错误: {e}，使用回退策略")
            return self._fallback_team_proposal(context, belief_system)
    
    def decide_vote(self, context: DecisionContext, belief_system: BeliefSystem,
                   all_players: List[Dict], proposed_team: List[int], 
                   mission_history: Optional[List[Dict]] = None) -> bool:
        """使用LLM决定是否投票同意"""
        if not self.client:
            return self._fallback_vote(context, belief_system, proposed_team)
        
        game_context = self._build_game_context_description(context, belief_system, all_players, 
                                                          proposed_team=proposed_team,
                                                          mission_history=mission_history)
        player_names = {p["player_id"]: p["name"] for p in all_players}
        team_names = [player_names.get(pid, f"玩家{pid}") for pid in proposed_team]
        
        # 构建角色信息描述
        role_info = f"你是{self.my_role.value}（{self.my_team.value}阵营）"
        if self.my_role == RoleType.MERLIN:
            role_info += "。你能看到所有坏人（除了莫德雷德），但必须隐藏身份，避免被刺客发现。"
        elif self.my_role == RoleType.PERCIVAL:
            role_info += "。你能看到梅林和莫甘娜（但不知道哪个是梅林），需要保护真正的梅林。"
        elif self.my_role == RoleType.ASSASSIN:
            role_info += "。你能看到所有坏人队友，你的目标是在游戏结束时刺杀梅林。"
        elif self.my_role == RoleType.MORGANA:
            role_info += "。你能看到所有坏人队友，并且会出现在派西维尔的视野中（伪装成梅林）。"
        elif self.my_team == Team.GOOD:
            role_info += "。你是好人阵营，需要帮助好人完成任务。"
        else:
            role_info += "。你是坏人阵营，需要破坏任务或刺杀梅林。"
        
        # 检查是否是队长
        is_leader = context.current_leader == self.my_player_id
        leader_note = ""
        if is_leader:
            leader_note = "\n重要：你是队长，你提议了这个队伍，所以你必须投票同意。"
        
        system_prompt = f"""你是一个阿瓦隆游戏中的玩家。
{role_info}
你的名字是{self.my_name}。
你的人格特质是{self.personality.value}。
{leader_note}

重要提醒：
- 如果这是第5次投票（vote_round >= 4），好人必须同意，否则流局会导致坏人直接获胜
- 如果这是第4次投票（vote_round >= 3），好人应该更倾向于同意，避免流局
- 前3次投票可以更谨慎地拒绝可疑的队伍
- **关键推理**：仔细分析任务历史中的失败票数。如果任务失败且失败票数等于队伍人数，说明队伍中所有人都是坏人！

请根据你的角色信息、可见信息、任务历史、游戏状态做出投票决定。"""
        
        user_prompt = f"""{game_context}

当前提议的队伍是：{', '.join(team_names)}

请以JSON格式返回你的决策，格式如下：
{{
    "reasoning": "你的推理过程（简要说明为什么同意或拒绝）",
    "vote": true 或 false
}}

只返回JSON，不要其他内容。"""
        
        try:
            response = self._call_llm(user_prompt, system_prompt)
            # 移除可能的markdown代码块标记
            response = response.strip()
            if response.startswith("```"):
                lines = response.split("\n")
                response = "\n".join(lines[1:-1])
            if response.startswith("```json"):
                lines = response.split("\n")
                response = "\n".join(lines[1:-1])
            
            decision = json.loads(response)
            vote = decision.get("vote", True)
            
            # 强制检查：第5次投票必须同意
            if context.vote_round >= 4:
                vote = True
            
            return bool(vote)
        except Exception as e:
            if "timeout" not in str(e).lower() and "timed out" not in str(e).lower():
                print(f"LLM投票决策错误: {e}，使用回退策略")
            return self._fallback_vote(context, belief_system, proposed_team)
    
    def decide_mission_vote(self, context: DecisionContext, belief_system: BeliefSystem,
                           all_players: List[Dict], mission_team: List[int],
                           mission_history: Optional[List[Dict]] = None) -> bool:
        """使用LLM决定任务投票（成功/失败）"""
        if not self.client:
            return self._fallback_mission_vote(context, belief_system, mission_team)
        
        # 好人总是投成功
        if self.my_team == Team.GOOD:
            return True
        
        # 坏人需要决定是否破坏
        game_context = self._build_game_context_description(context, belief_system, all_players, 
                                                          proposed_team=mission_team,
                                                          mission_history=context.mission_history if hasattr(context, 'mission_history') else None)
        player_names = {p["player_id"]: p["name"] for p in all_players}
        team_names = [player_names.get(pid, f"玩家{pid}") for pid in mission_team]
        
        system_prompt = f"""你是一个阿瓦隆游戏中的{self.my_role.value}（{self.my_team.value}阵营）。
你的名字是{self.my_name}。

作为坏人，你需要决定是否破坏任务（投失败票）。
策略考虑：
- 如果已经失败2次，可以考虑投成功以隐藏身份
- 如果这是前两轮，可以更积极地破坏
- 需要平衡破坏任务和隐藏身份"""
        
        user_prompt = f"""{game_context}

当前任务队伍是：{', '.join(team_names)}
需要 {context.mission_config.get('fails_needed', 1)} 张失败票才能破坏任务。

请以JSON格式返回你的决策，格式如下：
{{
    "reasoning": "你的推理过程",
    "success": true 或 false (true=投成功票, false=投失败票破坏任务)
}}

只返回JSON，不要其他内容。"""
        
        try:
            response = self._call_llm(user_prompt, system_prompt)
            response = response.strip()
            if response.startswith("```"):
                lines = response.split("\n")
                response = "\n".join(lines[1:-1])
            if response.startswith("```json"):
                lines = response.split("\n")
                response = "\n".join(lines[1:-1])
            
            decision = json.loads(response)
            success = decision.get("success", False)
            return bool(success)
        except Exception as e:
            print(f"LLM任务投票决策错误: {e}，使用回退策略")
            return self._fallback_mission_vote(context, belief_system, mission_team)
    
    def decide_assassination(self, context: DecisionContext, belief_system: BeliefSystem,
                            all_players: List[Dict], mission_history: Optional[List[Dict]] = None) -> Optional[int]:
        """使用LLM决定刺杀目标"""
        if self.my_role != RoleType.ASSASSIN or not self.client:
            return None
        
        game_context = self._build_game_context_description(context, belief_system, all_players,
                                                          mission_history=context.mission_history if hasattr(context, 'mission_history') else None)
        player_names = {p["player_id"]: p["name"] for p in all_players}
        all_player_list = [f"{pid}: {name}" for pid, name in player_names.items() if pid != self.my_player_id]
        
        system_prompt = f"""你是一个阿瓦隆游戏中的刺客（坏人阵营）。
好人已经完成了3个任务，现在你可以刺杀梅林。
如果成功刺杀梅林，坏人阵营获胜；如果刺杀错误，好人阵营获胜。

请根据你的判断选择最可能是梅林的玩家。"""
        
        user_prompt = f"""{game_context}

可选的玩家（不包括你自己）：
{chr(10).join(all_player_list)}

请以JSON格式返回你的决策，格式如下：
{{
    "reasoning": "你的推理过程（为什么选择这个玩家）",
    "target": 玩家ID
}}

只返回JSON，不要其他内容。"""
        
        try:
            response = self._call_llm(user_prompt, system_prompt)
            response = response.strip()
            if response.startswith("```"):
                lines = response.split("\n")
                response = "\n".join(lines[1:-1])
            if response.startswith("```json"):
                lines = response.split("\n")
                response = "\n".join(lines[1:-1])
            
            decision = json.loads(response)
            target = decision.get("target")
            return int(target) if target is not None else None
        except Exception as e:
            print(f"LLM刺杀决策错误: {e}")
            return None
    
    def generate_speech(self, context: DecisionContext, belief_system: BeliefSystem,
                       all_players: List[Dict], recent_speeches: List[Dict] = None,
                       mission_history: Optional[List[Dict]] = None) -> str:
        """使用LLM生成发言"""
        if not self.client:
            return "让我思考一下..."
        
        game_context = self._build_game_context_description(context, belief_system, all_players,
                                                          mission_history=context.mission_history if hasattr(context, 'mission_history') else None)
        player_names = {p["player_id"]: p["name"] for p in all_players}
        
        recent_speech_text = ""
        if recent_speeches:
            recent_speech_text = "\n最近的发言：\n"
            for speech in recent_speeches[-3:]:  # 只显示最近3条
                speaker_id = speech.get("player_id")
                speaker_name = player_names.get(speaker_id, f"玩家{speaker_id}")
                content = speech.get("content", "")
                recent_speech_text += f"- {speaker_name}: {content}\n"
        
        system_prompt = f"""你是一个阿瓦隆游戏中的{self.my_role.value}（{self.my_team.value}阵营）。
你的名字是{self.my_name}。
你的人格特质是{self.personality.value}。

请生成一段自然的发言，用于：
- 引导其他玩家
- 表达你的观点
- 分析局势
- 支持或质疑提议

发言应该：
- 符合你的角色身份和阵营目标
- 体现你的人格特质
- 简洁有力，不超过50字
- 使用中文"""
        
        user_prompt = f"""{game_context}
{recent_speech_text}

请生成你的发言（只返回发言内容，不要其他说明）："""
        
        try:
            response = self._call_llm(user_prompt, system_prompt)
            return response.strip()
        except Exception as e:
            # 如果LLM调用失败，使用回退策略生成简单发言
            if self.my_team == Team.GOOD:
                return "我需要更多信息才能做出判断。"
            else:
                return "让我观察一下局势。"
    
    # 回退策略（当LLM不可用时使用）
    def _fallback_team_proposal(self, context: DecisionContext, belief_system: BeliefSystem) -> List[int]:
        """回退策略：简单的队伍提议"""
        from agent.strategy import StrategyEngine
        fallback_engine = StrategyEngine(self.my_role, self.my_team, self.personality)
        return fallback_engine.decide_team_proposal(context, belief_system, self.my_player_id)
    
    def _fallback_vote(self, context: DecisionContext, belief_system: BeliefSystem,
                      proposed_team: List[int]) -> bool:
        """回退策略：简单的投票"""
        from agent.strategy import StrategyEngine
        fallback_engine = StrategyEngine(self.my_role, self.my_team, self.personality)
        return fallback_engine.decide_vote(context, belief_system, self.my_player_id, proposed_team)
    
    def _fallback_mission_vote(self, context: DecisionContext, belief_system: BeliefSystem,
                               mission_team: List[int]) -> bool:
        """回退策略：简单的任务投票"""
        from agent.strategy import StrategyEngine
        fallback_engine = StrategyEngine(self.my_role, self.my_team, self.personality)
        return fallback_engine.decide_mission_vote(context, belief_system, self.my_player_id, mission_team)

