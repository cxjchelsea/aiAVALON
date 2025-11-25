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
        
        # 记忆系统：存储对话历史和关键事件
        self.memory: List[str] = []
        self.max_memory_size = 20  # 限制记忆长度，防止Context窗口溢出
        
        # Prompt模板路径
        self.prompts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "prompts")
        
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
                # 对于qwen，base_url是必需的，如果未设置则使用默认值
                if not base_url:
                    base_url = "http://localhost:8000/v1"
            elif self.api_provider == "openai":
                api_key = api_key or os.getenv("OPENAI_API_KEY")
                base_url = None  # 使用OpenAI默认URL
            else:
                # 自定义API提供商
                api_key = api_key or os.getenv(f"{self.api_provider.upper()}_API_KEY") or os.getenv("OPENAI_API_KEY")
                base_url = os.getenv(f"{self.api_provider.upper()}_BASE_URL")
            
            # 判断是否能够初始化客户端
            can_init = False
            if self.api_provider == "qwen":
                # qwen只需要base_url即可
                can_init = bool(base_url)
            elif self.api_provider == "openai":
                # openai需要api_key
                can_init = bool(api_key)
            else:
                # 其他提供商需要api_key或base_url
                can_init = bool(api_key or base_url)
            
            if can_init:
                if base_url:
                    # 需要指定base_url（DeepSeek、Qwen或自定义）
                    # 为不同提供商设置不同的HTTP客户端配置
                    try:
                        import httpx
                        if self.api_provider == "qwen":
                            # 本地Qwen：更长的超时时间，允许更长的连接时间
                            http_client = httpx.Client(
                                timeout=httpx.Timeout(120.0, connect=30.0, read=120.0),
                                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
                            )
                            self.client = OpenAI(api_key=api_key, base_url=base_url, http_client=http_client)
                            print(f"Qwen客户端初始化: base_url={base_url}, 超时=120秒")
                        elif self.api_provider == "deepseek":
                            # DeepSeek：中等超时时间
                            http_client = httpx.Client(
                                timeout=httpx.Timeout(90.0, connect=20.0, read=90.0),
                                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
                            )
                            self.client = OpenAI(api_key=api_key, base_url=base_url, http_client=http_client)
                            print(f"DeepSeek客户端初始化: base_url={base_url}, 超时=90秒")
                        else:
                            # 其他提供商：默认配置
                            self.client = OpenAI(api_key=api_key, base_url=base_url)
                    except ImportError:
                        # 如果httpx不可用，使用默认配置
                        print(f"警告: httpx未安装，使用默认HTTP配置")
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
    
    def add_to_memory(self, event: str):
        """添加事件到记忆"""
        self.memory.append(event)
        # 限制记忆长度，只保留最近的N条
        if len(self.memory) > self.max_memory_size:
            self.memory = self.memory[-self.max_memory_size:]
    
    def get_memory_summary(self) -> str:
        """获取记忆摘要"""
        if not self.memory:
            return "暂无记忆。"
        return "\n".join([f"- {event}" for event in self.memory[-10:]])  # 只返回最近10条
    
    def _load_prompt_template(self, role_name: str, action_name: str) -> Optional[str]:
        """加载Prompt模板"""
        role_file = os.path.join(self.prompts_dir, "roles", f"{role_name}.md")
        action_file = os.path.join(self.prompts_dir, "actions", f"{action_name}.md")
        
        role_prompt = ""
        action_prompt = ""
        
        # 加载角色Prompt
        if os.path.exists(role_file):
            try:
                with open(role_file, "r", encoding="utf-8") as f:
                    role_prompt = f.read()
            except Exception as e:
                print(f"警告: 无法加载角色Prompt {role_file}: {e}")
        
        # 加载行为Prompt
        if os.path.exists(action_file):
            try:
                with open(action_file, "r", encoding="utf-8") as f:
                    action_prompt = f.read()
            except Exception as e:
                print(f"警告: 无法加载行为Prompt {action_file}: {e}")
        
        if role_prompt or action_prompt:
            return f"{role_prompt}\n\n{action_prompt}" if role_prompt and action_prompt else (role_prompt or action_prompt)
        
        return None
    
    def _build_fact_check_context(self, context: DecisionContext, 
                                  all_players: List[Dict],
                                  mission_history: Optional[List[Dict]] = None) -> Dict:
        """构建事实核查上下文（结构化数据）"""
        player_names = {p["player_id"]: p["name"] for p in all_players}
        
        facts = {
            "current_round": context.current_round,
            "current_phase": context.game_phase.value,
            "successful_missions": context.successful_missions,
            "failed_missions": context.failed_missions,
            "vote_round": context.vote_round,
            "current_leader": {
                "player_id": context.current_leader,
                "name": player_names.get(context.current_leader, f"玩家{context.current_leader}")
            },
            "mission_config": context.mission_config,
            "players": [
                {
                    "player_id": p["player_id"],
                    "name": p["name"]
                }
                for p in all_players
            ]
        }
        
        if mission_history:
            facts["mission_history"] = [
                {
                    "round": m["round"],
                    "team": m["team"],
                    "team_ids": m.get("team_ids", []),
                    "success": m["success"],
                    "fail_count": m.get("fail_count", 0),
                    "team_size": m.get("team_size", len(m["team"]))
                }
                for m in mission_history
            ]
        
        return facts
    
    def _get_timeout_for_provider(self) -> int:
        """根据API提供商返回合适的超时时间"""
        if self.api_provider == "qwen":
            # 本地Qwen可能需要更长时间
            return 120  # 2分钟
        elif self.api_provider == "deepseek":
            # DeepSeek可能需要更长时间
            return 90  # 1.5分钟
        else:
            # OpenAI等
            return 60  # 1分钟
    
    def _call_llm(self, prompt: str, system_prompt: str = None, max_retries: int = 3) -> str:
        """调用LLM，带重试机制和更好的错误处理"""
        if not self.client:
            raise RuntimeError("LLM客户端未初始化")
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # 根据提供商设置超时时间
        timeout = self._get_timeout_for_provider()
        
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                # 计算总token数（粗略估算）
                total_chars = len(system_prompt or "") + len(prompt)
                estimated_tokens = total_chars // 3  # 粗略估算：3个字符约等于1个token
                
                # 根据内容长度调整max_tokens
                max_tokens = min(2000, max(500, estimated_tokens + 200))  # 至少500，最多2000
                
                print(f"[{self.my_name}] 调用LLM ({self.api_provider}, 模型: {self.model}, 超时: {timeout}秒, 尝试 {attempt + 1}/{max_retries + 1})...")
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=max_tokens,
                    timeout=timeout
                )
                
                result = response.choices[0].message.content.strip()
                print(f"[{self.my_name}] LLM调用成功")
                return result
                
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                error_type = type(e).__name__
                
                # 判断错误类型
                is_connection_error = any(keyword in error_str for keyword in [
                    "connection", "connect", "network", "10054", "远程主机", 
                    "connection error", "connecterror", "refused"
                ])
                is_timeout_error = any(keyword in error_str for keyword in [
                    "timeout", "timed out", "read timeout", "timeout error"
                ])
                is_api_error = any(keyword in error_str for keyword in [
                    "api", "401", "403", "429", "500", "502", "503", "504"
                ])
                
                print(f"[{self.my_name}] LLM调用失败 (尝试 {attempt + 1}/{max_retries + 1}): {error_type}")
                print(f"  错误详情: {str(e)[:300]}")
                
                if attempt < max_retries:
                    # 根据错误类型决定等待时间
                    import time
                    if is_connection_error:
                        wait_time = 3.0 * (attempt + 1)  # 连接错误：3秒、6秒、9秒
                        print(f"  错误类型: 连接错误，{wait_time}秒后重试...")
                    elif is_timeout_error:
                        wait_time = 2.0 * (attempt + 1)  # 超时错误：2秒、4秒、6秒
                        print(f"  错误类型: 超时错误，{wait_time}秒后重试...")
                        # 超时错误时，可以尝试增加超时时间
                        if attempt == 1:  # 第二次重试时
                            timeout = min(timeout * 1.5, 180)  # 增加50%超时，最多3分钟
                            print(f"  增加超时时间到 {timeout}秒")
                    elif is_api_error:
                        wait_time = 5.0 * (attempt + 1)  # API错误：5秒、10秒、15秒
                        print(f"  错误类型: API错误，{wait_time}秒后重试...")
                    else:
                        wait_time = 1.0 * (attempt + 1)  # 其他错误：1秒、2秒、3秒
                        print(f"  错误类型: 其他错误，{wait_time}秒后重试...")
                    
                    time.sleep(wait_time)
                    continue
                else:
                    # 最后一次尝试失败，抛出异常
                    error_msg = f"LLM调用失败 (已重试 {max_retries + 1} 次)"
                    if is_connection_error:
                        error_msg += f": 连接错误 - 请检查网络连接和API服务状态"
                        if self.api_provider == "qwen":
                            error_msg += f"\n  提示: 如果是本地Qwen，请检查服务是否运行在 {self.client.base_url}"
                    elif is_timeout_error:
                        error_msg += f": 请求超时 ({timeout}秒) - 可能是网络慢或服务响应慢"
                        if self.api_provider == "deepseek":
                            error_msg += f"\n  提示: DeepSeek可能需要更长时间，可以尝试增加超时时间"
                    elif is_api_error:
                        error_msg += f": API错误 - 请检查API密钥和服务状态"
                    else:
                        error_msg += f": {error_type} - {str(e)[:200]}"
                    print(error_msg)
                    raise RuntimeError(error_msg) from e
    
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
        if mission_history and len(mission_history) > 0:
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
        else:
            # 明确说明没有任务历史（特别是第1轮）
            if context.current_round == 1:
                mission_history_desc = "\n⚠️ 重要：这是第1轮任务，**还没有任何任务历史**！\n"
                mission_history_desc += "你不能说\"前几轮\"、\"之前的表现\"、\"之前的任务\"等，因为这是第一轮。\n"
            else:
                mission_history_desc = "\n任务历史：暂无\n"
        
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
        """使用LLM决定提议的队伍（集成Prompt模板、记忆、事实核查、CoT推理）"""
        if not self.client:
            raise RuntimeError("LLM客户端未初始化，无法进行决策。请检查API配置。")
        
        # 1. 加载Prompt模板
        role_name_map = {
            RoleType.MERLIN: "merlin",
            RoleType.ASSASSIN: "assassin",
            RoleType.PERCIVAL: "percival",
            RoleType.MORGANA: "morgana",
            RoleType.SERVANT: "servant",
            RoleType.MORDRED: "mordred"
        }
        role_name = role_name_map.get(self.my_role, "servant")
        template = self._load_prompt_template(role_name, "team_proposal")
        
        # 2. 构建事实核查上下文（结构化数据）
        facts = self._build_fact_check_context(context, all_players, mission_history)
        facts_json = json.dumps(facts, ensure_ascii=False, indent=2)
        
        # 3. 构建游戏上下文描述
        game_context = self._build_game_context_description(context, belief_system, all_players, 
                                                          mission_history=mission_history)
        player_names = {p["player_id"]: p["name"] for p in all_players}
        
        # 4. 获取记忆
        memory_summary = self.get_memory_summary()
        
        # 5. 构建可见信息描述
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
        
        # 6. 构建System Prompt（使用模板或默认）
        if template:
            system_prompt = f"""{template}

你的名字是{self.my_name}。
你的人格特质是{self.personality.value}。

**重要：事实核查**
你的回答必须基于以下提供的游戏事实（JSON格式），不得编造信息：
{facts_json}

**记忆（之前的决策和行为）**：
{memory_summary}
"""
        else:
            # 回退到默认Prompt
            role_info = f"你是{self.my_role.value}（{self.my_team.value}阵营）"
            system_prompt = f"""你是一个阿瓦隆游戏中的玩家。
{role_info}
你的名字是{self.my_name}。
你的人格特质是{self.personality.value}。

**重要：事实核查**
你的回答必须基于以下提供的游戏事实（JSON格式），不得编造信息：
{facts_json}

**记忆（之前的决策和行为）**：
{memory_summary}

请按照思维链（Chain-of-Thought）进行推理，展示你的思考过程。"""
        
        # 7. 构建User Prompt（包含CoT要求）
        user_prompt = f"""{game_context}
{visible_info_desc}

所有玩家（你可以选择任意{context.mission_config.get('team_size', 2)}人，包括你自己）：
{chr(10).join([f"{pid}: {name}" for pid, name in player_names.items()])}

当前任务需要 {context.mission_config.get('team_size', 2)} 人。
注意：作为队长，你可以选择自己加入队伍。

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
    "team": [玩家ID列表，例如 [0, 1, 2]]
}}

只返回JSON，不要其他内容。"""
        
        try:
            response = self._call_llm(user_prompt, system_prompt)
            # 解析JSON
            response = response.strip()
            if response.startswith("```"):
                lines = response.split("\n")
                response = "\n".join(lines[1:-1])
            if response.startswith("```json"):
                lines = response.split("\n")
                response = "\n".join(lines[1:-1])
            
            decision = json.loads(response)
            team = decision.get("team", [])
            
            # 事实核查：验证队伍大小
            required_size = context.mission_config.get("team_size", 2)
            if len(team) != required_size:
                raise RuntimeError(f"LLM返回的队伍大小不正确：期望{required_size}人，实际{len(team)}人")
            
            # 事实核查：验证玩家ID有效性
            valid_player_ids = {p["player_id"] for p in all_players}
            if not all(pid in valid_player_ids for pid in team):
                raise RuntimeError(f"LLM返回的队伍包含无效的玩家ID")
            
            # 记录决策到记忆
            team_names = [player_names.get(pid, f"玩家{pid}") for pid in team]
            self.add_to_memory(f"第{context.current_round}轮：我提议了队伍 {', '.join(team_names)} (IDs: {team})")
            
            return team
        except Exception as e:
            raise RuntimeError(f"LLM队伍提议决策失败: {e}")
    
    def decide_vote(self, context: DecisionContext, belief_system: BeliefSystem,
                   all_players: List[Dict], proposed_team: List[int], 
                   mission_history: Optional[List[Dict]] = None) -> bool:
        """使用LLM决定是否投票同意（集成Prompt模板、记忆、事实核查、CoT推理）"""
        if not self.client:
            raise RuntimeError("LLM客户端未初始化，无法进行决策。请检查API配置。")
        
        # 1. 加载Prompt模板
        role_name_map = {
            RoleType.MERLIN: "merlin",
            RoleType.ASSASSIN: "assassin",
            RoleType.PERCIVAL: "percival",
            RoleType.MORGANA: "morgana",
            RoleType.SERVANT: "servant",
            RoleType.MORDRED: "mordred"
        }
        role_name = role_name_map.get(self.my_role, "servant")
        template = self._load_prompt_template(role_name, "vote")
        
        # 2. 构建事实核查上下文
        facts = self._build_fact_check_context(context, all_players, mission_history)
        facts["proposed_team"] = proposed_team
        facts_json = json.dumps(facts, ensure_ascii=False, indent=2)
        
        # 3. 构建游戏上下文
        game_context = self._build_game_context_description(context, belief_system, all_players, 
                                                          proposed_team=proposed_team,
                                                          mission_history=mission_history)
        player_names = {p["player_id"]: p["name"] for p in all_players}
        team_names = [player_names.get(pid, f"玩家{pid}") for pid in proposed_team]
        
        # 4. 获取记忆
        memory_summary = self.get_memory_summary()
        
        # 5. 检查是否是队长
        is_leader = context.current_leader == self.my_player_id
        
        # 6. 构建System Prompt
        if template:
            system_prompt = f"""{template}

你的名字是{self.my_name}。
你的人格特质是{self.personality.value}。

**重要：事实核查**
你的回答必须基于以下提供的游戏事实（JSON格式），不得编造信息：
{facts_json}

**记忆（之前的决策和行为）**：
{memory_summary}
"""
            if is_leader:
                system_prompt += "\n重要：你是队长，你提议了这个队伍，所以你必须投票同意。"
        else:
            # 回退到默认Prompt
            system_prompt = f"""你是一个阿瓦隆游戏中的玩家。
你的名字是{self.my_name}。
你的人格特质是{self.personality.value}。

**重要：事实核查**
你的回答必须基于以下提供的游戏事实（JSON格式），不得编造信息：
{facts_json}

**记忆（之前的决策和行为）**：
{memory_summary}

请按照思维链（Chain-of-Thought）进行推理。"""
            if is_leader:
                system_prompt += "\n重要：你是队长，你提议了这个队伍，所以你必须投票同意。"
        
        # 7. 构建User Prompt（包含CoT要求）
        user_prompt = f"""{game_context}

当前提议的队伍是：{', '.join(team_names)}

**请按照以下步骤思考（Chain-of-Thought）**：
1. 检查流局风险（关键！）
2. 分析提议的队伍
3. 分析任务历史
4. 考虑角色目标
5. 做出最终决策

请以JSON格式返回你的决策，格式如下：
{{
    "thinking_process": {{
        "step1_rejection_risk": "你的分析...",
        "step2_team_analysis": "你的分析...",
        "step3_mission_history": "你的分析...",
        "step4_role_objectives": "你的分析...",
        "step5_decision": "你的最终决策理由..."
    }},
    "vote": true 或 false
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
            vote = decision.get("vote", True)
            
            # 事实核查：第5次投票必须同意（流局保护）
            if context.vote_round >= 4:
                vote = True
            
            # 事实核查：队长必须同意自己提议的队伍
            if is_leader:
                vote = True
            
            # 记录决策到记忆
            vote_text = "同意" if vote else "拒绝"
            self.add_to_memory(f"第{context.current_round}轮投票：我对队伍 {', '.join(team_names)} 投了{vote_text}票")
            
            return bool(vote)
        except Exception as e:
            raise RuntimeError(f"LLM投票决策失败: {e}")
    
    def decide_mission_vote(self, context: DecisionContext, belief_system: BeliefSystem,
                           all_players: List[Dict], mission_team: List[int],
                           mission_history: Optional[List[Dict]] = None) -> bool:
        """使用LLM决定任务投票（成功/失败）（集成Prompt模板、记忆、事实核查、CoT推理）"""
        if not self.client:
            raise RuntimeError("LLM客户端未初始化，无法进行决策。请检查API配置。")
        
        # 好人总是投成功（事实核查）
        if self.my_team == Team.GOOD:
            self.add_to_memory(f"第{context.current_round}轮任务：我投了成功票（好人必须投成功）")
            return True
        
        # 坏人需要决定是否破坏
        # 1. 加载Prompt模板
        role_name_map = {
            RoleType.MERLIN: "merlin",
            RoleType.ASSASSIN: "assassin",
            RoleType.PERCIVAL: "percival",
            RoleType.MORGANA: "morgana",
            RoleType.SERVANT: "servant",
            RoleType.MORDRED: "mordred"
        }
        role_name = role_name_map.get(self.my_role, "servant")
        template = self._load_prompt_template(role_name, "mission_vote")
        
        # 2. 构建事实核查上下文
        facts = self._build_fact_check_context(context, all_players, mission_history)
        facts["mission_team"] = mission_team
        facts_json = json.dumps(facts, ensure_ascii=False, indent=2)
        
        # 3. 构建游戏上下文
        game_context = self._build_game_context_description(context, belief_system, all_players, 
                                                          proposed_team=mission_team,
                                                          mission_history=mission_history)
        player_names = {p["player_id"]: p["name"] for p in all_players}
        team_names = [player_names.get(pid, f"玩家{pid}") for pid in mission_team]
        
        # 4. 获取记忆
        memory_summary = self.get_memory_summary()
        
        # 5. 构建System Prompt
        if template:
            system_prompt = f"""{template}

你的名字是{self.my_name}。

**重要：事实核查**
你的回答必须基于以下提供的游戏事实（JSON格式），不得编造信息：
{facts_json}

**记忆（之前的决策和行为）**：
{memory_summary}
"""
        else:
            system_prompt = f"""你是一个阿瓦隆游戏中的{self.my_role.value}（{self.my_team.value}阵营）。
你的名字是{self.my_name}。

作为坏人，你需要决定是否破坏任务（投失败票）。

**重要：事实核查**
你的回答必须基于以下提供的游戏事实（JSON格式），不得编造信息：
{facts_json}

**记忆（之前的决策和行为）**：
{memory_summary}

请按照思维链（Chain-of-Thought）进行推理。"""
        
        # 6. 构建User Prompt（包含CoT要求）
        user_prompt = f"""{game_context}

当前任务队伍是：{', '.join(team_names)}
需要 {context.mission_config.get('fails_needed', 1)} 张失败票才能破坏任务。

**请按照以下步骤思考（Chain-of-Thought）**：
1. 检查阵营（好人必须投成功）
2. 分析当前局势
3. 分析任务队伍
4. 考虑隐藏身份
5. 做出最终决策

请以JSON格式返回你的决策，格式如下：
{{
    "thinking_process": {{
        "step1_team_check": "你的分析...",
        "step2_current_situation": "你的分析...",
        "step3_mission_team": "你的分析...",
        "step4_identity_hiding": "你的分析...",
        "step5_decision": "你的最终决策理由..."
    }},
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
            
            # 记录决策到记忆
            result_text = "成功" if success else "失败"
            self.add_to_memory(f"第{context.current_round}轮任务：我投了{result_text}票")
            
            return bool(success)
        except Exception as e:
            raise RuntimeError(f"LLM任务投票决策失败: {e}")
    
    def decide_assassination(self, context: DecisionContext, belief_system: BeliefSystem,
                            all_players: List[Dict], mission_history: Optional[List[Dict]] = None) -> Optional[int]:
        """使用LLM决定刺杀目标（集成Prompt模板、记忆、事实核查、CoT推理）"""
        if self.my_role != RoleType.ASSASSIN or not self.client:
            return None
        
        # 1. 加载Prompt模板
        template = self._load_prompt_template("assassin", "assassination")
        
        # 2. 构建事实核查上下文
        facts = self._build_fact_check_context(context, all_players, mission_history)
        facts_json = json.dumps(facts, ensure_ascii=False, indent=2)
        
        # 3. 构建游戏上下文
        game_context = self._build_game_context_description(context, belief_system, all_players,
                                                          mission_history=mission_history)
        player_names = {p["player_id"]: p["name"] for p in all_players}
        all_player_list = [f"{pid}: {name}" for pid, name in player_names.items() if pid != self.my_player_id]
        
        # 4. 获取记忆
        memory_summary = self.get_memory_summary()
        
        # 5. 构建System Prompt
        if template:
            system_prompt = f"""{template}

你的名字是{self.my_name}。

**重要：事实核查**
你的回答必须基于以下提供的游戏事实（JSON格式），不得编造信息：
{facts_json}

**记忆（之前的决策和行为）**：
{memory_summary}
"""
        else:
            system_prompt = f"""你是一个阿瓦隆游戏中的刺客（坏人阵营）。
好人已经完成了3个任务，现在你可以刺杀梅林。
如果成功刺杀梅林，坏人阵营获胜；如果刺杀错误，好人阵营获胜。

**重要：事实核查**
你的回答必须基于以下提供的游戏事实（JSON格式），不得编造信息：
{facts_json}

**记忆（之前的决策和行为）**：
{memory_summary}

请按照思维链（Chain-of-Thought）进行推理。"""
        
        # 6. 构建User Prompt（包含CoT要求）
        user_prompt = f"""{game_context}

可选的玩家（不包括你自己）：
{chr(10).join(all_player_list)}

**请按照以下步骤思考（Chain-of-Thought）**：
1. 回顾游戏历史
2. 分析梅林的特征
3. 排除不可能的人
4. 评估每个候选人
5. 做出最终决策

请以JSON格式返回你的决策，格式如下：
{{
    "thinking_process": {{
        "step1_game_history": "你的分析...",
        "step2_merlin_characteristics": "你的分析...",
        "step3_exclusion": "你排除了哪些人...",
        "step4_candidate_evaluation": "你对每个候选人的评估...",
        "step5_decision": "你的最终决策理由..."
    }},
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
            
            # 事实核查：验证目标玩家ID有效性
            if target is not None:
                target = int(target)
                valid_player_ids = {p["player_id"] for p in all_players}
                if target not in valid_player_ids or target == self.my_player_id:
                    raise RuntimeError(f"LLM返回的刺杀目标无效：{target}")
                
                # 记录决策到记忆
                target_name = player_names.get(target, f"玩家{target}")
                self.add_to_memory(f"刺杀阶段：我选择刺杀 {target_name} (ID: {target})")
            
            return target
        except Exception as e:
            raise RuntimeError(f"LLM刺杀决策失败: {e}")
    
    def generate_speech(self, context: DecisionContext, belief_system: BeliefSystem,
                       all_players: List[Dict], recent_speeches: List[Dict] = None,
                       mission_history: Optional[List[Dict]] = None) -> str:
        """使用LLM生成发言（集成Prompt模板、记忆、事实核查、CoT推理）"""
        if not self.client:
            raise RuntimeError("LLM客户端未初始化，无法生成发言。请检查API配置。")
        
        # 1. 加载Prompt模板
        role_name_map = {
            RoleType.MERLIN: "merlin",
            RoleType.ASSASSIN: "assassin",
            RoleType.PERCIVAL: "percival",
            RoleType.MORGANA: "morgana",
            RoleType.SERVANT: "servant",
            RoleType.MORDRED: "mordred"
        }
        role_name = role_name_map.get(self.my_role, "servant")
        template = self._load_prompt_template(role_name, "speech")
        
        # 2. 构建事实核查上下文
        facts = self._build_fact_check_context(context, all_players, mission_history)
        facts_json = json.dumps(facts, ensure_ascii=False, indent=2)
        
        # 3. 构建游戏上下文
        game_context = self._build_game_context_description(context, belief_system, all_players,
                                                          mission_history=mission_history)
        player_names = {p["player_id"]: p["name"] for p in all_players}
        
        # 4. 获取记忆
        memory_summary = self.get_memory_summary()
        
        # 5. 构建最近发言文本
        recent_speech_text = ""
        if recent_speeches:
            recent_speech_text = "\n最近的发言：\n"
            for speech in recent_speeches[-3:]:  # 只显示最近3条
                speaker_id = speech.get("player_id")
                speaker_name = player_names.get(speaker_id, f"玩家{speaker_id}")
                content = speech.get("speech", speech.get("content", ""))
                recent_speech_text += f"- {speaker_name}: {content}\n"
        
        # 6. 构建System Prompt
        if template:
            system_prompt = f"""{template}

你的名字是{self.my_name}。
你的人格特质是{self.personality.value}。

**重要：事实核查**
你的发言必须基于以下提供的游戏事实（JSON格式），不得编造信息：
{facts_json}

**记忆（之前的决策和行为）**：
{memory_summary}
"""
        else:
            system_prompt = f"""你是一个阿瓦隆游戏中的{self.my_role.value}（{self.my_team.value}阵营）。
你的名字是{self.my_name}。
你的人格特质是{self.personality.value}。

请生成一段自然的发言。

**重要：事实核查**
你的发言必须基于以下提供的游戏事实（JSON格式），不得编造信息：
{facts_json}

**记忆（之前的决策和行为）**：
{memory_summary}

请按照思维链（Chain-of-Thought）思考发言目的和内容。"""
        
        # 7. 构建User Prompt（包含CoT要求）
        # 特别强调第1轮的情况
        round_warning = ""
        if context.current_round == 1:
            round_warning = "\n⚠️ **特别提醒：这是第1轮任务，还没有任何任务历史！**\n"
            round_warning += "你不能在发言中提到\"前几轮\"、\"之前的表现\"、\"之前的任务\"等，因为这是第一轮。\n"
            round_warning += "只能说\"第一轮\"、\"刚开始\"、\"还没有历史\"等。\n"
        
        user_prompt = f"""{game_context}
{recent_speech_text}
{round_warning}
**请按照以下步骤思考（Chain-of-Thought）**：
1. 分析当前局势（特别注意：这是第{context.current_round}轮，是否有任务历史？）
2. 确定发言目的
3. 分析最近发言
4. 考虑角色身份
5. 生成发言（确保发言内容符合当前轮次，不要编造不存在的历史）

请生成你的发言（只返回发言内容，不要其他说明）："""
        
        try:
            response = self._call_llm(user_prompt, system_prompt)
            speech = response.strip()
            
            # 记录发言到记忆
            self.add_to_memory(f"第{context.current_round}轮讨论：我说了\"{speech[:30]}...\"")
            
            return speech
        except Exception as e:
            # 对于发言生成，如果LLM调用失败，提供回退方案
            error_str = str(e).lower()
            is_connection_error = any(keyword in error_str for keyword in [
                "connection", "connect", "network", "timeout"
            ])
            
            if is_connection_error:
                # 连接错误时，生成简单的回退发言
                print(f"警告: LLM连接失败，使用回退发言生成")
                fallback_speech = self._generate_fallback_speech(context, belief_system, all_players)
                self.add_to_memory(f"第{context.current_round}轮讨论：LLM连接失败，使用了回退发言")
                return fallback_speech
            else:
                # 其他错误，抛出异常
                raise RuntimeError(f"LLM发言生成失败: {e}")
    
    def _generate_fallback_speech(self, context: DecisionContext, belief_system: BeliefSystem,
                                  all_players: List[Dict]) -> str:
        """生成回退发言（当LLM不可用时使用）"""
        player_names = {p["player_id"]: p["name"] for p in all_players}
        
        # 根据角色和局势生成简单发言
        if self.my_team == Team.GOOD:
            if context.successful_missions >= 2:
                return "我们已经成功完成了两个任务，继续保持。"
            elif context.failed_missions >= 1:
                return "任务失败了，我们需要重新分析局势。"
            else:
                return "我们需要谨慎选择任务队伍，确保都是好人。"
        else:
            if context.failed_missions >= 2:
                return "我觉得我们需要重新考虑策略。"
            else:
                return "让我观察一下局势。"

