"""
智能体基类
整合所有核心模块：角色上下文、信念系统、策略引擎、沟通生成器
"""
from typing import Dict, List, Optional, Tuple
import random

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.rules import Team, GamePhase
from game.roles import RoleType
from agent.belief_system import BeliefSystem
from agent.strategy import StrategyEngine, Personality, DecisionContext
from agent.communication import CommunicationGenerator, SpeechContext, SpeechPurpose


class BaseAgent:
    """智能体基类"""
    
    def __init__(self, player_id: int, name: str, personality: Optional[Personality] = None,
                 use_llm: bool = False, llm_api_key: Optional[str] = None, 
                 llm_model: str = "gpt-4o-mini", llm_api_provider: str = "openai"):
        self.player_id = player_id
        self.name = name
        self.use_llm = use_llm
        self.llm_api_key = llm_api_key
        self.llm_model = llm_model
        self.llm_api_provider = llm_api_provider  # "openai" 或 "deepseek"
        
        # 角色信息（在游戏初始化时设置）
        self.role_type: Optional[RoleType] = None
        self.role = None
        self.team: Optional[Team] = None
        self.private_info: Optional[Dict] = None
        
        # 核心模块（在角色分配后初始化）
        self.belief_system: Optional[BeliefSystem] = None
        self.strategy_engine: Optional[StrategyEngine] = None
        self.llm_strategy_engine = None  # LLM策略引擎
        self.communication_generator: Optional[CommunicationGenerator] = None
        
        # 人格特质（如果没有指定，随机选择）
        if personality is None:
            personality = random.choice(list(Personality))
        self.personality = personality
        
        # 游戏状态记忆
        self.game_history: List[Dict] = []
    
    def initialize_role(self, role_type: RoleType, private_info: Dict):
        """
        初始化角色（由游戏引擎调用）
        private_info: 包含角色信息、可见玩家等私有信息
        """
        self.role_type = role_type
        from game.roles import get_role
        self.role = get_role(role_type)
        self.team = self.role.team
        self.private_info = private_info
        
        # 初始化核心模块
        # 获取所有玩家信息
        all_players = private_info.get("all_players", private_info.get("visible_players", []))
        visible_players = [p for p in all_players if p.get("is_self") or 
                          p.get("team") or p.get("possible_merlin")]
        
        self.belief_system = BeliefSystem(
            my_player_id=self.player_id,
            my_role=self.role_type,
            my_team=self.team,
            all_players=all_players,
            visible_players=visible_players
        )
        
        # 根据配置选择使用LLM策略引擎还是普通策略引擎
        if self.use_llm:
            try:
                from agent.llm_strategy import LLMStrategyEngine
                self.llm_strategy_engine = LLMStrategyEngine(
                    my_role=self.role_type,
                    my_team=self.team,
                    my_player_id=self.player_id,
                    my_name=self.name,
                    personality=self.personality,
                    api_key=self.llm_api_key,
                    model=self.llm_model,
                    api_provider=self.llm_api_provider
                )
                provider_name = "DeepSeek" if self.llm_api_provider == "deepseek" else "OpenAI"
                print(f"智能体 {self.name} 使用{provider_name} LLM策略引擎")
            except Exception as e:
                print(f"警告: 无法初始化LLM策略引擎 ({e})，使用普通策略引擎")
                self.use_llm = False
        
        # 仍然保留普通策略引擎作为回退
        if not self.strategy_engine:
            self.strategy_engine = StrategyEngine(
                    my_role=self.role_type,
                    my_team=self.team,
                    personality=self.personality
                )
        else:
            self.strategy_engine = StrategyEngine(
                my_role=self.role_type,
                my_team=self.team,
                personality=self.personality
            )
        
        self.communication_generator = CommunicationGenerator(
            my_role=self.role_type,
            my_team=self.team,
            personality=self.personality
        )
    
    def update_game_state(self, game_state: Dict):
        """更新游戏状态"""
        self.game_history.append(game_state)
        
        # 更新信念系统
        if self.belief_system and "recent_actions" in game_state:
            # 根据最近的行为更新信念
            for action in game_state["recent_actions"]:
                if action["type"] == "vote":
                    self.belief_system.update_belief_from_vote(
                        player_id=action["player_id"],
                        vote_decision=action["vote"],
                        team_proposal=action.get("team", []),
                        vote_result=action.get("result", False)
                    )
                elif action["type"] == "mission":
                    self.belief_system.update_belief_from_mission(
                        player_id=action["player_id"],
                        mission_success=action["success"],
                        mission_team=action.get("team", []),
                        mission_result=action.get("result", False)
                    )
    
    def propose_team(self, game_state: Dict) -> List[int]:
        """
        提议队伍
        返回: 提议的队伍成员ID列表
        """
        if not self.belief_system:
            return []
        
        context = DecisionContext(
            game_phase=GamePhase[game_state.get("current_phase", "DISCUSSION")],
            current_round=game_state.get("current_round", 1),
            successful_missions=game_state.get("successful_missions", 0),
            failed_missions=game_state.get("failed_missions", 0),
            current_leader=game_state.get("current_leader", 0),
            proposed_team=game_state.get("proposed_team", []),
            vote_round=game_state.get("vote_round", 0),
            mission_config=game_state.get("mission_config", {})
        )
        
        # 获取所有玩家信息
        all_players = self.private_info.get("all_players", []) if self.private_info else []
        # 获取任务历史
        mission_history = game_state.get("mission_history", [])
        
        # 如果使用LLM策略引擎
        if self.use_llm and self.llm_strategy_engine:
            try:
                return self.llm_strategy_engine.decide_team_proposal(
                    context=context,
                    belief_system=self.belief_system,
                    all_players=all_players,
                    mission_history=mission_history
                )
            except Exception as e:
                print(f"LLM决策失败，使用回退策略: {e}")
        
        # 使用普通策略引擎
        if not self.strategy_engine:
            return []
        
        return self.strategy_engine.decide_team_proposal(
            context=context,
            belief_system=self.belief_system,
            my_player_id=self.player_id
        )
    
    def vote_on_team(self, game_state: Dict, proposed_team: List[int]) -> bool:
        """
        对提议的队伍投票
        返回: True=同意, False=拒绝
        """
        if not self.belief_system:
            return True  # 默认同意
        
        context = DecisionContext(
            game_phase=GamePhase[game_state.get("current_phase", "VOTING")],
            current_round=game_state.get("current_round", 1),
            successful_missions=game_state.get("successful_missions", 0),
            failed_missions=game_state.get("failed_missions", 0),
            current_leader=game_state.get("current_leader", 0),
            proposed_team=proposed_team,
            vote_round=game_state.get("vote_round", 0),
            mission_config=game_state.get("mission_config", {})
        )
        
        # 获取所有玩家信息
        all_players = self.private_info.get("all_players", []) if self.private_info else []
        # 获取任务历史
        mission_history = game_state.get("mission_history", [])
        
        # 如果使用LLM策略引擎
        if self.use_llm and self.llm_strategy_engine:
            try:
                return self.llm_strategy_engine.decide_vote(
                    context=context,
                    belief_system=self.belief_system,
                    all_players=all_players,
                    proposed_team=proposed_team,
                    mission_history=mission_history
                )
            except Exception as e:
                print(f"LLM投票决策失败，使用回退策略: {e}")
        
        # 使用普通策略引擎
        if not self.strategy_engine:
            return True
        
        return self.strategy_engine.decide_vote(
            context=context,
            belief_system=self.belief_system,
            my_player_id=self.player_id,
            proposed_team=proposed_team
        )
    
    def vote_on_mission(self, game_state: Dict, mission_team: List[int]) -> bool:
        """
        任务投票（成功/失败）
        返回: True=成功, False=失败
        """
        if not self.belief_system:
            # 好人默认成功，坏人默认失败
            return self.team == Team.GOOD
        
        context = DecisionContext(
            game_phase=GamePhase[game_state.get("current_phase", "MISSION")],
            current_round=game_state.get("current_round", 1),
            successful_missions=game_state.get("successful_missions", 0),
            failed_missions=game_state.get("failed_missions", 0),
            current_leader=game_state.get("current_leader", 0),
            proposed_team=mission_team,
            vote_round=game_state.get("vote_round", 0),
            mission_config=game_state.get("mission_config", {})
        )
        
        # 获取所有玩家信息
        all_players = self.private_info.get("all_players", []) if self.private_info else []
        # 获取任务历史
        mission_history = game_state.get("mission_history", [])
        
        # 如果使用LLM策略引擎
        if self.use_llm and self.llm_strategy_engine:
            try:
                return self.llm_strategy_engine.decide_mission_vote(
                    context=context,
                    belief_system=self.belief_system,
                    all_players=all_players,
                    mission_team=mission_team,
                    mission_history=mission_history
                )
            except Exception as e:
                print(f"LLM任务投票决策失败，使用回退策略: {e}")
        
        # 使用普通策略引擎
        if not self.strategy_engine:
            return self.team == Team.GOOD
        
        return self.strategy_engine.decide_mission_vote(
            context=context,
            belief_system=self.belief_system,
            my_player_id=self.player_id,
            mission_team=mission_team
        )
    
    def assassinate(self, game_state: Dict) -> Optional[int]:
        """
        刺杀目标（仅刺客）
        返回: 目标玩家ID
        """
        if self.role_type != RoleType.ASSASSIN:
            return None
        
        if not self.belief_system:
            return None
        
        context = DecisionContext(
            game_phase=GamePhase[game_state.get("current_phase", "ASSASSINATION")],
            current_round=game_state.get("current_round", 1),
            successful_missions=game_state.get("successful_missions", 0),
            failed_missions=game_state.get("failed_missions", 0),
            current_leader=game_state.get("current_leader", 0),
            proposed_team=[],
            vote_round=0,
            mission_config={}
        )
        
        # 获取所有玩家信息
        all_players = self.private_info.get("all_players", []) if self.private_info else []
        # 获取任务历史
        mission_history = game_state.get("mission_history", [])
        
        # 如果使用LLM策略引擎
        if self.use_llm and self.llm_strategy_engine:
            try:
                return self.llm_strategy_engine.decide_assassination(
                    context=context,
                    belief_system=self.belief_system,
                    all_players=all_players,
                    mission_history=mission_history
                )
            except Exception as e:
                print(f"LLM刺杀决策失败，使用回退策略: {e}")
        
        # 使用普通策略引擎
        if not self.strategy_engine:
            return None
        
        return self.strategy_engine.decide_assassination(
            context=context,
            belief_system=self.belief_system
        )
    
    def generate_speech(self, game_state: Dict, recent_speeches: List[Dict] = None) -> str:
        """
        生成发言
        """
        if not self.belief_system:
            return "让我思考一下..."
        
        if recent_speeches is None:
            recent_speeches = []
        
        context = SpeechContext(
            game_phase=GamePhase[game_state.get("current_phase", "DISCUSSION")],
            current_round=game_state.get("current_round", 1),
            successful_missions=game_state.get("successful_missions", 0),
            failed_missions=game_state.get("failed_missions", 0),
            current_leader=game_state.get("current_leader", 0),
            proposed_team=game_state.get("proposed_team", []),
            recent_speeches=recent_speeches
        )
        
        # 获取所有玩家信息
        all_players = self.private_info.get("all_players", []) if self.private_info else []
        
        # 如果使用LLM策略引擎
        if self.use_llm and self.llm_strategy_engine:
            try:
                decision_context = DecisionContext(
                    game_phase=context.game_phase,
                    current_round=context.current_round,
                    successful_missions=context.successful_missions,
                    failed_missions=context.failed_missions,
                    current_leader=context.current_leader,
                    proposed_team=context.proposed_team,
                    vote_round=0,
                    mission_config={}
                )
                # 获取任务历史
                mission_history = game_state.get("mission_history", [])
                return self.llm_strategy_engine.generate_speech(
                    context=decision_context,
                    belief_system=self.belief_system,
                    all_players=all_players,
                    recent_speeches=recent_speeches,
                    mission_history=mission_history
                )
            except Exception as e:
                print(f"LLM发言生成失败，使用回退策略: {e}")
        
        # 使用普通沟通生成器
        if not self.communication_generator:
            return "让我思考一下..."
        
        speech = self.communication_generator.generate_speech(
            context=context,
            belief_system=self.belief_system,
            strategy_engine=self.strategy_engine,
            my_player_id=self.player_id
        )
        
        # 根据人格调整风格
        speech = self.communication_generator.adapt_speech_style(speech)
        
        return speech
    
    def get_belief_summary(self) -> Dict:
        """获取信念摘要（用于调试）"""
        if self.belief_system:
            return self.belief_system.get_belief_summary()
        return {}
    
    def get_strategy_priorities(self, game_state: Dict) -> List[str]:
        """获取当前策略优先级"""
        if not self.strategy_engine:
            return []
        
        context = DecisionContext(
            game_phase=GamePhase[game_state.get("current_phase", "DISCUSSION")],
            current_round=game_state.get("current_round", 1),
            successful_missions=game_state.get("successful_missions", 0),
            failed_missions=game_state.get("failed_missions", 0),
            current_leader=game_state.get("current_leader", 0),
            proposed_team=game_state.get("proposed_team", []),
            vote_round=game_state.get("vote_round", 0),
            mission_config=game_state.get("mission_config", {})
        )
        
        return self.strategy_engine.get_strategy_priority(context)

