"""
沟通生成器
生成有策略目的的发言内容
"""
from typing import List, Dict, Optional
from enum import Enum
from dataclasses import dataclass

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.rules import Team, GamePhase
from game.roles import RoleType
from agent.belief_system import BeliefSystem
from agent.strategy import Personality, StrategyEngine


class SpeechPurpose(Enum):
    """发言目的"""
    GUIDE = "引导"  # 引导其他玩家
    MISLEAD = "误导"  # 误导其他玩家
    PROBE = "试探"  # 试探其他玩家
    DEFEND = "辩护"  # 为自己辩护
    ACCUSE = "指控"  # 指控其他玩家
    SUPPORT = "支持"  # 支持其他玩家


@dataclass
class SpeechContext:
    """发言上下文"""
    game_phase: GamePhase
    current_round: int
    successful_missions: int
    failed_missions: int
    current_leader: int
    proposed_team: List[int]
    recent_speeches: List[Dict]  # 最近的发言记录


class CommunicationGenerator:
    """沟通生成器"""
    
    def __init__(self, my_role: RoleType, my_team: Team, personality: Personality = Personality.ANALYTICAL):
        self.my_role = my_role
        self.my_team = my_team
        self.personality = personality
    
    def generate_speech(self, context: SpeechContext, belief_system: BeliefSystem,
                       strategy_engine: StrategyEngine, my_player_id: int,
                       purpose: Optional[SpeechPurpose] = None) -> str:
        """
        生成发言内容
        """
        if purpose is None:
            # 根据局势自动决定发言目的
            purpose = self._determine_purpose(context, belief_system, strategy_engine, my_player_id)
        
        # 根据目的生成发言
        if purpose == SpeechPurpose.GUIDE:
            return self._generate_guide_speech(context, belief_system, my_player_id)
        elif purpose == SpeechPurpose.MISLEAD:
            return self._generate_mislead_speech(context, belief_system, my_player_id)
        elif purpose == SpeechPurpose.PROBE:
            return self._generate_probe_speech(context, belief_system, my_player_id)
        elif purpose == SpeechPurpose.DEFEND:
            return self._generate_defend_speech(context, belief_system, my_player_id)
        elif purpose == SpeechPurpose.ACCUSE:
            return self._generate_accuse_speech(context, belief_system, my_player_id)
        elif purpose == SpeechPurpose.SUPPORT:
            return self._generate_support_speech(context, belief_system, my_player_id)
        else:
            return self._generate_neutral_speech(context, belief_system, my_player_id)
    
    def _determine_purpose(self, context: SpeechContext, belief_system: BeliefSystem,
                          strategy_engine: StrategyEngine, my_player_id: int) -> SpeechPurpose:
        """根据局势决定发言目的"""
        if self.my_team == Team.GOOD:
            # 好人策略
            if context.proposed_team and my_player_id not in context.proposed_team:
                # 我不在提议的队伍中，可能需要引导或支持
                return SpeechPurpose.GUIDE
            elif context.proposed_team and my_player_id in context.proposed_team:
                # 我在队伍中，可能需要辩护
                return SpeechPurpose.DEFEND
            else:
                # 试探阶段
                return SpeechPurpose.PROBE
        else:
            # 坏人策略
            if context.failed_missions >= 2:
                # 已经失败2次，需要隐藏身份
                return SpeechPurpose.DEFEND
            elif context.proposed_team and my_player_id in context.proposed_team:
                # 我在队伍中，需要隐藏
                return SpeechPurpose.DEFEND
            else:
                # 误导阶段
                return SpeechPurpose.MISLEAD
    
    def _generate_guide_speech(self, context: SpeechContext, belief_system: BeliefSystem,
                              my_player_id: int) -> str:
        """生成引导性发言"""
        trusted_players = belief_system.get_most_trusted_players(count=2)
        suspicious_players = belief_system.get_most_suspicious_players(count=1)
        
        if trusted_players:
            trusted_name = belief_system.beliefs[trusted_players[0]].name
            if suspicious_players:
                suspicious_name = belief_system.beliefs[suspicious_players[0]].name
                return f"我认为{trusted_name}比较可信，建议让他参与任务。我对{suspicious_name}有些怀疑。"
            else:
                return f"我认为{trusted_name}比较可信，建议让他参与任务。"
        else:
            return "我们需要谨慎选择任务队伍，确保都是好人。"
    
    def _generate_mislead_speech(self, context: SpeechContext, belief_system: BeliefSystem,
                                my_player_id: int) -> str:
        """生成误导性发言"""
        trusted_players = belief_system.get_most_trusted_players(count=2)
        
        if trusted_players:
            # 误导：说好人是可疑的
            target_name = belief_system.beliefs[trusted_players[0]].name
            return f"我对{target_name}有些怀疑，他的行为不太对劲。"
        else:
            return "我觉得我们需要重新考虑队伍配置。"
    
    def _generate_probe_speech(self, context: SpeechContext, belief_system: BeliefSystem,
                              my_player_id: int) -> str:
        """生成试探性发言"""
        if context.recent_speeches:
            last_speaker = context.recent_speeches[-1].get("player_id")
            if last_speaker and last_speaker in belief_system.beliefs:
                speaker_name = belief_system.beliefs[last_speaker].name
                return f"我想听听{speaker_name}对这个队伍的看法。"
        
        return "大家对这个队伍有什么看法？"
    
    def _generate_defend_speech(self, context: SpeechContext, belief_system: BeliefSystem,
                               my_player_id: int) -> str:
        """生成辩护性发言"""
        if context.proposed_team and my_player_id in context.proposed_team:
            return "我认为这个队伍配置是合理的，我支持这个提议。"
        else:
            return "我的行为都是基于逻辑推理，希望大家相信我。"
    
    def _generate_accuse_speech(self, context: SpeechContext, belief_system: BeliefSystem,
                               my_player_id: int) -> str:
        """生成指控性发言"""
        suspicious_players = belief_system.get_most_suspicious_players(count=1)
        
        if suspicious_players:
            target_name = belief_system.beliefs[suspicious_players[0]].name
            return f"我认为{target_name}的行为很可疑，建议不要让他参与任务。"
        else:
            return "我觉得队伍中可能有问题，需要重新考虑。"
    
    def _generate_support_speech(self, context: SpeechContext, belief_system: BeliefSystem,
                                my_player_id: int) -> str:
        """生成支持性发言"""
        if context.proposed_team:
            return "我支持这个队伍配置，让我们试试看。"
        else:
            return "我同意当前的分析，让我们继续。"
    
    def _generate_neutral_speech(self, context: SpeechContext, belief_system: BeliefSystem,
                                my_player_id: int) -> str:
        """生成中性发言"""
        if context.current_round == 1:
            return "第一轮，我们需要谨慎选择。"
        elif context.successful_missions >= 2:
            return "我们已经成功完成了两个任务，继续保持。"
        elif context.failed_missions >= 1:
            return "任务失败了，我们需要重新分析局势。"
        else:
            return "让我们继续讨论。"
    
    def adapt_speech_style(self, speech: str) -> str:
        """根据人格特质调整发言风格"""
        if self.personality == Personality.AGGRESSIVE:
            # 激进：更直接、更强硬
            speech = speech.replace("我认为", "我确信")
            speech = speech.replace("建议", "必须")
        elif self.personality == Personality.CONSERVATIVE:
            # 保守：更谨慎、更委婉
            speech = speech.replace("我认为", "我觉得可能")
            speech = speech.replace("必须", "建议")
        elif self.personality == Personality.EMOTIONAL:
            # 情感：更感性
            speech = speech.replace("我认为", "我感觉")
            speech = speech.replace("逻辑", "直觉")
        
        return speech

