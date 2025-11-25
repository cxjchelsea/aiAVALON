"""
策略决策引擎
根据角色、局势和信念做出决策
"""
from typing import List, Dict, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
import random

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.rules import Team, GamePhase
from game.roles import RoleType
from agent.belief_system import BeliefSystem


class Personality(Enum):
    """人格特质"""
    AGGRESSIVE = "激进型"  # 激进，敢于冒险
    CONSERVATIVE = "保守型"  # 保守，谨慎行事
    ANALYTICAL = "分析型"  # 理性分析
    EMOTIONAL = "情感型"  # 感性判断


@dataclass
class DecisionContext:
    """决策上下文"""
    game_phase: GamePhase
    current_round: int
    successful_missions: int
    failed_missions: int
    current_leader: int
    proposed_team: List[int]
    vote_round: int
    mission_config: Dict  # 当前任务配置


class StrategyEngine:
    """策略决策引擎"""
    
    def __init__(self, my_role: RoleType, my_team: Team, personality: Personality = Personality.ANALYTICAL):
        self.my_role = my_role
        self.my_team = my_team
        self.personality = personality
    
    def decide_team_proposal(self, context: DecisionContext, belief_system: BeliefSystem,
                            my_player_id: int) -> List[int]:
        """
        决定提议的队伍
        返回: 提议的队伍成员ID列表
        """
        team_size = context.mission_config.get("team_size", 2)
        all_player_ids = list(belief_system.beliefs.keys())
        all_player_ids.remove(my_player_id)  # 排除自己
        
        if self.my_team == Team.GOOD:
            # 好人策略：选择最信任的玩家
            trusted_players = belief_system.get_most_trusted_players(count=team_size)
            
            # 如果信任的玩家不够，补充随机选择
            if len(trusted_players) < team_size:
                remaining = [pid for pid in all_player_ids if pid not in trusted_players]
                needed = team_size - len(trusted_players)
                trusted_players.extend(remaining[:needed])
            
            # 总是包含自己（如果队伍大小允许）
            if my_player_id not in trusted_players and len(trusted_players) < team_size:
                trusted_players.append(my_player_id)
            
            return trusted_players[:team_size]
        
        else:
            # 坏人策略：尽量包含坏人队友，但也要包含一些好人以隐藏身份
            suspicious_players = belief_system.get_most_suspicious_players(count=team_size - 1)
            trusted_players = belief_system.get_most_trusted_players(count=1)
            
            # 混合策略：包含一些可疑的（可能是队友）和一些信任的（好人）
            team = []
            
            # 根据人格调整策略
            if self.personality == Personality.AGGRESSIVE:
                # 激进：多包含可疑玩家（可能是队友）
                team = suspicious_players[:team_size - 1] + trusted_players[:1]
            elif self.personality == Personality.CONSERVATIVE:
                # 保守：多包含信任的玩家（好人），隐藏身份
                team = trusted_players[:team_size - 1] + suspicious_players[:1]
            else:
                # 平衡策略
                evil_count = min(1, team_size // 2)  # 至少包含1个坏人
                good_count = team_size - evil_count
                team = suspicious_players[:evil_count] + trusted_players[:good_count]
            
            # 如果队伍大小不够，补充
            if len(team) < team_size:
                remaining = [pid for pid in all_player_ids if pid not in team]
                needed = team_size - len(team)
                team.extend(remaining[:needed])
            
            return team[:team_size]
    
    def decide_vote(self, context: DecisionContext, belief_system: BeliefSystem,
                   my_player_id: int, proposed_team: List[int]) -> bool:
        """
        决定是否投票同意提议的队伍
        返回: True=同意, False=拒绝
        """
        if self.my_team == Team.GOOD:
            # 好人策略：需要考虑投票轮次，避免流局导致坏人直接获胜
            suspicious_players = belief_system.get_most_suspicious_players(count=len(proposed_team))
            
            # 检查队伍中是否有可疑玩家（基于信念系统）
            has_suspicious = any(pid in proposed_team for pid in suspicious_players[:2])
            
            # 对于梅林：如果能看到坏人，检查队伍中是否有确定的坏人
            if self.my_role == RoleType.MERLIN:
                # 梅林能看到坏人，检查队伍中是否有确定的坏人
                evil_in_team = False
                for pid in proposed_team:
                    if pid in belief_system.beliefs:
                        belief = belief_system.beliefs[pid]
                        # 如果坏人概率很高（>0.8），认为是确定的坏人
                        if belief.team_probabilities.get(Team.EVIL, 0) > 0.8:
                            evil_in_team = True
                            break
                
                # 关键：如果已经接近第5次投票（vote_round >= 4），必须同意
                if context.vote_round >= 4:
                    return True  # 第5次投票，必须同意
                
                # 第4次投票（vote_round >= 3），即使有坏人也要考虑同意
                if context.vote_round >= 3:
                    # 如果队伍中有确定的坏人，但已经是第4次投票，倾向于同意（避免流局）
                    if evil_in_team:
                        return random.random() < 0.3  # 30%概率同意（冒险但避免流局）
                    else:
                        return True  # 没有确定的坏人，同意
                
                # 前3次投票：如果队伍中有确定的坏人，拒绝
                if evil_in_team:
                    return False
                else:
                    return True  # 没有确定的坏人，同意
            
            # 其他好人（看不到坏人）的策略
            # 关键：如果已经接近第5次投票（vote_round >= 4），必须同意，避免流局
            if context.vote_round >= 4:
                # 第5次投票，必须同意，否则流局坏人直接获胜
                return True
            
            # 如果已经第4次投票（vote_round >= 3），更倾向于同意
            if context.vote_round >= 3:
                # 第4次投票，即使有可疑玩家，也倾向于同意（避免流局）
                if has_suspicious:
                    return random.random() < 0.6  # 60%概率同意（避免流局）
                else:
                    return True  # 没有可疑玩家，同意
            
            # 前3次投票：如果队伍中包含可疑玩家，拒绝
            if has_suspicious:
                # 有可疑玩家，拒绝
                return False
            
            # 如果我在队伍中，通常同意
            if my_player_id in proposed_team:
                return True
            
            # 如果队伍看起来都是好人，同意
            return True
        
        else:
            # 坏人策略：根据局势决定
            # 如果队伍中有坏人队友，通常同意
            # 如果任务已经失败2次，可以同意以隐藏身份
            if context.failed_missions >= 2:
                # 已经失败2次，可以同意以隐藏身份
                return True
            
            # 如果我在队伍中，通常同意（可以破坏任务）
            if my_player_id in proposed_team:
                return True
            
            # 如果队伍看起来都是好人，拒绝（阻止任务）
            trusted_players = belief_system.get_most_trusted_players(count=len(proposed_team))
            all_trusted = all(pid in trusted_players for pid in proposed_team)
            
            if all_trusted:
                return False  # 都是好人，拒绝
            
            # 其他情况，根据人格决定
            if self.personality == Personality.AGGRESSIVE:
                return False  # 激进：拒绝
            else:
                return True  # 保守：同意
    
    def decide_mission_vote(self, context: DecisionContext, belief_system: BeliefSystem,
                          my_player_id: int, mission_team: List[int]) -> bool:
        """
        决定任务投票（成功/失败）
        返回: True=成功, False=失败
        """
        if self.my_team == Team.GOOD:
            # 好人总是投成功
            return True
        
        else:
            # 坏人策略：根据局势决定是否破坏
            fails_needed = context.mission_config.get("fails_needed", 1)
            
            # 如果我已经在之前的任务中投过失败，可能需要隐藏
            # 这里简化处理：根据任务轮次和失败次数决定
            
            if context.failed_missions >= 2:
                # 已经失败2次，可以投成功以隐藏身份
                return True
            
            if context.current_round <= 2:
                # 前两轮，可以破坏
                return False
            
            # 根据人格决定
            if self.personality == Personality.AGGRESSIVE:
                return False  # 激进：破坏
            else:
                # 保守：可能投成功以隐藏
                return True
    
    def decide_assassination(self, context: DecisionContext, belief_system: BeliefSystem) -> Optional[int]:
        """
        决定刺杀目标（仅刺客）
        返回: 目标玩家ID，如果无法决定则返回None
        """
        if self.my_role != RoleType.ASSASSIN:
            return None
        
        # 选择最可能是梅林的玩家
        # 根据信念系统，选择信任度最高且可能是梅林的玩家
        beliefs = belief_system.get_belief_summary()
        
        # 按信任度排序，选择最信任的（可能是梅林）
        candidates = sorted(
            beliefs.items(),
            key=lambda x: x[1]["trust_score"],
            reverse=True
        )
        
        if candidates:
            return candidates[0][0]  # 返回最信任的玩家ID
        
        return None
    
    def get_strategy_priority(self, context: DecisionContext) -> List[str]:
        """
        获取当前局势下的策略优先级
        返回策略列表，按优先级排序
        """
        priorities = []
        
        if self.my_team == Team.GOOD:
            if context.successful_missions >= 2:
                priorities.append("保护梅林")
                priorities.append("完成任务")
            else:
                priorities.append("完成任务")
                priorities.append("识别坏人")
                priorities.append("保护梅林")
        else:
            if context.failed_missions >= 2:
                priorities.append("隐藏身份")
                priorities.append("刺杀梅林")
            else:
                priorities.append("破坏任务")
                priorities.append("隐藏身份")
                priorities.append("识别梅林")
        
        return priorities

