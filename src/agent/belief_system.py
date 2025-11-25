"""
动态信念系统
基于贝叶斯推理更新对其他玩家身份的信念
"""
from typing import Dict, List, Optional
import numpy as np
from dataclasses import dataclass, field

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.rules import Team
from game.roles import RoleType


@dataclass
class PlayerBelief:
    """对单个玩家的信念"""
    player_id: int
    name: str
    # 阵营概率分布 [好人概率, 坏人概率]
    team_probabilities: Dict[Team, float] = field(default_factory=lambda: {Team.GOOD: 0.5, Team.EVIL: 0.5})
    # 角色概率分布（可选，更细粒度）
    role_probabilities: Dict[RoleType, float] = field(default_factory=dict)
    # 信任度（0-1之间）
    trust_score: float = 0.5
    # 历史行为记录
    behavior_history: List[str] = field(default_factory=list)


class BeliefSystem:
    """动态信念系统"""
    
    def __init__(self, my_player_id: int, my_role: RoleType, my_team: Team, 
                 all_players: List[Dict], visible_players: List[Dict]):
        """
        初始化信念系统
        all_players: 所有玩家信息
        visible_players: 我能看到的玩家信息（包括自己）
        """
        self.my_player_id = my_player_id
        self.my_role = my_role
        self.my_team = my_team
        self.all_players = all_players
        self.visible_players = visible_players
        
        # 初始化信念
        self.beliefs: Dict[int, PlayerBelief] = {}
        self._initialize_beliefs()
    
    def _initialize_beliefs(self):
        """初始化信念"""
        # 根据可见信息初始化信念
        visible_player_ids = {p["player_id"] for p in self.visible_players}
        
        for player_info in self.all_players:
            player_id = player_info["player_id"]
            name = player_info["name"]
            
            belief = PlayerBelief(player_id=player_id, name=name)
            
            if player_id == self.my_player_id:
                # 自己：确定身份
                belief.team_probabilities[Team.GOOD] = 1.0 if self.my_team == Team.GOOD else 0.0
                belief.team_probabilities[Team.EVIL] = 1.0 if self.my_team == Team.EVIL else 0.0
                belief.trust_score = 1.0
            elif player_id in visible_player_ids:
                # 能看到：根据角色能力更新信念
                visible_info = next(p for p in self.visible_players if p["player_id"] == player_id)
                
                if "team" in visible_info and visible_info["team"]:
                    # 知道阵营
                    if visible_info["team"] == "好人":
                        belief.team_probabilities[Team.GOOD] = 0.9
                        belief.team_probabilities[Team.EVIL] = 0.1
                        belief.trust_score = 0.8
                    else:
                        belief.team_probabilities[Team.GOOD] = 0.1
                        belief.team_probabilities[Team.EVIL] = 0.9
                        belief.trust_score = 0.2
                elif "possible_merlin" in visible_info and visible_info["possible_merlin"]:
                    # 派西维尔视角：可能是梅林或莫甘娜
                    belief.team_probabilities[Team.GOOD] = 0.6  # 可能是梅林（好人）
                    belief.team_probabilities[Team.EVIL] = 0.4  # 可能是莫甘娜（坏人）
                    belief.trust_score = 0.5
            else:
                # 看不到：使用先验概率
                # 根据游戏配置，坏人数量通常是固定的
                total_players = len(self.all_players)
                evil_count = 2 if total_players <= 6 else (3 if total_players <= 8 else 4)
                good_count = total_players - evil_count - 1  # 减去自己
                
                # 先验：坏人概率 = 坏人数量 / 未知玩家数量
                unknown_count = total_players - len(visible_player_ids)
                if unknown_count > 0:
                    evil_prob = evil_count / unknown_count
                    evil_prob = min(0.8, max(0.2, evil_prob))  # 限制在合理范围
                    belief.team_probabilities[Team.EVIL] = evil_prob
                    belief.team_probabilities[Team.GOOD] = 1.0 - evil_prob
                else:
                    belief.team_probabilities[Team.GOOD] = 0.5
                    belief.team_probabilities[Team.EVIL] = 0.5
                
                belief.trust_score = 0.5
            
            self.beliefs[player_id] = belief
    
    def update_belief_from_vote(self, player_id: int, vote_decision: bool, 
                                team_proposal: List[int], vote_result: bool):
        """
        根据投票行为更新信念
        vote_decision: 该玩家的投票决定（True=同意，False=拒绝）
        team_proposal: 提议的队伍
        vote_result: 投票是否通过
        """
        if player_id not in self.beliefs:
            return
        
        belief = self.beliefs[player_id]
        
        # 分析投票行为
        if self.my_team == Team.GOOD:
            # 好人视角
            if player_id in team_proposal:
                # 在队伍中
                if vote_decision:
                    # 同意包含自己的队伍（可能是好人）
                    belief.team_probabilities[Team.GOOD] *= 1.1
                else:
                    # 拒绝包含自己的队伍（可疑）
                    belief.team_probabilities[Team.EVIL] *= 1.2
            else:
                # 不在队伍中
                if vote_decision:
                    # 同意这个队伍（可能是好人）
                    belief.team_probabilities[Team.GOOD] *= 1.05
                else:
                    # 拒绝这个队伍（可能是坏人，想阻止任务）
                    belief.team_probabilities[Team.EVIL] *= 1.1
        else:
            # 坏人视角
            if player_id in team_proposal:
                # 在队伍中
                if vote_decision:
                    # 同意包含自己的队伍（可能是坏人）
                    belief.team_probabilities[Team.EVIL] *= 1.1
                else:
                    # 拒绝包含自己的队伍（可能是好人）
                    belief.team_probabilities[Team.GOOD] *= 1.1
        
        # 归一化概率
        total = belief.team_probabilities[Team.GOOD] + belief.team_probabilities[Team.EVIL]
        belief.team_probabilities[Team.GOOD] /= total
        belief.team_probabilities[Team.EVIL] /= total
        
        # 更新信任度
        if belief.team_probabilities[Team.GOOD] > 0.7:
            belief.trust_score = min(1.0, belief.trust_score + 0.1)
        elif belief.team_probabilities[Team.EVIL] > 0.7:
            belief.trust_score = max(0.0, belief.trust_score - 0.1)
    
    def update_belief_from_mission(self, player_id: int, mission_success: bool,
                                   mission_team: List[int], mission_result: bool):
        """
        根据任务结果更新信念
        mission_success: 该玩家在任务中投的是成功还是失败
        mission_team: 任务队伍
        mission_result: 任务最终结果（成功/失败）
        """
        if player_id not in self.beliefs:
            return
        
        belief = self.beliefs[player_id]
        
        if player_id not in mission_team:
            return  # 不在任务中，无法推断
        
        if self.my_team == Team.GOOD:
            # 好人视角
            if not mission_result:
                # 任务失败，说明队伍中有坏人
                if mission_success:
                    # 投了成功，可能是好人
                    belief.team_probabilities[Team.GOOD] *= 1.2
                else:
                    # 投了失败，很可能是坏人
                    belief.team_probabilities[Team.EVIL] *= 1.5
            else:
                # 任务成功，队伍中可能都是好人
                if mission_success:
                    belief.team_probabilities[Team.GOOD] *= 1.1
        else:
            # 坏人视角（知道队友身份）
            if not mission_result:
                # 任务失败
                if not mission_success:
                    # 投了失败，可能是坏人队友
                    belief.team_probabilities[Team.EVIL] *= 1.2
        
        # 归一化概率
        total = belief.team_probabilities[Team.GOOD] + belief.team_probabilities[Team.EVIL]
        belief.team_probabilities[Team.GOOD] /= total
        belief.team_probabilities[Team.EVIL] /= total
    
    def update_belief_from_speech(self, player_id: int, speech_content: str, 
                                  speech_analysis: Dict):
        """
        根据发言内容更新信念
        speech_analysis: 对发言的分析结果（包含情感、立场等）
        """
        if player_id not in self.beliefs:
            return
        
        belief = self.beliefs[player_id]
        
        # 记录行为历史
        belief.behavior_history.append(speech_content[:50])  # 只保留前50个字符
        
        # 简单的发言分析（可以后续扩展）
        # 如果发言支持好人立场，增加好人概率
        # 如果发言可疑，增加坏人概率
        
        # 这里可以添加更复杂的NLP分析
        pass
    
    def get_most_trusted_players(self, count: int = 3) -> List[int]:
        """获取最信任的玩家"""
        sorted_players = sorted(
            self.beliefs.items(),
            key=lambda x: x[1].trust_score,
            reverse=True
        )
        return [pid for pid, _ in sorted_players[:count] if pid != self.my_player_id]
    
    def get_most_suspicious_players(self, count: int = 3) -> List[int]:
        """获取最可疑的玩家"""
        sorted_players = sorted(
            self.beliefs.items(),
            key=lambda x: x[1].team_probabilities[Team.EVIL],
            reverse=True
        )
        return [pid for pid, _ in sorted_players[:count] if pid != self.my_player_id]
    
    def get_belief_summary(self) -> Dict:
        """获取信念摘要"""
        return {
            player_id: {
                "name": belief.name,
                "good_prob": belief.team_probabilities[Team.GOOD],
                "evil_prob": belief.team_probabilities[Team.EVIL],
                "trust_score": belief.trust_score
            }
            for player_id, belief in self.beliefs.items()
        }

