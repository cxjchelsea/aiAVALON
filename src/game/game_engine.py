"""
中央游戏引擎
负责游戏状态管理、角色分发、信息过滤和胜负判定
"""
import random
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .rules import Team, GamePhase, MissionConfig, get_mission_configs, get_evil_count
from .roles import RoleType, Role, get_role, get_standard_roles


@dataclass
class Player:
    """玩家信息"""
    player_id: int
    name: str
    role_type: RoleType
    role: Role
    
    def __post_init__(self):
        if self.role is None:
            self.role = get_role(self.role_type)


@dataclass
class MissionResult:
    """任务结果"""
    round_number: int
    team_members: List[int]  # 参与任务的玩家ID
    votes: Dict[int, bool]  # 玩家ID -> 是否成功
    success: bool  # 任务是否成功
    fail_count: int  # 失败票数


@dataclass
class GameState:
    """游戏状态"""
    players: List[Player] = field(default_factory=list)
    current_phase: GamePhase = GamePhase.INITIALIZATION
    current_round: int = 0
    current_leader: int = 0  # 当前队长（玩家ID）
    mission_configs: List[MissionConfig] = field(default_factory=list)
    mission_results: List[MissionResult] = field(default_factory=list)
    proposed_team: List[int] = field(default_factory=list)  # 当前提议的队伍
    votes: Dict[int, bool] = field(default_factory=dict)  # 玩家ID -> 是否同意
    vote_round: int = 0  # 投票轮次（同一任务最多5次投票）
    successful_missions: int = 0  # 成功任务数
    failed_missions: int = 0  # 失败任务数
    game_over: bool = False
    winner: Optional[Team] = None
    assassination_target: Optional[int] = None  # 刺杀目标（玩家ID）


class RoleDistributor:
    """角色分发器"""
    
    @staticmethod
    def distribute_roles(player_count: int, player_names: List[str]) -> List[Tuple[int, str, RoleType]]:
        """
        随机分配角色
        返回: [(player_id, name, role_type), ...]
        """
        roles = get_standard_roles(player_count)
        random.shuffle(roles)
        
        assignments = []
        for i, (name, role_type) in enumerate(zip(player_names, roles)):
            assignments.append((i, name, role_type))
        
        return assignments


class InformationFilter:
    """信息过滤器 - 控制每个智能体接收到的信息范围"""
    
    @staticmethod
    def get_visible_players(player: Player, all_players: List[Player]) -> List[Player]:
        """
        获取玩家能看到的其他玩家
        返回玩家能看到的玩家列表（包括自己）
        """
        visible = [player]  # 自己总是可见
        
        for other_player in all_players:
            if other_player.player_id == player.player_id:
                continue
            
            if player.role.can_see(other_player.role):
                visible.append(other_player)
        
        return visible
    
    @staticmethod
    def get_private_info(player: Player, all_players: List[Player]) -> Dict:
        """
        获取玩家的私有信息
        返回一个字典，包含玩家能看到的信息
        """
        visible_players = InformationFilter.get_visible_players(player, all_players)
        
        # 构建可见玩家信息（不暴露具体角色，只暴露阵营或特殊信息）
        visible_info = []
        for visible_player in visible_players:
            if visible_player.player_id == player.player_id:
                visible_info.append({
                    "player_id": visible_player.player_id,
                    "name": visible_player.name,
                    "is_self": True,
                    "role": visible_player.role_type.value,
                    "team": visible_player.role.team.value
                })
            else:
                # 根据角色能力决定能看到什么信息
                if player.role_type == RoleType.PERCIVAL:
                    # 派西维尔看到的是"可能是梅林"的标记
                    visible_info.append({
                        "player_id": visible_player.player_id,
                        "name": visible_player.name,
                        "is_self": False,
                        "possible_merlin": True,  # 可能是梅林或莫甘娜
                        "team": None  # 不确定阵营
                    })
                else:
                    # 其他角色看到的是阵营信息
                    visible_info.append({
                        "player_id": visible_player.player_id,
                        "name": visible_player.name,
                        "is_self": False,
                        "team": visible_player.role.team.value,
                        "role": None  # 不暴露具体角色
                    })
        
        # 构建所有玩家列表（包括不可见的）
        all_players_info = []
        visible_ids = {p.player_id for p in visible_players}
        
        for p in all_players:
            if p.player_id in visible_ids:
                # 可见玩家，使用详细信息
                visible_player = next(vp for vp in visible_players if vp.player_id == p.player_id)
                if visible_player.player_id == player.player_id:
                    all_players_info.append({
                        "player_id": visible_player.player_id,
                        "name": visible_player.name,
                        "is_self": True,
                        "role": visible_player.role_type.value,
                        "team": visible_player.role.team.value
                    })
                else:
                    if player.role_type == RoleType.PERCIVAL:
                        all_players_info.append({
                            "player_id": visible_player.player_id,
                            "name": visible_player.name,
                            "is_self": False,
                            "possible_merlin": True,
                            "team": None
                        })
                    else:
                        all_players_info.append({
                            "player_id": visible_player.player_id,
                            "name": visible_player.name,
                            "is_self": False,
                            "team": visible_player.role.team.value,
                            "role": None
                        })
            else:
                # 不可见玩家，只提供基本信息
                all_players_info.append({
                    "player_id": p.player_id,
                    "name": p.name,
                    "is_self": False,
                    "team": None,
                    "role": None
                })
        
        return {
            "my_role": player.role_type.value,
            "my_team": player.role.team.value,
            "visible_players": visible_info,
            "all_players": all_players_info,
            "total_players": len(all_players)
        }


class WinConditionChecker:
    """胜负判定器"""
    
    @staticmethod
    def check_game_over(state: GameState) -> Tuple[bool, Optional[Team]]:
        """
        检查游戏是否结束
        返回: (是否结束, 获胜方)
        """
        # 检查任务结果
        if state.successful_missions >= 3:
            # 好人完成3个任务，进入刺杀阶段
            if state.current_phase == GamePhase.ASSASSINATION:
                # 刺杀阶段已结束，检查刺杀结果
                if state.assassination_target is not None:
                    target_player = next(p for p in state.players if p.player_id == state.assassination_target)
                    if target_player.role_type == RoleType.MERLIN:
                        # 成功刺杀梅林，坏人获胜
                        return True, Team.EVIL
                    else:
                        # 刺杀失败，好人获胜
                        return True, Team.GOOD
                else:
                    # 刺杀阶段还未执行
                    return False, None
            else:
                # 还未进入刺杀阶段
                return False, None
        
        if state.failed_missions >= 3:
            # 坏人破坏3个任务，坏人获胜
            return True, Team.EVIL
        
        # 检查是否达到最大投票轮次（5次投票都失败，坏人获胜）
        if state.vote_round >= 5 and state.current_round < len(state.mission_configs):
            return True, Team.EVIL
        
        return False, None


class GameEngine:
    """中央游戏引擎"""
    
    def __init__(self, player_count: int, player_names: List[str]):
        self.player_count = player_count
        self.state = GameState()
        self.role_distributor = RoleDistributor()
        self.info_filter = InformationFilter()
        self.win_checker = WinConditionChecker()
        
        # 初始化玩家和角色
        self._initialize_players(player_names)
        self._initialize_game()
    
    def _initialize_players(self, player_names: List[str]):
        """初始化玩家"""
        assignments = self.role_distributor.distribute_roles(self.player_count, player_names)
        
        for player_id, name, role_type in assignments:
            role = get_role(role_type)
            player = Player(player_id=player_id, name=name, role_type=role_type, role=role)
            self.state.players.append(player)
    
    def _initialize_game(self):
        """初始化游戏状态"""
        self.state.mission_configs = get_mission_configs(self.player_count)
        self.state.current_phase = GamePhase.DISCUSSION
        self.state.current_round = 1
        self.state.current_leader = 0  # 第一个玩家是队长
        self.state.vote_round = 0
    
    def get_player_info(self, player_id: int) -> Dict:
        """获取玩家的私有信息"""
        player = next(p for p in self.state.players if p.player_id == player_id)
        return self.info_filter.get_private_info(player, self.state.players)
    
    def propose_team(self, leader_id: int, team_members: List[int]) -> bool:
        """
        提议队伍
        返回是否成功
        """
        if leader_id != self.state.current_leader:
            return False
        
        current_config = self.state.mission_configs[self.state.current_round - 1]
        if len(team_members) != current_config.team_size:
            return False
        
        # 检查所有成员是否有效
        valid_player_ids = {p.player_id for p in self.state.players}
        if not all(mid in valid_player_ids for mid in team_members):
            return False
        
        self.state.proposed_team = team_members
        self.state.current_phase = GamePhase.VOTING
        return True
    
    def vote_on_team(self, player_id: int, approve: bool) -> bool:
        """
        对提议的队伍投票
        返回是否成功
        """
        if self.state.current_phase != GamePhase.VOTING:
            return False
        
        if player_id in self.state.votes:
            return False  # 已经投过票了
        
        self.state.votes[player_id] = approve
        return True
    
    def process_voting_result(self) -> Tuple[bool, bool]:
        """
        处理投票结果
        返回: (投票是否完成, 是否通过)
        """
        if len(self.state.votes) < len(self.state.players):
            return False, False  # 投票未完成
        
        approve_count = sum(1 for v in self.state.votes.values() if v)
        reject_count = len(self.state.votes) - approve_count
        
        passed = approve_count > reject_count
        
        if passed:
            # 投票通过，进入任务执行阶段
            self.state.current_phase = GamePhase.MISSION
            self.state.vote_round = 0
        else:
            # 投票未通过
            self.state.vote_round += 1
            if self.state.vote_round >= 5:
                # 5次投票都失败，游戏结束
                self.state.game_over = True
                self.state.winner = Team.EVIL
                self.state.current_phase = GamePhase.FINISHED
            else:
                # 下一轮投票，更换队长
                self.state.current_leader = (self.state.current_leader + 1) % len(self.state.players)
                self.state.current_phase = GamePhase.DISCUSSION
                self.state.votes = {}
        
        return True, passed
    
    def execute_mission(self, player_id: int, success: bool) -> bool:
        """
        执行任务投票
        返回是否成功
        """
        if self.state.current_phase != GamePhase.MISSION:
            return False
        
        if player_id not in self.state.proposed_team:
            return False  # 不在任务队伍中
        
        # 这里应该由任务结果来收集，暂时简化处理
        return True
    
    def submit_mission_result(self, mission_votes: Dict[int, bool]) -> bool:
        """
        提交任务结果
        mission_votes: 玩家ID -> 是否成功
        """
        if self.state.current_phase != GamePhase.MISSION:
            return False
        
        # 验证所有任务成员都投票了
        if set(mission_votes.keys()) != set(self.state.proposed_team):
            return False
        
        current_config = self.state.mission_configs[self.state.current_round - 1]
        fail_count = sum(1 for v in mission_votes.values() if not v)
        success = fail_count < current_config.fails_needed
        
        # 记录任务结果
        mission_result = MissionResult(
            round_number=self.state.current_round,
            team_members=self.state.proposed_team.copy(),
            votes=mission_votes.copy(),
            success=success,
            fail_count=fail_count
        )
        self.state.mission_results.append(mission_result)
        
        if success:
            self.state.successful_missions += 1
        else:
            self.state.failed_missions += 1
        
        # 检查好人是否完成3个任务（需要进入刺杀阶段）
        if self.state.successful_missions >= 3:
            # 好人完成3个任务，进入刺杀阶段
            self.state.current_phase = GamePhase.ASSASSINATION
            # 不增加轮次，等待刺杀结果
            # 重置投票轮次，为下一轮做准备（如果刺杀失败）
            self.state.vote_round = 0
        elif self.state.failed_missions >= 3:
            # 坏人破坏3个任务，坏人获胜
            self.state.game_over = True
            self.state.winner = Team.EVIL
            self.state.current_phase = GamePhase.FINISHED
        else:
            # 进入下一轮
            self.state.current_round += 1
            # 检查是否还有下一轮任务
            if self.state.current_round > len(self.state.mission_configs):
                # 没有更多任务了，检查胜负
                if self.state.successful_missions > self.state.failed_missions:
                    self.state.game_over = True
                    self.state.winner = Team.GOOD
                    self.state.current_phase = GamePhase.FINISHED
                else:
                    self.state.game_over = True
                    self.state.winner = Team.EVIL
                    self.state.current_phase = GamePhase.FINISHED
            else:
                # 进入下一轮讨论
                self.state.current_leader = (self.state.current_leader + 1) % len(self.state.players)
                self.state.current_phase = GamePhase.DISCUSSION
                self.state.proposed_team = []
                self.state.votes = {}
                self.state.vote_round = 0
        
        return True
    
    def assassinate(self, target_player_id: int) -> bool:
        """
        刺杀目标
        """
        if self.state.current_phase != GamePhase.ASSASSINATION:
            return False
        
        self.state.assassination_target = target_player_id
        
        # 检查游戏结果
        game_over, winner = self.win_checker.check_game_over(self.state)
        if game_over:
            self.state.game_over = True
            self.state.winner = winner
            self.state.current_phase = GamePhase.FINISHED
        
        return True
    
    def get_game_state_summary(self, player_id: Optional[int] = None) -> Dict:
        """
        获取游戏状态摘要
        如果提供player_id，则返回该玩家视角的信息
        """
        # 构建任务历史信息
        mission_history = []
        for result in self.state.mission_results:
            team_names = [self.state.players[pid].name for pid in result.team_members]
            mission_history.append({
                "round": result.round_number,
                "team": team_names,
                "team_ids": result.team_members,
                "success": result.success,
                "fail_count": result.fail_count,
                "team_size": len(result.team_members)
            })
        
        summary = {
            "current_phase": self.state.current_phase.name,  # 使用枚举名称而不是值
            "current_round": self.state.current_round,
            "current_leader": self.state.current_leader,
            "successful_missions": self.state.successful_missions,
            "failed_missions": self.state.failed_missions,
            "vote_round": self.state.vote_round,
            "game_over": self.state.game_over,
            "winner": self.state.winner.name if self.state.winner else None,  # 使用枚举名称
            "mission_history": mission_history  # 添加任务历史
        }
        
        if player_id is not None:
            # 添加玩家私有信息
            summary["private_info"] = self.get_player_info(player_id)
        
        return summary

