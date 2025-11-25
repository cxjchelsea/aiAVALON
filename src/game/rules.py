"""
阿瓦隆游戏规则定义
"""
from enum import Enum
from typing import List, Dict, Tuple
from dataclasses import dataclass


class Team(Enum):
    """阵营"""
    GOOD = "好人"
    EVIL = "坏人"


class GamePhase(Enum):
    """游戏阶段"""
    INITIALIZATION = "初始化"
    DISCUSSION = "讨论阶段"
    VOTING = "投票阶段"
    MISSION = "任务执行阶段"
    ASSASSINATION = "刺杀阶段"
    FINISHED = "游戏结束"


@dataclass
class MissionConfig:
    """任务配置"""
    round_number: int  # 第几轮
    team_size: int  # 队伍人数
    fails_needed: int  # 需要几个失败才能破坏任务


# 5人局配置
MISSION_CONFIGS_5 = [
    MissionConfig(round_number=1, team_size=2, fails_needed=1),
    MissionConfig(round_number=2, team_size=3, fails_needed=1),
    MissionConfig(round_number=3, team_size=2, fails_needed=1),
    MissionConfig(round_number=4, team_size=3, fails_needed=1),
    MissionConfig(round_number=5, team_size=3, fails_needed=1),
]

# 6人局配置
MISSION_CONFIGS_6 = [
    MissionConfig(round_number=1, team_size=2, fails_needed=1),
    MissionConfig(round_number=2, team_size=3, fails_needed=1),
    MissionConfig(round_number=3, team_size=4, fails_needed=1),
    MissionConfig(round_number=4, team_size=3, fails_needed=1),
    MissionConfig(round_number=5, team_size=4, fails_needed=1),
]

# 7人局配置
MISSION_CONFIGS_7 = [
    MissionConfig(round_number=1, team_size=2, fails_needed=1),
    MissionConfig(round_number=2, team_size=3, fails_needed=1),
    MissionConfig(round_number=3, team_size=3, fails_needed=1),
    MissionConfig(round_number=4, team_size=4, fails_needed=2),
    MissionConfig(round_number=5, team_size=4, fails_needed=1),
]

# 8人局配置
MISSION_CONFIGS_8 = [
    MissionConfig(round_number=1, team_size=3, fails_needed=1),
    MissionConfig(round_number=2, team_size=4, fails_needed=1),
    MissionConfig(round_number=3, team_size=4, fails_needed=1),
    MissionConfig(round_number=4, team_size=5, fails_needed=2),
    MissionConfig(round_number=5, team_size=5, fails_needed=1),
]

# 9人局配置
MISSION_CONFIGS_9 = [
    MissionConfig(round_number=1, team_size=3, fails_needed=1),
    MissionConfig(round_number=2, team_size=4, fails_needed=1),
    MissionConfig(round_number=3, team_size=4, fails_needed=1),
    MissionConfig(round_number=4, team_size=5, fails_needed=2),
    MissionConfig(round_number=5, team_size=5, fails_needed=1),
]

# 10人局配置
MISSION_CONFIGS_10 = [
    MissionConfig(round_number=1, team_size=3, fails_needed=1),
    MissionConfig(round_number=2, team_size=4, fails_needed=1),
    MissionConfig(round_number=3, team_size=4, fails_needed=1),
    MissionConfig(round_number=4, team_size=5, fails_needed=2),
    MissionConfig(round_number=5, team_size=5, fails_needed=1),
]


def get_mission_configs(player_count: int) -> List[MissionConfig]:
    """根据玩家数量获取任务配置"""
    configs = {
        5: MISSION_CONFIGS_5,
        6: MISSION_CONFIGS_6,
        7: MISSION_CONFIGS_7,
        8: MISSION_CONFIGS_8,
        9: MISSION_CONFIGS_9,
        10: MISSION_CONFIGS_10,
    }
    return configs.get(player_count, MISSION_CONFIGS_5)


def get_evil_count(player_count: int) -> int:
    """根据玩家数量获取坏人数量"""
    evil_counts = {
        5: 2,
        6: 2,
        7: 3,
        8: 3,
        9: 3,
        10: 4,
    }
    return evil_counts.get(player_count, 2)

