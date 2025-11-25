"""
阿瓦隆角色定义
"""
from enum import Enum
from typing import List, Set, Optional
from dataclasses import dataclass
from .rules import Team


class RoleType(Enum):
    """角色类型"""
    MERLIN = "梅林"  # 能看到所有坏人（除莫甘娜）
    PERCIVAL = "派西维尔"  # 能看到梅林和莫甘娜
    SERVANT = "忠臣"  # 普通好人
    ASSASSIN = "刺客"  # 坏人，最后可以刺杀梅林
    MORGANA = "莫甘娜"  # 坏人，在派西维尔眼中显示为梅林
    MORDRED = "莫德雷德"  # 坏人，梅林看不到
    OBERON = "奥伯伦"  # 坏人，看不到其他坏人，其他坏人也看不到他
    MINION = "爪牙"  # 普通坏人


@dataclass
class Role:
    """角色信息"""
    role_type: RoleType
    team: Team
    description: str
    special_abilities: List[str]
    win_condition: str
    
    def can_see(self, other_role: 'Role') -> bool:
        """判断是否能看到另一个角色"""
        if self.role_type == RoleType.MERLIN:
            # 梅林能看到所有坏人，除了莫德雷德和莫甘娜
            return (other_role.team == Team.EVIL and 
                   other_role.role_type not in [RoleType.MORDRED, RoleType.MORGANA])
        
        elif self.role_type == RoleType.PERCIVAL:
            # 派西维尔能看到梅林和莫甘娜（但分不清哪个是哪个）
            return other_role.role_type in [RoleType.MERLIN, RoleType.MORGANA]
        
        elif self.role_type == RoleType.ASSASSIN:
            # 刺客能看到其他坏人（除了奥伯伦）
            return (other_role.team == Team.EVIL and 
                   other_role.role_type != RoleType.OBERON)
        
        elif self.role_type == RoleType.MORGANA:
            # 莫甘娜能看到其他坏人（除了奥伯伦）
            return (other_role.team == Team.EVIL and 
                   other_role.role_type != RoleType.OBERON)
        
        elif self.role_type == RoleType.MINION:
            # 爪牙能看到其他坏人（除了奥伯伦）
            return (other_role.team == Team.EVIL and 
                   other_role.role_type != RoleType.OBERON)
        
        elif self.role_type == RoleType.MORDRED:
            # 莫德雷德能看到其他坏人（除了奥伯伦）
            return (other_role.team == Team.EVIL and 
                   other_role.role_type != RoleType.OBERON)
        
        elif self.role_type == RoleType.OBERON:
            # 奥伯伦看不到任何人
            return False
        
        else:  # SERVANT
            # 忠臣看不到任何人
            return False


# 角色定义
ROLE_DEFINITIONS = {
    RoleType.MERLIN: Role(
        role_type=RoleType.MERLIN,
        team=Team.GOOD,
        description="梅林是好人阵营的领袖，能看到所有坏人（除了莫德雷德和莫甘娜）",
        special_abilities=["能看到大部分坏人", "必须隐藏身份避免被刺杀"],
        win_condition="好人阵营完成3个任务且自己不被刺杀"
    ),
    RoleType.PERCIVAL: Role(
        role_type=RoleType.PERCIVAL,
        team=Team.GOOD,
        description="派西维尔能看到梅林和莫甘娜，但分不清哪个是哪个",
        special_abilities=["能看到梅林和莫甘娜（但分不清）", "需要保护真正的梅林"],
        win_condition="好人阵营完成3个任务"
    ),
    RoleType.SERVANT: Role(
        role_type=RoleType.SERVANT,
        team=Team.GOOD,
        description="忠臣是普通的好人，没有任何特殊能力",
        special_abilities=[],
        win_condition="好人阵营完成3个任务"
    ),
    RoleType.ASSASSIN: Role(
        role_type=RoleType.ASSASSIN,
        team=Team.EVIL,
        description="刺客是坏人阵营的领袖，最后可以刺杀梅林",
        special_abilities=["能看到其他坏人", "最后可以刺杀梅林"],
        win_condition="破坏3个任务，或成功刺杀梅林"
    ),
    RoleType.MORGANA: Role(
        role_type=RoleType.MORGANA,
        team=Team.EVIL,
        description="莫甘娜在派西维尔眼中显示为梅林，用来迷惑派西维尔",
        special_abilities=["能看到其他坏人", "在派西维尔眼中显示为梅林"],
        win_condition="破坏3个任务，或成功刺杀梅林"
    ),
    RoleType.MORDRED: Role(
        role_type=RoleType.MORDRED,
        team=Team.EVIL,
        description="莫德雷德是梅林看不到的坏人",
        special_abilities=["梅林看不到自己", "能看到其他坏人"],
        win_condition="破坏3个任务，或成功刺杀梅林"
    ),
    RoleType.OBERON: Role(
        role_type=RoleType.OBERON,
        team=Team.EVIL,
        description="奥伯伦看不到其他坏人，其他坏人也看不到他",
        special_abilities=["独立行动", "看不到其他坏人"],
        win_condition="破坏3个任务，或成功刺杀梅林"
    ),
    RoleType.MINION: Role(
        role_type=RoleType.MINION,
        team=Team.EVIL,
        description="爪牙是普通的坏人",
        special_abilities=["能看到其他坏人"],
        win_condition="破坏3个任务，或成功刺杀梅林"
    ),
}


def get_role(role_type: RoleType) -> Role:
    """获取角色定义"""
    return ROLE_DEFINITIONS[role_type]


def get_standard_roles(player_count: int) -> List[RoleType]:
    """根据玩家数量获取标准角色配置"""
    if player_count == 5:
        return [RoleType.MERLIN, RoleType.ASSASSIN, RoleType.SERVANT, RoleType.MORGANA, RoleType.PERCIVAL]
    elif player_count == 6:
        return [RoleType.MERLIN, RoleType.ASSASSIN, RoleType.PERCIVAL, RoleType.SERVANT, RoleType.SERVANT, RoleType.MORGANA]
    elif player_count == 7:
        return [RoleType.MERLIN, RoleType.ASSASSIN, RoleType.PERCIVAL, RoleType.MORGANA, RoleType.SERVANT, RoleType.SERVANT, RoleType.MINION]
    elif player_count == 8:
        return [RoleType.MERLIN, RoleType.ASSASSIN, RoleType.PERCIVAL, RoleType.MORGANA, RoleType.SERVANT, RoleType.SERVANT, RoleType.SERVANT, RoleType.MINION]
    elif player_count == 9:
        return [RoleType.MERLIN, RoleType.ASSASSIN, RoleType.PERCIVAL, RoleType.MORGANA, RoleType.MORDRED, RoleType.SERVANT, RoleType.SERVANT, RoleType.SERVANT, RoleType.MINION]
    elif player_count == 10:
        return [RoleType.MERLIN, RoleType.ASSASSIN, RoleType.PERCIVAL, RoleType.MORGANA, RoleType.MORDRED, RoleType.OBERON, RoleType.SERVANT, RoleType.SERVANT, RoleType.SERVANT, RoleType.MINION]
    else:
        # 默认5人配置
        return [RoleType.MERLIN, RoleType.ASSASSIN, RoleType.SERVANT, RoleType.MORGANA, RoleType.PERCIVAL]

