"""
使用 LangGraph 优化的游戏引擎
提供更清晰的状态管理和更好的可扩展性
"""
from typing import Dict, List, Optional, TypedDict, Annotated, Literal
from dataclasses import dataclass

from game.rules import GamePhase, Team
from game.game_engine import GameEngine
from agent.base_agent import BaseAgent


class GameStateGraph(TypedDict):
    """LangGraph 游戏状态"""
    engine: GameEngine
    agents: List[BaseAgent]
    verbose: bool
    round_count: int
    max_rounds: int


class LangGraphGameEngine:
    """使用 LangGraph 的游戏引擎"""
    
    def __init__(self, game_engine: GameEngine, agents: List[BaseAgent], verbose: bool = True):
        self.engine = game_engine
        self.agents = agents
        self.verbose = verbose
        
        # 构建状态图
        try:
            from langgraph.graph import StateGraph, END
            self.graph = self._build_graph()
            self.use_langgraph = True
        except ImportError:
            print("警告: LangGraph未安装，将使用传统游戏循环")
            self.use_langgraph = False
            self.graph = None
    
    def _build_graph(self):
        """构建游戏状态图"""
        from langgraph.graph import StateGraph, END
        
        workflow = StateGraph(GameStateGraph)
        
        # 添加节点
        workflow.add_node("check_win", self._check_win_node)
        workflow.add_node("discussion", self._discussion_node)
        workflow.add_node("voting", self._voting_node)
        workflow.add_node("mission", self._mission_node)
        workflow.add_node("assassination", self._assassination_node)
        workflow.add_node("finished", self._finished_node)
        
        # 设置入口点
        workflow.set_entry_point("check_win")
        
        # 添加条件边
        workflow.add_conditional_edges(
            "check_win",
            self._route_after_check,
            {
                "discussion": "discussion",
                "voting": "voting",
                "mission": "mission",
                "assassination": "assassination",
                "finished": "finished",
                "end": END
            }
        )
        
        # 添加边
        workflow.add_edge("discussion", "voting")
        workflow.add_edge("voting", "check_win")
        workflow.add_edge("mission", "check_win")
        workflow.add_edge("assassination", "finished")
        workflow.add_edge("finished", END)
        
        return workflow.compile()
    
    def _check_win_node(self, state: GameStateGraph) -> GameStateGraph:
        """检查游戏是否结束"""
        # 检查是否超过最大轮次
        if state["round_count"] >= state["max_rounds"]:
            state["engine"].state.game_over = True
            state["engine"].state.current_phase = GamePhase.FINISHED
            return state
        
        # 检查游戏结束条件
        game_over, winner = state["engine"].win_checker.check_game_over(state["engine"].state)
        if game_over:
            state["engine"].state.game_over = True
            state["engine"].state.winner = winner
            state["engine"].state.current_phase = GamePhase.FINISHED
        
        return state
    
    def _route_after_check(self, state: GameStateGraph) -> Literal["discussion", "voting", "mission", "assassination", "finished", "end"]:
        """根据游戏状态路由到下一个节点"""
        if state["engine"].state.game_over:
            return "finished"
        
        phase = state["engine"].state.current_phase
        if phase == GamePhase.DISCUSSION:
            return "discussion"
        elif phase == GamePhase.VOTING:
            return "voting"
        elif phase == GamePhase.MISSION:
            return "mission"
        elif phase == GamePhase.ASSASSINATION:
            return "assassination"
        elif phase == GamePhase.FINISHED:
            return "finished"
        else:
            return "end"
    
    def _discussion_node(self, state: GameStateGraph) -> GameStateGraph:
        """讨论阶段节点"""
        if state["verbose"]:
            print("\n[讨论阶段]")
        
        leader_id = state["engine"].state.current_leader
        recent_speeches = []
        
        # 准备游戏状态
        base_game_state = state["engine"].get_game_state_summary(leader_id)
        if state["engine"].state.current_round <= len(state["engine"].state.mission_configs):
            current_config = state["engine"].state.mission_configs[state["engine"].state.current_round - 1]
            base_game_state["mission_config"] = {
                "team_size": current_config.team_size,
                "fails_needed": current_config.fails_needed
            }
        else:
            base_game_state["mission_config"] = {
                "team_size": 2,
                "fails_needed": 1
            }
        
        # 找到队长位置
        leader_index = next(i for i, agent in enumerate(state["agents"]) if agent.player_id == leader_id)
        leader_agent = state["agents"][leader_index]
        
        # 从队长的下一位开始发言
        for i in range(len(state["agents"])):
            agent_index = (leader_index + 1 + i) % len(state["agents"])
            agent = state["agents"][agent_index]
            
            game_state = state["engine"].get_game_state_summary(agent.player_id)
            game_state["mission_config"] = base_game_state["mission_config"]
            game_state["mission_history"] = base_game_state.get("mission_history", [])
            
            if state["verbose"]:
                speech = agent.generate_speech(game_state, recent_speeches)
                print(f"{agent.name}: {speech}")
                recent_speeches.append({"player_id": agent.player_id, "name": agent.name, "speech": speech})
        
        # 队长决定队伍
        leader_game_state = state["engine"].get_game_state_summary(leader_id)
        leader_game_state["mission_config"] = base_game_state["mission_config"]
        leader_game_state["mission_history"] = base_game_state.get("mission_history", [])
        
        proposed_team = leader_agent.propose_team(leader_game_state)
        
        if state["verbose"]:
            leader_name = state["engine"].state.players[leader_id].name
            team_names = [state["engine"].state.players[pid].name for pid in proposed_team]
            print(f"\n{leader_name} 根据讨论决定队伍: {', '.join(team_names)}")
        
        state["engine"].propose_team(leader_id, proposed_team)
        return state
    
    def _voting_node(self, state: GameStateGraph) -> GameStateGraph:
        """投票阶段节点"""
        if state["verbose"]:
            print("\n[投票阶段]")
        
        proposed_team = state["engine"].state.proposed_team
        team_names = [state["engine"].state.players[pid].name for pid in proposed_team]
        leader_id = state["engine"].state.current_leader
        
        if state["verbose"]:
            print(f"对队伍进行投票: {', '.join(team_names)}")
        
        votes = {}
        for agent in state["agents"]:
            game_state = state["engine"].get_game_state_summary(agent.player_id)
            if state["engine"].state.current_round <= len(state["engine"].state.mission_configs):
                current_config = state["engine"].state.mission_configs[state["engine"].state.current_round - 1]
                game_state["mission_config"] = {
                    "team_size": current_config.team_size,
                    "fails_needed": current_config.fails_needed
                }
            else:
                game_state["mission_config"] = {"team_size": 2, "fails_needed": 1}
            
            # 队长必须同意自己提议的队伍
            if agent.player_id == leader_id:
                vote = True
            else:
                vote = agent.vote_on_team(game_state, proposed_team)
            
            votes[agent.player_id] = vote
            state["engine"].vote_on_team(agent.player_id, vote)
            
            if state["verbose"]:
                vote_text = "同意" if vote else "拒绝"
                print(f"{agent.name}: {vote_text}")
        
        # 处理投票结果
        voting_complete, passed = state["engine"].process_voting_result()
        
        if state["verbose"]:
            approve_count = sum(1 for v in votes.values() if v)
            reject_count = len(votes) - approve_count
            print(f"\n投票结果: {approve_count} 同意, {reject_count} 拒绝")
            print(f"结果: {'通过' if passed else '未通过'}")
        
        return state
    
    def _mission_node(self, state: GameStateGraph) -> GameStateGraph:
        """任务执行节点"""
        if state["verbose"]:
            print("\n[任务执行阶段]")
        
        mission_team = state["engine"].state.proposed_team
        team_names = [state["engine"].state.players[pid].name for pid in mission_team]
        
        if state["verbose"]:
            print(f"执行任务的队伍: {', '.join(team_names)}")
        
        mission_votes = {}
        for agent in state["agents"]:
            if agent.player_id in mission_team:
                game_state = state["engine"].get_game_state_summary(agent.player_id)
                if state["engine"].state.current_round <= len(state["engine"].state.mission_configs):
                    current_config = state["engine"].state.mission_configs[state["engine"].state.current_round - 1]
                    game_state["mission_config"] = {
                        "team_size": current_config.team_size,
                        "fails_needed": current_config.fails_needed
                    }
                else:
                    game_state["mission_config"] = {"team_size": 2, "fails_needed": 1}
                
                success = agent.vote_on_mission(game_state, mission_team)
                mission_votes[agent.player_id] = success
                
                if state["verbose"]:
                    result_text = "成功" if success else "失败"
                    print(f"{agent.name}: {result_text}")
        
        # 提交任务结果
        state["engine"].submit_mission_result(mission_votes)
        
        # 更新所有智能体的信念系统
        if state["engine"].state.mission_results:
            last_result = state["engine"].state.mission_results[-1]
            for agent in state["agents"]:
                for player_id, vote_success in mission_votes.items():
                    agent.belief_system.update_belief_from_mission(
                        player_id=player_id,
                        mission_success=vote_success,
                        mission_team=mission_team,
                        mission_result=last_result.success
                    )
            
            if state["verbose"]:
                result_text = "成功" if last_result.success else "失败"
                print(f"\n任务结果: {result_text} (失败票数: {last_result.fail_count})")
        
        # 增加轮次计数
        state["round_count"] += 1
        
        return state
    
    def _assassination_node(self, state: GameStateGraph) -> GameStateGraph:
        """刺杀阶段节点"""
        if state["verbose"]:
            print("\n[刺杀阶段]")
            print("坏人阵营可以刺杀梅林...")
        
        # 找到刺客
        assassin_agent = None
        for agent in state["agents"]:
            from game.roles import RoleType
            if agent.role_type == RoleType.ASSASSIN:
                assassin_agent = agent
                break
        
        if assassin_agent:
            game_state = state["engine"].get_game_state_summary(assassin_agent.player_id)
            target_id = assassin_agent.assassinate(game_state)
            
            if target_id is not None:
                target_name = state["engine"].state.players[target_id].name
                if state["verbose"]:
                    print(f"{assassin_agent.name} 选择刺杀: {target_name}")
                
                state["engine"].assassinate(target_id)
            else:
                if state["verbose"]:
                    print(f"{assassin_agent.name} 无法决定刺杀目标")
        
        return state
    
    def _finished_node(self, state: GameStateGraph) -> GameStateGraph:
        """游戏结束节点"""
        if state["verbose"]:
            self._print_game_result(state)
        
        return state
    
    def _print_game_result(self, state: GameStateGraph):
        """打印游戏结果"""
        print("\n" + "=" * 60)
        print("游戏结束！")
        print("=" * 60)
        
        if state["engine"].state.winner:
            winner_text = "好人阵营" if state["engine"].state.winner == Team.GOOD else "坏人阵营"
            print(f"获胜方: {winner_text}")
        else:
            print("游戏未正常结束")
        
        print(f"\n最终统计:")
        print(f"  成功任务: {state['engine'].state.successful_missions}")
        print(f"  失败任务: {state['engine'].state.failed_missions}")
        print(f"  总轮次: {len(state['engine'].state.mission_results)}")
        
        print("\n任务历史:")
        for i, result in enumerate(state["engine"].state.mission_results, 1):
            result_text = "成功" if result.success else "失败"
            team_names = [state["engine"].state.players[pid].name for pid in result.team_members]
            print(f"  第{i}轮: {result_text} - 队伍: {', '.join(team_names)}")
    
    def run(self):
        """运行游戏"""
        if not self.use_langgraph:
            # 回退到传统游戏循环
            from src.main import AvalonGame
            game = AvalonGame(
                player_count=len(self.engine.state.players),
                player_names=[p.name for p in self.engine.state.players],
                use_llm=True
            )
            game.engine = self.engine
            game.agents = self.agents
            game.run_game(verbose=self.verbose)
            return
        
        if self.verbose:
            print("=" * 60)
            print("游戏开始！")
            print("=" * 60)
            self._print_role_assignment()
            print()
        
        initial_state: GameStateGraph = {
            "engine": self.engine,
            "agents": self.agents,
            "verbose": self.verbose,
            "round_count": 0,
            "max_rounds": 20
        }
        
        # 运行状态图
        try:
            final_state = self.graph.invoke(initial_state)
            return final_state
        except Exception as e:
            print(f"LangGraph执行错误: {e}")
            raise
    
    def _print_role_assignment(self):
        """打印角色分配（仅用于调试）"""
        print("\n角色分配（调试信息）:")
        for player in self.engine.state.players:
            print(f"  {player.name}: {player.role_type.value} ({player.role.team.value})")
