"""
AI阿瓦隆多智能体系统 - 主程序入口
"""
import random
import sys
import os
from typing import List, Dict, Optional

# 添加src目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from game.game_engine import GameEngine
from agent.base_agent import BaseAgent
from game.rules import GamePhase, Team


class AvalonGame:
    """阿瓦隆游戏主类"""
    
    def __init__(self, player_count: int = 5, player_names: List[str] = None,
                 use_llm: bool = False, llm_api_key: Optional[str] = None,
                 llm_model: str = "gpt-4o-mini", llm_api_provider: str = "openai"):
        if player_names is None:
            player_names = [f"玩家{i+1}" for i in range(player_count)]
        
        if len(player_names) != player_count:
            raise ValueError(f"玩家名称数量({len(player_names)})与玩家数量({player_count})不匹配")
        
        self.player_count = player_count
        self.player_names = player_names
        self.use_llm = use_llm
        self.llm_api_key = llm_api_key
        self.llm_model = llm_model
        self.llm_api_provider = llm_api_provider
        
        # 初始化游戏引擎
        self.engine = GameEngine(player_count, player_names)
        
        # 初始化智能体
        self.agents: List[BaseAgent] = []
        self._initialize_agents()
    
    def _initialize_agents(self):
        """初始化智能体"""
        for player in self.engine.state.players:
            agent = BaseAgent(
                player_id=player.player_id,
                name=player.name,
                use_llm=self.use_llm,
                llm_api_key=self.llm_api_key,
                llm_model=self.llm_model,
                llm_api_provider=self.llm_api_provider
            )
            
            # 获取玩家的私有信息
            private_info = self.engine.get_player_info(player.player_id)
            
            # 初始化角色
            agent.initialize_role(player.role_type, private_info)
            
            self.agents.append(agent)
    
    def get_agent(self, player_id: int) -> BaseAgent:
        """获取指定玩家的智能体"""
        return next(agent for agent in self.agents if agent.player_id == player_id)
    
    def run_game(self, verbose: bool = True):
        """运行游戏"""
        if verbose:
            print("=" * 60)
            print("游戏开始！")
            print("=" * 60)
            self._print_role_assignment()
            print()
        
        # 游戏主循环
        max_rounds = 20  # 防止无限循环
        round_count = 0
        
        while not self.engine.state.game_over and round_count < max_rounds:
            round_count += 1
            
            # 检查游戏是否结束
            game_over, winner = self.engine.win_checker.check_game_over(self.engine.state)
            if game_over:
                self.engine.state.game_over = True
                self.engine.state.winner = winner
                self.engine.state.current_phase = GamePhase.FINISHED
                break
            
            if verbose:
                print(f"\n--- 第 {self.engine.state.current_round} 轮任务 ---")
                print(f"当前队长: {self.engine.state.players[self.engine.state.current_leader].name}")
                print(f"成功任务: {self.engine.state.successful_missions}, "
                      f"失败任务: {self.engine.state.failed_missions}")
            
            # 讨论阶段
            if self.engine.state.current_phase == GamePhase.DISCUSSION:
                self._handle_discussion_phase(verbose)
            
            # 投票阶段
            elif self.engine.state.current_phase == GamePhase.VOTING:
                self._handle_voting_phase(verbose)
                # 投票后检查是否游戏结束（流局5次）
                if self.engine.state.game_over:
                    break
            
            # 任务执行阶段
            elif self.engine.state.current_phase == GamePhase.MISSION:
                self._handle_mission_phase(verbose)
                # 任务后检查是否游戏结束
                game_over, winner = self.engine.win_checker.check_game_over(self.engine.state)
                if game_over:
                    self.engine.state.game_over = True
                    self.engine.state.winner = winner
                    self.engine.state.current_phase = GamePhase.FINISHED
                    break
                if self.engine.state.game_over:
                    break
            
            # 刺杀阶段
            elif self.engine.state.current_phase == GamePhase.ASSASSINATION:
                self._handle_assassination_phase(verbose)
                # 刺杀后游戏必然结束
                break
            
            # 游戏结束
            elif self.engine.state.current_phase == GamePhase.FINISHED:
                break
        
        # 显示游戏结果
        if verbose:
            self._print_game_result()
    
    def _handle_discussion_phase(self, verbose: bool):
        """处理讨论阶段"""
        if verbose:
            print("\n[讨论阶段]")
        
        # 所有玩家都可以发言
        leader_id = self.engine.state.current_leader
        recent_speeches = []
        
        # 准备游戏状态（所有玩家共享相同的基础信息）
        base_game_state = self.engine.get_game_state_summary(leader_id)
        if self.engine.state.current_round <= len(self.engine.state.mission_configs):
            current_config = self.engine.state.mission_configs[self.engine.state.current_round - 1]
            base_game_state["mission_config"] = {
                "team_size": current_config.team_size,
                "fails_needed": current_config.fails_needed
            }
        else:
            base_game_state["mission_config"] = {
                "team_size": 2,
                "fails_needed": 1
            }
        
        # 第一阶段：所有玩家依次发言讨论（从队长的下一位开始，队长最后发言）
        # 找到队长在agents列表中的位置
        leader_index = next(i for i, agent in enumerate(self.agents) if agent.player_id == leader_id)
        leader_agent = self.agents[leader_index]  # 先获取队长智能体
        
        # 从队长的下一位开始发言
        for i in range(len(self.agents)):
            agent_index = (leader_index + 1 + i) % len(self.agents)
            agent = self.agents[agent_index]
            
            # 获取该玩家的游戏状态（包含私有信息）
            game_state = self.engine.get_game_state_summary(agent.player_id)
            # 合并基础信息
            game_state["mission_config"] = base_game_state["mission_config"]
            game_state["mission_history"] = base_game_state.get("mission_history", [])
            
            if verbose:
                speech = agent.generate_speech(game_state, recent_speeches)
                print(f"{agent.name}: {speech}")
                recent_speeches.append({"player_id": agent.player_id, "name": agent.name, "speech": speech})
        
        # 第二阶段：讨论结束后，队长根据讨论内容决定队伍
        leader_game_state = self.engine.get_game_state_summary(leader_id)
        leader_game_state["mission_config"] = base_game_state["mission_config"]
        leader_game_state["mission_history"] = base_game_state.get("mission_history", [])
        
        # 队长根据讨论内容决定队伍
        proposed_team = leader_agent.propose_team(leader_game_state)
        
        if verbose:
            leader_name = self.engine.state.players[leader_id].name
            team_names = [self.engine.state.players[pid].name for pid in proposed_team]
            print(f"\n{leader_name} 根据讨论决定队伍: {', '.join(team_names)}")
        
        # 提交提议
        self.engine.propose_team(leader_id, proposed_team)
    
    def _handle_voting_phase(self, verbose: bool):
        """处理投票阶段"""
        if verbose:
            print("\n[投票阶段]")
        
        proposed_team = self.engine.state.proposed_team
        team_names = [self.engine.state.players[pid].name for pid in proposed_team]
        
        if verbose:
            print(f"对队伍进行投票: {', '.join(team_names)}")
        
        # 收集所有玩家的投票
        votes = {}
        leader_id = self.engine.state.current_leader
        
        for agent in self.agents:
            game_state = self.engine.get_game_state_summary(agent.player_id)
            # 检查是否还有任务配置
            if self.engine.state.current_round <= len(self.engine.state.mission_configs):
                current_config = self.engine.state.mission_configs[self.engine.state.current_round - 1]
                game_state["mission_config"] = {
                    "team_size": current_config.team_size,
                    "fails_needed": current_config.fails_needed
                }
            else:
                # 没有更多任务了，使用默认配置
                game_state["mission_config"] = {
                    "team_size": 2,
                    "fails_needed": 1
                }
            
            # 队长必须同意自己提议的队伍
            if agent.player_id == leader_id:
                vote = True
            else:
                vote = agent.vote_on_team(game_state, proposed_team)
            
            votes[agent.player_id] = vote
            
            if verbose:
                vote_text = "同意" if vote else "拒绝"
                print(f"{agent.name}: {vote_text}")
            
            self.engine.vote_on_team(agent.player_id, vote)
        
        # 处理投票结果
        voting_complete, passed = self.engine.process_voting_result()
        
        if verbose:
            approve_count = sum(1 for v in votes.values() if v)
            reject_count = len(votes) - approve_count
            print(f"\n投票结果: {approve_count} 同意, {reject_count} 拒绝")
            print(f"结果: {'通过' if passed else '未通过'}")
    
    def _handle_mission_phase(self, verbose: bool):
        """处理任务执行阶段"""
        if verbose:
            print("\n[任务执行阶段]")
        
        mission_team = self.engine.state.proposed_team
        team_names = [self.engine.state.players[pid].name for pid in mission_team]
        
        if verbose:
            print(f"执行任务的队伍: {', '.join(team_names)}")
        
        # 收集任务投票
        mission_votes = {}
        for agent in self.agents:
            if agent.player_id in mission_team:
                game_state = self.engine.get_game_state_summary(agent.player_id)
                # 检查是否还有任务配置
                if self.engine.state.current_round <= len(self.engine.state.mission_configs):
                    current_config = self.engine.state.mission_configs[self.engine.state.current_round - 1]
                    game_state["mission_config"] = {
                        "team_size": current_config.team_size,
                        "fails_needed": current_config.fails_needed
                    }
                else:
                    # 没有更多任务了，使用默认配置
                    game_state["mission_config"] = {
                        "team_size": 2,
                        "fails_needed": 1
                    }
                
                success = agent.vote_on_mission(game_state, mission_team)
                mission_votes[agent.player_id] = success
                
                if verbose:
                    result_text = "成功" if success else "失败"
                    print(f"{agent.name}: {result_text}")
        
        # 提交任务结果
        self.engine.submit_mission_result(mission_votes)
        
        # 显示任务结果
        if self.engine.state.mission_results:
            last_result = self.engine.state.mission_results[-1]
            if verbose:
                result_text = "成功" if last_result.success else "失败"
                print(f"\n任务结果: {result_text} (失败票数: {last_result.fail_count})")
            
            # 更新所有智能体的信念系统
            for agent in self.agents:
                for player_id, vote_success in mission_votes.items():
                    agent.belief_system.update_belief_from_mission(
                        player_id=player_id,
                        mission_success=vote_success,
                        mission_team=mission_team,
                        mission_result=last_result.success
                    )
    
    def _handle_assassination_phase(self, verbose: bool):
        """处理刺杀阶段"""
        if verbose:
            print("\n[刺杀阶段]")
            print("坏人阵营可以刺杀梅林...")
        
        # 找到刺客
        assassin_agent = None
        for agent in self.agents:
            from game.roles import RoleType
            if agent.role_type == RoleType.ASSASSIN:
                assassin_agent = agent
                break
        
        if assassin_agent:
            game_state = self.engine.get_game_state_summary(assassin_agent.player_id)
            target_id = assassin_agent.assassinate(game_state)
            
            if target_id is not None:
                target_name = self.engine.state.players[target_id].name
                if verbose:
                    print(f"{assassin_agent.name} 选择刺杀: {target_name}")
                
                self.engine.assassinate(target_id)
            else:
                if verbose:
                    print(f"{assassin_agent.name} 无法决定刺杀目标")
    
    def _print_role_assignment(self):
        """打印角色分配（仅用于调试，实际游戏中不应显示）"""
        print("\n角色分配（调试信息）:")
        for player in self.engine.state.players:
            print(f"  {player.name}: {player.role_type.value} ({player.role.team.value})")
    
    def _print_game_result(self):
        """打印游戏结果"""
        print("\n" + "=" * 60)
        print("游戏结束！")
        print("=" * 60)
        
        if self.engine.state.winner:
            winner_text = "好人阵营" if self.engine.state.winner == Team.GOOD else "坏人阵营"
            print(f"获胜方: {winner_text}")
        else:
            print("游戏未正常结束")
        
        print(f"\n最终统计:")
        print(f"  成功任务: {self.engine.state.successful_missions}")
        print(f"  失败任务: {self.engine.state.failed_missions}")
        print(f"  总轮次: {len(self.engine.state.mission_results)}")
        
        print("\n任务历史:")
        for i, result in enumerate(self.engine.state.mission_results, 1):
            result_text = "成功" if result.success else "失败"
            team_names = [self.engine.state.players[pid].name for pid in result.team_members]
            print(f"  第{i}轮: {result_text} - 队伍: {', '.join(team_names)}")


def main():
    """主函数"""
    import os
    from dotenv import load_dotenv
    
    # 加载环境变量
    load_dotenv()
    
    # 检查是否使用LLM（可以通过环境变量USE_LLM或直接设置）
    use_llm = os.getenv("USE_LLM", "false").lower() == "true"
    llm_api_provider = os.getenv("LLM_API_PROVIDER", "openai").lower()  # "openai", "deepseek", "qwen"
    use_langgraph = os.getenv("USE_LANGGRAPH", "false").lower() == "true"
    
    # 根据提供商选择API密钥和模型
    if llm_api_provider == "deepseek":
        llm_api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
        default_model = "deepseek-chat"
        provider_name = "DeepSeek"
        env_var_name = "DEEPSEEK_API_KEY"
    elif llm_api_provider == "qwen":
        llm_api_key = os.getenv("QWEN_API_KEY", "not-needed")
        default_model = os.getenv("QWEN_MODEL", "qwen")
        provider_name = "Qwen (本地)"
        env_var_name = "QWEN_API_KEY"
        base_url = os.getenv("QWEN_BASE_URL", "http://localhost:8000/v1")
        if use_llm:
            print(f"使用本地Qwen模型: {base_url}")
    else:
        llm_api_key = os.getenv("OPENAI_API_KEY")
        default_model = "gpt-4o-mini"
        provider_name = "OpenAI"
        env_var_name = "OPENAI_API_KEY"
    
    llm_model = os.getenv("LLM_MODEL", default_model)
    
    if use_llm and llm_api_provider != "qwen" and not llm_api_key:
        print(f"警告: 设置了USE_LLM=true但未设置{env_var_name}，将使用普通策略引擎")
        use_llm = False
    
    # 创建游戏（5人局）
    player_names = ["Alice", "Bob", "Charlie", "David", "Eve"]
    game = AvalonGame(
        player_count=5, 
        player_names=player_names,
        use_llm=use_llm,
        llm_api_key=llm_api_key,
        llm_model=llm_model,
        llm_api_provider=llm_api_provider
    )
    
    if use_llm:
        print(f"使用{provider_name} LLM策略引擎 (模型: {llm_model})")
    else:
        print("使用普通策略引擎")
    
    # 选择运行方式
    if use_langgraph:
        try:
            from game.langgraph_game import LangGraphGameEngine
            print("使用LangGraph游戏引擎")
            langgraph_engine = LangGraphGameEngine(game.engine, game.agents, verbose=True)
            langgraph_engine.run()
        except ImportError:
            print("警告: LangGraph未安装，使用传统游戏循环")
            print("安装命令: pip install langgraph")
            game.run_game(verbose=True)
        except Exception as e:
            print(f"LangGraph执行错误: {e}，回退到传统游戏循环")
            game.run_game(verbose=True)
    else:
        # 运行游戏
        game.run_game(verbose=True)


if __name__ == "__main__":
    main()

