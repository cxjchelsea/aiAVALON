"""
Flask后端API服务器
提供阿瓦隆游戏的RESTful API接口
"""
import os
import sys
import uuid
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from typing import Dict, Optional
import json

# 添加src目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
# 确保src目录在路径中，这样相对导入才能工作
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# 使用绝对导入
from src.main import AvalonGame
from src.game.rules import GamePhase, Team

# 加载环境变量
load_dotenv()

app = Flask(__name__)
app.config['DEBUG'] = True  # 启用调试模式以显示详细错误
CORS(app)  # 允许跨域请求

# 存储游戏实例（实际应用中应使用数据库或Redis）
games: Dict[str, AvalonGame] = {}


def serialize_game_state(game: AvalonGame) -> Dict:
    """序列化游戏状态为JSON可序列化的格式"""
    state = game.engine.state
    
    # 序列化玩家信息
    players = []
    for player in state.players:
        players.append({
            "player_id": player.player_id,
            "name": player.name,
            "role_type": player.role_type.value,
            "team": player.role.team.value
        })
    
    # 序列化任务结果
    mission_results = []
    for result in state.mission_results:
        mission_results.append({
            "round_number": result.round_number,
            "team_members": result.team_members,
            "team_member_names": [state.players[pid].name for pid in result.team_members],
            "success": result.success,
            "fail_count": result.fail_count
        })
    
    # 序列化投票信息
    votes = {}
    for player_id, vote in state.votes.items():
        votes[player_id] = {
            "player_id": player_id,
            "player_name": state.players[player_id].name,
            "approve": vote
        }
    
    # 获取当前任务配置
    mission_config = None
    if state.current_round <= len(state.mission_configs):
        config = state.mission_configs[state.current_round - 1]
        mission_config = {
            "round_number": config.round_number,
            "team_size": config.team_size,
            "fails_needed": config.fails_needed
        }
    
    # 获取提议队伍的名称
    proposed_team_names = []
    if state.proposed_team:
        proposed_team_names = [state.players[pid].name for pid in state.proposed_team]
    
    # 获取当前队长的名称
    current_leader_name = state.players[state.current_leader].name if state.players else None
    
    # 获取刺杀目标名称
    assassination_target_name = None
    if state.assassination_target is not None:
        assassination_target_name = state.players[state.assassination_target].name
    
    # 获取游戏历史记录
    game_history = getattr(game, 'game_history', [])
    
    return {
            "game_id": None,  # 将在调用时设置
            "players": players,
            "current_phase": state.current_phase.name,
            "current_phase_display": state.current_phase.value,
            "current_round": state.current_round,
            "current_leader": state.current_leader,
            "current_leader_name": current_leader_name,
            "mission_config": mission_config,
            "mission_results": mission_results,
            "proposed_team": state.proposed_team,
            "proposed_team_names": proposed_team_names,
            "votes": votes,
            "vote_round": state.vote_round,
            "successful_missions": state.successful_missions,
            "failed_missions": state.failed_missions,
            "game_over": state.game_over,
            "winner": state.winner.name if state.winner else None,
            "winner_display": state.winner.value if state.winner else None,
            "assassination_target": state.assassination_target,
            "assassination_target_name": assassination_target_name,
            "game_history": game_history  # 添加游戏历史
        }


@app.route('/api/games', methods=['POST'])
def create_game():
    """创建新游戏"""
    try:
        data = request.get_json() or {}
        player_count = data.get('player_count', 5)
        player_names = data.get('player_names', None)
        use_llm = data.get('use_llm', True)
        
        # 获取LLM配置
        llm_api_provider = os.getenv("LLM_API_PROVIDER", "openai").lower()
        
        if llm_api_provider == "deepseek":
            llm_api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
            default_model = "deepseek-chat"
        elif llm_api_provider == "qwen":
            llm_api_key = os.getenv("QWEN_API_KEY", "not-needed")
            default_model = os.getenv("QWEN_MODEL", "qwen")
        else:
            llm_api_key = os.getenv("OPENAI_API_KEY")
            default_model = "gpt-4o-mini"
        
        llm_model = os.getenv("LLM_MODEL", default_model)
        
        # 检查LLM配置
        if llm_api_provider != "qwen" and not llm_api_key:
            return jsonify({
                "error": f"请配置LLM API密钥。设置{llm_api_provider.upper()}_API_KEY环境变量"
            }), 400
        
        # 如果没有提供玩家名称，生成默认名称
        if player_names is None:
            player_names = [f"玩家{i+1}" for i in range(player_count)]
        
        # 创建游戏
        game = AvalonGame(
            player_count=player_count,
            player_names=player_names,
            use_llm=use_llm,
            llm_api_key=llm_api_key,
            llm_model=llm_model,
            llm_api_provider=llm_api_provider
        )
        
        # 生成游戏ID
        game_id = str(uuid.uuid4())
        games[game_id] = game
        
        # 获取游戏状态
        game_state = serialize_game_state(game)
        game_state["game_id"] = game_id
        
        return jsonify({
            "success": True,
            "game_id": game_id,
            "game_state": game_state
        }), 201
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"创建游戏时发生错误: {str(e)}")
        print(f"错误堆栈:\n{error_trace}")
        return jsonify({
            "error": str(e),
            "traceback": error_trace if app.debug else None
        }), 500


@app.route('/api/games/<game_id>', methods=['GET'])
def get_game_state(game_id: str):
    """获取游戏状态"""
    if game_id not in games:
        return jsonify({"error": "游戏不存在"}), 404
    
    game = games[game_id]
    game_state = serialize_game_state(game)
    game_state["game_id"] = game_id
    
    return jsonify({
        "success": True,
        "game_state": game_state
    })


@app.route('/api/games/<game_id>/player/<player_id>', methods=['GET'])
def get_player_info(game_id: str, player_id: int):
    """获取玩家私有信息"""
    if game_id not in games:
        return jsonify({"error": "游戏不存在"}), 404
    
    game = games[game_id]
    player_id = int(player_id)
    
    # 检查玩家是否存在
    if player_id >= len(game.engine.state.players):
        return jsonify({"error": "玩家不存在"}), 404
    
    # 获取玩家私有信息
    private_info = game.engine.get_player_info(player_id)
    
    return jsonify({
        "success": True,
        "player_id": player_id,
        "private_info": private_info
    })


@app.route('/api/games/<game_id>/step', methods=['POST'])
def execute_game_step(game_id: str):
    """执行游戏步骤（自动进行一个阶段）"""
    if game_id not in games:
        return jsonify({"error": "游戏不存在"}), 404
    
    game = games[game_id]
    
    # 检查游戏是否已结束
    if game.engine.state.game_over:
        game_state = serialize_game_state(game)
        game_state["game_id"] = game_id
        return jsonify({
            "success": True,
            "message": "游戏已结束",
            "game_state": game_state
        })
    
    try:
        # 根据当前阶段执行相应的操作
        phase = game.engine.state.current_phase
        phase_name = phase.name if hasattr(phase, 'name') else str(phase)
        
        print(f"执行游戏步骤 - 游戏ID: {game_id}, 当前阶段: {phase_name}")
        print(f"阶段类型: {type(phase)}, 阶段值: {phase}, 阶段名称: {phase_name}")
        print(f"GamePhase.DISCUSSION: {GamePhase.DISCUSSION}, 类型: {type(GamePhase.DISCUSSION)}")
        print(f"比较结果: phase == GamePhase.DISCUSSION = {phase == GamePhase.DISCUSSION}")
        
        # 使用阶段名称进行比较（更可靠）
        if phase_name == "INITIALIZATION" or phase == GamePhase.INITIALIZATION:
            # 初始化阶段，应该自动转到讨论阶段
            print("游戏初始化，转到讨论阶段...")
            game.engine.state.current_phase = GamePhase.DISCUSSION
        elif phase_name == "DISCUSSION" or phase == GamePhase.DISCUSSION:
            # 讨论阶段
            print("执行讨论阶段...")
            game._handle_discussion_phase(verbose=False)
        elif phase_name == "VOTING" or phase == GamePhase.VOTING:
            # 投票阶段
            print("执行投票阶段...")
            game._handle_voting_phase(verbose=False)
        elif phase_name == "MISSION" or phase == GamePhase.MISSION:
            # 任务执行阶段
            print("执行任务阶段...")
            game._handle_mission_phase(verbose=False)
        elif phase_name == "ASSASSINATION" or phase == GamePhase.ASSASSINATION:
            # 刺杀阶段
            print("执行刺杀阶段...")
            game._handle_assassination_phase(verbose=False)
        elif phase_name == "FINISHED" or phase == GamePhase.FINISHED:
            # 游戏已结束
            game_state = serialize_game_state(game)
            game_state["game_id"] = game_id
            return jsonify({
                "success": True,
                "message": "游戏已结束",
                "game_state": game_state
            })
        else:
            error_msg = f"无法执行阶段: {phase_name} (值: {phase.value if hasattr(phase, 'value') else phase})"
            print(f"错误: {error_msg}")
            print(f"当前游戏状态: game_over={game.engine.state.game_over}, round={game.engine.state.current_round}")
            return jsonify({
                "error": error_msg,
                "current_phase": phase_name,
                "game_over": game.engine.state.game_over
            }), 400
        
        # 检查游戏是否结束
        game_over, winner = game.engine.win_checker.check_game_over(game.engine.state)
        if game_over:
            game.engine.state.game_over = True
            game.engine.state.winner = winner
            game.engine.state.current_phase = GamePhase.FINISHED
        
        # 返回更新后的游戏状态
        game_state = serialize_game_state(game)
        game_state["game_id"] = game_id
        
        print(f"步骤执行成功，新阶段: {game.engine.state.current_phase.name}")
        
        return jsonify({
            "success": True,
            "game_state": game_state
        })
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        error_msg = str(e)
        print(f"执行游戏步骤时发生错误: {error_msg}")
        print(f"错误堆栈:\n{error_trace}")
        return jsonify({
            "error": error_msg,
            "traceback": error_trace if app.debug else None
        }), 500


@app.route('/api/games/<game_id>/history', methods=['GET'])
def get_game_history(game_id: str):
    """获取游戏历史记录（包括发言、投票等）"""
    if game_id not in games:
        return jsonify({"error": "游戏不存在"}), 404
    
    game = games[game_id]
    
    # 这里可以扩展为记录更详细的历史
    # 目前返回任务历史
    history = {
        "mission_results": [],
        "speeches": []  # 可以扩展记录发言历史
    }
    
    for result in game.engine.state.mission_results:
        history["mission_results"].append({
            "round_number": result.round_number,
            "team_members": result.team_members,
            "team_member_names": [game.engine.state.players[pid].name for pid in result.team_members],
            "success": result.success,
            "fail_count": result.fail_count
        })
    
    return jsonify({
        "success": True,
        "history": history
    })


@app.route('/api/games/<game_id>/auto-play', methods=['POST'])
def auto_play_game(game_id: str):
    """自动运行游戏直到结束（用于演示）"""
    if game_id not in games:
        return jsonify({"error": "游戏不存在"}), 404
    
    game = games[game_id]
    
    # 运行游戏（不输出到控制台）
    max_rounds = 20
    round_count = 0
    steps = []
    
    while not game.engine.state.game_over and round_count < max_rounds:
        round_count += 1
        
        # 检查游戏是否结束
        game_over, winner = game.engine.win_checker.check_game_over(game.engine.state)
        if game_over:
            game.engine.state.game_over = True
            game.engine.state.winner = winner
            game.engine.state.current_phase = GamePhase.FINISHED
            break
        
        phase = game.engine.state.current_phase
        
        # 记录步骤
        step_state = serialize_game_state(game)
        step_state["game_id"] = game_id
        steps.append({
            "step": round_count,
            "phase": phase.name,
            "game_state": step_state
        })
        
        # 执行阶段
        if phase == GamePhase.DISCUSSION:
            game._handle_discussion_phase(verbose=False)
        elif phase == GamePhase.VOTING:
            game._handle_voting_phase(verbose=False)
            if game.engine.state.game_over:
                break
        elif phase == GamePhase.MISSION:
            game._handle_mission_phase(verbose=False)
            game_over, winner = game.engine.win_checker.check_game_over(game.engine.state)
            if game_over:
                game.engine.state.game_over = True
                game.engine.state.winner = winner
                game.engine.state.current_phase = GamePhase.FINISHED
                break
        elif phase == GamePhase.ASSASSINATION:
            game._handle_assassination_phase(verbose=False)
            break
        elif phase == GamePhase.FINISHED:
            break
    
    # 最终状态
    final_state = serialize_game_state(game)
    final_state["game_id"] = game_id
    
    return jsonify({
        "success": True,
        "steps": steps,
        "final_state": final_state
    })


@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({
        "status": "ok",
        "message": "阿瓦隆游戏API服务运行正常"
    })


@app.route('/api/test-import', methods=['GET'])
def test_import():
    """测试导入是否正常"""
    try:
        from src.main import AvalonGame
        from src.game.rules import GamePhase, Team
        return jsonify({
            "success": True,
            "message": "导入成功",
            "sys_path": sys.path[:5]  # 只返回前5个路径
        })
    except Exception as e:
        import traceback
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


if __name__ == '__main__':
    # 开发环境配置
    app.run(debug=True, host='0.0.0.0', port=5000)

