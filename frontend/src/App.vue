<template>
  <div class="app-container">
    <!-- 顶部标题栏 -->
    <div class="header">
      <h1>🎮 AI阿瓦隆游戏</h1>
      <p>多智能体推理对话系统</p>
    </div>

    <!-- 错误提示 -->
    <div v-if="error" class="error-message">
      {{ error }}
    </div>

    <!-- 游戏创建界面 -->
    <div v-if="!gameState" class="create-game-panel">
      <div class="panel-content">
        <h2>创建新游戏</h2>
        <div class="form-group">
          <label>玩家数量</label>
          <select v-model="playerCount" class="form-control">
            <option :value="5">5人局</option>
            <option :value="6">6人局</option>
          </select>
        </div>
        <button class="btn btn-primary" @click="createGame" :disabled="loading">
          {{ loading ? '创建中...' : '开始游戏' }}
        </button>
      </div>
    </div>

    <!-- 游戏主界面 -->
    <div v-else class="game-container">
      <!-- 左侧：游戏信息面板 -->
      <div class="sidebar">
        <div class="info-panel">
          <h3>游戏信息</h3>
          <div class="info-item">
            <span class="label">当前阶段</span>
            <span :class="['phase-badge', `phase-${gameState.current_phase.toLowerCase()}`]">
              {{ gameState.current_phase_display }}
            </span>
          </div>
          <div class="info-item">
            <span class="label">第</span>
            <span class="value">{{ gameState.current_round }}</span>
            <span class="label">轮</span>
          </div>
          <div class="info-item">
            <span class="label">成功任务</span>
            <span class="value success">{{ gameState.successful_missions }}</span>
          </div>
          <div class="info-item">
            <span class="label">失败任务</span>
            <span class="value danger">{{ gameState.failed_missions }}</span>
          </div>
          <div class="info-item" v-if="gameState.mission_config">
            <span class="label">队伍人数</span>
            <span class="value">{{ gameState.mission_config.team_size }}</span>
          </div>
        </div>

        <div class="players-panel">
          <h3>玩家列表</h3>
          <div class="player-list">
            <div 
              v-for="player in gameState.players" 
              :key="player.player_id"
              :class="['player-item', { leader: player.player_id === gameState.current_leader }]"
            >
              <div class="player-avatar">
                {{ player.name.charAt(0) }}
              </div>
              <div class="player-info">
                <div class="player-name">
                  {{ player.name }}
                  <span v-if="player.player_id === gameState.current_leader" class="leader-badge">👑</span>
                </div>
                <div class="player-role">{{ player.role_type }}</div>
                <div :class="['player-team', `team-${player.team === '好人' ? 'good' : 'evil'}`]">
                  {{ player.team }}
                </div>
              </div>
            </div>
          </div>
        </div>

        <div class="control-panel">
          <button 
            class="btn btn-primary btn-block" 
            @click="executeStep" 
            :disabled="loading || gameState.game_over"
          >
            {{ loading ? '执行中...' : '执行下一步' }}
          </button>
          <button 
            class="btn btn-secondary btn-block" 
            @click="autoPlay" 
            :disabled="loading || gameState.game_over"
          >
            {{ loading ? '运行中...' : '自动运行' }}
          </button>
          <button class="btn btn-danger btn-block" @click="resetGame">
            重置游戏
          </button>
        </div>
      </div>

      <!-- 右侧：对话区域 -->
      <div class="main-content">
        <div class="chat-container">
          <div class="chat-header">
            <h2>游戏对话</h2>
            <div class="round-info">
              第 {{ gameState.current_round }} 轮 · {{ gameState.current_phase_display }}
            </div>
          </div>
          
          <div class="chat-messages" ref="chatMessages">
            <!-- 游戏开始消息 -->
            <div class="message system-message">
              <div class="message-content">
                <strong>🎮 游戏开始！</strong> 第{{ gameState.current_round }}轮任务即将开始。
              </div>
            </div>

            <!-- 历史消息 -->
            <div 
              v-for="(item, index) in gameState.game_history" 
              :key="index"
              :class="['message', getMessageClass(item)]"
            >
              <div class="message-avatar" v-if="item.player_name">
                {{ item.player_name.charAt(0) }}
              </div>
              <div class="message-content">
                <div class="message-header" v-if="item.player_name">
                  <span class="message-author">{{ item.player_name }}</span>
                  <span class="message-type">{{ getMessageTypeLabel(item.type) }}</span>
                </div>
                <div class="message-text">{{ item.content }}</div>
                <div class="message-meta" v-if="item.type === 'team_proposal' && item.team_member_names">
                  <span class="team-members">队伍成员: {{ item.team_member_names.join(', ') }}</span>
                </div>
                <div class="message-meta" v-if="item.type === 'vote_result'">
                  <span class="vote-result">
                    {{ item.approve_count }} 同意 · {{ item.reject_count }} 拒绝
                    <span :class="['result', item.passed ? 'success' : 'danger']">
                      {{ item.passed ? '✓ 通过' : '✗ 未通过' }}
                    </span>
                  </span>
                </div>
              </div>
            </div>

            <!-- 当前阶段提示 -->
            <div v-if="!gameState.game_over" class="message system-message">
              <div class="message-content">
                <strong>⏳ 等待执行下一步...</strong>
              </div>
            </div>

            <!-- 游戏结束消息 -->
            <div v-if="gameState.game_over" class="message system-message game-over">
              <div class="message-content">
                <strong>🎉 游戏结束！</strong>
                <div class="winner-announcement">
                  获胜方：<span :class="['winner', gameState.winner === 'GOOD' ? 'good' : 'evil']">
                    {{ gameState.winner_display }}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import axios from 'axios'

const API_BASE = '/api'

export default {
  name: 'App',
  data() {
    return {
      gameState: null,
      gameId: null,
      loading: false,
      error: null,
      playerCount: 5
    }
  },
  methods: {
    async createGame() {
      this.loading = true
      this.error = null
      
      try {
        const playerNames = []
        for (let i = 0; i < this.playerCount; i++) {
          playerNames.push(`玩家${i + 1}`)
        }
        
        const response = await axios.post(`${API_BASE}/games`, {
          player_count: this.playerCount,
          player_names: playerNames,
          use_llm: true
        })
        
        if (response.data.success) {
          this.gameId = response.data.game_id
          this.gameState = response.data.game_state
          this.$nextTick(() => {
            this.scrollToBottom()
          })
        } else {
          this.error = '创建游戏失败'
        }
      } catch (err) {
        this.error = err.response?.data?.error || err.message || '创建游戏时发生错误'
        console.error('创建游戏错误:', err)
      } finally {
        this.loading = false
      }
    },
    
    async executeStep() {
      if (!this.gameId) return
      
      this.loading = true
      this.error = null
      
      try {
        const response = await axios.post(`${API_BASE}/games/${this.gameId}/step`)
        
        if (response.data.success) {
          this.gameState = response.data.game_state
          this.$nextTick(() => {
            this.scrollToBottom()
          })
        } else {
          this.error = '执行步骤失败'
        }
      } catch (err) {
        this.error = err.response?.data?.error || err.message || '执行步骤时发生错误'
        console.error('执行步骤错误:', err)
      } finally {
        this.loading = false
      }
    },
    
    async autoPlay() {
      if (!this.gameId) return
      
      this.loading = true
      this.error = null
      
      try {
        const response = await axios.post(`${API_BASE}/games/${this.gameId}/auto-play`)
        
        if (response.data.success) {
          this.gameState = response.data.final_state
          this.$nextTick(() => {
            this.scrollToBottom()
          })
        } else {
          this.error = '自动运行失败'
        }
      } catch (err) {
        this.error = err.response?.data?.error || err.message || '自动运行时发生错误'
        console.error('自动运行错误:', err)
      } finally {
        this.loading = false
      }
    },
    
    resetGame() {
      this.gameState = null
      this.gameId = null
      this.error = null
    },
    
    getMessageClass(item) {
      const classes = ['message']
      if (item.type === 'speech') {
        classes.push('speech-message')
      } else if (item.type === 'team_proposal') {
        classes.push('proposal-message')
      } else if (item.type === 'vote') {
        classes.push('vote-message')
      } else if (item.type === 'vote_result') {
        classes.push('result-message')
      } else if (item.type === 'mission_vote') {
        classes.push('mission-message')
      }
      return classes.join(' ')
    },
    
    getMessageTypeLabel(type) {
      const labels = {
        'speech': '💬 发言',
        'team_proposal': '👥 提议',
        'vote': '🗳️ 投票',
        'vote_result': '📊 结果',
        'mission_vote': '⚔️ 任务'
      }
      return labels[type] || type
    },
    
    scrollToBottom() {
      const chatMessages = this.$refs.chatMessages
      if (chatMessages) {
        chatMessages.scrollTop = chatMessages.scrollHeight
      }
    }
  },
  watch: {
    'gameState.game_history'() {
      this.$nextTick(() => {
        this.scrollToBottom()
      })
    }
  }
}
</script>
