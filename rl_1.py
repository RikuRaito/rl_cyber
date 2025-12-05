import nasim
import numpy as np
import random

class QLearningAgent:
    def __init__(self, env_name="small", episodes=1000, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.env_name = env_name
        self.episodes = episodes
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.env = nasim.make_benchmark(
            self.env_name,
            fully_obs=True,
            flat_actions=True,
            flat_obs=True
        )
        self.q_table = {}

    def _state_to_key(self, state):
        """状態をハッシュ可能なキーに変換"""
        return tuple(state.astype(np.int32))

    def _get_q(self, state, action):
        # まだ知らない状態なら 0.0 を返す
        state_key = self._state_to_key(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.env.action_space.n)
        return self.q_table[state_key][action]

    def _update_q(self, state, action, reward, next_state):
        # Q学習の数式 (ベルマン方程式)
        state_key = self._state_to_key(state)
        next_state_key = self._state_to_key(next_state)
        
        # 次の状態での最大Q値を取得 (未来の予測)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.env.action_space.n)
        max_next_q = np.max(self.q_table[next_state_key])
        
        # 現在のQ値
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.env.action_space.n)
        current_q = self.q_table[state_key][action]
        
        # 更新
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state_key][action] = new_q

    def train(self):
        # --- 学習開始 ---
        print(f"--- {self.env_name} の攻略を開始します (alpha={self.alpha}, gamma={self.gamma}, epsilon={self.epsilon}) ---")
        reward_history = []

        for episode in range(self.episodes):
            state, _ = self.env.reset()
            total_reward = 0
            done = False
            steps = 0
            
            while not done:
                # 行動を決める
                state_key = self._state_to_key(state)
                if random.uniform(0, 1) < self.epsilon:
                    # ランダム (探索)
                    action = int(self.env.action_space.sample())
                else:
                    # 今一番良いと思う行動を選ぶ (活用)
                    if state_key not in self.q_table:
                        action = int(self.env.action_space.sample())
                    else:
                        action = int(np.argmax(self.q_table[state_key]))

                # 実行
                next_state, reward, done, step_limit_reached, _ = self.env.step(action)
                done = done or step_limit_reached
                
                # 学習
                self._update_q(state, action, reward, next_state)
                
                state = next_state
                total_reward += reward
                steps += 1

            reward_history.append(total_reward)
            # 100回ごとに進捗を表示
            if (episode + 1) % 100 == 0:
                print(f"Episode {episode + 1}: 手数={steps}, スコア={total_reward}")

        print("--- 学習終了 ---")
        return reward_history

    def test_policy(self):
        # --- 成果確認 (テストプレイ) ---
        print("\n--- 賢くなったAIのプレイ ---")
        state, _ = self.env.reset()
        done = False
        steps = 0
        total_test_reward = 0

        while not done:
            state_key = self._state_to_key(state)
            if state_key in self.q_table:
                action = int(np.argmax(self.q_table[state_key]))
            else:
                action = int(self.env.action_space.sample())
            
            next_state, reward, done, step_limit_reached, _ = self.env.step(action)
            done = done or step_limit_reached
            total_test_reward += reward
            print(f"Step {steps+1}: Action {action} -> Reward {reward}")
            state = next_state
            steps += 1

        print(f"\nテスト結果: 総報酬={total_test_reward}, ステップ数={steps}")

#データのプロット方法はわからない
import matplotlib.pyplot as plt

def plot_rewards(histories, labels, moving_avg_window=100):
    """複数の学習履歴をプロットする"""
    plt.figure(figsize=(12, 6))
    
    for history, label in zip(histories, labels):
        # 移動平均を計算してグラフを滑らかにする
        moving_avg = np.convolve(history, np.ones(moving_avg_window)/moving_avg_window, mode='valid')
        plt.plot(moving_avg, label=label)
        
    plt.title("Learning Curve (Moving Average)")
    plt.xlabel(f"Episodes (averaged over {moving_avg_window} episodes)")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True)
    
    # グラフを画像として保存
    filename = "results/rl_1_learning_curve.png"
    plt.savefig(filename)
    print(f"\nグラフを {filename} として保存しました。")

if __name__ == "__main__":
    # --- 実験パラメータ ---
    alpha_to_test = [0.1, 0.4, 0.8]
    gamma = 0.99
    epsilon = 0.1
    episode= 8000

    all_histories = []
    all_labels = []

    for alpha in alpha_to_test:
        agent = QLearningAgent(
            episodes=episode,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon
        )
        print(f"\n--- Experimenting with alpha ---")
        history = agent.train()
        # agent.test_policy() # 最終テストは一旦コメントアウト

        all_histories.append(history)
        all_labels.append(f"alpha = {alpha}")
    
    plot_rewards(all_histories, all_labels)
