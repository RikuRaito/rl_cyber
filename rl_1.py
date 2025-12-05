import nasim
import numpy as np
import random

# --- 設定 ---
ENV_NAME = "tiny"  # ステージ
EPISODES = 1000    # 学習する回数
ALPHA = 0.1        # 学習率 
GAMMA = 0.99       # 割引率 
EPSILON = 0.1      # 探索率 

# 環境を作成（flat_actions=Trueでアクションを整数で扱えるように）
env = nasim.make_benchmark(
    ENV_NAME,
    fully_obs=True,      # 完全観測モード
    flat_actions=True,   # フラットなアクション空間
    flat_obs=True        # フラットな観測空間
)

# NASimの状態は複雑なので、タプル化してハッシュ可能にする
q_table = {}

def state_to_key(state):
    """状態をハッシュ可能なキーに変換"""
    # numpy配列をタプルに変換（メモリ効率が良い）
    return tuple(state.astype(np.int32))

#行動を価値化するためのテーブルを作る
def get_q(state, action):
    # まだ知らない状態なら 0.0 を返す
    state_key = state_to_key(state)
    if state_key not in q_table:
        q_table[state_key] = np.zeros(env.action_space.n)
    return q_table[state_key][action]

def update_q(state, action, reward, next_state):
    # Q学習の数式 (ベルマン方程式)
    state_key = state_to_key(state)
    next_state_key = state_to_key(next_state)
    
    # 次の状態での最大Q値を取得 (未来の予測)
    if next_state_key not in q_table:
        q_table[next_state_key] = np.zeros(env.action_space.n)
    max_next_q = np.max(q_table[next_state_key])
    
    # 現在のQ値
    if state_key not in q_table:
        q_table[state_key] = np.zeros(env.action_space.n)
    current_q = q_table[state_key][action]
    
    # 更新
    new_q = current_q + ALPHA * (reward + GAMMA * max_next_q - current_q)
    q_table[state_key][action] = new_q

# --- 学習開始 ---
print(f"--- {ENV_NAME} の攻略を開始します ---")

for episode in range(EPISODES):
    state, _ = env.reset()
    total_reward = 0
    done = False
    steps = 0
    
    while not done:
        # 行動を決める (Epsilon-Greedy法)
        state_key = state_to_key(state)
        if random.uniform(0, 1) < EPSILON:
            # ランダム (探索)
            action = int(env.action_space.sample())
        else:
            # 今一番良いと思う行動を選ぶ (活用)
            if state_key not in q_table:
                action = int(env.action_space.sample())
            else:
                action = int(np.argmax(q_table[state_key]))

        # 実行！
        next_state, reward, done, step_limit_reached, _ = env.step(action)
        done = done or step_limit_reached
        
        # 学習！ (経験を脳に書き込む)
        update_q(state, action, reward, next_state)
        
        state = next_state
        total_reward += reward
        steps += 1

    # 100回ごとに進捗を表示
    if (episode + 1) % 100 == 0:
        print(f"Episode {episode + 1}: 手数={steps}, スコア={total_reward}")

print("--- 学習終了 ---")

# --- 成果確認 (テストプレイ) ---
print("\n--- 賢くなったAIのプレイ ---")
state, _ = env.reset()
done = False
steps = 0
total_test_reward = 0

while not done:
    state_key = state_to_key(state)
    if state_key in q_table:
        action = int(np.argmax(q_table[state_key]))
    else:
        action = int(env.action_space.sample())
    
    next_state, reward, done, step_limit_reached, _ = env.step(action)
    done = done or step_limit_reached
    total_test_reward += reward
    print(f"Step {steps+1}: Action {action} -> Reward {reward}")
    state = next_state
    steps += 1

print(f"\nテスト結果: 総報酬={total_test_reward}, ステップ数={steps}")