import nasim

# 環境を作る
env = nasim.make_benchmark("tiny")
state, info= env.reset()

print("--- AIが見ているデータの形 ---")
print(f"型: {type(state)}")
print(f"サイズ: {state.shape}")
print("\n--- 生データの中身（先頭20個）---")
print(state[:20])
print("\n--infoの中身---")
print(f"{info}")