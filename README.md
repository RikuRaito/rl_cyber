# NASim Reinforcement Learning

NASim環境を使った強化学習プロジェクト

## セットアップ

### 1. 仮想環境の作成とアクティベート

```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
# または
venv\Scripts\activate  # Windows
```

### 2. 依存関係のインストール

```bash
pip install -r requirements.txt
```

## 使用方法

仮想環境をアクティベートした状態で実行：

```bash
source venv/bin/activate
python rl_1.py
```

## 注意事項

- 仮想環境（`venv/`）は `.gitignore` に含まれているため、Gitにはコミットされません
- 各環境で `python -m venv venv` を実行して仮想環境を作成してください

