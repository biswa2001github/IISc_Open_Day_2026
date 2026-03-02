# Games for IISc Open Day 2026 — AI vs Human

This repository contains three small games that let an AI play against a human.

## Installation

1. Create and activate a Python virtual environment (recommended):

```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

> The project was developed using Python 3.13 and uses packages listed in `requirements.txt`.

## Games and how to run them

- **AI vs Human — Brick Breaker**

  Run from the project root:

  ```bash
  python3 ai_vs_human_brick_breaker.py
  ```

  File: [ai_vs_human_brick_breaker.py](ai_vs_human_brick_breaker.py)

- **AI vs Human — Pong**

  Run from the project root:

  ```bash
  python3 ai_vs_human_pong.py
  ```

  File: [ai_vs_human_pong.py](ai_vs_human_pong.py)

- **AI vs Human — Snake (Easy)**

  Change into the snake easy folder and run the gameplay script:

  ```bash
  cd Snake_Game/snake_game_AI_easy
  python3 ai_vs_human_gameplay.py
  ```

  File: [Snake_Game/snake_game_AI_easy/ai_vs_human_gameplay.py](Snake_Game/snake_game_AI_easy/ai_vs_human_gameplay.py)

## Notes

- If you experience issues launching the games, confirm your virtual environment is active and the packages in `requirements.txt` installed correctly.
- These games rely on `pygame` and other dependencies listed in `requirements.txt`.

