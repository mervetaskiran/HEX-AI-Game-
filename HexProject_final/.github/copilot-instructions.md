<!-- Copilot / AI Agent instructions for HexProject -->
# Copilot Instructions — HexProject

Purpose: help an AI coding agent be immediately productive in this small Hex game project.

- **Big picture**: The project has three main parts:
  - `hex_engine.py` — core game model (`HexGame`, board state, rules, win check). Uses `numpy` arrays and BFS for win detection.
  - `hybrid_agent.py` — the AI agent (class `HybridAI`) combining MCTS and a Dijkstra-based heuristic (`evaluate_board`, `dijkstra_distance`). Uses `copy.deepcopy` for state cloning and numpy for heuristics.
  - `main.py` — a `pygame` UI and game loop that instantiates `HexGame` and `HybridAI`. It contains rendering, input handling, and mode/difficulty controls.

- **Key conventions & patterns**:
  - Players are integers: `PLAYER_1 = 1`, `PLAYER_2 = -1`, `EMPTY = 0`.
  - Board is a `numpy` 2D array of ints: `game.board`.
  - `get_valid_moves()` returns a list of `(row, col)` tuples built with `list(zip(*np.where(self.board == EMPTY)))`.
  - `make_move(row, col)` places `current_player` and flips `current_player *= -1`. `game.winner` is set when a win is detected.
  - Hex adjacency is expressed with the 6-direction neighbor list: `directions = [(-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0)]` (used in both engine and heuristic).

- **Important coupling / gotchas**:
  - `hex_engine.HexGame` supports variable `size`, but `main.py` frequently assumes a fixed 11x11 board (loops and `pixel_to_hex` use range(11)). If you change `BOARD_SIZE` update `main.py` accordingly.
  - `hybrid_agent.HybridAI` uses `copy.deepcopy` extensively; performance-sensitive changes should avoid deep copies or replace them with lightweight state clones.
  - The AI rollout depth is limited to 10 moves in `HybridAI` — a tuning parameter that affects evaluation speed vs. quality.
  - Difficulty maps to iteration counts: `Easy=100`, `Medium=500`, `Hard=1500`.

- **How to run locally (Windows PowerShell)**:
  - Create a venv (optional): `python -m venv .venv; .\.venv\Scripts\Activate.ps1`
  - Install dependencies: `pip install numpy pygame`
  - Run the UI: `python .\main.py`

- **Where to make common edits**:
  - Change board logic or rules: edit `hex_engine.py` (BFS win check, `get_valid_moves`, `make_move`).
  - Tune AI: edit `hybrid_agent.py` — adjust `iterations`, `rollout` depth, `evaluate_board`, or replace `dijkstra_distance` with a different heuristic.
  - Change visuals or board-size coupling: edit `main.py` (constants `BOARD_SIZE`-related code, `pixel_to_hex`, loops over range(11), `get_hex_center`).

- **Examples of useful code edits an agent may be asked to implement**:
  - Replace `copy.deepcopy` with a lightweight `clone()` in `HexGame` (add a method that copies `board`, `current_player`, `winner`) and update `hybrid_agent.py` to call it.
  - Parameterize `main.py` loops to use `game.size` instead of hardcoded `11` (update `pixel_to_hex`, draw loops, and `draw_borders`).
  - Add a CLI flag to run `AIvAI` headless for faster testing (move the game loop logic into a runnable function and bypass pygame rendering when headless).

- **Testing & debugging tips for the repo**:
  - To quickly inspect game logic without the UI, import `HexGame` in a REPL or small script and call `make_move` / `check_win` with deterministic moves.
  - For AI profiling: run `HybridAI.get_best_move` on a small `HexGame` state in a loop (outside pygame) and measure time to tune `iterations`.
  - Watch for heavy `deepcopy` calls in `hybrid_agent.py` — these dominate CPU/memory when `iterations` is large.

- **Files to reference while modifying code**:
  - `hex_engine.py` — canonical implementation of rules and board representation.
  - `hybrid_agent.py` — where MCTS, heuristic, and tuning constants live.
  - `main.py` — UI, input mapping, and mode/difficulty wiring.

If anything here is unclear or you want more detail (for example, a recommended lightweight `clone()` implementation or a headless AI-vs-AI runner), tell me which area to expand and I will update this file.
