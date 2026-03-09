# BattleSnake Elite AI — Rust

High-performance BattleSnake AI targeting 20,000 turn survival.
Written in pure Rust with zero heap allocation in the hot path.

## Algorithm Stack

| Algorithm | Purpose |
|-----------|---------|
| **Move classifier** | Labels every move Safe / Risky / Lethal before scoring |
| **Flood fill** | BitBoard BFS — prevents traps, measures reachable space |
| **A\* pathfinding** | Finds shortest path to nearest food |
| **Minimax + α-β** | Depth 5–8 lookahead with alpha-beta pruning |
| **Voronoi territory** | BFS from all heads simultaneously — territory control |
| **Coiling** | Follow own tail when trapped in small space |
| **1v1 endgame** | Switches to territory-delta + chase/flee when last enemy |
| **Offense/Defence** | Cuts off enemies when larger; maximises own space when threatened |

## Game Rules Implemented

- **Food restores full health (100 HP)**
- **Health depletes 1 HP per turn**
- **11 × 11 grid**
- Target: survive **20,000 turns**

## Performance

All hot-path data structures are stack-allocated:
- `BitBoard` — 121-cell board in 2×u64 (16 bytes), no heap
- `Snake` body — 121-byte ring buffer on stack, no `Vec`
- Flood fill — 121-byte queue array on stack
- A\* — stack-allocated g/f cost arrays (121×u32 each)
- Voronoi — stack-allocated owner/queue arrays

Typical move latency: **< 1 ms** at minimax depth 5–8.

## Build

```bash
# Install Rust (if not installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Debug build (faster compile, slower runtime)
cargo build

# Release build (maximum performance — use this for competition)
cargo build --release
```

## Run

### HTTP Server (BattleSnake API)
```bash
cargo run --release
# Listens on http://0.0.0.0:8080

# Custom port
PORT=3000 cargo run --release
```

API endpoints:
- `GET  /`      — snake info (color, head, tail)
- `POST /start` — game start notification
- `POST /move`  — returns move decision
- `POST /end`   — game end notification

### Self-Play Mode (benchmark / test survival)
```bash
# Run to 20,000 turns
cargo run --release -- selfplay 20000

# Quick test run
cargo run --release -- selfplay 1000
```

Example output:
```
=== BattleSnake Self-Play ===
Target: 20000 turns | Rules: food=+100HP, -1HP/turn
Grid: 11x11
----------------------------
T   500 | HP: 87 | len:  8 | mode:CONTROL    | food:12 | kills:1 | enemies:2
T  1000 | HP: 43 | len: 14 | mode:FEED       | food:28 | kills:2 | enemies:1
T  1500 | HP: 72 | len: 19 | mode:1v1        | food:44 | kills:2 | enemies:1
...
```

## Run Tests

```bash
cargo test
```

Tests cover:
- Flood fill (open board, blocked start, walled-off sections)
- Cell encoding correctness (no collisions for all 121 cells)
- Obstacle map never includes head
- Wall avoidance
- Neck reversal prevention
- Critical HP → food seeking
- Dead-end avoidance (space heuristic)
- A\* pathfinding + blocked start guard
- Simulation: food restores full health, starvation, head-on collisions
- Voronoi territory even split
- BitBoard set/get/iter for all 121 cells

## Deploy to BattleSnake Platform

```bash
# Build release binary
cargo build --release

# The binary is at ./target/release/battlesnake
# Deploy it anywhere that can serve HTTP — Heroku, Railway, Fly.io, etc.

# Example: Fly.io
fly launch
fly deploy
```

### Dockerfile (optional)
```dockerfile
FROM rust:1.75 as builder
WORKDIR /app
COPY . .
RUN cargo build --release

FROM debian:bookworm-slim
COPY --from=builder /app/target/release/battlesnake /usr/local/bin/
CMD ["battlesnake"]
```

## Architecture

```
src/main.rs
├── Grid encoding      cell(x,y) = y*11+x  →  u8
├── BitBoard           2×u64, stack only, bit-scan iterator
├── Snake              121-byte ring buffer body, no Vec
├── State              [Snake;4] + BitBoard food + turn
├── flood_fill()       BFS with 121-byte stack queue
├── space_after()      flood fill after hypothetical move
├── voronoi()          simultaneous BFS from all heads
├── astar()            A* with stack-allocated cost arrays
├── coil_dir()         tail-following when trapped
├── classify()         Safe/Risky/Lethal move labels
├── enemy_move()       Hunter / Territory / Ambush AI
├── evaluate()         Heuristic: fill+food+voronoi+length+danger
├── minimax()          Alpha-beta depth 5-8
├── choose_move()      Master selector: crit→coil→filter→minimax
├── HTTP handlers      actix-web /info /start /move /end
└── run_selfplay()     Standalone benchmark mode
```
