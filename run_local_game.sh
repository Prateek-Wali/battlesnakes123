#!/bin/bash
# ============================================================
#  Local BattleSnake Game Runner
#  Runs your bot + the hackathon game engine via Docker
# ============================================================

set -e

TEAM_NAME="${1:-APEX}"
REPLAYS_DIR="/opt/battlesnake/replays"

echo "🐍 BattleSnake Local Game Runner"
echo "================================"
echo "Team: $TEAM_NAME"
echo ""

# Step 1: Create Docker network (if it doesn't exist)
docker network inspect snakenet &>/dev/null || {
    echo "📡 Creating Docker network 'snakenet'..."
    docker network create snakenet
}

# Step 2: Build your bot image
echo "🔨 Building your bot Docker image..."
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
docker build -t my_snake "$SCRIPT_DIR"

# Step 3: Start your bot container
echo "🚀 Starting your bot on snakenet..."
docker rm -f my_snake_container 2>/dev/null || true
docker run -d --name my_snake_container --network snakenet my_snake
echo "   ✅ Bot running at http://my_snake_container:8080"

# Step 4: Create replays directory
sudo mkdir -p "$REPLAYS_DIR" 2>/dev/null || mkdir -p "$REPLAYS_DIR" 2>/dev/null || true

# Step 5: Run the game
REPLAY_FILE="/replays/battlesnake_replay_$(date +%Y%m%d-%H%M%S).json"
echo ""
echo "🎮 Starting game..."
echo "   Board: 11x11 | Timeout: 5000ms"
echo "   Replay: $REPLAY_FILE"
echo ""

docker run --rm --network snakenet \
    -v "$REPLAYS_DIR:/replays" \
    battlesnake_board \
    battlesnake play \
    -W 11 -H 11 \
    --timeout 5000 \
    --output "$REPLAY_FILE" \
    --name "$TEAM_NAME" --url http://my_snake_container:8080 \
    2>&1

echo ""
echo "🏁 Game complete! Replay saved."

# Cleanup
echo "🧹 Cleaning up bot container..."
docker stop my_snake_container 2>/dev/null && docker rm my_snake_container 2>/dev/null || true
echo "✅ Done!"
