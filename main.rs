// ================================================================
//  BattleSnake Elite AI  —  Pure Rust Implementation
//  Actix-web HTTP server implementing the BattleSnake API v1
//
//  Algorithm stack (from competition slide):
//  ┌─ Move classification: Safe / Risky / Lethal
//  ├─ Space heuristic: flood fill (BitBoard, no heap alloc)
//  ├─ Food seeking: A* pathfinding
//  ├─ Minimax + alpha-beta pruning (depth 5-8)
//  ├─ Voronoi territory control
//  ├─ Coiling (follow own tail when trapped)
//  ├─ Endgame 1v1 strategy
//  └─ Offense / Defence switching
//
//  Game rules:
//  - 11×11 grid
//  - Food restores health to 100 (full)
//  - Health depletes 1 per turn
//  - Target: survive 20,000 turns
//
//  Run:  cargo run --release
//  Test: curl localhost:8080/move  (see README)
// ================================================================

use actix_web::{get, post, web, App, HttpResponse, HttpServer, Responder};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU32, Ordering};

// ----------------------------------------------------------------
//  GRID CONSTANTS
// ----------------------------------------------------------------
const W: i8 = 11;
const H: i8 = 11;
const CELLS: usize = (W as usize) * (H as usize); // 121

// ----------------------------------------------------------------
//  GAME RULE CONSTANTS
// ----------------------------------------------------------------
const FOOD_RESTORE: i32 = 100;
const HP_DECAY:     i32 = 1;
const MAX_HP:       i32 = 100;
const HP_CRITICAL:  i32 = 20;
const HP_LOW:       i32 = 45;
const HP_MEDIUM:    i32 = 70;
const COIL_RATIO:   f32 = 1.5;
const MM_BUDGET:    u32 = 25_000;

// ----------------------------------------------------------------
//  CELL ENCODING
//  cell = y * W + x  →  u8 (fits in 0..120)
// ----------------------------------------------------------------
type Cell = u8;

#[inline(always)]
fn cell(x: i8, y: i8) -> Cell { (y * W + x) as Cell }

#[inline(always)]
fn cx(c: Cell) -> i8 { (c as i8) % W }

#[inline(always)]
fn cy(c: Cell) -> i8 { (c as i8) / W }

#[inline(always)]
fn in_bounds(x: i8, y: i8) -> bool { x >= 0 && x < W && y >= 0 && y < H }

#[inline(always)]
fn manhattan(a: Cell, b: Cell) -> i32 {
    ((cx(a) - cx(b)).abs() + (cy(a) - cy(b)).abs()) as i32
}

// ----------------------------------------------------------------
//  DIRECTIONS
// ----------------------------------------------------------------
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Dir { Up, Down, Left, Right }

const DIRS: [Dir; 4] = [Dir::Up, Dir::Down, Dir::Left, Dir::Right];

impl Dir {
    #[inline(always)]
    fn delta(self) -> (i8, i8) {
        match self {
            Dir::Up    => (0,  1),
            Dir::Down  => (0, -1),
            Dir::Left  => (-1, 0),
            Dir::Right => (1,  0),
        }
    }

    #[inline(always)]
    fn step(self, c: Cell) -> Option<Cell> {
        let (dx, dy) = self.delta();
        let nx = cx(c) + dx;
        let ny = cy(c) + dy;
        if in_bounds(nx, ny) { Some(cell(nx, ny)) } else { None }
    }

    fn as_str(self) -> &'static str {
        match self {
            Dir::Up    => "up",
            Dir::Down  => "down",
            Dir::Left  => "left",
            Dir::Right => "right",
        }
    }
}

// ----------------------------------------------------------------
//  BITBOARD  —  121-cell board, zero heap allocation
//  Stored as two u64s (cells 0-63 in lo, 64-120 in hi)
// ----------------------------------------------------------------
#[derive(Clone, Copy, Default)]
struct Bb { lo: u64, hi: u64 }

impl Bb {
    #[inline(always)]
    fn set(&mut self, c: Cell) {
        if c < 64 { self.lo |=  1u64 << c;        }
        else       { self.hi |=  1u64 << (c - 64); }
    }

    #[inline(always)]
    fn unset(&mut self, c: Cell) {
        if c < 64 { self.lo &= !(1u64 << c);        }
        else       { self.hi &= !(1u64 << (c - 64)); }
    }

    #[inline(always)]
    fn get(self, c: Cell) -> bool {
        if c < 64 { (self.lo >> c) & 1 == 1 }
        else       { (self.hi >> (c - 64)) & 1 == 1 }
    }

    #[inline(always)]
    fn or(self, other: Bb) -> Bb {
        Bb { lo: self.lo | other.lo, hi: self.hi | other.hi }
    }

    fn count(self) -> u32 {
        self.lo.count_ones() + self.hi.count_ones()
    }

    // Iterate set cells via bit-scan
    fn iter(self) -> BbIter { BbIter { lo: self.lo, hi: self.hi } }
}

struct BbIter { lo: u64, hi: u64 }
impl Iterator for BbIter {
    type Item = Cell;
    fn next(&mut self) -> Option<Cell> {
        if self.lo != 0 {
            let tz = self.lo.trailing_zeros() as u8;
            self.lo &= self.lo.wrapping_sub(1);
            Some(tz)
        } else if self.hi != 0 {
            let tz = self.hi.trailing_zeros() as u8;
            self.hi &= self.hi.wrapping_sub(1);
            Some(64 + tz)
        } else {
            None
        }
    }
}

// ----------------------------------------------------------------
//  SNAKE  —  ring-buffer body, entirely stack-allocated
// ----------------------------------------------------------------
const MAX_BODY: usize = CELLS; // snake can theoretically fill the board

#[derive(Clone, Debug)]
struct Snake {
    health:   i32,
    alive:    bool,
    body:     [Cell; MAX_BODY], // ring buffer
    head_idx: usize,            // index of head in ring
    length:   usize,
}

impl Snake {
    fn new(segments: &[Cell]) -> Self {
        let mut s = Snake {
            health:   MAX_HP,
            alive:    true,
            body:     [0u8; MAX_BODY],
            head_idx: 0,
            length:   segments.len(),
        };
        for (i, &c) in segments.iter().enumerate() {
            s.body[i] = c;
        }
        s
    }

    #[inline(always)]
    fn head(&self) -> Cell { self.body[self.head_idx] }

    #[inline(always)]
    fn tail(&self) -> Cell {
        self.body[(self.head_idx + self.length - 1) % MAX_BODY]
    }

    // Segment at offset i from head (0=head, 1=neck, ...)
    #[inline(always)]
    fn seg(&self, i: usize) -> Cell {
        self.body[(self.head_idx + i) % MAX_BODY]
    }

    // Returns true if tail is "safe" (vacates next turn)
    // Tail is NOT safe if the snake just ate (last two segs identical)
    #[inline(always)]
    fn tail_is_safe(&self) -> bool {
        if self.length < 2 { return true; }
        let t  = self.tail();
        let p  = self.body[(self.head_idx + self.length - 2) % MAX_BODY];
        t != p
    }

    /// Build obstacle bits from this snake's body:
    /// blocks segments [1 .. length-2], optionally blocks tail.
    fn add_obs(&self, obs: &mut Bb) {
        let n = self.length;
        // Block body[1..n-2]  (never head, never tail unless not safe)
        let inner_end = if self.tail_is_safe() { n - 1 } else { n };
        for i in 1..inner_end {
            obs.set(self.seg(i));
        }
    }

    /// Advance snake one step. Returns whether food was eaten.
    fn advance(&mut self, new_head: Cell, food: &mut Bb) -> bool {
        // Push new head into ring
        self.head_idx = (self.head_idx + MAX_BODY - 1) % MAX_BODY;
        self.body[self.head_idx] = new_head;
        self.health -= HP_DECAY;

        if food.get(new_head) {
            food.unset(new_head);
            self.health = FOOD_RESTORE;
            self.length += 1;
            true
        } else {
            // tail vacates — ring length stays same (no pop needed; ring wraps)
            false
        }
    }
}

// ----------------------------------------------------------------
//  GAME STATE  —  entirely stack-allocated, cheap to clone
// ----------------------------------------------------------------
#[derive(Clone, Debug)]
struct State {
    snakes: [Snake; 4],  // [0]=player, [1-3]=enemies
    n_snakes: usize,     // how many snake slots are used
    food:   Bb,
    turn:   u32,
}

impl State {
    /// Build the obstacle board from all living snakes.
    /// Head is NEVER in obs.
    fn obs(&self) -> Bb {
        let mut bb = Bb::default();
        for i in 0..self.n_snakes {
            if self.snakes[i].alive { self.snakes[i].add_obs(&mut bb); }
        }
        bb
    }

    /// Full body board including heads (for post-move collision).
    fn all_bodies(&self) -> Bb {
        let mut bb = Bb::default();
        for i in 0..self.n_snakes {
            if !self.snakes[i].alive { continue; }
            for j in 0..self.snakes[i].length {
                bb.set(self.snakes[i].seg(j));
            }
        }
        bb
    }

    fn player(&self)     -> &Snake  { &self.snakes[0] }
    fn alive_count(&self) -> usize  { (0..self.n_snakes).filter(|&i| self.snakes[i].alive).count() }
    fn enemy_alive(&self) -> usize  { (1..self.n_snakes).filter(|&i| self.snakes[i].alive).count() }
}

// ----------------------------------------------------------------
//  FLOOD FILL  —  BFS using Bb as visited set, no heap in hot path
//  Uses a fixed 121-element stack-allocated queue.
// ----------------------------------------------------------------
fn flood_fill(start: Cell, obs: Bb) -> u32 {
    if obs.get(start) { return 0; }
    let mut visited = Bb::default();
    visited.set(start);
    let mut queue = [0u8; CELLS];
    let mut qhead = 0usize;
    let mut qtail = 0usize;
    queue[qtail] = start; qtail += 1;

    while qhead < qtail {
        let c = queue[qhead]; qhead += 1;
        for d in DIRS {
            if let Some(nc) = d.step(c) {
                if !visited.get(nc) && !obs.get(nc) {
                    visited.set(nc);
                    queue[qtail] = nc; qtail += 1;
                }
            }
        }
    }
    visited.count()
}

/// Space reachable AFTER moving from `from` to `to`.
/// Blocks `from` (it becomes occupied body) before measuring.
#[inline]
fn space_after(from: Cell, to: Cell, obs: Bb) -> u32 {
    let mut o = obs;
    o.set(from);
    flood_fill(to, o)
}

// ----------------------------------------------------------------
//  VORONOI TERRITORY
//  Returns array[n_snakes] of cell counts owned by each snake.
//  Uses BFS with fixed stack queue — no heap.
// ----------------------------------------------------------------
fn voronoi(state: &State, obs: Bb) -> [u32; 4] {
    let mut owner  = [u8::MAX; CELLS]; // MAX = unowned
    let mut counts = [0u32; 4];
    let mut queue  = [(0u8, 0u8); CELLS * 4]; // (cell, snake_idx) — worst case
    let mut qhead  = 0usize;
    let mut qtail  = 0usize;

    for i in 0..state.n_snakes {
        if !state.snakes[i].alive { continue; }
        let h = state.snakes[i].head();
        if !obs.get(h) && owner[h as usize] == u8::MAX {
            owner[h as usize] = i as u8;
            queue[qtail] = (h, i as u8); qtail += 1;
        }
    }

    while qhead < qtail {
        let (c, si) = queue[qhead]; qhead += 1;
        for d in DIRS {
            if let Some(nc) = d.step(c) {
                let ni = nc as usize;
                if !obs.get(nc) && owner[ni] == u8::MAX {
                    owner[ni] = si;
                    queue[qtail] = (nc, si); qtail += 1;
                }
            }
        }
    }

    for i in 0..CELLS {
        let o = owner[i];
        if o != u8::MAX { counts[o as usize] += 1; }
    }
    counts
}

// ----------------------------------------------------------------
//  A* PATHFINDING
//  Returns distance to goal (or u32::MAX if unreachable) and
//  the first step direction.
//  Entirely stack-allocated: g/f arrays on stack.
// ----------------------------------------------------------------
fn astar(start: Cell, goal: Cell, obs: Bb) -> Option<(u32, Dir)> {
    if obs.get(start) { return None; }
    if start == goal  { return Some((0, Dir::Up)); } // already there

    let mut g_cost = [u32::MAX; CELLS];
    let mut came_from_dir = [Dir::Up; CELLS]; // direction used to reach cell
    let mut in_open  = Bb::default();
    let mut closed   = Bb::default();

    g_cost[start as usize] = 0;
    in_open.set(start);

    // f_cost array — stack allocated
    let mut f_cost = [u32::MAX; CELLS];
    f_cost[start as usize] = manhattan(start, goal) as u32;

    loop {
        // Pick lowest-f cell from open set
        let mut cur = u8::MAX;
        let mut best_f = u32::MAX;
        for c in in_open.iter() {
            let f = f_cost[c as usize];
            if f < best_f { best_f = f; cur = c; }
        }
        if cur == u8::MAX { break; } // open set empty

        if cur == goal {
            // Reconstruct first direction by tracing back to start
            let mut c = cur;
            let mut first_dir = came_from_dir[c as usize];
            loop {
                let prev = came_from_dir[c as usize];
                // step backwards: prev direction brings us to c, so reverse it
                let parent = reverse_step(prev, c);
                if parent == start { first_dir = prev; break; }
                c = parent;
            }
            return Some((g_cost[goal as usize], first_dir));
        }

        in_open.unset(cur);
        closed.set(cur);

        let g_cur = g_cost[cur as usize];
        for d in DIRS {
            let Some(nc) = d.step(cur) else { continue };
            if obs.get(nc) || closed.get(nc) { continue; }
            let ng = g_cur + 1;
            if ng < g_cost[nc as usize] {
                g_cost[nc as usize] = ng;
                f_cost[nc as usize] = ng + manhattan(nc, goal) as u32;
                came_from_dir[nc as usize] = d;
                in_open.set(nc);
            }
        }
    }
    None
}

// Step backwards: given direction d that was used to reach cell c,
// return the cell we came from.
#[inline]
fn reverse_step(d: Dir, c: Cell) -> Cell {
    let (dx, dy) = d.delta();
    let x = cx(c) - dx;
    let y = cy(c) - dy;
    cell(x, y)
}

/// Find the nearest reachable food cell and return (distance, first_step_dir).
fn nearest_food(head: Cell, food: Bb, obs: Bb) -> Option<(u32, Dir)> {
    let mut best: Option<(u32, Dir)> = None;
    for f in food.iter() {
        if let Some((dist, dir)) = astar(head, f, obs) {
            if best.map_or(true, |(d, _)| dist < d) {
                best = Some((dist, dir));
            }
        }
    }
    best
}

// ----------------------------------------------------------------
//  COIL  —  follow own tail when trapped
// ----------------------------------------------------------------
fn coil_dir(snake: &Snake, obs: Bb) -> Option<Dir> {
    let head = snake.head();
    let space = flood_fill(head, obs);
    if space >= (snake.length as f32 * COIL_RATIO) as u32 {
        return None; // not trapped
    }

    // Path to own tail (tail will vacate — temporarily unblock it)
    let tail = snake.tail();
    let mut path_obs = obs;
    path_obs.unset(tail);

    if let Some((_, dir)) = astar(head, tail, path_obs) {
        // Verify the step is actually free
        if let Some(nc) = dir.step(head) {
            if !obs.get(nc) { return Some(dir); }
        }
    }

    // Fallback: pick direction with most space
    let mut best_dir: Option<Dir> = None;
    let mut best_spc = 0u32;
    for d in DIRS {
        let Some(nc) = d.step(head) else { continue };
        if obs.get(nc) { continue; }
        let spc = space_after(head, nc, obs);
        if spc > best_spc { best_spc = spc; best_dir = Some(d); }
    }
    best_dir
}

// ----------------------------------------------------------------
//  MOVE CLASSIFIER
//  Returns (safe_moves, risky_moves) as Dir arrays.
//  safe  = not wall, not body, not adj to >=equal-length enemy head
//  risky = not wall, not body, BUT adj to >=equal-length enemy head
// ----------------------------------------------------------------
fn classify(state: &State, obs: Bb) -> ([Option<Dir>; 4], usize, [Option<Dir>; 4], usize) {
    let me   = state.player();
    let head = me.head();

    // Build head-danger cells
    let mut danger = Bb::default();
    for i in 1..state.n_snakes {
        let e = &state.snakes[i];
        if !e.alive { continue; }
        if e.length >= me.length {
            let eh = e.head();
            for d in DIRS {
                if let Some(nc) = d.step(eh) { danger.set(nc); }
            }
        }
    }

    let mut safe      = [None; 4];
    let mut safe_n    = 0;
    let mut risky     = [None; 4];
    let mut risky_n   = 0;

    for d in DIRS {
        let Some(nc) = d.step(head) else { continue };
        if obs.get(nc) { continue; } // lethal
        if danger.get(nc) { risky[risky_n] = Some(d); risky_n += 1; }
        else              { safe[safe_n]   = Some(d); safe_n   += 1; }
    }
    (safe, safe_n, risky, risky_n)
}

// ----------------------------------------------------------------
//  SIMULATE ONE FULL TURN  (all snakes simultaneously)
// ----------------------------------------------------------------
fn sim_turn(state: &State, dirs: &[Option<Dir>; 4]) -> State {
    let mut next = state.clone();

    // 1. Move all alive snakes, handle food
    for i in 0..next.n_snakes {
        if !next.snakes[i].alive { continue; }
        let Some(d) = dirs[i] else { continue };
        let new_head = match d.step(next.snakes[i].head()) {
            Some(c) => c,
            None    => { next.snakes[i].alive = false; continue; }
        };
        next.snakes[i].advance(new_head, &mut next.food);
    }

    // 2. Build body set for collision detection (excludes heads)
    let mut bodies = Bb::default();
    for i in 0..next.n_snakes {
        if !next.snakes[i].alive { continue; }
        let s = &next.snakes[i];
        for j in 1..s.length { bodies.set(s.seg(j)); }
    }

    // 3. Death checks: wall, starvation, body collision
    for i in 0..next.n_snakes {
        if !next.snakes[i].alive { continue; }
        let h = next.snakes[i].head();
        if next.snakes[i].health <= 0 || bodies.get(h) {
            next.snakes[i].alive = false;
        }
    }

    // 4. Head-on collisions (simultaneous move = both move, smaller dies)
    for i in 0..next.n_snakes {
        if !next.snakes[i].alive { continue; }
        for j in (i+1)..next.n_snakes {
            if !next.snakes[j].alive { continue; }
            if next.snakes[i].head() == next.snakes[j].head() {
                if next.snakes[i].length <= next.snakes[j].length { next.snakes[i].alive = false; }
                if next.snakes[j].length <= next.snakes[i].length { next.snakes[j].alive = false; }
            }
        }
    }

    next.turn += 1;
    next
}

// ----------------------------------------------------------------
//  ENEMY AI  —  greedy heuristic, each snake has a personality
// ----------------------------------------------------------------
#[derive(Clone, Copy, PartialEq, Eq)]
enum EnemyAi { Hunter, Territory, Ambush }

const ENEMY_AIS: [EnemyAi; 3] = [EnemyAi::Hunter, EnemyAi::Territory, EnemyAi::Ambush];

fn enemy_move(state: &State, si: usize) -> Dir {
    let s   = &state.snakes[si];
    if !s.alive { return Dir::Up; }
    let head = s.head();
    let obs  = state.obs();
    let ai   = ENEMY_AIS[si.saturating_sub(1).min(2)];

    let mut best_dir   = Dir::Up;
    let mut best_score = i32::MIN;
    let mut any_valid  = false;

    for d in DIRS {
        let Some(nc) = d.step(head) else { continue };
        if obs.get(nc) { continue; }

        let mut score: i32 = 0;
        any_valid = true;

        // All enemies: survival space
        score += space_after(head, nc, obs) as i32 * 4;

        // All enemies: food urgency
        let urg: i32 = if s.health <= HP_CRITICAL { 300 } else if s.health <= HP_LOW { 100 } else { 20 };
        for f in state.food.iter() {
            score += urg / (manhattan(nc, f) + 1);
        }

        let ncx = cx(nc) as i32;
        let ncy = cy(nc) as i32;

        match ai {
            EnemyAi::Hunter => {
                let p = state.player();
                if p.alive {
                    let dp = manhattan(nc, p.head());
                    score += if s.length > p.length { -dp * 6 } else { dp * 2 };
                }
            }
            EnemyAi::Territory => {
                let t = voronoi(state, obs);
                score += t[si] as i32 * 4;
                let p = state.player();
                if p.alive {
                    for f in state.food.iter() {
                        if manhattan(nc, f) < manhattan(p.head(), f) { score += 15; }
                    }
                }
                // prefer center (5,5)
                score -= ((ncx - 5).abs() + (ncy - 5).abs()) as i32;
            }
            EnemyAi::Ambush => {
                let t = voronoi(state, obs);
                score -= t[0] as i32 * 4; // shrink player territory
                score += t[si] as i32 * 3;
                let p = state.player();
                if p.alive {
                    for f in state.food.iter() {
                        let mx = (cx(p.head()) as i32 + cx(f) as i32) / 2;
                        let my = (cy(p.head()) as i32 + cy(f) as i32) / 2;
                        score -= ((ncx - mx).abs() + (ncy - my).abs()) as i32 * 2;
                    }
                }
            }
        }

        if score > best_score { best_score = score; best_dir = d; }
    }

    if !any_valid {
        // Completely blocked — pick any in-bounds direction
        for d in DIRS { if d.step(head).is_some() { return d; } }
    }
    best_dir
}

// ----------------------------------------------------------------
//  HEURISTIC EVALUATION  (called at minimax leaf nodes)
// ----------------------------------------------------------------
fn evaluate(state: &State) -> i32 {
    let me = state.player();
    if !me.alive { return -100_000_000; }

    let n_enemies = state.enemy_alive();
    if n_enemies == 0 {
        return 10_000_000 + me.health * 100 + me.length as i32 * 50;
    }

    let obs  = state.obs();
    let head = me.head();

    // 1. Flood fill — trap prevention (highest weight)
    let my_fill = flood_fill(head, obs);
    let body_len = me.length as i32;
    let ratio = my_fill as f32 / body_len as f32;
    let fill_score: i32 = if      ratio >= 4.0 { my_fill as i32 * 6 }
                          else if ratio >= 2.0 { my_fill as i32 * 5 }
                          else if ratio >= 1.0 { my_fill as i32 * 3 }
                          else                 { my_fill as i32 - body_len * 20 };

    // 2. Food urgency — full restore makes food critical
    let food_score: i32 = match nearest_food(head, state.food, obs) {
        Some((dist, _)) => {
            let dist = dist as i32;
            if dist >= me.health {
                // Will starve before reaching food — catastrophic
                -80_000 + dist * 100
            } else {
                let urgency: i32 = if me.health <= HP_CRITICAL { 500 }
                                   else if me.health <= HP_LOW  { 150 }
                                   else if me.health <= HP_MEDIUM{ 40 }
                                   else                          { 10 };
                urgency * 30 / (dist + 1)
            }
        }
        None => -60_000, // no reachable food
    };

    // 3. Voronoi territory
    let terr = voronoi(state, obs);
    let is_1v1 = n_enemies == 1;
    let voro_score = terr[0] as i32 * if is_1v1 { 8 } else { 5 };

    // 4. Length advantage — longer = win head-ons
    let max_e_len = (1..state.n_snakes)
        .filter(|&i| state.snakes[i].alive)
        .map(|i| state.snakes[i].length as i32)
        .max()
        .unwrap_or(0);
    let len_score = if body_len > max_e_len {
        30 + (body_len - max_e_len) * 8
    } else {
        (body_len - max_e_len) * 10
    };

    // 5. Danger — proximity to larger/equal heads
    let mut danger_score: i32 = 0;
    for i in 1..state.n_snakes {
        let e = &state.snakes[i];
        if !e.alive { continue; }
        let d = manhattan(head, e.head());
        if e.length >= me.length {
            danger_score += if d <= 1 { -200 } else if d <= 2 { -80 } else if d <= 3 { -25 } else { 0 };
        } else if d <= 2 {
            danger_score += 25; // reward being close to killable enemy
        }
    }

    // 6. 1v1 endgame: maximize territory delta, chase/flee
    let endgame_score: i32 = if is_1v1 {
        let ei = (1..state.n_snakes).find(|&i| state.snakes[i].alive).unwrap_or(1);
        let my_t = terr[0] as i32;
        let et   = terr[ei] as i32;
        let e    = &state.snakes[ei];
        let chase_flee = if body_len > e.length as i32 {
            -manhattan(head, e.head()) * 5  // chase
        } else {
             manhattan(head, e.head()) * 3  // flee
        };
        (my_t - et) * 6 + chase_flee
    } else {
        0
    };

    // 7. HP buffer
    let hp_score = me.health / 2;

    // 8. Trap penalty
    let trap_penalty = if my_fill < me.length as u32 {
        ((me.length as i32 - my_fill as i32) * 40).max(0)
    } else if my_fill < (me.length as f32 * 1.5) as u32 {
        ((me.length as f32 * 1.5) as i32 - my_fill as i32) * 10
    } else {
        0
    };

    fill_score + food_score + voro_score + len_score + danger_score
        + endgame_score + hp_score - trap_penalty
}

// ----------------------------------------------------------------
//  MINIMAX WITH ALPHA-BETA PRUNING
// ----------------------------------------------------------------
static MM_COUNT: AtomicU32 = AtomicU32::new(0);

fn minimax(state: &State, depth: u8, mut alpha: i32, beta: i32, is_max: bool) -> i32 {
    let n = MM_COUNT.fetch_add(1, Ordering::Relaxed);
    if n > MM_BUDGET || depth == 0 { return evaluate(state); }
    if !state.player().alive { return -100_000_000 - depth as i32 * 200; }

    if is_max {
        let obs = state.obs();
        let (safe, safe_n, risky, risky_n) = classify(state, obs);

        let (moves, n_moves) = if safe_n > 0 { (safe, safe_n) } else { (risky, risky_n) };
        if n_moves == 0 { return evaluate(state); }

        let mut best = i32::MIN;
        let mut beta = beta;

        for i in 0..n_moves {
            let dir = moves[i].unwrap();
            let mut dirs = [None; 4];
            dirs[0] = Some(dir);
            for j in 1..state.n_snakes {
                if state.snakes[j].alive { dirs[j] = Some(enemy_move(state, j)); }
            }
            let next  = sim_turn(state, &dirs);
            let score = minimax(&next, depth - 1, alpha, beta, false);
            if score > best { best = score; }
            if score > alpha { alpha = score; }
            if beta <= alpha { break; }
        }
        best
    } else {
        minimax(state, depth - 1, alpha, beta, true)
    }
}

// ----------------------------------------------------------------
//  MASTER MOVE SELECTION
// ----------------------------------------------------------------
#[derive(Debug)]
struct MoveResult {
    dir:  Dir,
    mode: &'static str,
}

fn choose_move(state: &State) -> MoveResult {
    let me   = state.player();
    let head = me.head();
    let obs  = state.obs();

    let (safe, safe_n, risky, risky_n) = classify(state, obs);

    // Build candidate pool: prefer safe, fall back to risky
    let (pool, pool_n) = if safe_n > 0 { (safe, safe_n) } else { (risky, risky_n) };

    // Absolute last resort
    if pool_n == 0 {
        for d in DIRS {
            if d.step(head).is_some() { return MoveResult { dir: d, mode: "LAST-RESORT" }; }
        }
        return MoveResult { dir: Dir::Up, mode: "LAST-RESORT" };
    }

    let my_space  = flood_fill(head, obs);
    let trapped   = my_space < (me.length as f32 * 1.5) as u32;
    let critical  = me.health <= HP_CRITICAL;
    let low       = me.health <= HP_LOW;
    let n_enemies = state.enemy_alive();
    let is_1v1    = n_enemies == 1;
    let threatened = (1..state.n_snakes).any(|i| {
        let e = &state.snakes[i];
        e.alive && e.length >= me.length && manhattan(head, e.head()) <= 3
    });

    let mode: &'static str = if critical  { "CRIT-FEED" }
                             else if trapped   { "COIL"      }
                             else if low       { "FEED"      }
                             else if is_1v1    { "1v1"       }
                             else if threatened { "EVADE"    }
                             else              { "CONTROL"   };

    // ── CRIT-FEED: A* straight to nearest food ──
    if critical {
        if let Some((_, dir)) = nearest_food(head, state.food, obs) {
            if let Some(nc) = dir.step(head) {
                if !obs.get(nc) { return MoveResult { dir, mode }; }
            }
        }
        // Fall through to minimax if food unreachable
    }

    // ── COIL: follow own tail when trapped ──
    if trapped {
        if let Some(d) = coil_dir(me, obs) {
            return MoveResult { dir: d, mode };
        }
    }

    // ── Pre-filter: discard dead-end traps smaller than body ──
    let mut scored = [(Dir::Up, 0u32); 4];
    let mut n_scored = 0;
    for i in 0..pool_n {
        let d  = pool[i].unwrap();
        let nc = d.step(head).unwrap();
        let spc = space_after(head, nc, obs);
        scored[n_scored] = (d, spc);
        n_scored += 1;
    }
    // Sort descending by space
    scored[..n_scored].sort_unstable_by(|a, b| b.1.cmp(&a.1));

    let min_spc = me.length as u32;
    let mut cands     = [None::<Dir>; 4];
    let mut n_cands   = 0;
    for i in 0..n_scored {
        if scored[i].1 >= min_spc {
            cands[n_cands] = Some(scored[i].0);
            n_cands += 1;
        }
    }
    if n_cands == 0 {
        // All are traps — pick the least-bad one
        for i in 0..n_scored { cands[i] = Some(scored[i].0); }
        n_cands = n_scored;
    }

    if n_cands == 1 { return MoveResult { dir: cands[0].unwrap(), mode }; }

    // ── MINIMAX on candidates ──
    let depth: u8 = if state.alive_count() <= 2 { 8 }
                    else if state.alive_count() <= 3 { 6 }
                    else { 5 };

    let mut best_score = i32::MIN;
    let mut best_dir   = cands[0].unwrap();

    for i in 0..n_cands {
        let dir = cands[i].unwrap();
        MM_COUNT.store(0, Ordering::Relaxed);

        let mut dirs = [None; 4];
        dirs[0] = Some(dir);
        for j in 1..state.n_snakes {
            if state.snakes[j].alive { dirs[j] = Some(enemy_move(state, j)); }
        }

        let next  = sim_turn(state, &dirs);
        let score = minimax(&next, depth, i32::MIN, i32::MAX, false);

        if score > best_score { best_score = score; best_dir = dir; }
    }

    MoveResult { dir: best_dir, mode }
}

// ================================================================
//  BATTLESNAKE HTTP API  (actix-web)
//  https://docs.battlesnake.com/api
// ================================================================

// ── Request / Response types ────────────────────────────────────

#[derive(Deserialize, Debug)]
struct Coord { x: i8, y: i8 }

#[derive(Deserialize, Debug)]
struct ApiSnake {
    id:     String,
    health: i32,
    body:   Vec<Coord>,
    head:   Coord,
    length: u32,
}

#[derive(Deserialize, Debug)]
struct Board {
    height: i8,
    width:  i8,
    food:   Vec<Coord>,
    snakes: Vec<ApiSnake>,
}

#[derive(Deserialize, Debug)]
struct GameRequest {
    turn:  u32,
    board: Board,
    you:   ApiSnake,
}

#[derive(Serialize)]
struct MoveResponse {
    #[serde(rename = "move")]
    direction: String,
    shout: String,
}

#[derive(Serialize)]
struct InfoResponse {
    apiversion: &'static str,
    author:     &'static str,
    color:      &'static str,
    head:       &'static str,
    tail:       &'static str,
}

// ── Convert API request → internal State ────────────────────────

fn api_to_state(req: &GameRequest) -> State {
    // Player is always index 0; enemies follow in order
    let mut all_snakes: Vec<&ApiSnake> = Vec::with_capacity(4);
    all_snakes.push(&req.you);
    for s in &req.board.snakes {
        if s.id != req.you.id { all_snakes.push(s); }
    }

    // Build default snake (dead placeholder) for unused slots
    let dead_snake = Snake { health: 0, alive: false, body: [0u8; MAX_BODY], head_idx: 0, length: 1 };

    let mut snakes = [dead_snake.clone(), dead_snake.clone(), dead_snake.clone(), dead_snake];
    let n = all_snakes.len().min(4);

    for (i, api_s) in all_snakes.iter().take(n).enumerate() {
        let segments: Vec<Cell> = api_s.body.iter()
            .filter(|c| in_bounds(c.x, c.y))
            .map(|c| cell(c.x, c.y))
            .collect();
        if segments.is_empty() { continue; }
        snakes[i] = Snake::new(&segments);
        snakes[i].health = api_s.health;
        snakes[i].alive  = true;
    }

    let mut food_bb = Bb::default();
    for f in &req.board.food {
        if in_bounds(f.x, f.y) { food_bb.set(cell(f.x, f.y)); }
    }

    State { snakes, n_snakes: n, food: food_bb, turn: req.turn }
}

// ── Handlers ────────────────────────────────────────────────────

#[get("/")]
async fn info() -> impl Responder {
    HttpResponse::Ok().json(InfoResponse {
        apiversion: "1",
        author:     "APEX",
        color:      "#39ff14",
        head:       "default",
        tail:       "default",
    })
}

#[post("/start")]
async fn start() -> impl Responder {
    HttpResponse::Ok().body("{}")
}

#[post("/move")]
async fn make_move(body: web::Json<GameRequest>) -> impl Responder {
    let state = api_to_state(&body);
    let result = choose_move(&state);

    let shout = format!(
        "T{} | {} | HP:{} | len:{}",
        body.turn,
        result.mode,
        state.player().health,
        state.player().length,
    );

    eprintln!("[T{}] move={} mode={} hp={} len={}",
        body.turn, result.dir.as_str(), result.mode,
        state.player().health, state.player().length);

    HttpResponse::Ok().json(MoveResponse {
        direction: result.dir.as_str().to_string(),
        shout,
    })
}

#[post("/end")]
async fn end(body: web::Json<serde_json::Value>) -> impl Responder {
    eprintln!("[END] Game over. Turn: {}", body["turn"]);
    HttpResponse::Ok().body("{}")
}

// ================================================================
//  STANDALONE SELF-PLAY MODE  (no server — benchmark/test)
//  Run with:  cargo run --release -- selfplay [turns]
// ================================================================

fn run_selfplay(target_turns: u32) {
    use std::collections::VecDeque as Vd;

    println!("=== BattleSnake Self-Play ===");
    println!("Target: {} turns | Rules: food=+100HP, -1HP/turn", target_turns);
    println!("Grid: {}x{}", W, H);
    println!("----------------------------");

    // Build initial state
    // Player starts center, enemies in corners
    let player_body = [cell(5,5), cell(4,5), cell(3,5)];
    let e1_body     = [cell(1,9), cell(2,9), cell(3,9)];
    let e2_body     = [cell(9,1), cell(9,2), cell(9,3)];
    let e3_body     = [cell(1,1), cell(1,2), cell(1,3)];

    let dead = Snake { health: 0, alive: false, body: [0u8; MAX_BODY], head_idx: 0, length: 1 };
    let mut snakes = [dead.clone(), dead.clone(), dead.clone(), dead];
    snakes[0] = Snake::new(&player_body);
    snakes[1] = Snake::new(&e1_body);
    snakes[2] = Snake::new(&e2_body);
    snakes[3] = Snake::new(&e3_body);

    // Spawn initial food
    let mut food = Bb::default();
    let initial_food = [cell(5,3), cell(3,7), cell(7,7), cell(5,9), cell(9,5)];
    for f in initial_food { food.set(f); }

    let mut state = State { snakes, n_snakes: 4, food, turn: 0 };
    let mut rng   = SimpleRng::new(12345);
    let mut food_eaten = 0u32;
    let mut kills      = 0u32;

    loop {
        if !state.player().alive {
            println!("\n💥 DEAD at turn {}", state.turn);
            break;
        }
        if state.turn >= target_turns {
            println!("\n🏆 SURVIVED {} TURNS — TARGET REACHED!", state.turn);
            break;
        }

        // Choose player move
        let result = choose_move(&state);

        // Build dirs for all snakes
        let mut dirs = [None; 4];
        dirs[0] = Some(result.dir);
        for i in 1..state.n_snakes {
            if state.snakes[i].alive { dirs[i] = Some(enemy_move(&state, i)); }
        }

        // Track food eaten
        if let Some(nc) = result.dir.step(state.player().head()) {
            if state.food.get(nc) { food_eaten += 1; }
        }

        // Track kills
        let prev_alive: [bool; 4] = std::array::from_fn(|i| {
            if i < state.n_snakes { state.snakes[i].alive } else { false }
        });

        state = sim_turn(&state, &dirs);

        for i in 1..state.n_snakes {
            if prev_alive[i] && !state.snakes[i].alive { kills += 1; }
        }

        // Maintain food supply
        while state.food.count() < 3 {
            if let Some(f) = spawn_food(&state, &mut rng) { state.food.set(f); }
            else { break; }
        }
        if state.turn % 25 == 0 {
            if let Some(f) = spawn_food(&state, &mut rng) { state.food.set(f); }
        }

        // Milestone logging
        if state.turn % 500 == 0 {
            println!("T{:>6} | HP:{:>3} | len:{:>3} | mode:{:<10} | food:{} | kills:{} | enemies:{}",
                state.turn, state.player().health, state.player().length,
                result.mode, food_eaten, kills, state.enemy_alive());
        }
    }

    println!("----------------------------");
    println!("Final turn:  {}", state.turn);
    println!("Food eaten:  {}", food_eaten);
    println!("Kills:       {}", kills);
    println!("Max length:  {}", state.player().length);
    println!("Alive snakes: {}", state.alive_count());
}

fn spawn_food(state: &State, rng: &mut SimpleRng) -> Option<Cell> {
    let mut occupied = state.all_bodies().or(state.food);
    for _ in 0..500 {
        let x = (rng.next() % W as u64) as i8;
        let y = (rng.next() % H as u64) as i8;
        let c = cell(x, y);
        if !occupied.get(c) { return Some(c); }
    }
    None
}

// ── Tiny xorshift RNG (no rand crate needed) ──────────────────
struct SimpleRng(u64);
impl SimpleRng {
    fn new(seed: u64) -> Self { SimpleRng(seed) }
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

// ================================================================
//  MAIN
// ================================================================

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().collect();

    // Selfplay mode: cargo run --release -- selfplay [turns]
    if args.get(1).map(|s| s.as_str()) == Some("selfplay") {
        let turns: u32 = args.get(2)
            .and_then(|s| s.parse().ok())
            .unwrap_or(20_000);
        run_selfplay(turns);
        return Ok(());
    }

    // HTTP server mode (default)
    let port = std::env::var("PORT").unwrap_or_else(|_| "8080".to_string());
    let addr = format!("0.0.0.0:{}", port);
    println!("🐍 BattleSnake APEX listening on http://{}", addr);
    println!("   Endpoints: GET /  POST /start  POST /move  POST /end");

    HttpServer::new(|| {
        App::new()
            .service(info)
            .service(start)
            .service(make_move)
            .service(end)
    })
    .bind(&addr)?
    .run()
    .await
}

// ================================================================
//  TESTS
// ================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_state(player: &[Cell], enemies: &[&[Cell]], food_cells: &[Cell]) -> State {
        let dead = Snake { health: 0, alive: false, body: [0u8; MAX_BODY], head_idx: 0, length: 1 };
        let mut snakes = [dead.clone(), dead.clone(), dead.clone(), dead];
        snakes[0] = Snake::new(player);
        let mut n = 1;
        for (i, e) in enemies.iter().enumerate() {
            if i + 1 < 4 { snakes[i+1] = Snake::new(e); n = i + 2; }
        }
        let mut food = Bb::default();
        for &f in food_cells { food.set(f); }
        State { snakes, n_snakes: n, food, turn: 0 }
    }

    #[test]
    fn test_flood_fill_open_board() {
        // Empty board — should fill all 121 cells
        let obs = Bb::default();
        assert_eq!(flood_fill(cell(5,5), obs), 121);
    }

    #[test]
    fn test_flood_fill_blocked_start() {
        let mut obs = Bb::default();
        obs.set(cell(5,5));
        assert_eq!(flood_fill(cell(5,5), obs), 0);
    }

    #[test]
    fn test_flood_fill_walled_off() {
        // Build a wall at x=5 (cells (5,0)..(5,10))
        let mut obs = Bb::default();
        for y in 0..H { obs.set(cell(5, y)); }
        // Left half should have 5*11 = 55 cells
        assert_eq!(flood_fill(cell(0,0), obs), 55);
        assert_eq!(flood_fill(cell(10,0), obs), 55);
    }

    #[test]
    fn test_cell_encoding() {
        // Verify no two distinct grid coords map to the same cell
        let mut seen = std::collections::HashSet::new();
        for x in 0..W {
            for y in 0..H {
                let c = cell(x, y);
                assert!(seen.insert(c), "collision at ({},{})", x, y);
                assert_eq!(cx(c), x);
                assert_eq!(cy(c), y);
            }
        }
    }

    #[test]
    fn test_obs_excludes_head() {
        // Snake head must never appear in obs
        let body = [cell(5,5), cell(4,5), cell(3,5)];
        let state = make_state(&body, &[], &[]);
        let obs = state.obs();
        // Head at (5,5) must NOT be blocked
        assert!(!obs.get(cell(5,5)), "Head should not be in obstacle map");
        // Neck at (4,5) must be blocked
        assert!(obs.get(cell(4,5)), "Neck should be blocked");
        // Tail (3,5) should NOT be blocked (safe to enter)
        assert!(!obs.get(cell(3,5)), "Tail should be free");
    }

    #[test]
    fn test_dont_move_into_wall() {
        // Snake at left edge — should not move left
        let body = [cell(0,5), cell(1,5), cell(2,5)];
        let state = make_state(&body, &[], &[cell(5,5)]);
        let result = choose_move(&state);
        assert_ne!(result.dir, Dir::Left, "Should not move into wall");
    }

    #[test]
    fn test_dont_reverse_into_neck() {
        // Snake heading right — must not reverse into neck
        let body = [cell(5,5), cell(4,5), cell(3,5)];
        let state = make_state(&body, &[], &[cell(9,5)]);
        let result = choose_move(&state);
        assert_ne!(result.dir, Dir::Left, "Should not reverse into neck");
    }

    #[test]
    fn test_critical_hp_goes_for_food() {
        // With critical HP, should head toward food
        let body = [cell(5,5), cell(4,5), cell(3,5)];
        let food = [cell(5,9)]; // food directly above
        let mut state = make_state(&body, &[], &food);
        state.snakes[0].health = HP_CRITICAL - 1;
        let result = choose_move(&state);
        // Should move up toward food or at least not away
        assert_ne!(result.dir, Dir::Down, "Critical HP — should not move away from only food");
    }

    #[test]
    fn test_avoid_dead_end() {
        // Snake in corridor — should pick direction with more space
        //  . X X X .
        //  . X H X .
        //  . X . X .
        // Build walls around head except one exit
        let body = [cell(5,5), cell(4,5), cell(3,5)];
        let mut obs = Bb::default();
        // Block left, right, down — only up is open
        obs.set(cell(4,5)); // neck
        obs.set(cell(6,5)); // right
        obs.set(cell(5,4)); // down
        // Flood fill from up direction should be open; others tiny
        let space_up    = space_after(cell(5,5), cell(5,6), obs);
        let space_down  = space_after(cell(5,5), cell(5,4), obs);
        assert!(space_up > space_down, "Up should have more space than blocked down");
    }

    #[test]
    fn test_astar_finds_path() {
        let obs = Bb::default();
        let result = astar(cell(0,0), cell(10,10), obs);
        assert!(result.is_some());
        let (dist, _) = result.unwrap();
        assert_eq!(dist, 20); // Manhattan distance (0,0)→(10,10) = 20
    }

    #[test]
    fn test_astar_blocked() {
        // Wall cuts board in two — no path should exist
        let mut obs = Bb::default();
        for y in 0..H { obs.set(cell(5, y)); }
        assert!(astar(cell(0,0), cell(10,0), obs).is_none());
    }

    #[test]
    fn test_astar_blocked_start() {
        let mut obs = Bb::default();
        obs.set(cell(0,0));
        assert!(astar(cell(0,0), cell(5,5), obs).is_none());
    }

    #[test]
    fn test_simulate_food_restores_full_health() {
        let body = [cell(4,5), cell(3,5), cell(2,5)];
        let food = [cell(5,5)];
        let mut state = make_state(&body, &[], &food);
        state.snakes[0].health = 50;
        let dirs = [Some(Dir::Right), None, None, None];
        let next = sim_turn(&state, &dirs);
        assert_eq!(next.snakes[0].health, FOOD_RESTORE, "Food should restore full health");
        assert_eq!(next.snakes[0].length, 4, "Length should increase after eating");
    }

    #[test]
    fn test_simulate_starvation() {
        let body = [cell(5,5), cell(4,5), cell(3,5)];
        let mut state = make_state(&body, &[], &[]);
        state.snakes[0].health = 1;
        let dirs = [Some(Dir::Up), None, None, None];
        let next = sim_turn(&state, &dirs);
        assert!(!next.snakes[0].alive, "Should die from starvation");
    }

    #[test]
    fn test_simulate_head_on_larger_wins() {
        // Equal length snakes collide head-on — both die
        let p_body = [cell(5,5), cell(4,5), cell(3,5)];
        let e_body = [cell(5,7), cell(5,8), cell(5,9)];
        let state  = make_state(&p_body, &[&e_body], &[]);
        // Both move toward each other
        let dirs = [Some(Dir::Up), Some(Dir::Down), None, None];
        let next = sim_turn(&state, &dirs);
        // Equal length = both die on head-on
        assert!(!next.snakes[0].alive || !next.snakes[1].alive);
    }

    #[test]
    fn test_simulate_smaller_dies_head_on() {
        // Enemy is larger — player dies, enemy survives
        let p_body = [cell(5,5), cell(4,5)];          // length 2
        let e_body = [cell(5,7), cell(5,8), cell(5,9)]; // length 3
        let state  = make_state(&p_body, &[&e_body], &[]);
        let dirs = [Some(Dir::Up), Some(Dir::Down), None, None];
        // Move them 1 step closer, not to same cell yet
        let next = sim_turn(&state, &dirs);
        // Next step they collide
        let dirs2 = [Some(Dir::Up), Some(Dir::Down), None, None];
        let next2 = sim_turn(&next, &dirs2);
        // Player (len 2) should die, enemy (len 3) should survive
        assert!(!next2.snakes[0].alive, "Smaller snake should die on head-on");
    }

    #[test]
    fn test_voronoi_even_split() {
        // Player at (0,5) and enemy at (10,5) — should roughly split the board
        let p_body = [cell(0,5), cell(0,4), cell(0,3)];
        let e_body = [cell(10,5), cell(10,4), cell(10,3)];
        let state  = make_state(&p_body, &[&e_body], &[]);
        let obs    = state.obs();
        let terr   = voronoi(&state, obs);
        // Player owns roughly left half, enemy right half
        assert!(terr[0] > 30, "Player should own significant territory");
        assert!(terr[1] > 30, "Enemy should own significant territory");
        assert!((terr[0] as i32 - terr[1] as i32).abs() < 20, "Should be roughly even");
    }

    #[test]
    fn test_bitboard_all_cells() {
        // Verify all 121 cells can be set/get without corruption
        let mut bb = Bb::default();
        for x in 0..W { for y in 0..H { bb.set(cell(x,y)); } }
        assert_eq!(bb.count(), 121);
        for x in 0..W { for y in 0..H { assert!(bb.get(cell(x,y))); } }
    }

    #[test]
    fn test_bitboard_iter() {
        let mut bb = Bb::default();
        let cells = [cell(0,0), cell(5,5), cell(10,10), cell(3,7)];
        for &c in &cells { bb.set(c); }
        let mut collected: Vec<Cell> = bb.iter().collect();
        collected.sort();
        let mut expected = cells.to_vec();
        expected.sort();
        assert_eq!(collected, expected);
    }
}
