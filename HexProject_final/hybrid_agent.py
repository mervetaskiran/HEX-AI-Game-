import random
import copy
import numpy as np
import heapq
import pickle
import os
import math
import time
from hex_engine import PLAYER_1, PLAYER_2, EMPTY

# --- CONSTANTS ---
INF = 9999

# --- RL WEIGHTS MANAGEMENT ---
RL_WEIGHTS = np.array([-4.0, 8.0, 2.0, 5.0, 3.0]) 
RL_ENABLED = False 

if os.path.exists("rl_weights.pkl"):
    try:
        with open("rl_weights.pkl", "rb") as f:
            RL_WEIGHTS = pickle.load(f)
            RL_ENABLED = True
            print(f"Loaded RL Weights: {RL_WEIGHTS}")
    except Exception as e:
        print(f"Could not load weights: {e}")

# --- A* (A-STAR) PATHFINDING ---
def a_star_distance(game, player):
    size = game.size
    pq = []
    dist = np.full((size, size), INF)
    
    def heuristic(r, c):
        if player == PLAYER_1: return (size - 1) - r
        else: return (size - 1) - c

    if player == PLAYER_1: 
        for c in range(size):
            cost = 0 if game.board[0, c] == player else 1
            if game.board[0, c] != -player:
                dist[0, c] = cost
                heapq.heappush(pq, (cost + heuristic(0, c), 0, c))
    else: 
        for r in range(size):
            cost = 0 if game.board[r, 0] == player else 1
            if game.board[r, 0] != -player:
                dist[r, 0] = cost
                heapq.heappush(pq, (cost + heuristic(r, 0), r, 0))
    
    directions = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]
    
    while pq:
        if not pq: break
        f_val, r, c = heapq.heappop(pq)
        
        if player == PLAYER_1 and r == size - 1: return dist[r, c]
        if player == PLAYER_2 and c == size - 1: return dist[r, c]
        
        if f_val - heuristic(r,c) > dist[r, c]: continue
        
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < size and 0 <= nc < size:
                cell = game.board[nr, nc]
                if cell == -player: continue 
                
                move_cost = 1
                if cell == player: move_cost = 0 
                
                if dist[r, c] + move_cost < dist[nr, nc]:
                    dist[nr, nc] = dist[r, c] + move_cost
                    heapq.heappush(pq, (dist[nr, nc] + heuristic(nr, nc), nr, nc))      
    return INF

# --- PATH VISUALIZATION ---
def get_visual_path(game, player):
    """
    Compute a visualization path for the given player.

    Design goals:
    - Path walks only on player's stones and empty cells (never on opponent stones).
    - Start from the player's current stones (cluster-based), not from an empty edge.
    - Prefer to stay close to the player's stone cluster.
    - Strongly avoid cells that are adjacent to opponent stones (wall effect).
    - After a path is found to the goal edge, extend it visually to both edges.
    """
    size = game.size
    directions = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]

    # --- 1) Player stones (cluster) ---
    stones = [(r, c) for r in range(size) for c in range(size)
              if game.board[r, c] == player]

    # Eğer henüz taş yoksa: ortadan düz bir çizgi çiz
    if not stones:
        path = []
        if player == PLAYER_1:
            col = size // 2
            for r in range(size):
                path.append((r, col))
        else:
            row = size // 2
            for c in range(size):
                path.append((row, c))
        return path

    # Küme merkezi (taşların ağırlık merkezi)
    cr = sum(r for r, _ in stones) / len(stones)
    cc = sum(c for _, c in stones) / len(stones)

    def heuristic(r, c):
        # Hedef kenara uzaklık
        if player == PLAYER_1:
            goal_dist = (size - 1) - r    # alta ne kadar uzak
        else:
            goal_dist = (size - 1) - c    # sağa ne kadar uzak
        # Taş kümesine uzaklık (path'i taşlara yakın tutmak için)
        cluster_dist = abs(r - cr) + abs(c - cc)
        return goal_dist + 0.4 * cluster_dist

    # --- 2) A* başlangıcı: tüm taşlar ---
    pq = []
    came_from = {}
    cost_so_far = {}

    for (r, c) in stones:
        start = (r, c)
        cost_so_far[start] = 0.0
        came_from[start] = None
        heapq.heappush(pq, (heuristic(r, c), start))

    final_node = None

    # --- 3) A* araması ---
    while pq:
        _, current = heapq.heappop(pq)
        r, c = current

        # Hedef kenara ulaştı mı?
        if (player == PLAYER_1 and r == size - 1) or \
           (player == PLAYER_2 and c == size - 1):
            final_node = current
            break

        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < size and 0 <= nc < size:
                cell = game.board[nr, nc]

                # Rakip taşın üstüne ASLA basma
                if cell == -player:
                    continue

                # Temel adım maliyeti
                step_cost = 0.0 if cell == player else 1.0

                # Rakip taşlara KOMŞU hücrelere ekstra ceza (duvar etkisi)
                neighbor_penalty = 0.0
                for dr2, dc2 in directions:
                    ar, ac = nr + dr2, nc + dc2
                    if 0 <= ar < size and 0 <= ac < size:
                        if game.board[ar, ac] == -player:
                            neighbor_penalty += 3.0

                new_cost = cost_so_far[current] + step_cost + neighbor_penalty
                next_node = (nr, nc)

                if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                    cost_so_far[next_node] = new_cost
                    priority = new_cost + heuristic(nr, nc)
                    heapq.heappush(pq, (priority, next_node))
                    came_from[next_node] = current

    # --- 4) Kümeden hedef kenara path'i geri kur ---
    path = []
    if final_node is not None:
        cur = final_node
        while cur is not None:
            path.append(cur)
            cur = came_from[cur]
        path.reverse()
    else:
        # Nadiren: hedef kenara ulaşamazsa, hedefe en yakın taşı tek path yap
        if player == PLAYER_1:
            best = min(stones, key=lambda rc: (size - 1) - rc[0])
        else:
            best = min(stones, key=lambda rc: (size - 1) - rc[1])
        path = [best]

    # --- 5) Path'i iki kenara görsel olarak uzat ---
    if path:
        first = path[0]
        last = path[-1]
        if player == PLAYER_1:
            # Üst kenara
            for r in range(first[0] - 1, -1, -1):
                path.insert(0, (r, first[1]))
            # Alt kenara
            for r in range(last[0] + 1, size):
                path.append((r, last[1]))
        else:
            # Sol kenara
            for c in range(first[1] - 1, -1, -1):
                path.insert(0, (first[0], c))
            # Sağ kenara
            for c in range(last[1] + 1, size):
                path.append((last[0], c))

    return path


def find_path_to_goal(game, player, start_stone):
    """
    Find path from a stone to the goal edge, aiming for the closest edge point.
    """
    size = game.size
    pq = []
    came_from = {}
    cost_so_far = {}
    
    # Start from the stone
    cost_so_far[start_stone] = 0
    came_from[start_stone] = None
    
    def heuristic(r, c):
        # For goal edge, prefer path that goes straight from stone
        if player == PLAYER_1:
            return (size - 1 - r)
        else:
            return (size - 1 - c)
    
    heapq.heappush(pq, (heuristic(start_stone[0], start_stone[1]), start_stone))
    
    directions = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]
    final_node = None
    visited = set()
    
    while pq:
        _, current = heapq.heappop(pq)
        
        if current in visited:
            continue
        visited.add(current)
        
        r, c = current
        
        # Check if reached goal edge
        if (player == PLAYER_1 and r == size - 1) or (player == PLAYER_2 and c == size - 1):
            final_node = current
            break
        
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < size and 0 <= nc < size:
                if (nr, nc) in visited:
                    continue
                    
                cell = game.board[nr, nc]
                
                if cell == -player:
                    continue
                
                # Prefer own stones, penalize empty
                step_cost = 0.1 if cell == player else 1.0
                new_cost = cost_so_far[current] + step_cost
                next_node = (nr, nc)
                
                if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                    cost_so_far[next_node] = new_cost
                    priority = new_cost + heuristic(nr, nc)
                    heapq.heappush(pq, (priority, next_node))
                    came_from[next_node] = current
    
    # Reconstruct path
    path = []
    if final_node is not None:
        curr = final_node
        while curr is not None:
            path.append(curr)
            curr = came_from.get(curr)
        path.reverse()
    
    return path

def find_path_to_point(game, player, start_point, end_point):
    """
    Find path using A*.
    If start_point is None, start from appropriate edge.
    If end_point is None, go to goal edge.
    """
    size = game.size
    pq = []
    came_from = {}
    cost_so_far = {}
    
    def heuristic(r, c):
        if end_point:
            # Distance to specific point
            return abs(r - end_point[0]) + abs(c - end_point[1])
        else:
            # Distance to goal edge - prefer straight paths
            if player == PLAYER_1:
                # Distance to bottom edge, slight preference for staying in same column
                return (size - 1 - r)
            else:
                # Distance to right edge, slight preference for staying in same row  
                return (size - 1 - c)
    
    # Initialize starting position
    if start_point:
        # Start from specific stone
        cost_so_far[start_point] = 0
        came_from[start_point] = None
        heapq.heappush(pq, (heuristic(start_point[0], start_point[1]), start_point))
    else:
        # Start from edge - find point closest to end_point
        if player == PLAYER_1:
            # Red: start from top edge
            if end_point:
                # Find closest edge point to target stone
                best_col = end_point[1]
                start = (0, best_col) if 0 <= best_col < size else (0, size // 2)
            else:
                start = (0, size // 2)
            
            if game.board[start[0], start[1]] != -player:
                cost = 0 if game.board[start[0], start[1]] == player else 1
                cost_so_far[start] = cost
                came_from[start] = None
                heapq.heappush(pq, (cost + heuristic(start[0], start[1]), start))
        else:
            # Blue: start from left edge
            if end_point:
                # Find closest edge point to target stone
                best_row = end_point[0]
                start = (best_row, 0) if 0 <= best_row < size else (size // 2, 0)
            else:
                start = (size // 2, 0)
            
            if game.board[start[0], start[1]] != -player:
                cost = 0 if game.board[start[0], start[1]] == player else 1
                cost_so_far[start] = cost
                came_from[start] = None
                heapq.heappush(pq, (cost + heuristic(start[0], start[1]), start))
    
    directions = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]
    final_node = None
    visited = set()
    
    # A* search
    while pq:
        _, current = heapq.heappop(pq)
        
        if current in visited:
            continue
        visited.add(current)
        
        r, c = current
        
        # Check if reached goal
        if end_point:
            if current == end_point:
                final_node = current
                break
        else:
            # Check if reached goal edge
            if (player == PLAYER_1 and r == size - 1) or (player == PLAYER_2 and c == size - 1):
                final_node = current
                break
        
        # Explore neighbors
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < size and 0 <= nc < size:
                if (nr, nc) in visited:
                    continue
                    
                cell = game.board[nr, nc]
                
                if cell == -player:
                    continue
                
                # Prefer own stones (low cost), penalize empty
                step_cost = 0.1 if cell == player else 1.0
                new_cost = cost_so_far[current] + step_cost
                next_node = (nr, nc)
                
                if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                    cost_so_far[next_node] = new_cost
                    priority = new_cost + heuristic(nr, nc)
                    heapq.heappush(pq, (priority, next_node))
                    came_from[next_node] = current
    
    # Reconstruct path
    path = []
    if final_node is not None:
        curr = final_node
        while curr is not None:
            path.append(curr)
            curr = came_from.get(curr)
        path.reverse()
    
    return path

# --- FEATURE EXTRACTION ---
def count_bridges(game, player):
    bridges = 0
    size = game.size
    bridge_patterns = [(1, -2), (2, -1), (1, 1), (-1, 2), (-2, 1), (-1, -1)]
    visited = set()
    
    for r in range(size):
        for c in range(size):
            if game.board[r, c] == player:
                for dr, dc in bridge_patterns:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < size and 0 <= nc < size:
                        if game.board[nr, nc] == player:
                            if ((r, c), (nr, nc)) not in visited and ((nr, nc), (r, c)) not in visited:
                                bridges += 1
                                visited.add(((r, c), (nr, nc)))
    return bridges // 2 

def calculate_connectivity(game, player):
    score = 0
    size = game.size
    directions = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]
    
    for r in range(size):
        for c in range(size):
            if game.board[r, c] == player:
                neighbors = 0
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < size and 0 <= nc < size and game.board[nr, nc] == player:
                        neighbors += 1
                if neighbors > 0:
                    score += 1 + (neighbors * 0.5)
    return score

def get_features_vector(game, player):
    my_dist = a_star_distance(game, player)
    if my_dist >= INF: my_dist = game.size * 2
    op_dist = a_star_distance(game, -player)
    if op_dist >= INF: op_dist = game.size * 2
    center_score = 0
    center = game.size // 2
    for r in range(center-2, center+3):
        for c in range(center-2, center+3):
            if 0 <= r < game.size and 0 <= c < game.size:
                if game.board[r, c] == player: center_score += 1
    bridge_count = count_bridges(game, player)
    connectivity = calculate_connectivity(game, player)
    return np.array([my_dist / game.size, op_dist / game.size, center_score / 9.0, bridge_count / game.size, connectivity / (game.size * 2)])

def evaluate_board(game, player):
    features = get_features_vector(game, player)
    return np.dot(RL_WEIGHTS, features)

# --- MCTS NODE ---
class MCTSNode:
    def __init__(self, game_state, parent=None, move=None):
        self.game = game_state
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.wins = 0
        self.untried_moves = game_state.get_valid_moves()

    def uct_select_child(self, uct_constant=2.0):
        best_score = -float('inf')
        best_child = None
        for child in self.children:
            if child.visits == 0: score = float('inf') 
            else: score = child.wins / child.visits + uct_constant * math.sqrt(math.log(self.visits) / child.visits)
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def expand(self, difficulty="Medium"):
        if not self.untried_moves: return None
        
        if difficulty == "Easy":
            if random.random() < 0.5:
                center = self.game.size // 2
                self.untried_moves.sort(key=lambda m: abs(m[0]-center) + abs(m[1]-center))
        
        elif len(self.untried_moves) > 5: 
            me = self.game.current_player
            opp = -me
            
            opp_path = get_visual_path(self.game, opp)
            opp_path_set = set(opp_path) if opp_path else set()
            
            my_neighbors = set()
            directions = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]
            for r in range(self.game.size):
                for c in range(self.game.size):
                    if self.game.board[r, c] == me:
                        for dr, dc in directions:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < self.game.size and 0 <= nc < self.game.size:
                                if self.game.board[nr, nc] == EMPTY:
                                    my_neighbors.add((nr, nc))

            priority_moves = []
            if difficulty == "Hard":
                priority_moves = list(opp_path_set.intersection(my_neighbors)) + list(opp_path_set)
            else: 
                priority_moves = list(opp_path_set.intersection(my_neighbors)) + list(my_neighbors)

            if priority_moves:
                unique_priority = list(set(priority_moves)) 
                final_priority = [m for m in self.untried_moves if m in unique_priority]
                others = [m for m in self.untried_moves if m not in unique_priority]
                random.shuffle(final_priority)
                self.untried_moves = final_priority + others
            else:
                center = self.game.size // 2
                self.untried_moves.sort(key=lambda m: abs(m[0]-center) + abs(m[1]-center))

        move = self.untried_moves.pop(0)
        new_game = copy.deepcopy(self.game)
        new_game.make_move(move[0], move[1])
        child = MCTSNode(new_game, parent=self, move=move)
        self.children.append(child)
        return child

# --- HYBRID AGENT ---
class HybridAI:
    def __init__(self, difficulty="Medium"):
        self.difficulty = difficulty
        
        if difficulty == "Easy": 
            self.iterations = 50
            self.uct_constant = 5.0 
            self.time_limit = 0.3
        elif difficulty == "Medium": 
            self.iterations = 550 
            self.uct_constant = 1.4 
            self.time_limit = 1.2
        else: # Hard
            self.iterations = 2500 
            self.uct_constant = 0.8 
            self.time_limit = 3.5

    def check_immediate_threats(self, game):
        me = game.current_player
        opp = -me
        valid_moves = game.get_valid_moves()
        
        for move in valid_moves:
            game.board[move[0], move[1]] = me
            if game.check_win(me):
                game.board[move[0], move[1]] = 0
                return move, "WINNING_MOVE"
            game.board[move[0], move[1]] = 0

        for move in valid_moves:
            game.board[move[0], move[1]] = opp
            if game.check_win(opp):
                game.board[move[0], move[1]] = 0
                return move, "FORCED_BLOCK"
            game.board[move[0], move[1]] = 0

        if self.difficulty == "Easy": return None, None

        # SPEARHEAD
        if self.difficulty in ["Hard", "Medium"]:
            opp_stones = []
            for r in range(game.size):
                for c in range(game.size):
                    if game.board[r, c] == opp:
                        opp_stones.append((r, c))
            
            if opp_stones:
                if opp == PLAYER_1:
                    spearhead = max(opp_stones, key=lambda x: x[0])
                    critical_next_steps = [(spearhead[0]+1, spearhead[1]), (spearhead[0]+1, spearhead[1]-1)]
                else:
                    spearhead = max(opp_stones, key=lambda x: x[1])
                    critical_next_steps = [(spearhead[0], spearhead[1]+1), (spearhead[0]-1, spearhead[1]+1)]

                for step in critical_next_steps:
                    if 0 <= step[0] < game.size and 0 <= step[1] < game.size:
                        if game.board[step[0], step[1]] == EMPTY:
                            if self.difficulty == "Hard":
                                return step, "SPEARHEAD_BLOCK"
                            elif self.difficulty == "Medium" and random.random() < 0.5:
                                return step, "SPEARHEAD_BLOCK"

        # SHADOW DEFENSE
        if self.difficulty in ["Hard", "Medium"]:
            opp_dist = a_star_distance(game, opp)
            trigger_dist = INF if self.difficulty == "Hard" else 5
            
            if opp_dist <= trigger_dist:
                opp_path = get_visual_path(game, opp)
                if opp_path:
                    best_block = None
                    max_impact = -1
                    candidates = [m for m in valid_moves if m in opp_path]
                    
                    if not candidates and self.difficulty == "Hard":
                         ops = set(opp_path)
                         directions = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]
                         for r, c in valid_moves:
                             for dr, dc in directions:
                                 if (r+dr, c+dc) in ops:
                                     candidates.append((r,c))
                                     break

                    for move in candidates:
                        game.board[move[0], move[1]] = me
                        new_opp_dist = a_star_distance(game, opp)
                        damage = new_opp_dist - opp_dist
                        
                        connectivity_bonus = 0
                        if self.difficulty == "Hard":
                            opp_neighbors = 0
                            directions = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]
                            for dr, dc in directions:
                                nr, nc = move[0] + dr, move[1] + dc
                                if 0 <= nr < game.size and 0 <= nc < game.size:
                                    if game.board[nr, nc] == opp: opp_neighbors += 1
                            if opp_neighbors >= 2: connectivity_bonus += 3.0
                        
                        total_impact = damage + connectivity_bonus
                        game.board[move[0], move[1]] = 0
                        
                        if total_impact > max_impact:
                            max_impact = total_impact
                            best_block = move
                    
                    threshold = 0.5 if self.difficulty == "Hard" else 1.0
                    if best_block and max_impact >= threshold:
                        return best_block, f"SHADOW_DEFENSE (Impact: {max_impact})"

        return None, None

    def get_smart_simulation_move(self, valid_moves, game, player):
        if self.difficulty == "Easy": return random.choice(valid_moves)

        smart_prob = 0.95 if self.difficulty == "Hard" else 0.60
        
        if random.random() < smart_prob:
            sample_size = 15 if self.difficulty == "Hard" else 8
            sample_moves = random.sample(valid_moves, min(len(valid_moves), sample_size))
            
            best_move = None
            best_score = -999
            
            for move in sample_moves:
                score = 0
                r, c = move
                directions = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]
                
                my_neighbors = 0
                opp_neighbors = 0
                
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < game.size and 0 <= nc < game.size:
                        if game.board[nr, nc] == player: my_neighbors += 1
                        elif game.board[nr, nc] == -player: opp_neighbors += 1
                
                if my_neighbors == 0:
                    bridge_potential = 0
                    bridge_patterns = [(1, -2), (2, -1), (1, 1), (-1, 2), (-2, 1), (-1, -1)]
                    for dr, dc in bridge_patterns:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < game.size and 0 <= nc < game.size:
                             if game.board[nr, nc] == player: bridge_potential += 1
                    if bridge_potential > 0: score += 3 
                
                if self.difficulty == "Hard":
                    if opp_neighbors > 0: score += 4 + opp_neighbors 
                    if my_neighbors > 0: score += 2 + my_neighbors   
                    if my_neighbors >= 2: score += 5                 
                else:
                    if my_neighbors > 0: score += 3 + my_neighbors
                    if opp_neighbors > 0: score += 1
                    
                if score > best_score:
                    best_score = score
                    best_move = move
            
            if best_move and best_score > 0:
                return best_move

        return random.choice(valid_moves)

    def get_best_move(self, game):
        reflex_move, reason = self.check_immediate_threats(game)
        if reflex_move:
            print(f"AI Reflex [{self.difficulty}]: {reason}")
            return reflex_move, {
                "iterations": 0, "total_simulations": 0,
                "candidates": [{"move": reflex_move, "final_score": 1.0, "confidence": 100}]
            }

        start_time = time.time()
        root = MCTSNode(game_state=copy.deepcopy(game))
        iteration_count = 0
        
        for i in range(self.iterations):
            if (i % 50 == 0) and (time.time() - start_time > self.time_limit): break
            node = root
            while not node.untried_moves and node.children:
                node = node.uct_select_child(self.uct_constant)
            if node.untried_moves: node = node.expand(difficulty=self.difficulty)
            
            temp_game = copy.deepcopy(node.game)
            depth = 0
            max_depth = 45 if self.difficulty == "Hard" else (30 if self.difficulty == "Medium" else 10)
            
            while temp_game.winner is None and depth < max_depth:
                moves = temp_game.get_valid_moves()
                if not moves: break
                move = self.get_smart_simulation_move(moves, temp_game, temp_game.current_player)
                temp_game.make_move(move[0], move[1])
                depth += 1
            
            if temp_game.winner:
                result = 1.0 if temp_game.winner == root.game.current_player else 0.0
            else:
                raw_score = evaluate_board(temp_game, root.game.current_player)
                if self.difficulty == "Easy": raw_score += random.uniform(-3, 3) 
                result = (math.tanh(raw_score / 5.0) + 1) / 2.0
            
            while node:
                node.visits += 1
                node.wins += result
                node = node.parent
            iteration_count = i + 1
        
        if not root.children: return None, {}
        candidates = sorted(root.children, key=lambda c: c.visits, reverse=True)[:5]
        best_move = candidates[0].move
        
        debug_candidates = []
        for c in candidates:
            debug_candidates.append({
                "move": c.move, "visits": c.visits,
                "final_score": c.wins / c.visits if c.visits > 0 else 0,
                "confidence": (c.visits / root.visits) * 100,
            })
        return best_move, {"iterations": iteration_count, "total_simulations": root.visits, "candidates": debug_candidates}

def update_rl_weights(game_history, winner):
    global RL_WEIGHTS
    learning_rate, gamma = 0.02, 0.95
    print(f"Updating RL weights based on {len(game_history)} moves...")
    G = 1.0
    for i in range(len(game_history) - 1, -1, -1):
        features, player = game_history[i]
        reward = G if player == winner else -G
        prediction = np.dot(RL_WEIGHTS, features)
        error = reward - prediction
        RL_WEIGHTS += learning_rate * error * features
        G *= gamma
    RL_WEIGHTS = np.clip(RL_WEIGHTS, -10.0, 10.0)
    try:
        with open("rl_weights.pkl", "wb") as f: pickle.dump(RL_WEIGHTS, f)
        print(f"Weights saved successfully: {RL_WEIGHTS}")
    except Exception as e: print(f"Error saving weights: {e}")
    print(f"Updated RL Weights: {RL_WEIGHTS}")