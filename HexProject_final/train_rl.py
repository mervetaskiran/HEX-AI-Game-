import random
import numpy as np
import pickle
import os
import copy
import time

from hex_engine import HexGame
from hybrid_agent import get_features_vector, update_rl_weights, HybridAI, RL_WEIGHTS

def train_offline(episodes=100):
    print(f"🚀 Starting Offline Training ({episodes} episodes)...")
    print("=" * 60)
    
    # 1. Agentları Medium zekasıyla başlatıyoruz
    agent_p1 = HybridAI(difficulty="Medium")
    agent_p2 = HybridAI(difficulty="Medium")
    
    # --- TURBO MOD AYARI ---
    TURBO_ITERATIONS = 50
    TURBO_TIME = 0.1
    
    agent_p1.iterations = TURBO_ITERATIONS
    agent_p1.time_limit = TURBO_TIME
    agent_p2.iterations = TURBO_ITERATIONS
    agent_p2.time_limit = TURBO_TIME
    
    print(f"⚡ Training Speed: {TURBO_ITERATIONS} iterations/move (Overclocked)")
    
    wins_p1 = 0
    wins_p2 = 0
    start_time = time.time()
    
    for episode in range(episodes):
        game = HexGame(11)
        history = []
        
        while game.winner is None:
            player = game.current_player
            
            # 1. State Kaydı
            features = get_features_vector(game, player)
            history.append((features, player))
            
            # 2. Agent Seçimi
            active_agent = agent_p1 if player == 1 else agent_p2
            
            # 3. Hamle Seçimi (Epsilon-Greedy)
            # %30 Keşif (Rastgele), %70 Sömürü (AI)
            if random.random() < 0.3:  
                moves = game.get_valid_moves()
                move = random.choice(moves) if moves else None
            else: 
                # AI hamlesini hesaplat
                move, _ = active_agent.get_best_move(game)
            
            if move:
                game.make_move(move[0], move[1])
            else:
                break
        
        # 4. Oyun Bitti - Ağırlıkları Güncelle
        if game.winner:
            if game.winner == 1: wins_p1 += 1
            else: wins_p2 += 1
            
            update_rl_weights(history, game.winner)
        
        # Raporlama
        if (episode + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Episode {episode + 1}/{episodes} | "
                  f"P1: {wins_p1} - P2: {wins_p2} | "
                  f"Time: {elapsed:.1f}s")
    
    print("=" * 60)
    print("Training Complete!")
    print(f"Final Stats - P1: {wins_p1} | P2: {wins_p2}")
    
    try:
        with open("rl_weights.pkl", "rb") as f:
            final_weights = pickle.load(f)
            print(f"Final Learned Weights: {final_weights}")
    except:
        pass

if __name__ == "__main__":
    train_offline(100)