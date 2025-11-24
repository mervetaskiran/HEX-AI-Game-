import random
import numpy as np
import pickle
import os
import copy

from hex_engine import HexGame
from hybrid_agent import get_features_vector, update_rl_weights, HybridAI

def train_offline(episodes=500):
    print("Starting Offline Training...")
    print("=" * 60)
    
    agent_p1 = HybridAI(difficulty="Medium")
    agent_p2 = HybridAI(difficulty="Medium")
    
    wins_p1 = 0
    wins_p2 = 0
    
    for episode in range(episodes):
        game = HexGame(11)
        history = []
        
        while game.winner is None:
            player = game.current_player
            
            # 1. Record State
            features = get_features_vector(game, player)
            history.append((features, player))
            
            # 2. Doğru agent'ı seç (DÜZELTİLMİŞ)
            active_agent = agent_p1 if player == 1 else agent_p2
            
            # 3. Make Move (30% exploration, 70% exploitation)
            if random.random() < 0.3:  # Exploration
                moves = game.get_valid_moves()
                move = random.choice(moves) if moves else None
            else:  # Exploitation
                move, _ = active_agent.get_best_move(game)
            
            if move:
                game.make_move(move[0], move[1])
            else:
                break
        
        # 3. Update Weights at end of game
        if game.winner:
            if game.winner == 1:
                wins_p1 += 1
            else:
                wins_p2 += 1
            
            update_rl_weights(history, game.winner)
        
        # Progress report
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{episodes} | "
                  f"P1 Wins: {wins_p1} | P2 Wins: {wins_p2} | "
                  f"Win Rate P1: {wins_p1/(episode+1)*100:.1f}%")
    
    print("=" * 60)
    print("Training Complete!")
    print(f"Final Stats - P1: {wins_p1} wins | P2: {wins_p2} wins")
    
    # Load final weights
    try:
        with open("rl_weights.pkl", "rb") as f:
            final_weights = pickle.load(f)
            print(f"Final Weights: {final_weights}")
    except:
        pass

if __name__ == "__main__":
    # 100 episode ile hızlı test
    train_offline(100)