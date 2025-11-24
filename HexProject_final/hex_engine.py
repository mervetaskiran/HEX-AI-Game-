import numpy as np
from collections import deque

# Tahta Boyutu (Standart Hex 11x11)
BOARD_SIZE = 11

# Oyuncular
EMPTY = 0
PLAYER_1 = 1   # Kırmızı (Dikey - Yukarıdan Aşağıya)
PLAYER_2 = -1  # Mavi (Yatay - Soldan Sağa)

class HexGame:
    def __init__(self, size=BOARD_SIZE):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        self.current_player = PLAYER_1
        self.winner = None

    def get_valid_moves(self):
        # Boş olan tüm karelerin listesini döndür (Row, Col)
        return list(zip(*np.where(self.board == EMPTY)))

    def make_move(self, row, col):
        if self.board[row, col] == EMPTY:
            self.board[row, col] = self.current_player
            
            # Kazanma kontrolü
            if self.check_win(self.current_player):
                self.winner = self.current_player
            
            # Sırayı değiştir
            self.current_player *= -1
            return True
        return False

    def check_win(self, player):
        # BFS Algoritması ile kenardan kenara yol var mı bakar
        visited = set()
        queue = deque()

        # Başlangıç kenarındaki taşları kuyruğa ekle
        if player == PLAYER_1:  # Dikey (Row 0 -> Row 10)
            starts = [(0, c) for c in range(self.size) if self.board[0, c] == player]
            target_row = self.size - 1
        else:  # Yatay (Col 0 -> Col 10)
            starts = [(r, 0) for r in range(self.size) if self.board[r, 0] == player]
            target_col = self.size - 1

        for start in starts:
            queue.append(start)
            visited.add(start)

        # Komşuluk yönleri (Hexagonal grid için 6 yön)
        directions = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]

        while queue:
            r, c = queue.popleft()

            # Hedefe ulaşıldı mı?
            if player == PLAYER_1 and r == target_row:
                return True
            if player == PLAYER_2 and c == target_col:
                return True

            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.size and 0 <= nc < self.size:
                    if self.board[nr, nc] == player and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append((nr, nc))
        return False