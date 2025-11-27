import pygame
import pygame.gfxdraw
import math
import random
import numpy as np 
import os
import time
from hex_engine import HexGame, PLAYER_1, PLAYER_2
from hybrid_agent import HybridAI, get_visual_path, update_rl_weights, RL_ENABLED, get_features_vector

# --- SETTINGS ---
WIDTH, HEIGHT = 1400, 800 
BG_COLOR = (5, 5, 10)
HEX_SIZE = 30
BOARD_SIZE = 11

OFFSET_X = 200 
OFFSET_Y = 160

# --- COLORS ---
C_NEON_BLUE = (0, 255, 255)
C_NEON_PINK = (255, 0, 128)
C_NEON_ORANGE = (255, 165, 0)
C_BRIGHT_WHITE = (255, 255, 255)
C_DARK_PANEL = (10, 15, 25, 240) 
C_TEXT_WHITE = (220, 230, 255)
C_TEXT_DIM = (100, 110, 130)
C_GREEN_MATRIX = (0, 255, 65)
C_YELLOW_PATH = (255, 220, 0) 
C_SHADOW = (0, 0, 0, 150)
C_ALERT = (255, 50, 50) 
C_LOG_TEXT = (210, 220, 230) 
C_SCROLL_BAR = (60, 70, 80)
C_SCROLL_THUMB = (0, 200, 255)
C_MODAL_BG = (15, 20, 30, 250)
C_CARD_BG = (30, 35, 45)

pygame.init()
pygame.mixer.init() 

# --- SYNTHESIZER SOUND MANAGER ---
class SoundManager:
    def __init__(self):
        self.sounds = {}
        self.enabled = True
        try:
            pygame.mixer.set_num_channels(8)
            self.sounds["click"] = self._generate_beep(frequency=800, duration=0.05, volume=0.1)
            self.sounds["win"] = self._generate_chord([523.25, 659.25, 783.99, 1046.50], duration=0.6, volume=0.1)
            self.sounds["lose"] = self._generate_slide(start_freq=400, end_freq=100, duration=0.6, volume=0.15) 
        except Exception as e:
            print(f"Sound Gen Error: {e}")

    def play(self, name):
        if self.enabled and name in self.sounds and self.sounds[name]:
            self.sounds[name].play()

    def _generate_beep(self, frequency=440, duration=0.1, volume=0.1):
        sample_rate = 44100
        n_samples = int(sample_rate * duration)
        t = np.linspace(0, duration, n_samples, False)
        wave = np.sin(2 * np.pi * frequency * t)
        wave = wave * np.linspace(1, 0, n_samples)
        audio_data = (wave * volume * 32767).astype(np.int16)
        stereo_data = np.column_stack((audio_data, audio_data))
        return pygame.sndarray.make_sound(stereo_data)

    def _generate_chord(self, frequencies, duration=0.5, volume=0.1):
        sample_rate = 44100
        n_samples = int(sample_rate * duration)
        t = np.linspace(0, duration, n_samples, False)
        final_wave = np.zeros(n_samples)
        for freq in frequencies:
            wave = np.sin(2 * np.pi * freq * t)
            final_wave += wave
        final_wave /= len(frequencies)
        final_wave = final_wave * np.linspace(1, 0, n_samples)
        audio_data = (final_wave * volume * 32767).astype(np.int16)
        stereo_data = np.column_stack((audio_data, audio_data))
        return pygame.sndarray.make_sound(stereo_data)

    def _generate_slide(self, start_freq, end_freq, duration=0.5, volume=0.1):
        sample_rate = 44100
        n_samples = int(sample_rate * duration)
        freqs = np.linspace(start_freq, end_freq, n_samples)
        phases = np.cumsum(2 * np.pi * freqs / sample_rate)
        wave = np.sin(phases)
        wave = wave * np.linspace(1, 0, n_samples)
        audio_data = (wave * volume * 32767).astype(np.int16)
        stereo_data = np.column_stack((audio_data, audio_data))
        return pygame.sndarray.make_sound(stereo_data)

sound_manager = SoundManager()

# --- SCREEN SETUP ---
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE) 
pygame.display.set_caption("HEX BOARD GAME - AI PROJECT")
is_fullscreen = False

# --- FONT MANAGEMENT ---
font_names = ["Arial Black", "Impact", "Verdana", "Consolas", "Arial"] 
tech_font_name = pygame.font.match_font(font_names[0]) 

def get_font(size, bold=True):
    try: 
        if tech_font_name:
            return pygame.font.Font(tech_font_name, size)
        else:
            return pygame.font.SysFont("Arial", size, bold=bold)
    except: 
        return pygame.font.SysFont("Arial", size, bold=bold) 

font_xs = get_font(14, bold=True) 
font_sm = get_font(16, bold=True)
font_md = get_font(20, bold=True)
font_lg = get_font(40, bold=True)
font_xl_fancy = get_font(70, bold=True) 
font_xxl_splash = get_font(90, bold=True)

# --- LOGGING & GRAPH SYSTEM ---
game_logs = []
win_probability_history = [0.5] * 20
dev_scroll_y = 0
tree_scroll_y = 0
MAX_LOG_HISTORY = 50   

def add_log(text, category="SYS"):
    global dev_scroll_y 
    formatted_text = f"[{category}] {text}"
    game_logs.append(formatted_text)
    if len(game_logs) > MAX_LOG_HISTORY: 
        game_logs.pop(0)
    dev_scroll_y = 0

# --- AI PERSONALITY ---
def get_ai_phrase(event_type):
    phrases = {
        "THINKING": [
            "Scanning neural pathways...",
            "Calculating trajectory...",
            "Simulating 500 outcomes...",
            "Reading the board...",
            "Optimizing strategy..."
        ],
        "BLOCK": [
            "Access Denied.",
            "Blockade established.",
            "Not so fast!",
            "Path obstructed.",
            "Defense protocols active."
        ],
        "WIN": [
            "Checkmate sequence.",
            "Victory is calculated.",
            "Resistance is futile.",
            "Optimization: Complete.",
            "Game Over."
        ]
    }
    return random.choice(phrases.get(event_type, ["Thinking..."]))

# --- GLOBAL STATE ---
STATE_SPLASH = -1 
STATE_MENU = 0
STATE_GAME = 1
current_state = STATE_SPLASH 

game_mode = "HvAI"
difficulty = "Medium"
dev_mode = False 
show_path_mode = False 
show_tree_modal = False 
show_heatmap = False 
game_paused = False 

current_ai_path = [] 
last_ai_analysis = None 
last_move_time = 0

confetti_pieces = [] 
game_history = [] 
game_over_sound_played = False 
win_start_time = None

game = None
ai_agent, ai_agent_2 = None, None

# --- AI TURN TIMER STATE ---
# AI'ın bekleme süresini bloklamadan yönetmek için
ai_turn_state = {
    "start_time": None,
    "delay_needed": 0
}

# --- BACKGROUND EFFECTS ---
bg_hexes = [{"x": c, "y": r, "pulse": random.uniform(0,6), 
             "speed": random.uniform(0.02,0.05),
             "filled": random.random() < 0.2, 
             "fill_color_offset": random.uniform(0, 2*math.pi)} 
            for r in range(0, HEIGHT+200, 60) for c in range(0, WIDTH+200, 60)]

def draw_background(is_splash=False):
    w, h = screen.get_size()
    cx, cy = w//2, h//2
    time_now = time.time()

    for hex_obj in bg_hexes:
        if not is_splash and (hex_obj["x"] > w + 50 or hex_obj["y"] > h + 50): continue

        hex_obj["pulse"] += hex_obj["speed"]
        alpha = int((math.sin(hex_obj["pulse"]) + 1) * 15) + 10 
        
        draw_x, draw_y = hex_obj["x"], hex_obj["y"]
        if is_splash:
            dx = hex_obj["x"] - cx
            dy = hex_obj["y"] - cy
            dist = math.sqrt(dx*dx + dy*dy)
            angle = math.atan2(dy, dx)
            angle += 0.005 * dist * math.sin(time_now * 0.5) 
            draw_x = cx + dist * math.cos(angle)
            draw_y = cy + dist * math.sin(angle)

        points = []
        for i in range(6):
            ang = math.radians(60 * i)
            px = int(draw_x + 25 * math.cos(ang))
            py = int(draw_y + 25 * math.sin(ang))
            points.append((px, py))
        
        if hex_obj["filled"]:
            color_intensity = (math.sin(hex_obj["pulse"] + hex_obj["fill_color_offset"]) + 1) / 2 
            fill_col = (int(color_intensity * 50) + 20, 
                        int(color_intensity * 150) + 50, 
                        int(color_intensity * 200) + 50, alpha)
            try: pygame.gfxdraw.filled_polygon(screen, points, fill_col)
            except: pass
        try:
            pygame.gfxdraw.aapolygon(screen, points, (0, 200, 255, alpha))
        except: pass

# --- CUSTOM CURSOR (YENİ TATLI VERSİYON) ---
def draw_custom_cursor():
    mp = pygame.mouse.get_pos()
    
    # Oyuncu sırasına göre renk seçimi
    if game and game.current_player == PLAYER_1:
        cursor_col = C_NEON_PINK
    else:
        cursor_col = C_NEON_BLUE
        
    # --- 1. Katman: Dış "Nefes Alan" Hare ---
    t = time.time() * 6  # Hız
    radius_variation = math.sin(t) * 2.5
    outer_radius = 14 + radius_variation
    
    s_outer = pygame.Surface((40, 40), pygame.SRCALPHA)
    pygame.draw.circle(s_outer, (cursor_col[0], cursor_col[1], cursor_col[2], 40), (20, 20), outer_radius)
    screen.blit(s_outer, (mp[0]-20, mp[1]-20))
    
    # --- 2. Katman: Orta Parlaklık (Ana Gövde) ---
    s_middle = pygame.Surface((26, 26), pygame.SRCALPHA)
    pygame.draw.circle(s_middle, (cursor_col[0], cursor_col[1], cursor_col[2], 150), (13, 13), 11)
    screen.blit(s_middle, (mp[0]-13, mp[1]-13))

    # --- 3. Katman: Merkez Çekirdek ---
    pygame.draw.circle(screen, cursor_col, mp, 6)
    
    # --- 4. Katman: Parlak Beyaz Nokta ---
    pygame.draw.circle(screen, C_BRIGHT_WHITE, mp, 3)

# --- CONFETTI CLASS ---
class Confetti:
    def __init__(self, x, y):
        self.pos = pygame.Vector2(x, y)
        self.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        self.size = random.randint(3, 8)
        self.velocity = pygame.Vector2(random.uniform(-3, 3), random.uniform(-10, -5))
        self.gravity = 0.5
        self.lifetime = random.randint(60, 180)
        self.rotation = random.uniform(0, 2*math.pi)
        self.rotation_speed = random.uniform(-0.1, 0.1)

    def update(self):
        self.velocity.y += self.gravity
        self.pos += self.velocity
        self.rotation += self.rotation_speed
        self.lifetime -= 1

    def draw(self, surface):
        if self.lifetime > 0:
            s = pygame.Surface((self.size, self.size), pygame.SRCALPHA)
            pygame.draw.rect(s, self.color, (0, 0, self.size, self.size))
            rotated_s = pygame.transform.rotate(s, math.degrees(self.rotation))
            surface.blit(rotated_s, rotated_s.get_rect(center=self.pos))

def spawn_confetti(count, x_range, y_range):
    global confetti_pieces
    for _ in range(count):
        confetti_pieces.append(Confetti(random.randint(x_range[0], x_range[1]), random.randint(y_range[0], y_range[1])))

# --- UI FUNCTIONS ---
def draw_neon_text(surface, text, font, x, y, color, center=True, shadow=False):
    if shadow:
        shadow_surf = font.render(text, True, C_SHADOW)
        shadow_rect = shadow_surf.get_rect()
        if center: shadow_rect.center = (x + 5, y + 5) 
        else: shadow_rect.topleft = (x + 5, y + 5)
        surface.blit(shadow_surf, shadow_rect)
    text_surf = font.render(text, True, color)
    rect = text_surf.get_rect()
    if center: rect.center = (x, y)
    else: rect.topleft = (x, y)
    surface.blit(text_surf, rect)

def draw_cyber_button(text, x, y, w, h, active=False, special=False):
    rect = pygame.Rect(x, y, w, h)
    mp = pygame.mouse.get_pos()
    hover = rect.collidepoint(mp)
    base_col, border_col = ((20, 30, 40), (60, 70, 80))
    
    # Special: Heatmap veya Hint aktifse turuncu/mor olsun
    if special: base_col, border_col = (150, 80, 0), C_NEON_ORANGE
    elif active: base_col, border_col = C_NEON_BLUE, C_NEON_BLUE
    
    if hover and not active: base_col = (40, 50, 60)
    
    pygame.draw.rect(screen, base_col, rect, border_radius=5)
    pygame.draw.rect(screen, border_col, rect, 2, border_radius=5)
    txt_col = (255, 255, 255) if (active or special) else (150, 150, 150)
    if active and not special: txt_col = (0, 0, 0)
    text_surf = font_md.render(text, True, txt_col)
    screen.blit(text_surf, (x + (w - text_surf.get_width())//2, y + (h - text_surf.get_height())//2))
    clicked = hover and pygame.mouse.get_pressed()[0]
    return clicked

# --- FIXED HEATMAP (STRONGER LOCAL IMPACT) ---
def draw_influence_heatmap():
    """
    Draws a strategic heatmap with emphasized local stone influence.
    """
    s = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if game.board[r, c] != 0: continue

            x, y = get_hex_center(r, c)

            # 1. Global Goal Potential (Static - Lower weight)
            dist_to_red_edge = min(r, BOARD_SIZE - 1 - r)
            red_static = (1.0 - (dist_to_red_edge / (BOARD_SIZE / 2))) ** 2.0
            dist_to_blue_edge = min(c, BOARD_SIZE - 1 - c)
            blue_static = (1.0 - (dist_to_blue_edge / (BOARD_SIZE / 2))) ** 2.0

            red_inf = red_static * 1.2 # Ağırlık düşürüldü (1.5 -> 1.2)
            blue_inf = blue_static * 1.2

            # 2. Local Stone Influence (Dynamic - HIGHER WEIGHT)
            for dist in range(1, 4):
                # Ağırlık artırıldı (8.0 -> 12.0) - Taş etkisi daha baskın
                weight = 12.0 / (dist * dist) 
                
                for dr in range(-dist, dist + 1):
                    for dc in range(-dist, dist + 1):
                        if max(abs(dr), abs(dc), abs(dr + dc)) != dist: continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                            cell = game.board[nr, nc]
                            if cell == PLAYER_1: red_inf += weight
                            elif cell == PLAYER_2: blue_inf += weight

            # Calculate Ratio & Draw
            total = red_inf + blue_inf + 1e-6
            red_ratio = red_inf / total
            blue_ratio = blue_inf / total
            
            threshold = 0.51 # Eşik biraz düşürüldü
            max_alpha = 210  # Maksimum parlaklık artırıldı

            if red_ratio > threshold and red_ratio > blue_ratio:
                strength = (red_ratio - threshold) / (1 - threshold)
                # Parlama eşiği düşürüldü (10.0 -> 7.0) - Daha çabuk parlar
                alpha_scale = min(1.0, total / 7.0) 
                alpha = int(max_alpha * strength * alpha_scale)
                pygame.gfxdraw.filled_circle(s, int(x), int(y), int(HEX_SIZE * 0.85), (255, 20, 100, alpha))
                    
            elif blue_ratio > threshold and blue_ratio > red_ratio:
                strength = (blue_ratio - threshold) / (1 - threshold)
                alpha_scale = min(1.0, total / 7.0)
                alpha = int(max_alpha * strength * alpha_scale)
                pygame.gfxdraw.filled_circle(s, int(x), int(y), int(HEX_SIZE * 0.85), (0, 200, 255, alpha))

    screen.blit(s, (0, 0))

# --- SCROLLABLE DEV PANEL ---
def draw_analysis_panel():
    global dev_scroll_y
    
    panel_w = 400
    panel_rect = pygame.Rect(WIDTH - panel_w - 10, 80, panel_w, HEIGHT - 100)
    
    s = pygame.Surface((panel_rect.w, panel_rect.h), pygame.SRCALPHA)
    s.fill(C_DARK_PANEL)
    screen.blit(s, panel_rect)
    pygame.draw.rect(screen, C_NEON_BLUE, panel_rect, 2)
    
    content_h = 1300 
    content_surf = pygame.Surface((panel_w - 20, content_h), pygame.SRCALPHA)
    
    y_curr = 10
    draw_neon_text(content_surf, ">> SYSTEM KERNEL <<", font_md, (panel_w-20)//2, y_curr, C_NEON_BLUE)
    y_curr += 40
    
    draw_neon_text(content_surf, f"ALGORITHM: Hybrid MCTS + A*", font_sm, 10, y_curr, C_TEXT_WHITE, False)
    y_curr += 25
    draw_neon_text(content_surf, f"DIFFICULTY: {difficulty.upper()}", font_sm, 10, y_curr, C_TEXT_WHITE, False)
    y_curr += 25
    rl_status = "RL: ACTIVE" if RL_ENABLED else "RL: DISABLED"
    rl_col = C_GREEN_MATRIX if RL_ENABLED else (100, 100, 100)
    draw_neon_text(content_surf, rl_status, font_sm, 10, y_curr, rl_col, False)
    y_curr += 40

    # Win Graph
    graph_h = 100
    graph_rect = pygame.Rect(10, y_curr, panel_w - 40, graph_h)
    pygame.draw.rect(content_surf, (10, 20, 30), graph_rect)
    pygame.draw.rect(content_surf, (50, 60, 70), graph_rect, 1)
    mid_y = graph_rect.centery
    pygame.draw.line(content_surf, (40, 50, 60), (graph_rect.left, mid_y), (graph_rect.right, mid_y), 1)
    
    padding_x, padding_y = 10, 10
    avail_w = graph_rect.width - (2 * padding_x)
    avail_h = graph_rect.height - (2 * padding_y)

    if len(win_probability_history) > 1:
        points = []
        for i, val in enumerate(win_probability_history):
            px = graph_rect.left + padding_x + (i / (len(win_probability_history) - 1)) * avail_w
            clamped_val = max(0.0, min(1.0, val))
            py = (graph_rect.bottom - padding_y) - (clamped_val * avail_h)
            points.append((px, py))
        last_val = win_probability_history[-1]
        line_col = C_GREEN_MATRIX if last_val >= 0.5 else C_ALERT
        if len(points) > 1:
            pygame.draw.lines(content_surf, line_col, False, points, 2)
            last_pt = points[-1]
            pygame.draw.circle(content_surf, line_col, (int(last_pt[0]), int(last_pt[1])), 4)
            draw_neon_text(content_surf, f"{last_val*100:.0f}%", font_xs, last_pt[0] - 25, last_pt[1] - 15, line_col, False)
    
    y_curr += 130

    if last_ai_analysis:
        info = last_ai_analysis
        draw_neon_text(content_surf, f"ITERATIONS: {info.get('iterations',0)}", font_sm, 10, y_curr, C_GREEN_MATRIX, False)
        y_curr += 25
        draw_neon_text(content_surf, f"SIMULATIONS: {info.get('total_simulations',0)}", font_sm, 10, y_curr, C_GREEN_MATRIX, False)
        y_curr += 40
        draw_neon_text(content_surf, "TOP MOVES SCORE CONF", font_xs, 10, y_curr, (100, 200, 255), False)
        y_curr += 20
        for cand in info.get('candidates', []):
            move_str = f"[{cand['move'][0]},{cand['move'][1]}]"
            score_str = f"{cand['final_score']:.2f}"
            conf_str = f"{cand['confidence']:.1f}%"
            col = C_GREEN_MATRIX if cand == info.get('candidates', [])[0] else C_TEXT_WHITE
            row_txt = f"{move_str:<8} {score_str:<8} {conf_str}"
            draw_neon_text(content_surf, row_txt, font_sm, 10, y_curr, col, False)
            y_curr += 25
    else:
        draw_neon_text(content_surf, "WAITING FOR AI...", font_sm, 10, y_curr, (100, 100, 100), False)
        y_curr += 50

    y_curr += 20
    pygame.draw.line(content_surf, (50, 60, 80), (10, y_curr), (panel_w-30, y_curr), 1)
    y_curr += 30

    # LIVE STATS
    draw_neon_text(content_surf, "LIVE STATISTICS", font_md, (panel_w-20)//2, y_curr, C_NEON_ORANGE, True)
    y_curr += 40
    moves_made = np.count_nonzero(game.board != 0)
    moves_left = (BOARD_SIZE * BOARD_SIZE) - moves_made
    game_progress = moves_made / (BOARD_SIZE * BOARD_SIZE)
    ai_time = last_move_time if last_move_time > 0 else 0.0
    branching_factor = moves_left
    
    stats_data = [
        {"label": "AI TIME", "val": f"{ai_time:.2f}s", "max": 2.0, "col": C_NEON_BLUE, "val_num": ai_time},
        {"label": "BRANCHES", "val": f"{branching_factor}", "max": 121, "col": C_NEON_PINK, "val_num": branching_factor},
        {"label": "REMAINING", "val": f"{moves_left}", "max": 121, "col": C_GREEN_MATRIX, "val_num": moves_left},
        {"label": "PROGRESS", "val": f"{game_progress*100:.0f}%", "max": 1.0, "col": C_YELLOW_PATH, "val_num": game_progress},
    ]

    for stat in stats_data:
        draw_neon_text(content_surf, stat["label"], font_xs, 10, y_curr, C_TEXT_WHITE, False)
        draw_neon_text(content_surf, stat["val"], font_xs, panel_w - 80, y_curr, stat["col"], False)
        y_curr += 20
        bar_max_w = panel_w - 40
        bar_h = 8
        pygame.draw.rect(content_surf, (40, 40, 50), (10, y_curr, bar_max_w, bar_h), border_radius=4)
        fill_pct = min(1.0, stat["val_num"] / stat["max"]) if stat["max"] > 0 else 0
        fill_w = int(bar_max_w * fill_pct)
        if fill_w > 0:
            pygame.draw.rect(content_surf, stat["col"], (10, y_curr, fill_w, bar_h), border_radius=4)
        y_curr += 25

    y_curr += 20
    pygame.draw.line(content_surf, (50, 60, 80), (10, y_curr), (panel_w-30, y_curr), 1)
    y_curr += 30

    draw_neon_text(content_surf, "SYSTEM LOGS", font_md, (panel_w-20)//2, y_curr, C_TEXT_DIM, True)
    y_curr += 30
    for log in reversed(game_logs): 
        col = C_LOG_TEXT
        if "[WIN]" in log: col = (255, 215, 0)
        elif "[MCTS]" in log: col = C_NEON_BLUE
        elif "[RFLX]" in log: col = C_ALERT
        elif "[RL]" in log: col = C_GREEN_MATRIX
        elif "[AI]" in log: col = C_NEON_PINK 
        draw_neon_text(content_surf, log, font_xs, 10, y_curr, col, False)
        y_curr += 20

    max_scroll = max(0, y_curr - panel_rect.h + 20)
    dev_scroll_y = max(0, min(dev_scroll_y, max_scroll))
    view_rect = pygame.Rect(0, dev_scroll_y, panel_w - 20, panel_rect.h - 20)
    screen.blit(content_surf, (panel_rect.x + 10, panel_rect.y + 10), view_rect)
    if max_scroll > 0:
        sb_h = panel_rect.h - 20
        thumb_h = max(30, (panel_rect.h / y_curr) * sb_h)
        thumb_y = panel_rect.y + 10 + (dev_scroll_y / max_scroll) * (sb_h - thumb_h)
        pygame.draw.rect(screen, C_SCROLL_BAR, (panel_rect.right - 8, panel_rect.y + 10, 6, sb_h), border_radius=3)
        pygame.draw.rect(screen, C_SCROLL_THUMB, (panel_rect.right - 8, thumb_y, 6, thumb_h), border_radius=3)


# --- DIRECTORY STYLE TREE MODAL ---
def draw_tree_modal():
    global tree_scroll_y
    
    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    overlay.fill((5, 5, 10, 230)) 
    screen.blit(overlay, (0, 0))
    
    modal_w, modal_h = 1000, 700
    modal_x, modal_y = (WIDTH - modal_w)//2, (HEIGHT - modal_h)//2
    
    modal_rect = pygame.Rect(modal_x, modal_y, modal_w, modal_h)
    pygame.draw.rect(screen, C_MODAL_BG, modal_rect, border_radius=12)
    
    analysis_owner = -game.current_player 
    theme_color = C_NEON_PINK if analysis_owner == PLAYER_1 else C_NEON_BLUE
    owner_name = "RED PLAYER (AI)" if analysis_owner == PLAYER_1 else "BLUE PLAYER (AI)"
    
    pygame.draw.rect(screen, theme_color, modal_rect, 2, border_radius=12) 
    draw_neon_text(screen, "DECISION TREE ANALYSIS", font_lg, modal_rect.centerx, modal_y + 40, theme_color)
    draw_neon_text(screen, f"Branching Analysis for: {owner_name}", font_sm, modal_rect.centerx, modal_y + 80, C_TEXT_DIM)

    content_x = modal_x + 50
    content_y = modal_y + 120
    content_w = modal_w - 100
    content_h = modal_h - 180
    
    virtual_h = 2000 
    v_surf = pygame.Surface((content_w, virtual_h), pygame.SRCALPHA)
    
    vy = 20
    
    root_pos = (60, vy + 30)
    pygame.draw.circle(v_surf, theme_color, root_pos, 12)
    pygame.draw.circle(v_surf, C_BRIGHT_WHITE, root_pos, 6)
    draw_neon_text(v_surf, "ROOT STATE", font_md, root_pos[0] + 30, root_pos[1] - 10, C_TEXT_WHITE, False)
    
    trunk_top = (root_pos[0], root_pos[1] + 12)
    
    vy += 80
    
    if last_ai_analysis and 'candidates' in last_ai_analysis:
        candidates = last_ai_analysis['candidates']
        last_child_y = vy 
        
        for i, cand in enumerate(candidates):
            branch_y = vy + 40 
            last_child_y = branch_y
            
            branch_start = (root_pos[0], branch_y)
            branch_end = (root_pos[0] + 60, branch_y)
            
            confidence = cand['confidence'] / 100.0
            line_width = max(2, int(confidence * 6))
            line_col = (
                int(theme_color[0] * 0.5 + 100 * confidence), 
                int(theme_color[1] * 0.5 + 100 * confidence), 
                int(theme_color[2] * 0.5 + 100 * confidence)
            )
            line_col = (min(line_col[0], 255), min(line_col[1], 255), min(line_col[2], 255))

            pygame.draw.line(v_surf, line_col, branch_start, branch_end, line_width)
            pygame.draw.circle(v_surf, line_col, branch_end, 6)
            
            card_x = branch_end[0] + 20
            card_w = 700
            card_h = 80
            card_rect = pygame.Rect(card_x, vy, card_w, card_h)
            
            pygame.draw.rect(v_surf, C_CARD_BG, card_rect, border_radius=8)
            
            if i == 0:
                pygame.draw.rect(v_surf, C_NEON_ORANGE, card_rect, 2, border_radius=8)
                draw_neon_text(v_surf, "★ BEST", font_xs, card_rect.right - 60, card_rect.y + 10, C_NEON_ORANGE, False)
            else:
                pygame.draw.rect(v_surf, line_col, card_rect, 1, border_radius=8)

            raw_move = cand['move']
            move_str = f"MOVE: Hex[{int(raw_move[0]) + 1}, {int(raw_move[1]) + 1}]"
            score_str = f"SCORE: {cand['final_score']:.4f}"
            
            draw_neon_text(v_surf, move_str, font_md, card_rect.x + 20, card_rect.centery - 10, C_BRIGHT_WHITE, False)
            draw_neon_text(v_surf, score_str, font_sm, card_rect.x + 250, card_rect.centery - 10, C_TEXT_DIM, False)
            
            bar_w = 200
            bar_h = 10
            bar_x = card_rect.right - 280
            bar_y = card_rect.centery - 5
            
            pygame.draw.rect(v_surf, (20,20,30), (bar_x, bar_y, bar_w, bar_h), border_radius=5)
            pygame.draw.rect(v_surf, theme_color, (bar_x, bar_y, int(bar_w * confidence), bar_h), border_radius=5)
            
            draw_neon_text(v_surf, f"{cand['confidence']:.1f}%", font_xs, bar_x + bar_w + 10, bar_y - 2, theme_color, False)

            vy += card_h + 20
        
        trunk_bottom = (root_pos[0], last_child_y)
        pygame.draw.line(v_surf, theme_color, trunk_top, trunk_bottom, 4)
            
    else:
        draw_neon_text(v_surf, "No Analysis Data.", font_md, content_w//2, 100, C_TEXT_DIM)

    max_tree_scroll = max(0, vy - content_h)
    tree_scroll_y = max(0, min(tree_scroll_y, max_tree_scroll))
    
    view_rect = pygame.Rect(0, tree_scroll_y, content_w, content_h)
    screen.blit(v_surf, (content_x, content_y), view_rect)
    
    if max_tree_scroll > 0:
        sb_h = content_h
        thumb_h = max(40, (content_h / vy) * sb_h)
        thumb_y = content_y + (tree_scroll_y / max_tree_scroll) * (sb_h - thumb_h)
        pygame.draw.rect(screen, C_SCROLL_BAR, (content_x + content_w + 5, content_y, 8, sb_h), border_radius=4)
        pygame.draw.rect(screen, C_SCROLL_THUMB, (content_x + content_w + 5, thumb_y, 8, thumb_h), border_radius=4)

    close_btn_rect = pygame.Rect(modal_rect.centerx - 60, modal_rect.bottom - 60, 120, 45)
    pygame.draw.rect(screen, (50, 20, 20), close_btn_rect, border_radius=5)
    pygame.draw.rect(screen, C_ALERT, close_btn_rect, 2, border_radius=5)
    draw_neon_text(screen, "CLOSE", font_md, close_btn_rect.centerx, close_btn_rect.centery, C_TEXT_WHITE)
    
    return close_btn_rect

# --- GAME HELPERS ---
def get_hex_center(row, col):
    x = OFFSET_X + (col * HEX_SIZE * math.sqrt(3)) + (row * HEX_SIZE * math.sqrt(3) / 2)
    y = OFFSET_Y + (row * HEX_SIZE * 1.5)
    return x, y

def pixel_to_hex(x, y):
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            cx, cy = get_hex_center(r, c)
            if math.hypot(x - cx, y - cy) < HEX_SIZE * 0.9: return r, c
    return None

def draw_glowing_hex(surface, color, x, y, size, glow=False):
    points = []
    for i in range(6):
        ang = math.radians(60 * i + 30)
        px = int(x + size * math.cos(ang))
        py = int(y + size * math.sin(ang))
        points.append((px, py))
    if glow:
        gp = []
        for i in range(6):
            ang = math.radians(60 * i + 30)
            px = int(x + (size+4) * math.cos(ang))
            py = int(y + (size+4) * math.sin(ang))
            gp.append((px, py))
        try: pygame.gfxdraw.filled_polygon(surface, gp, (color[0], color[1], color[2], 100))
        except: pass
    try:
        pygame.gfxdraw.filled_polygon(surface, points, color)
        pygame.gfxdraw.aapolygon(surface, points, (255, 255, 255))
    except: pass

def toggle_fullscreen():
    global screen, is_fullscreen, WIDTH, HEIGHT
    is_fullscreen = not is_fullscreen
    if is_fullscreen:
        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        WIDTH, HEIGHT = screen.get_size() 
    else:
        WIDTH, HEIGHT = 1400, 800
        screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)

def reset_game():
    global game, ai_agent, ai_agent_2, last_ai_analysis, current_ai_path, confetti_pieces, game_history, game_paused, game_over_sound_played, game_logs, win_probability_history, log_scroll_offset, last_move_time, win_start_time, ai_turn_state
    game = HexGame(BOARD_SIZE)
    last_ai_analysis = None
    current_ai_path = [] 
    confetti_pieces = [] 
    game_history = [] 
    game_paused = False
    game_over_sound_played = False
    win_start_time = None
    game_logs = [] 
    win_probability_history = [0.5] * 20
    log_scroll_offset = 0
    last_move_time = 0
    # Zamanlayıcıyı sıfırla
    ai_turn_state = {"start_time": None, "delay_needed": 0}
    add_log("System initialized.", "SYS")
    
    if game_mode == "HvAI": ai_agent = HybridAI(difficulty)
    elif game_mode == "AIvAI":
        ai_agent = HybridAI(difficulty)
        ai_agent_2 = HybridAI(difficulty)

# --- RL STATE RECORDING ---
def record_game_state(game, player):
    if RL_ENABLED:
        features = get_features_vector(game, player)
        game_history.append((features, player))

# --- MAIN LOOP ---
running = True
clock = pygame.time.Clock()
mouse_clicked_this_frame = False
start_time_splash = time.time()
pygame.mouse.set_visible(False) 

while running:
    screen.fill(BG_COLOR) 
    
    if current_state == STATE_SPLASH:
        draw_background(is_splash=True)
        pulse = (math.sin(time.time() * 5) + 1) / 2
        title_col = (int(C_NEON_BLUE[0] + pulse * 50), int(C_NEON_BLUE[1]), int(C_NEON_BLUE[2] + pulse * 50))
        title_col = (min(title_col[0], 255), min(title_col[1], 255), min(title_col[2], 255))
        draw_neon_text(screen, "HEX AI PROJECT", font_xxl_splash, WIDTH//2, HEIGHT//2 - 20, title_col, shadow=True)
        draw_neon_text(screen, "PRESS ANY KEY TO START", font_md, WIDTH//2, HEIGHT//2 + 50, C_BRIGHT_WHITE, shadow=True)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN:
                current_state = STATE_MENU
                sound_manager.play("click")
        if time.time() - start_time_splash > 4.0:
            current_state = STATE_MENU

        if pygame.mouse.get_focused():
         draw_custom_cursor()

        pygame.display.flip()
        clock.tick(60)
        continue 

    draw_background(is_splash=False) 

    mouse_clicked_this_frame = False
    for event in pygame.event.get():
        if event.type == pygame.QUIT: running = False
        if event.type == pygame.VIDEORESIZE:
            if not is_fullscreen:
                WIDTH, HEIGHT = event.w, event.h
                screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
        
        if event.type == pygame.MOUSEWHEEL:
            if show_tree_modal:
                tree_scroll_y -= event.y * 20 
            elif dev_mode:
                dev_scroll_y -= event.y * 15

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_f or event.key == pygame.K_F11: toggle_fullscreen()
            if event.key == pygame.K_ESCAPE: 
                if show_tree_modal: show_tree_modal = False
            if event.key == pygame.K_p: 
                if current_state == STATE_GAME:
                    game_paused = not game_paused
                    sound_manager.play("click")
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_clicked_this_frame = True
            
            if show_tree_modal:
                pass 
            elif current_state == STATE_GAME:
                if game and not game.winner and not game_paused:
                    is_human_turn = False
                    if game_mode == "HvH": is_human_turn = True
                    elif game_mode == "HvAI" and game.current_player == PLAYER_1: is_human_turn = True
                    
                    if is_human_turn:
                        mp = pygame.mouse.get_pos()
                        panel_w = 400
                        if dev_mode and mp[0] > WIDTH - panel_w - 10:
                            pass
                        elif mp[1] > 60: 
                            gp = pixel_to_hex(mp[0], mp[1])
                            if gp:
                                record_game_state(game, game.current_player)
                                if game.make_move(gp[0], gp[1]): pass 

    if current_state == STATE_MENU:
        cx, cy = WIDTH//2, HEIGHT//2
        pygame.draw.rect(screen, C_DARK_PANEL, (cx-250, cy-220, 500, 500), border_radius=10)
        pygame.draw.rect(screen, C_NEON_BLUE, (cx-250, cy-220, 500, 500), 2, border_radius=10)
        draw_neon_text(screen, "HEX BOARD GAME", font_xl_fancy, cx, cy-260, C_NEON_BLUE, shadow=True)
        curr_y = cy - 180
        draw_neon_text(screen, "SELECT MODE", font_sm, cx, curr_y, C_TEXT_WHITE)
        curr_y += 30
        if draw_cyber_button("HUMAN vs AI", cx-200, curr_y, 190, 50, game_mode=="HvAI"): 
            if mouse_clicked_this_frame:
                game_mode="HvAI"
                sound_manager.play("click")
        if draw_cyber_button("AI vs AI", cx+10, curr_y, 190, 50, game_mode=="AIvAI"): 
            if mouse_clicked_this_frame:
                game_mode="AIvAI"
                sound_manager.play("click")
        curr_y += 60
        if draw_cyber_button("HUMAN vs HUMAN", cx-200, curr_y, 400, 50, game_mode=="HvH"): 
            if mouse_clicked_this_frame:
                game_mode="HvH"
                sound_manager.play("click")
        curr_y += 70
        draw_neon_text(screen, "DIFFICULTY", font_sm, cx, curr_y, C_TEXT_WHITE)
        curr_y += 30
        dx = cx - 190
        for d in ["Easy", "Medium", "Hard"]:
            is_hvh = (game_mode == "HvH")
            btn_active = (not is_hvh and difficulty == d)
            if is_hvh:
                draw_cyber_button(d, dx, curr_y, 120, 40, False) 
                s = pygame.Surface((120, 40), pygame.SRCALPHA)
                s.fill((0, 0, 0, 150))
                screen.blit(s, (dx, curr_y))
            else:
                if draw_cyber_button(d, dx, curr_y, 120, 40, btn_active): 
                    if mouse_clicked_this_frame:
                        difficulty = d
                        sound_manager.play("click")
            dx += 130   
        if draw_cyber_button("START", cx-150, cy+180, 300, 60, False, True):
            if mouse_clicked_this_frame:
                sound_manager.play("click")
                reset_game()
                current_state = STATE_GAME
        fs_text = "EXIT FULLSCREEN" if is_fullscreen else "FULLSCREEN"
        if draw_cyber_button(fs_text, WIDTH - 160, HEIGHT - 60, 150, 40):
            if mouse_clicked_this_frame:
                toggle_fullscreen()
                sound_manager.play("click")

    elif current_state == STATE_GAME:
        pygame.draw.rect(screen, C_DARK_PANEL, (0, 0, WIDTH, 60))
        pygame.draw.line(screen, C_NEON_BLUE, (0, 60), (WIDTH, 60), 2)
        
        turn_txt = "RED TURN" if game.current_player == 1 else "BLUE TURN"
        turn_col = C_NEON_PINK if game.current_player == 1 else C_NEON_BLUE
        status_text = ">> SYSTEM PAUSED <<" if game_paused else f">> {turn_txt} <<"
        status_col = C_ALERT if game_paused else turn_col
        draw_neon_text(screen, status_text, font_md, 120, 30, status_col, center=True) 

        btn_h = 40
        btn_y = 10
        margin_right = 20
        gap = 10
        current_x = WIDTH - margin_right
        
        fs_w = 50
        current_x -= fs_w
        if draw_cyber_button("[ ]" if is_fullscreen else "[F]", current_x, btn_y, fs_w, btn_h, is_fullscreen):
            if mouse_clicked_this_frame:
                toggle_fullscreen()
                sound_manager.play("click")
        current_x -= gap
        
        exit_w = 100
        current_x -= exit_w
        if draw_cyber_button("EXIT", current_x, btn_y, exit_w, btn_h):
             if mouse_clicked_this_frame:
                sound_manager.play("click")
                current_state = STATE_MENU
        current_x -= gap
        
        std_btn_w = 140 
        
        current_x -= std_btn_w
        if draw_cyber_button("SHOW PATH", current_x, btn_y, std_btn_w, btn_h, show_path_mode):
            if mouse_clicked_this_frame:
                show_path_mode = not show_path_mode
                sound_manager.play("click")
        current_x -= gap
        
        current_x -= std_btn_w
        if draw_cyber_button("HEATMAP", current_x, btn_y, std_btn_w, btn_h, show_heatmap):
            if mouse_clicked_this_frame:
                show_heatmap = not show_heatmap
                sound_manager.play("click")
        current_x -= gap

        current_x -= std_btn_w
        if draw_cyber_button("TREE VIEW", current_x, btn_y, std_btn_w, btn_h, show_tree_modal):
            if mouse_clicked_this_frame:
                show_tree_modal = not show_tree_modal
                sound_manager.play("click")
        current_x -= gap

        current_x -= std_btn_w
        if draw_cyber_button("DEV MODE", current_x, btn_y, std_btn_w, btn_h, dev_mode):
            if mouse_clicked_this_frame:
                dev_mode = not dev_mode
                sound_manager.play("click")
        current_x -= gap
        
        current_x -= std_btn_w
        pause_label = "RESUME" if game_paused else "PAUSE"
        if draw_cyber_button(pause_label, current_x, btn_y, std_btn_w, btn_h, game_paused):
            if mouse_clicked_this_frame:
                game_paused = not game_paused
                sound_manager.play("click")

        x1_rt_top, y1_rt_top = get_hex_center(0, 0)
        x2_rt_top, y2_rt_top = get_hex_center(0, BOARD_SIZE - 1)
        pygame.draw.line(screen, C_NEON_PINK, (int(x1_rt_top - HEX_SIZE / math.sqrt(3)), int(y1_rt_top - HEX_SIZE)), (int(x2_rt_top + HEX_SIZE / math.sqrt(3)), int(y2_rt_top - HEX_SIZE)), 4)
        x1_rt_bot, y1_rt_bot = get_hex_center(BOARD_SIZE - 1, 0)
        x2_rt_bot, y2_rt_bot = get_hex_center(BOARD_SIZE - 1, BOARD_SIZE - 1)
        pygame.draw.line(screen, C_NEON_PINK, (int(x1_rt_bot - HEX_SIZE / math.sqrt(3)), int(y1_rt_bot + HEX_SIZE)), (int(x2_rt_bot + HEX_SIZE / math.sqrt(3)), int(y2_rt_bot + HEX_SIZE)), 4)
        x1_bl_left, y1_bl_left = get_hex_center(0, 0)
        x2_bl_left, y2_bl_left = get_hex_center(BOARD_SIZE - 1, 0)
        pygame.draw.line(screen, C_NEON_BLUE, (int(x1_bl_left - HEX_SIZE / math.sqrt(3) - 5), int(y1_bl_left - HEX_SIZE + 20)), (int(x2_bl_left - HEX_SIZE / math.sqrt(3) - 5), int(y2_bl_left + HEX_SIZE - 20)), 4)
        x1_bl_right, y1_bl_right = get_hex_center(0, BOARD_SIZE - 1)
        x2_bl_right, y2_bl_right = get_hex_center(BOARD_SIZE - 1, BOARD_SIZE - 1)
        pygame.draw.line(screen, C_NEON_BLUE, (int(x1_bl_right + HEX_SIZE / math.sqrt(3) + 5), int(y1_bl_right - HEX_SIZE + 20)), (int(x2_bl_right + HEX_SIZE / math.sqrt(3) + 5), int(y2_bl_right + HEX_SIZE - 20)), 4)
        
        for c in range(BOARD_SIZE):
            cx, cy = get_hex_center(0, c)
            draw_neon_text(screen, str(c+1), font_sm, cx, cy - 40, C_TEXT_DIM)
        for r in range(BOARD_SIZE):
            cx, cy = get_hex_center(r, 0)
            draw_neon_text(screen, str(r+1), font_sm, cx - 45, cy, C_TEXT_DIM)

        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                x, y = get_hex_center(r, c)
                val = game.board[r, c]
                col = (30, 35, 45) 
                if val == PLAYER_1: col = C_NEON_PINK
                elif val == PLAYER_2: col = C_NEON_BLUE
                mp = pygame.mouse.get_pos()
                is_hover = False
                if not game_paused and not show_tree_modal and val == 0 and math.hypot(mp[0]-x, mp[1]-y) < HEX_SIZE*0.9:
                    if game_mode == "HvH": is_hover = True
                    elif game_mode == "HvAI" and game.current_player == PLAYER_1: is_hover = True
                if is_hover: col = (60, 60, 70)
                draw_glowing_hex(screen, col, x, y, HEX_SIZE-2, glow=(val!=0))
                if dev_mode and last_ai_analysis and val == 0 and not show_tree_modal:
                    for cand in last_ai_analysis.get('candidates', []):
                        if cand['move'] == (r,c):
                            score_txt = f"{cand['final_score']:.1f}"
                            draw_neon_text(screen, score_txt, font_xs, x, y, C_GREEN_MATRIX)

        if game and not game.winner and show_path_mode and not show_tree_modal:
            if np.any(game.board != 0):
                current_ai_path = get_visual_path(game, game.current_player)
                path_color = C_NEON_PINK if game.current_player == PLAYER_1 else C_NEON_BLUE
                if current_ai_path and len(current_ai_path) > 1:
                    points = []
                    for (r, c) in current_ai_path:
                        points.append(get_hex_center(r, c))
                    if len(points) > 1:
                        glow_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
                        r_val, g_val, b_val = path_color
                        pygame.draw.lines(glow_surf, (r_val, g_val, b_val, 50), False, points, 20)
                        pygame.draw.lines(glow_surf, (r_val, g_val, b_val, 100), False, points, 10)
                        screen.blit(glow_surf, (0,0))
                        pygame.draw.lines(screen, path_color, False, points, 3)
                        for p in points:
                            pygame.gfxdraw.filled_circle(screen, int(p[0]), int(p[1]), 8, (r_val, g_val, b_val, 100))
                            bright_col = (min(r_val+100,255), min(g_val+100,255), min(b_val+100,255))
                            pygame.draw.circle(screen, bright_col, (int(p[0]), int(p[1])), 4)

        if show_heatmap:
            draw_influence_heatmap()

        if dev_mode:
            draw_analysis_panel()

        if show_tree_modal:
            close_rect = draw_tree_modal()
            if mouse_clicked_this_frame:
                mp = pygame.mouse.get_pos()
                if close_rect.collidepoint(mp):
                    show_tree_modal = False
                    sound_manager.play("click")

        if not game.winner and not game_paused and not show_tree_modal:
            move = None
            is_ai_turn = False
            active_agent = None
            if game_mode == "HvAI" and game.current_player == PLAYER_2:
                is_ai_turn = True
                active_agent = ai_agent
            elif game_mode == "AIvAI":
                is_ai_turn = True
                active_agent = ai_agent if game.current_player == PLAYER_1 else ai_agent_2
            
            if is_ai_turn:
                # --- BLOKLAMAYAN ZAMANLAYICI MANTIĞI ---
                current_time = time.time()
                
                # 1. Yeni bir tur başladıysa zamanlayıcıyı başlat
                if ai_turn_state["start_time"] is None:
                    ai_turn_state["start_time"] = current_time
                    if game_mode == "AIvAI":
                        ai_turn_state["delay_needed"] = 0.5
                    elif difficulty == "Easy":
                        ai_turn_state["delay_needed"] = random.uniform(0.1, 0.2)
                    elif difficulty == "Medium":
                        ai_turn_state["delay_needed"] = random.uniform(0.2, 0.4)
                    else:
                        ai_turn_state["delay_needed"] = 0.7

                # 2. Bekleme süresi doldu mu kontrol et
                if current_time - ai_turn_state["start_time"] >= ai_turn_state["delay_needed"]:
                    t_start = time.time()
                    move, info = active_agent.get_best_move(game)
                    t_end = time.time()
                    last_move_time = t_end - t_start 
                    last_ai_analysis = info
                    
                    # ... (Loglama ve hareket kodları aynı) ...
                    if info and "candidates" in info and len(info["candidates"]) > 0:
                        best_score = info["candidates"][0]["final_score"]
                        win_probability_history.append(best_score)
                        if len(win_probability_history) > 40:
                             win_probability_history.pop(0)

                    if "log" in info:
                        raw_log = info["log"]
                        personality_msg = ""
                        if "Thinking" in raw_log or "MCTS" in raw_log:
                            personality_msg = get_ai_phrase("THINKING")
                        elif "BLOCK" in raw_log:
                            personality_msg = get_ai_phrase("BLOCK")
                        elif "WIN" in raw_log:
                            personality_msg = get_ai_phrase("WIN")
                        
                        if personality_msg:
                            add_log(personality_msg, "AI")
                        
                        clean_text = raw_log.replace("AI Reflex [Easy]: ", "").replace("AI Reflex [Medium]: ", "").replace("AI Reflex [Hard]: ", "").replace("AI MCTS Strategy", "MCTS Search")
                        category = "RFLX" if "Reflex" in raw_log else "MCTS"
                        add_log(clean_text, category)
                    
                    if move: 
                        record_game_state(game, game.current_player)
                        game.make_move(move[0], move[1])
                    
                    # 3. Zamanlayıcıyı bir sonraki tur için sıfırla
                    ai_turn_state["start_time"] = None
                    ai_turn_state["delay_needed"] = 0
                else:
                    pass
        
        # --- GAME OVER ---
        if game.winner:
            if win_start_time is None:
                win_start_time = time.time()

            if not confetti_pieces: 
                spawn_confetti(200, (0, WIDTH), (0, HEIGHT // 2))
                
            if not game_over_sound_played:
                if game_mode == "HvAI":
                    if game.winner == PLAYER_1: sound_manager.play("win")
                    else: sound_manager.play("lose")
                else: 
                    sound_manager.play("win")
                
                if RL_ENABLED and game_history:
                    add_log(f"Winner: {'RED' if game.winner == PLAYER_1 else 'BLUE'}", "WIN")
                    rl_log = update_rl_weights(game_history, game.winner)
                    add_log("RL Weights Updated", "RL")
                
                game_over_sound_played = True

            if time.time() - win_start_time > 1.2:
                s = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
                s.fill((0,0,0, 200))
                screen.blit(s, (0,0))
                
                w_txt = "RED WINS" if game.winner == PLAYER_1 else "BLUE WINS"
                w_col = C_NEON_PINK if game.winner == PLAYER_1 else C_NEON_BLUE
                draw_neon_text(screen, w_txt, font_xl_fancy, WIDTH//2, HEIGHT//2 - 40, w_col, shadow=True) 
                
                if draw_cyber_button("RESTART", WIDTH//2 - 100, HEIGHT//2 + 60, 200, 50, False, True):
                    if mouse_clicked_this_frame:
                        sound_manager.play("click")
                        reset_game()
                if draw_cyber_button("MAIN MENU", WIDTH//2 - 100, HEIGHT//2 + 130, 200, 50, False, False):
                    if mouse_clicked_this_frame:
                        sound_manager.play("click")
                        current_state = STATE_MENU
                        confetti_pieces = []
            else:
                # Bekleme süresinde sadece mevcut ekranı görmeye devam ederiz
                pass

    for piece in list(confetti_pieces): 
        piece.update()
        piece.draw(screen)
        if piece.lifetime <= 0 or not (0 < piece.pos.x < WIDTH and 0 < piece.pos.y < HEIGHT):
            confetti_pieces.remove(piece)
    
    # --- DRAW CUSTOM CURSOR (EN ÜSTE) ---
    if pygame.mouse.get_focused():
        draw_custom_cursor()

    pygame.display.flip()
    clock.tick(60)

pygame.quit()