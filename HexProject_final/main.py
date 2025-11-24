import pygame
import pygame.gfxdraw
import math
import random
import numpy as np 
import os
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
C_DARK_PANEL = (10, 15, 25, 240) 
C_TEXT_WHITE = (220, 230, 255)
C_TEXT_DIM = (100, 110, 130)
C_GREEN_MATRIX = (0, 255, 65)
C_YELLOW_PATH = (255, 220, 0) # Artık dinamik değişecek ama default kalsın
C_SHADOW = (0, 0, 0, 150)
C_ALERT = (255, 50, 50) 

pygame.init()
pygame.mixer.init() # Ses motorunu başlat

# --- SOUND MANAGER ---
# Ses dosyaları yoksa oyunun çökmesini engellemek için güvenli yükleme
class SoundManager:
    def __init__(self):
        self.sounds = {}
        self.enabled = True
        self.load_sound("click", "click.wav")
        self.load_sound("win", "win.wav")
        self.load_sound("lose", "lose.wav")

    def load_sound(self, name, filename):
        if os.path.exists(filename):
            try:
                self.sounds[name] = pygame.mixer.Sound(filename)
                # Ses seviyelerini ayarla
                if name == "click": self.sounds[name].set_volume(0.4)
                if name == "win": self.sounds[name].set_volume(0.6)
                if name == "lose": self.sounds[name].set_volume(0.6)
            except:
                print(f"Warning: Could not load sound {filename}")
                self.sounds[name] = None
        else:
            self.sounds[name] = None

    def play(self, name):
        if self.enabled and name in self.sounds and self.sounds[name]:
            self.sounds[name].play()

sound_manager = SoundManager()

# --- SCREEN SETUP ---
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE) # Resizable yaptık
pygame.display.set_caption("HEX BOARD GAME - AI PROJECT")
is_fullscreen = False

# --- FONT MANAGEMENT ---
font_names = ["Montserrat Black", "Impact", "Arial Black", "Orbitron", "Exo 2", "Rajdhani", "Consolas", "Lucida Console", "Courier New", "Arial"]
tech_font_name = pygame.font.match_font(font_names[0]) 

def get_font(size, bold=True):
    try: return pygame.font.Font(tech_font_name, size, bold=bold)
    except: return pygame.font.SysFont("Arial", size, bold=bold) 

font_xs = get_font(14)
font_sm = get_font(16)
font_md = get_font(20)
font_lg = get_font(40)
font_xl_fancy = get_font(70, bold=True) 

# --- GLOBAL STATE ---
STATE_MENU, STATE_GAME = 0, 1
current_state = STATE_MENU
game_mode = "HvAI"
difficulty = "Medium"
dev_mode = False 
show_path_mode = False 
game_paused = False 
current_ai_path = [] 
last_ai_analysis = None 
confetti_pieces = [] 
game_history = [] 
game_over_sound_played = False # Sesin bir kere çalması için flag

game = None
ai_agent, ai_agent_2 = None, None

# --- BACKGROUND EFFECTS ---
bg_hexes = [{"x": c, "y": r, "pulse": random.uniform(0,6), 
             "speed": random.uniform(0.02,0.05),
             "filled": random.random() < 0.2, 
             "fill_color_offset": random.uniform(0, 2*math.pi)} 
            for r in range(0, HEIGHT+50, 50) for c in range(0, WIDTH+50, 50)]

def draw_background():
    # Tam ekranda arkaplanın tüm ekranı kaplaması için dinamik boyut
    w, h = screen.get_size()
    
    for hex_obj in bg_hexes:
        # Eğer hex ekran dışındaysa çizme (performans)
        if hex_obj["x"] > w + 50 or hex_obj["y"] > h + 50: continue

        hex_obj["pulse"] += hex_obj["speed"]
        alpha = int((math.sin(hex_obj["pulse"]) + 1) * 15) + 10 
        
        points = []
        for i in range(6):
            ang = math.radians(60 * i)
            px = int(hex_obj["x"] + 25 * math.cos(ang))
            py = int(hex_obj["y"] + 25 * math.sin(ang))
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
def draw_neon_text(text, font, x, y, color, center=True, shadow=False):
    if shadow:
        shadow_surf = font.render(text, True, C_SHADOW)
        shadow_rect = shadow_surf.get_rect()
        if center: shadow_rect.center = (x + 5, y + 5) 
        else: shadow_rect.topleft = (x + 5, y + 5)
        screen.blit(shadow_surf, shadow_rect)

    text_surf = font.render(text, True, color)
    rect = text_surf.get_rect()
    if center: rect.center = (x, y)
    else: rect.topleft = (x, y)
    screen.blit(text_surf, rect)

def draw_cyber_button(text, x, y, w, h, active=False, special=False):
    rect = pygame.Rect(x, y, w, h)
    mp = pygame.mouse.get_pos()
    hover = rect.collidepoint(mp)
    
    base_col, border_col = ((20, 30, 40), (60, 70, 80))
    
    if special: base_col, border_col = (150, 0, 80), (255, 100, 200)
    elif active: base_col, border_col = C_NEON_BLUE, C_NEON_BLUE
    
    if hover and not active: base_col = (40, 50, 60)
    
    pygame.draw.rect(screen, base_col, rect, border_radius=5)
    pygame.draw.rect(screen, border_col, rect, 2, border_radius=5)
    
    txt_col = (255, 255, 255) if (active or special) else (150, 150, 150)
    if active and not special: txt_col = (0, 0, 0)

    text_surf = font_md.render(text, True, txt_col)
    screen.blit(text_surf, (x + (w - text_surf.get_width())//2, y + (h - text_surf.get_height())//2))
    
    # Click Logic & Sound
    clicked = hover and pygame.mouse.get_pressed()[0]
    if clicked:
         # Butona basıldığı an ses çalması için (tekrarı önlemek için main loopta kontrol edilir ama burada basitlik için)
         # Bu fonksiyon her frame çağrıldığı için sesi burada çalmak "trrrrr" diye ses yapabilir.
         # O yüzden main loop'ta `mouse_clicked_this_frame` ile birleştireceğiz.
         pass
         
    return clicked

def draw_analysis_panel():
    panel_w = 380
    panel_rect = pygame.Rect(WIDTH - panel_w - 20, 80, panel_w, HEIGHT - 100)
    
    s = pygame.Surface((panel_rect.w, panel_rect.h), pygame.SRCALPHA)
    s.fill(C_DARK_PANEL)
    screen.blit(s, panel_rect)
    pygame.draw.rect(screen, C_NEON_BLUE, panel_rect, 2)
    
    x_start = panel_rect.x + 20
    y_curr = panel_rect.y + 20
    
    draw_neon_text(">> SYSTEM KERNEL <<", font_md, panel_rect.centerx, y_curr, C_NEON_BLUE)
    y_curr += 40
    
    draw_neon_text(f"ALGORITHM: Hybrid MCTS + A*", font_sm, x_start, y_curr, C_TEXT_WHITE, False)
    y_curr += 25
    draw_neon_text(f"HEURISTIC: Pathfinding + Bridge", font_sm, x_start, y_curr, C_TEXT_WHITE, False)
    y_curr += 25
    draw_neon_text(f"DIFFICULTY: {difficulty.upper()}", font_sm, x_start, y_curr, C_TEXT_WHITE, False)
    y_curr += 30

    rl_status = "RL: ACTIVE" if RL_ENABLED else "RL: DISABLED"
    rl_col = C_GREEN_MATRIX if RL_ENABLED else (100, 100, 100)
    draw_neon_text(rl_status, font_sm, x_start, y_curr, rl_col, False)
    y_curr += 25

    status_txt = "PATH VISUAL: ON" if show_path_mode else "PATH VISUAL: OFF"
    status_col = C_YELLOW_PATH if show_path_mode else (100, 100, 100)
    draw_neon_text(status_txt, font_sm, x_start, y_curr, status_col, False)
    y_curr += 25
    
    pause_txt = "STATUS: PAUSED" if game_paused else "STATUS: RUNNING"
    pause_col = C_ALERT if game_paused else C_GREEN_MATRIX
    draw_neon_text(pause_txt, font_sm, x_start, y_curr, pause_col, False)
    
    pygame.draw.line(screen, (50, 60, 80), (x_start, y_curr+30), (panel_rect.right-20, y_curr+30), 1)
    y_curr += 50
    
    if last_ai_analysis:
        info = last_ai_analysis
        draw_neon_text(f"ITERATIONS: {info['iterations']}", font_sm, x_start, y_curr, C_GREEN_MATRIX, False)
        y_curr += 25
        draw_neon_text(f"SIMULATIONS: {info['total_simulations']}", font_sm, x_start, y_curr, C_GREEN_MATRIX, False)
        y_curr += 40
        
        draw_neon_text("MOVE   SCORE   CONFIDENCE", font_xs, x_start, y_curr, (100, 200, 255), False)
        y_curr += 20
        
        for cand in info['candidates']:
            move_str = f"[{cand['move'][0]},{cand['move'][1]}]"
            score_str = f"{cand['final_score']:.2f}"
            conf_str = f"{cand['confidence']:.1f}%"
            col = C_GREEN_MATRIX if cand == info['candidates'][0] else C_TEXT_WHITE
            
            row_txt = f"{move_str:<8} {score_str:<8} {conf_str}"
            draw_neon_text(row_txt, font_sm, x_start, y_curr, col, False)
            
            bar_w = int((cand['confidence']/100) * 100)
            pygame.draw.rect(screen, (50, 50, 50), (panel_rect.right - 120, y_curr+5, 100, 5))
            pygame.draw.rect(screen, col, (panel_rect.right - 120, y_curr+5, bar_w, 5))
            
            y_curr += 25
    else:
        draw_neon_text("WAITING FOR AI INPUT...", font_sm, x_start, y_curr, (100, 100, 100), False)

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
        WIDTH, HEIGHT = screen.get_size() # Ekran boyutlarını güncelle
    else:
        WIDTH, HEIGHT = 1400, 800
        screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)

def reset_game():
    global game, ai_agent, ai_agent_2, last_ai_analysis, current_ai_path, confetti_pieces, game_history, game_paused, game_over_sound_played
    game = HexGame(BOARD_SIZE)
    last_ai_analysis = None
    current_ai_path = [] 
    confetti_pieces = [] 
    game_history = [] 
    game_paused = False
    game_over_sound_played = False
    
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

while running:
    screen.fill(BG_COLOR) 
    draw_background() 

    mouse_clicked_this_frame = False
    for event in pygame.event.get():
        if event.type == pygame.QUIT: running = False
        
        # Ekran boyutu değişirse (Windowed modda)
        if event.type == pygame.VIDEORESIZE:
            if not is_fullscreen:
                WIDTH, HEIGHT = event.w, event.h
                screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)

        # Klavye Kısayolları
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_f or event.key == pygame.K_F11: # F veya F11 ile tam ekran
                toggle_fullscreen()
            if event.key == pygame.K_p: # P ile Pause
                if current_state == STATE_GAME:
                    game_paused = not game_paused
                    sound_manager.play("click")
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_clicked_this_frame = True
            
            if current_state == STATE_GAME:
                if game and not game.winner and not game_paused:
                    # Human Turn
                    is_human_turn = False
                    if game_mode == "HvH":
                        is_human_turn = True
                    elif game_mode == "HvAI" and game.current_player == PLAYER_1:
                        is_human_turn = True
                    
                    if is_human_turn:
                        mp = pygame.mouse.get_pos()
                        if mp[1] > 60: # Menü çubuğuna tıklamıyorsa
                            gp = pixel_to_hex(mp[0], mp[1])
                            if gp:
                                record_game_state(game, game.current_player)
                                if game.make_move(gp[0], gp[1]):
                                    pass # Move successful

    # --- MENU DESIGN ---
    if current_state == STATE_MENU:
        cx, cy = WIDTH//2, HEIGHT//2
        
        pygame.draw.rect(screen, C_DARK_PANEL, (cx-250, cy-220, 500, 500), border_radius=10)
        pygame.draw.rect(screen, C_NEON_BLUE, (cx-250, cy-220, 500, 500), 2, border_radius=10)
        
        draw_neon_text("HEX BOARD GAME", font_xl_fancy, cx, cy-260, C_NEON_BLUE, shadow=True)
        
        curr_y = cy - 180
        draw_neon_text("SELECT MODE", font_sm, cx, curr_y, C_TEXT_WHITE)
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
        draw_neon_text("DIFFICULTY", font_sm, cx, curr_y, C_TEXT_WHITE)
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
        
        # Fullscreen Toggle in Menu
        fs_text = "EXIT FULLSCREEN" if is_fullscreen else "FULLSCREEN"
        if draw_cyber_button(fs_text, WIDTH - 160, HEIGHT - 60, 150, 40):
            if mouse_clicked_this_frame:
                toggle_fullscreen()
                sound_manager.play("click")


    elif current_state == STATE_GAME:
        pygame.draw.rect(screen, C_DARK_PANEL, (0, 0, WIDTH, 60))
        pygame.draw.line(screen, C_NEON_BLUE, (0, 60), (WIDTH, 60), 2)
        
        # TOP MENU BUTTONS
        
        # Fullscreen Button (Top Right most)
        fs_label = "[ ]" if is_fullscreen else "[F]"
        if draw_cyber_button(fs_label, WIDTH - 60, 10, 50, 40, is_fullscreen):
            if mouse_clicked_this_frame:
                toggle_fullscreen()
                sound_manager.play("click")

        pause_label = "RESUME" if game_paused else "PAUSE"
        if draw_cyber_button(pause_label, WIDTH - 600, 10, 120, 40, game_paused):
            if mouse_clicked_this_frame:
                game_paused = not game_paused
                sound_manager.play("click")

        dev_col = C_GREEN_MATRIX if dev_mode else (50, 50, 50)
        if draw_cyber_button("DEV MODE", WIDTH - 460, 10, 120, 40, dev_mode):
            if mouse_clicked_this_frame:
                dev_mode = not dev_mode
                sound_manager.play("click")
        if dev_mode:
            path_col = C_YELLOW_PATH if show_path_mode else (50, 50, 50)
            if draw_cyber_button("SHOW PATH", WIDTH - 320, 10, 120, 40, show_path_mode):
                if mouse_clicked_this_frame:
                    show_path_mode = not show_path_mode
                    sound_manager.play("click")
        
        if draw_cyber_button("EXIT", WIDTH - 180, 10, 100, 40):
             if mouse_clicked_this_frame:
                sound_manager.play("click")
                current_state = STATE_MENU

        turn_txt = "RED TURN" if game.current_player == 1 else "BLUE TURN"
        turn_col = C_NEON_PINK if game.current_player == 1 else C_NEON_BLUE
        
        if game_paused:
            draw_neon_text(">> SYSTEM PAUSED <<", font_md, WIDTH//2, 30, C_ALERT)
        else:
            draw_neon_text(f">> {turn_txt} <<", font_md, WIDTH//2, 30, turn_col)

        # BOARD BORDERS
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
            draw_neon_text(str(c+1), font_sm, cx, cy - 40, C_TEXT_DIM)
        
        for r in range(BOARD_SIZE):
            cx, cy = get_hex_center(r, 0)
            draw_neon_text(str(r+1), font_sm, cx - 45, cy, C_TEXT_DIM)

        # DRAW HEXES
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                x, y = get_hex_center(r, c)
                val = game.board[r, c]
                col = (30, 35, 45) 
                if val == PLAYER_1: col = C_NEON_PINK
                elif val == PLAYER_2: col = C_NEON_BLUE
                
                mp = pygame.mouse.get_pos()
                is_hover = False
                if not game_paused and val == 0 and math.hypot(mp[0]-x, mp[1]-y) < HEX_SIZE*0.9:
                    if game_mode == "HvH": is_hover = True
                    elif game_mode == "HvAI" and game.current_player == PLAYER_1: is_hover = True
                
                if is_hover: col = (60, 60, 70)
                draw_glowing_hex(screen, col, x, y, HEX_SIZE-2, glow=(val!=0))
                
                if dev_mode and last_ai_analysis and val == 0:
                    for cand in last_ai_analysis['candidates']:
                        if cand['move'] == (r,c):
                            score_txt = f"{cand['final_score']:.1f}"
                            draw_neon_text(score_txt, font_xs, x, y, C_GREEN_MATRIX)

        # --- PATH VISUALIZATION CHANGES ---
        if game and not game.winner and show_path_mode:
            # 1. Check if board is empty (Tahta boşsa çizme)
            if np.any(game.board != 0):
                current_ai_path = get_visual_path(game, game.current_player)
                
                # 2. Dynamic Path Color (Sıradaki oyuncuya göre renk)
                path_color = C_NEON_PINK if game.current_player == PLAYER_1 else C_NEON_BLUE
                
                if current_ai_path and len(current_ai_path) > 1:
                    points = []
                    for (r, c) in current_ai_path:
                        points.append(get_hex_center(r, c))
                    
                    if len(points) > 1:
                        # Çizgiyi oyuncu renginde çiz
                        pygame.draw.lines(screen, path_color, False, points, 4)
                        for p in points:
                            pygame.draw.circle(screen, path_color, (int(p[0]), int(p[1])), 6)
                            # İçindeki nokta biraz daha açık renk olsun
                            highlight_color = (min(path_color[0]+50, 255), min(path_color[1]+50, 255), min(path_color[2]+50, 255))
                            pygame.draw.circle(screen, highlight_color, (int(p[0]), int(p[1])), 3)

        if dev_mode:
            draw_analysis_panel()

        # AI TURN
        if not game.winner and not game_paused:
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
                pygame.display.flip() 
                move, info = active_agent.get_best_move(game)
                last_ai_analysis = info
                if move: 
                    record_game_state(game, game.current_player)
                    game.make_move(move[0], move[1])
        
        # --- GAME OVER LOGIC WITH SOUND ---
        if game.winner:
            if not confetti_pieces: 
                spawn_confetti(200, (0, WIDTH), (0, HEIGHT // 2))
                
                # --- PLAY SOUND ONCE ---
                if not game_over_sound_played:
                    if game_mode == "HvAI":
                        if game.winner == PLAYER_1: # Human (Red) Wins
                             sound_manager.play("win")
                        else: # AI (Blue) Wins
                             sound_manager.play("lose")
                    else: # HvH or AIvAI -> Applause anyway
                        sound_manager.play("win")
                    
                    game_over_sound_played = True

                if RL_ENABLED and game_history:
                    print(f"🎓 Updating RL Weights... Winner: {'RED' if game.winner == PLAYER_1 else 'BLUE'}")
                    update_rl_weights(game_history, game.winner)
                    print("✓ Weights Updated!")
            
            s = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            s.fill((0,0,0, 200))
            screen.blit(s, (0,0))
            
            w_txt = "RED WINS" if game.winner == PLAYER_1 else "BLUE WINS"
            w_col = C_NEON_PINK if game.winner == PLAYER_1 else C_NEON_BLUE
            draw_neon_text(w_txt, font_xl_fancy, WIDTH//2, HEIGHT//2 - 40, w_col, shadow=True) 
            
            if draw_cyber_button("RESTART", WIDTH//2 - 100, HEIGHT//2 + 60, 200, 50, False, True):
                if mouse_clicked_this_frame:
                    sound_manager.play("click")
                    reset_game()
            
            if draw_cyber_button("MAIN MENU", WIDTH//2 - 100, HEIGHT//2 + 130, 200, 50, False, False):
                if mouse_clicked_this_frame:
                    sound_manager.play("click")
                    current_state = STATE_MENU
                    confetti_pieces = []

    for piece in list(confetti_pieces): 
        piece.update()
        piece.draw(screen)
        if piece.lifetime <= 0 or not (0 < piece.pos.x < WIDTH and 0 < piece.pos.y < HEIGHT):
            confetti_pieces.remove(piece)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()