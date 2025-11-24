import pygame
import pygame.gfxdraw
import math
import random
import time
import numpy as np # Ses üretimi için gerekli
from hex_engine import HexGame, PLAYER_1, PLAYER_2
from hybrid_agent import HybridAI, get_visual_path, update_rl_weights, RL_ENABLED

# --- AYARLAR ---
WIDTH, HEIGHT = 1400, 800 
BG_COLOR = (5, 5, 10)
HEX_SIZE = 30
BOARD_SIZE = 11

OFFSET_X = 200 
OFFSET_Y = 160

# --- RENKLER ---
C_NEON_BLUE = (0, 255, 255)
C_NEON_PINK = (255, 0, 128)
C_DARK_PANEL = (10, 15, 25, 240) 
C_TEXT_WHITE = (220, 230, 255)
C_TEXT_DIM = (100, 110, 130)
C_GREEN_MATRIX = (0, 255, 65)
C_YELLOW_PATH = (255, 220, 0)
C_SHADOW = (0, 0, 0, 150)

pygame.init()
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512) # Ses motorunu başlat
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("HEX BOARD GAME")

# --- FONT YÖNETİMİ ---
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
STATE_MENU, STATE_GAME, STATE_PAUSE = 0, 1, 2
current_state = STATE_MENU
game_mode = "HvAI"
difficulty = "Medium"
dev_mode = False 
show_path_mode = False 
current_ai_path = [] 
last_ai_analysis = None 
confetti_pieces = []
board_changed = True
game_history = []
last_hovered_button = None # Ses için hover takibi

game = None
ai_agent, ai_agent_2 = None, None

# --- SES YÖNETİCİSİ (SENTETİK) ---
# --- GELİŞMİŞ SES YÖNETİCİSİ (SOFT & FX) ---
class SoundManager:
    def __init__(self):
        # Frekansı biraz düşürdük, daha doğal tınlaması için
        try:
            pygame.mixer.quit() # Öncekini kapat
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        except: pass
        self.sounds = {}
        self.generate_sounds()

    def generate_wave(self, duration, freq, wave_type="sine", vol=0.5, slide_to=None):
        sample_rate = 44100
        n_samples = int(sample_rate * duration)
        t = np.linspace(0, duration, n_samples, False)
        
        if wave_type == "noise":
            # Alkış için Beyaz/Pembe gürültü
            wave = np.random.uniform(-1, 1, n_samples)
        else:
            # Frekans kayması (Slide) varsa (Örn: Su damlası veya Üzgün ses için)
            if slide_to:
                # Frekans zamanla değişir
                inst_freq = np.linspace(freq, slide_to, n_samples)
                phase = 2 * np.pi * np.cumsum(inst_freq) / sample_rate
                wave = np.sin(phase)
            else:
                wave = np.sin(2 * np.pi * freq * t)

        # ENVELOPE (Zarf) - Sesi yumuşatmak için en önemli kısım
        # Sert başlangıç ve bitişleri yok eder (Fade In / Fade Out)
        attack = int(n_samples * 0.1)  # %10 açılış
        decay = int(n_samples * 0.9)   # %90 kapanış
        
        env = np.ones(n_samples)
        # Yumuşak giriş
        env[:attack] = np.linspace(0, 1, attack)
        # Yumuşak çıkış (Exponential decay daha doğal duyulur)
        env[-decay:] = np.linspace(1, 0, decay) ** 2 
        
        wave = wave * env
        
        # Stereo (2 kanal) ve 16-bit dönüşüm
        wave = np.column_stack((wave, wave))
        wave = (wave * 32767 * vol).astype(np.int16)
        return pygame.sndarray.make_sound(wave)

    def generate_applause(self):
        # Alkış, rastgele aralıklarla patlayan gürültülerdir
        duration = 3.0 # 3 saniye alkış
        sample_rate = 44100
        n_samples = int(sample_rate * duration)
        master_wave = np.zeros(n_samples)
        
        # 50 farklı "el çırpma" sesi oluşturup rastgele zamanlara yerleştiriyoruz
        for _ in range(50):
            clap_len = int(sample_rate * random.uniform(0.05, 0.15))
            start = random.randint(0, n_samples - clap_len)
            
            # Bir el çırpması = Hızlı gürültü patlaması
            clap = np.random.uniform(-1, 1, clap_len)
            env = np.linspace(1, 0, clap_len) ** 4 # Hızlı sönümlenme
            clap = clap * env
            
            # Ana sese ekle
            master_wave[start:start+clap_len] += clap * 0.3

        # Normalize et (Patlamaması için)
        max_val = np.max(np.abs(master_wave))
        if max_val > 0: master_wave /= max_val
        
        # Stereo yap
        master_wave = np.column_stack((master_wave, master_wave))
        master_wave = (master_wave * 32767 * 0.5).astype(np.int16)
        return pygame.sndarray.make_sound(master_wave)

    def generate_sad_sound(self):
        # "Wah-wah-wah" efekti (Azalan 3 nota)
        s1 = self.generate_wave(0.4, 400, "sine", 0.5, slide_to=300) # Düşen notalar
        s2 = self.generate_wave(0.4, 300, "sine", 0.5, slide_to=200)
        s3 = self.generate_wave(0.8, 200, "sine", 0.5, slide_to=100)
        
        # Sesleri birleştirme yeteneği pygame mixer'da sınırlı olduğu için
        # Basitçe en uzun olanı döndüreceğiz ama arka arkaya çalacağız.
        # Burada tek bir uzun "Ahhh" benzeri düşen ses yapalım.
        return self.generate_wave(1.5, 400, "sine", 0.6, slide_to=100)

    def generate_sounds(self):
        try:
            # Move: Su damlası gibi (Yüksekten düşüğe hızlı kayma)
            self.sounds['move'] = self.generate_wave(0.15, 800, "sine", 0.4, slide_to=300)
            
            # Hover: Çok hafif, yumuşak bir ton
            self.sounds['hover'] = self.generate_wave(0.05, 400, "sine", 0.1)
            
            # Click: Biraz daha tok
            self.sounds['click'] = self.generate_wave(0.1, 600, "sine", 0.3)
            
            # Win: Alkış efekti
            self.sounds['win'] = self.generate_applause()
            
            # Lose: Hüzünlü kayma sesi
            self.sounds['lose'] = self.generate_sad_sound()
            
        except Exception as e:
            print(f"Ses hatası: {e}")

    def play(self, name):
        if name in self.sounds:
            # Üst üste binmeyi engellemek için (özellikle hover'da)
            # Ama win/lose için durdurma yapma
            if name == 'hover': 
                self.sounds[name].stop()
            self.sounds[name].play()

sound_manager = SoundManager()

# --- ARKA PLAN ---
bg_hexes = [{"x": c, "y": r, "pulse": random.uniform(0,6), 
             "speed": random.uniform(0.02,0.05),
             "filled": random.random() < 0.2, 
             "fill_color_offset": random.uniform(0, 2*math.pi)} 
            for r in range(0, HEIGHT+50, 50) for c in range(0, WIDTH+50, 50)]

def draw_background():
    for h in bg_hexes:
        h["pulse"] += h["speed"]
        alpha = int((math.sin(h["pulse"]) + 1) * 15) + 10 
        
        points = []
        for i in range(6):
            ang = math.radians(60 * i)
            px = int(h["x"] + 25 * math.cos(ang))
            py = int(h["y"] + 25 * math.sin(ang))
            points.append((px, py))
        
        if h["filled"]:
            color_intensity = (math.sin(h["pulse"] + h["fill_color_offset"]) + 1) / 2 
            fill_col = (int(color_intensity * 50) + 20, 
                        int(color_intensity * 150) + 50, 
                        int(color_intensity * 200) + 50, alpha)
            try: pygame.gfxdraw.filled_polygon(screen, points, fill_col)
            except: pass

        try:
            pygame.gfxdraw.aapolygon(screen, points, (0, 200, 255, alpha))
        except: pass

# --- KONFETİ SINIFI ---
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

# --- UI FONKSİYONLARI ---
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

def draw_cyber_button(text, x, y, w, h, active=False, special=False, btn_id=None):
    global last_hovered_button
    rect = pygame.Rect(x, y, w, h)
    mp = pygame.mouse.get_pos()
    hover = rect.collidepoint(mp)
    
    # Hover sesi kontrolü
    if hover and last_hovered_button != btn_id:
        sound_manager.play('hover')
        last_hovered_button = btn_id
    
    if special: base_col, border_col = (150, 0, 80), (255, 100, 200)
    else: base_col, border_col = ((20, 30, 40), (60, 70, 80)) if not active else (C_NEON_BLUE, C_NEON_BLUE)
    
    if hover and not active: base_col = (40, 50, 60)
    
    pygame.draw.rect(screen, base_col, rect, border_radius=5)
    pygame.draw.rect(screen, border_col, rect, 2, border_radius=5)
    
    txt_col = (255, 255, 255) if (active or special) else (150, 150, 150)
    text_surf = font_md.render(text, True, txt_col)
    screen.blit(text_surf, (x + (w - text_surf.get_width())//2, y + (h - text_surf.get_height())//2))
    
    if hover and pygame.mouse.get_pressed()[0]:
        sound_manager.play('click') # Tıklama sesi
        return True
    return False

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
    draw_neon_text(f"HEURISTIC: Pathfinding", font_sm, x_start, y_curr, C_TEXT_WHITE, False)
    y_curr += 25
    draw_neon_text(f"DIFFICULTY: {difficulty.upper()}", font_sm, x_start, y_curr, C_TEXT_WHITE, False)
    y_curr += 30

    status_txt = "PATH VISUAL: ON" if show_path_mode else "PATH VISUAL: OFF"
    status_col = C_YELLOW_PATH if show_path_mode else (100, 100, 100)
    draw_neon_text(status_txt, font_sm, x_start, y_curr, status_col, False)
    
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

# --- OYUN YARDIMCILARI ---
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

def record_game_state(game, player):
    if RL_ENABLED:
        from hybrid_agent import get_features_vector
        features = get_features_vector(game, player)
        game_history.append((features, player))

def reset_game():
    global game, ai_agent, ai_agent_2, last_ai_analysis, current_ai_path, confetti_pieces, board_changed, game_history
    game = HexGame(BOARD_SIZE)
    last_ai_analysis = None
    current_ai_path = [] 
    confetti_pieces = [] 
    game_history = []
    board_changed = True 
    
    if game_mode == "HvAI": ai_agent = HybridAI(difficulty)
    elif game_mode == "AIvAI":
        ai_agent = HybridAI(difficulty)
        ai_agent_2 = HybridAI(difficulty)

# --- ANA DÖNGÜ ---
running = True
clock = pygame.time.Clock()

while running:
    screen.fill(BG_COLOR) 
    draw_background() 
    
    # Mouse hover takibi için (boş yere) sıfırlama
    hovering_any = False

    for event in pygame.event.get():
        if event.type == pygame.QUIT: running = False
        if event.type == pygame.MOUSEBUTTONDOWN and current_state == STATE_GAME:
            if game and not game.winner:
                is_human_turn = False
                if game_mode == "HvH":
                    is_human_turn = True
                elif game_mode == "HvAI" and game.current_player == PLAYER_1:
                    is_human_turn = True
                
                if is_human_turn:
                    mp = pygame.mouse.get_pos()
                    gp = pixel_to_hex(mp[0], mp[1])
                    if gp: 
                        if game.make_move(gp[0], gp[1]):
                            sound_manager.play('move') # Hamle sesi
                            record_game_state(game, game.current_player)
                            board_changed = True 

    # --- MENÜ TASARIMI ---
    if current_state == STATE_MENU:
        cx, cy = WIDTH//2, HEIGHT//2
        
        pygame.draw.rect(screen, C_DARK_PANEL, (cx-250, cy-220, 500, 500), border_radius=10)
        pygame.draw.rect(screen, C_NEON_BLUE, (cx-250, cy-220, 500, 500), 2, border_radius=10)
        
        draw_neon_text("HEX BOARD GAME", font_xl_fancy, cx, cy-260, C_NEON_BLUE, shadow=True)
        
        curr_y = cy - 180
        draw_neon_text("SELECT MODE", font_sm, cx, curr_y, C_TEXT_WHITE)
        curr_y += 30
        
        if draw_cyber_button("HUMAN vs AI", cx-200, curr_y, 190, 50, game_mode=="HvAI", btn_id="hvai"): game_mode="HvAI"
        if draw_cyber_button("AI vs AI", cx+10, curr_y, 190, 50, game_mode=="AIvAI", btn_id="aivai"): game_mode="AIvAI"
        
        curr_y += 60
        if draw_cyber_button("HUMAN vs HUMAN", cx-200, curr_y, 400, 50, game_mode=="HvH", btn_id="hvh"): game_mode="HvH"
        
        curr_y += 70
        draw_neon_text("DIFFICULTY", font_sm, cx, curr_y, C_TEXT_WHITE)
        curr_y += 30
        dx = cx - 190
        for d in ["Easy", "Medium", "Hard"]:
            is_active_difficulty = (game_mode != "HvH" and difficulty == d)
            if game_mode == "HvH":
                draw_cyber_button(d, dx, curr_y, 120, 40, False, btn_id=f"diff_{d}") 
                s = pygame.Surface((120, 40), pygame.SRCALPHA)
                s.fill((0, 0, 0, 150)) 
                screen.blit(s, (dx, curr_y))
            else:
                if draw_cyber_button(d, dx, curr_y, 120, 40, is_active_difficulty, btn_id=f"diff_{d}"): 
                    difficulty = d
            dx += 130
            
        if draw_cyber_button("START", cx-150, cy+180, 300, 60, False, True, btn_id="start"):
            reset_game()
            current_state = STATE_GAME

    elif current_state == STATE_GAME:
        pygame.draw.rect(screen, C_DARK_PANEL, (0, 0, WIDTH, 60))
        pygame.draw.line(screen, C_NEON_BLUE, (0, 60), (WIDTH, 60), 2)
        
        dev_col = C_GREEN_MATRIX if dev_mode else (50, 50, 50)
        if draw_cyber_button("DEV MODE", WIDTH - 420, 10, 120, 40, dev_mode, btn_id="dev"):
            dev_mode = not dev_mode
        if dev_mode:
            path_col = C_YELLOW_PATH if show_path_mode else (50, 50, 50)
            if draw_cyber_button("SHOW PATH", WIDTH - 280, 10, 120, 40, show_path_mode, btn_id="path"):
                show_path_mode = not show_path_mode
        if draw_cyber_button("PAUSE", WIDTH - 150, 10, 100, 40, btn_id="pause"):
            current_state = STATE_PAUSE

        turn_txt = "RED TURN" if game.current_player == 1 else "BLUE TURN"
        turn_col = C_NEON_PINK if game.current_player == 1 else C_NEON_BLUE
        draw_neon_text(f">> {turn_txt} <<", font_md, WIDTH//2, 30, turn_col)

        # --- ÇERÇEVE ÇİZGİLERİ ---
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
        
        # --- NUMARALANDIRMA ---
        for c in range(BOARD_SIZE):
            cx, cy = get_hex_center(0, c)
            draw_neon_text(str(c+1), font_sm, cx, cy - 40, C_TEXT_DIM)
        for r in range(BOARD_SIZE):
            cx, cy = get_hex_center(r, 0)
            draw_neon_text(str(r+1), font_sm, cx - 45, cy, C_TEXT_DIM)

        # --- 1. TAHTA ÇİZİMİ ---
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                x, y = get_hex_center(r, c)
                val = game.board[r, c]
                col = (30, 35, 45) 
                if val == PLAYER_1: col = C_NEON_PINK
                elif val == PLAYER_2: col = C_NEON_BLUE
                
                mp = pygame.mouse.get_pos()
                is_hover = False
                if val == 0 and math.hypot(mp[0]-x, mp[1]-y) < HEX_SIZE*0.9:
                    if game_mode == "HvH": is_hover = True
                    elif game_mode == "HvAI" and game.current_player == PLAYER_1: is_hover = True
                
                if is_hover: col = (60, 60, 70)
                draw_glowing_hex(screen, col, x, y, HEX_SIZE-2, glow=(val!=0))
                
                if dev_mode and last_ai_analysis and val == 0:
                    for cand in last_ai_analysis['candidates']:
                        if cand['move'] == (r,c):
                            score_txt = f"{cand['final_score']:.1f}"
                            draw_neon_text(score_txt, font_xs, x, y, C_GREEN_MATRIX)

        # --- 2. YOL ÇİZİMİ (ÜSTE GELMESİ İÇİN) ---
        if game and not game.winner and show_path_mode:
            if board_changed:
                if game_mode != "HvH" and game.current_player != PLAYER_1 and last_ai_analysis and "predicted_path" in last_ai_analysis:
                    current_ai_path = last_ai_analysis["predicted_path"]
                else:
                    current_ai_path = get_visual_path(game, game.current_player)
                board_changed = False 
            
            if current_ai_path and len(current_ai_path) > 1:
                points = []
                for (r, c) in current_ai_path:
                    points.append(get_hex_center(r, c))
                if len(points) > 1:
                    path_color = C_NEON_PINK if game.current_player == PLAYER_1 else C_NEON_BLUE
                    pygame.draw.lines(screen, path_color, False, points, 6)
                    for p in points:
                        pygame.draw.circle(screen, path_color, (int(p[0]), int(p[1])), 6)
                        pygame.draw.circle(screen, (255, 255, 255), (int(p[0]), int(p[1])), 3)

        if dev_mode:
            draw_analysis_panel()

        # AI HAMLESİ
        if not game.winner:
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
                if difficulty == "Hard":
                    time.sleep(1.0)
                move, info = active_agent.get_best_move(game)
                last_ai_analysis = info
                if move: 
                    game.make_move(move[0], move[1])
                    sound_manager.play('move') # Hamle sesi
                    record_game_state(game, game.current_player)
                    board_changed = True

        if game.winner:
            if not confetti_pieces: 
                # SES MANTIĞI (YENİ)
                # Eğer HvAI modundaysak ve kazanan Player 1 (İnsan) ise -> ALKIŞ
                if game_mode == "HvAI":
                    if game.winner == PLAYER_1:
                        sound_manager.play('win') # Alkış
                    else:
                        sound_manager.play('lose') # Hüzünlü ses (Ahhh)
                # Diğer modlarda (HvH veya AIvAI) her zaman alkış çalsın
                else:
                    sound_manager.play('win')

                spawn_confetti(200, (0, WIDTH), (0, HEIGHT // 2))
                if RL_ENABLED and game_history:
                    update_rl_weights(game_history, game.winner)
            
            s = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            s.fill((0,0,0, 200))
            screen.blit(s, (0, 0))
            
            w_txt = "RED WINS" if game.winner == PLAYER_1 else "BLUE WINS"
            w_col = C_NEON_PINK if game.winner == PLAYER_1 else C_NEON_BLUE
            draw_neon_text(w_txt, font_xl_fancy, WIDTH//2, HEIGHT//2 - 40, w_col, shadow=True) 
            
            if draw_cyber_button("RESTART", WIDTH//2 - 100, HEIGHT//2 + 60, 200, 50, False, True, btn_id="restart"):
                reset_game()
            
            if draw_cyber_button("MAIN MENU", WIDTH//2 - 100, HEIGHT//2 + 130, 200, 50, False, False, btn_id="menu"):
                current_state = STATE_MENU
                confetti_pieces = [] 

    # --- PAUSE MENÜSÜ ---
    elif current_state == STATE_PAUSE:
        s = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        s.fill((0, 0, 0, 150))
        screen.blit(s, (0, 0))
        
        panel_w, panel_h = 400, 350
        panel_x, panel_y = WIDTH//2 - panel_w//2, HEIGHT//2 - panel_h//2
        
        pygame.draw.rect(screen, C_DARK_PANEL, (panel_x, panel_y, panel_w, panel_h), border_radius=10)
        pygame.draw.rect(screen, C_NEON_BLUE, (panel_x, panel_y, panel_w, panel_h), 2, border_radius=10)
        
        draw_neon_text("GAME PAUSED", font_lg, WIDTH//2, panel_y + 50, C_TEXT_WHITE, shadow=True)
        
        curr_y = panel_y + 120
        if draw_cyber_button("RESUME", WIDTH//2 - 100, curr_y, 200, 50, False, True, btn_id="resume"):
            current_state = STATE_GAME
            
        curr_y += 70
        if draw_cyber_button("RESTART", WIDTH//2 - 100, curr_y, 200, 50, btn_id="p_restart"):
            reset_game()
            current_state = STATE_GAME
            
        curr_y += 70
        if draw_cyber_button("MAIN MENU", WIDTH//2 - 100, curr_y, 200, 50, btn_id="p_menu"):
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