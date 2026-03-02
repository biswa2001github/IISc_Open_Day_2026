import gymnasium as gym
import pygame
import sys
import numpy as np
import ale_py

# ================= CONFIG & STYLE =================
WIDTH, HEIGHT = 1000, 700  # High-res window
FPS = 30 # increase or decrease ball speed

# Colors (Retro Synthwave Theme)
BG_COLOR = (15, 15, 25)
NEON_BLUE = (0, 255, 255)
NEON_PINK = (255, 0, 255)
WHITE = (240, 240, 240)
OVERLAY = (0, 0, 0, 160)

MAX_SCORE = 5 # Game ends when someone hits this

# ================= INITIALIZE =================
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("🕹️ Cyber-Pong: Human vs Atari")
clock = pygame.time.Clock()

font_huge = pygame.font.SysFont("Impact", 100)
font_big = pygame.font.SysFont("Impact", 50)
font_med = pygame.font.SysFont("Arial", 30, bold=True)

# Initialize Gymnasium with rgb_array so we can upscale it
env = gym.make("ALE/Pong-v5", render_mode="rgb_array", frameskip=1)

# ================= UTILS =================
def draw_text(text, font, color, x, y):
    img = font.render(text, True, color)
    rect = img.get_rect(center=(x, y))
    screen.blit(img, rect)

def get_action():
    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP] or keys[pygame.K_w]: return 2  # Up
    if keys[pygame.K_DOWN] or keys[pygame.K_s]: return 3  # Down
    if keys[pygame.K_SPACE]: return 1  # Serve
    return 0

# ================= SCREENS =================
def splash_screen():
    while True:
        screen.fill(BG_COLOR)
        # Draw some decorative lines
        for i in range(0, WIDTH, 40):
            pygame.draw.line(screen, (30, 30, 60), (i, 0), (i, HEIGHT))
        
        draw_text("CYBER PONG", font_huge, NEON_BLUE, WIDTH//2, HEIGHT//2 - 50)
        draw_text("PRESS SPACE TO START", font_med, NEON_PINK, WIDTH//2, HEIGHT//2 + 50)
        draw_text("SHIFT: RESET ANYTIME  |  WASD: MOVE", font_med, WHITE, WIDTH//2, HEIGHT - 50)
        
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                return

def game_over_screen(winner_text):
    while True:
        # Create a blur/overlay effect
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill(OVERLAY)
        screen.blit(overlay, (0, 0))
        
        draw_text(winner_text, font_huge, WHITE, WIDTH//2, HEIGHT//2 - 50)
        draw_text("PRESS SHIFT FOR NEW GAME", font_med, NEON_BLUE, WIDTH//2, HEIGHT//2 + 50)
        
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_LSHIFT, pygame.K_RSHIFT]:
                    return # Reset game

# ================= MAIN LOOP =================
def play_game():
    obs, info = env.reset()
    player_score = 0
    cpu_score = 0
    
    while True:
        # 1. Input Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); sys.exit()
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            return # Global Reset

        # 2. Step Environment
        action = get_action()
        obs, reward, terminated, truncated, info = env.step(action)
        
        # In Pong-v5: reward +1 means player won point, -1 means CPU won
        if reward > 0: player_score += 1
        if reward < 0: cpu_score += 1

        # 3. Graphics Processing
        # Get the RGB frame from Atari and convert to Pygame Surface
        frame = env.render()
        frame = np.transpose(frame, (1, 0, 2)) # Flip for Pygame
        surf = pygame.surfarray.make_surface(frame)
        
        # Upscale the tiny Atari screen to fill our high-res window
        upscaled_surf = pygame.transform.scale(surf, (WIDTH, HEIGHT))
        screen.blit(upscaled_surf, (0, 0))

        # 4. Custom UI Overlay (Beautification)
        # Scoreboard
        pygame.draw.rect(screen, BG_COLOR, (WIDTH//2 - 100, 10, 200, 60), border_radius=10)
        pygame.draw.rect(screen, NEON_BLUE, (WIDTH//2 - 100, 10, 200, 60), 3, border_radius=10)
        draw_text(f"{cpu_score}  |  {player_score}", font_big, WHITE, WIDTH//2, 40)
        
        # Side labels
        draw_text("AI", font_med, NEON_PINK, 100, 40)
        draw_text("PLAYER", font_med, NEON_BLUE, WIDTH - 100, 40)

        # 5. Win Logic
        if player_score >= MAX_SCORE:
            game_over_screen("PLAYER WINS!")
            return
        elif cpu_score >= MAX_SCORE:
            game_over_screen("AI WINS!")
            return

        if terminated or truncated:
            env.reset()

        pygame.display.flip()
        clock.tick(FPS)
        

# Run State Machine
while True:
    splash_screen()
    play_game()