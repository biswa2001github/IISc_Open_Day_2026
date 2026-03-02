import pygame
import sys
import numpy as np
from stable_baselines3 import PPO

# ================= CONFIG =================
WIDTH, HEIGHT = 1200, 600
HALF_WIDTH = WIDTH // 2

WHITE = (255, 255, 255)
BALL_COLOR = (255, 80, 80)
PADDLE_COLOR = (50, 120, 255)
BRICK_COLOR = (80, 220, 120)
TEXT_COLOR = (20, 20, 20)
DIVIDER_COLOR = (180, 180, 180)
OVERLAY_COLOR = (0, 0, 0, 150) # Semi-transparent for "Lost" side

PADDLE_WIDTH = 100
PADDLE_HEIGHT = 12
BALL_RADIUS = 10
BALL_SPEED = 5
PADDLE_SPEED = 10

BRICK_WIDTH = 80
BRICK_HEIGHT = 20
BRICK_ROWS = 5  # Increased rows for a better challenge
BRICK_COLUMNS = 6
BRICK_SPACING = 10

# ================= INIT =================
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Human vs AI Brick Breaker")
clock = pygame.time.Clock()

font_big = pygame.font.Font(None, 72)
font_med = pygame.font.Font(None, 36)

# ================= LOAD AI =================
# Ensure "brick_ai.zip" is in the same folder!
try:
    model = PPO.load("brick_ai")
except:
    print("AI Model not found! Please ensure 'brick_ai.zip' exists.")
    model = None

# ================= BACKGROUND =================
def draw_background():
    screen.fill((240, 240, 240)) # Light clean background
    pygame.draw.rect(screen, (220, 230, 255), (0, 0, HALF_WIDTH, HEIGHT)) # Human side tint
    pygame.draw.rect(screen, (255, 230, 230), (HALF_WIDTH, 0, HALF_WIDTH, HEIGHT)) # AI side tint

# ================= GAME STATE =================
def create_bricks(x_offset):
    bricks = []
    # Center the brick layout in the half-screen
    total_brick_w = BRICK_COLUMNS * (BRICK_WIDTH + BRICK_SPACING) - BRICK_SPACING
    start_x = x_offset + (HALF_WIDTH - total_brick_w) // 2
    
    for r in range(BRICK_ROWS):
        for c in range(BRICK_COLUMNS):
            x = start_x + c * (BRICK_WIDTH + BRICK_SPACING)
            y = r * (BRICK_HEIGHT + BRICK_SPACING) + 80
            bricks.append(pygame.Rect(x, y, BRICK_WIDTH, BRICK_HEIGHT))
    return bricks

def reset_side(x_offset):
    return {
        "ball_x": x_offset + HALF_WIDTH // 2,
        "ball_y": HEIGHT // 2,
        "ball_dx": np.random.choice([-1, 1]) * BALL_SPEED,
        "ball_dy": -BALL_SPEED,
        "paddle_x": x_offset + (HALF_WIDTH - PADDLE_WIDTH) // 2,
        "bricks": create_bricks(x_offset),
        "score": 0,
        "done": False,
        "won": False
    }

# ================= SCREENS =================
def wait_for_start():
    while True:
        screen.fill((15, 15, 40))
        t1 = font_big.render("BRICK BREAKER: HUMAN VS AI", True, WHITE)
        t2 = font_med.render("Press SPACE to Start", True, WHITE)
        t3 = font_med.render("Press SHIFT anytime to Reset", True, (150, 150, 150))
        
        screen.blit(t1, (WIDTH//2 - t1.get_width()//2, HEIGHT//2 - 60))
        screen.blit(t2, (WIDTH//2 - t2.get_width()//2, HEIGHT//2 + 20))
        screen.blit(t3, (WIDTH//2 - t3.get_width()//2, HEIGHT//2 + 60))
        pygame.display.flip()

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if e.type == pygame.KEYDOWN and e.key == pygame.K_SPACE:
                return reset_side(0), reset_side(HALF_WIDTH)

# ================= PHYSICS =================
def update_state(state, action, x_offset):
    if state["done"] or state["won"]:
        return

    # Paddle movement
    if action == 1: state["paddle_x"] -= PADDLE_SPEED
    elif action == 2: state["paddle_x"] += PADDLE_SPEED

    state["paddle_x"] = np.clip(state["paddle_x"], x_offset, x_offset + HALF_WIDTH - PADDLE_WIDTH)

    # Ball Movement
    state["ball_x"] += state["ball_dx"]
    state["ball_y"] += state["ball_dy"]

    # Wall collisions
    if state["ball_x"] <= x_offset or state["ball_x"] >= x_offset + HALF_WIDTH - BALL_RADIUS:
        state["ball_dx"] *= -1
    if state["ball_y"] <= 0:
        state["ball_dy"] *= -1

    # Paddle collision
    paddle_rect = pygame.Rect(state["paddle_x"], HEIGHT - 30, PADDLE_WIDTH, PADDLE_HEIGHT)
    ball_rect = pygame.Rect(state["ball_x"] - BALL_RADIUS, state["ball_y"] - BALL_RADIUS, BALL_RADIUS*2, BALL_RADIUS*2)
    
    if ball_rect.colliderect(paddle_rect) and state["ball_dy"] > 0:
        state["ball_dy"] *= -1
        # Add slight angle variation based on hit position
        hit_pos = (state["ball_x"] - state["paddle_x"]) / PADDLE_WIDTH
        state["ball_dx"] = (hit_pos - 0.5) * 10

    # Brick collision
    for b in state["bricks"][:]:
        if ball_rect.colliderect(b):
            state["bricks"].remove(b)
            state["ball_dy"] *= -1
            state["score"] += 1
            break

    # Win/Loss check
    if not state["bricks"]:
        state["won"] = True
    if state["ball_y"] > HEIGHT:
        state["done"] = True

# ================= DRAW =================
def draw_side(state, label, x_offset):
    # Draw Game Elements
    pygame.draw.circle(screen, BALL_COLOR, (int(state["ball_x"]), int(state["ball_y"])), BALL_RADIUS)
    pygame.draw.rect(screen, PADDLE_COLOR, (state["paddle_x"], HEIGHT - 30, PADDLE_WIDTH, PADDLE_HEIGHT), border_radius=5)
    
    for b in state["bricks"]:
        pygame.draw.rect(screen, BRICK_COLOR, b, border_radius=3)

    # Draw Lost/Won Overlays
    if state["done"] or state["won"]:
        overlay = pygame.Surface((HALF_WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180)) 
        screen.blit(overlay, (x_offset, 0))
        
        msg = "LOST" if state["done"] else "WINS!"
        color = (255, 100, 100) if state["done"] else (100, 255, 100)
        txt = font_big.render(f"{label} {msg}", True, color)
        screen.blit(txt, (x_offset + HALF_WIDTH//2 - txt.get_width()//2, HEIGHT//2 - 30))

# ================= MAIN LOOP =================
left, right = wait_for_start()

while True:
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            pygame.quit(); sys.exit()

    keys = pygame.key.get_pressed()

    # --- GLOBAL RESET KEY ---
    if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
        left, right = wait_for_start()

    # Human Controls
    human_action = 0
    if keys[pygame.K_LEFT]: human_action = 1
    elif keys[pygame.K_RIGHT]: human_action = 2

    # AI Controls
    ai_action = 0
    if model and not right["done"] and not right["won"]:
        obs = np.array([
            (right["ball_x"] - HALF_WIDTH) / HALF_WIDTH,
            right["ball_y"] / HEIGHT,
            right["ball_dx"] / 10,
            right["ball_dy"] / 10,
            (right["paddle_x"] - HALF_WIDTH) / HALF_WIDTH,
            len(right["bricks"]) / (BRICK_ROWS * BRICK_COLUMNS),
        ], dtype=np.float32)
        ai_action, _ = model.predict(obs, deterministic=True)

    # Update Physics
    update_state(left, human_action, 0)
    update_state(right, ai_action, HALF_WIDTH)

    # Drawing
    draw_background()
    draw_side(left, "HUMAN", 0)
    draw_side(right, "AI", HALF_WIDTH)

    # UI Elements
    pygame.draw.line(screen, DIVIDER_COLOR, (HALF_WIDTH, 0), (HALF_WIDTH, HEIGHT), 5)
    screen.blit(font_med.render(f"Human Score: {left['score']}", True, TEXT_COLOR), (20, 20))
    screen.blit(font_med.render(f"AI Score: {right['score']}", True, TEXT_COLOR), (HALF_WIDTH + 20, 20))

    # If Human wins or both finished, offer restart
    if left["won"] or (left["done"] and right["done"]):
        prompt = font_med.render("Press SPACE for New Game", True, (50, 50, 50))
        screen.blit(prompt, (WIDTH//2 - prompt.get_width()//2, HEIGHT - 50))
        if keys[pygame.K_SPACE]:
            left, right = wait_for_start()

    pygame.display.flip()
    clock.tick(60)