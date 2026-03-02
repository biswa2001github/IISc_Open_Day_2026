import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum
from collections import namedtuple, deque

pygame.init()

# Try to load arial font, fallback to system font
try:
    font = pygame.font.Font('arial.ttf', 22)
    font_large = pygame.font.Font('arial.ttf', 48)
    font_medium = pygame.font.Font('arial.ttf', 30)
    font_small = pygame.font.Font('arial.ttf', 18)
except:
    font = pygame.font.SysFont('arial', 22)
    font_large = pygame.font.SysFont('arial', 48)
    font_medium = pygame.font.SysFont('arial', 30)
    font_small = pygame.font.SysFont('arial', 18)

# ── Colors ──────────────────────────────────────────────────────────────────
WHITE      = (255, 255, 255)
BLACK      = (0,   0,   0  )
RED        = (200, 0,   0  )
GREEN      = (0,   200, 0  )
DARK_GREEN = (0,   140, 0  )
BLUE1      = (0,   0,   255)
BLUE2      = (0,   100, 255)
YELLOW     = (255, 220, 0  )
ORANGE     = (255, 140, 0  )
GREY       = (60,  60,  60 )
DARK_GREY  = (30,  30,  30 )
LIGHT_GREY = (80,  80,  80 )
PURPLE     = (160, 32,  240)
CYAN       = (0,   200, 200)

BLOCK_SIZE = 20
SPEED      = 12

# ── Direction / Point ────────────────────────────────────────────────────────
class Direction(Enum):
    RIGHT = 1
    LEFT  = 2
    UP    = 3
    DOWN  = 4

Point = namedtuple('Point', 'x, y')

# ── Neural-network model (mirrors model.py) ──────────────────────────────────
class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        return self.linear2(x)

# ── Single-side snake game ───────────────────────────────────────────────────
class SnakeSide:
    """Encapsulates one snake (human or AI) playing inside a sub-rectangle."""

    def __init__(self, x_offset, w, h):
        self.x_offset = x_offset   # pixel offset so we draw in the right panel
        self.w = w                  # width of this panel
        self.h = h
        self.reset()

    def reset(self):
        self.direction    = Direction.RIGHT
        self.head         = Point(self.w // 2, self.h // 2)
        self.snake        = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - 2 * BLOCK_SIZE, self.head.y),
        ]
        self.score         = 0
        self.game_over     = False
        self.food          = None
        self.frame_iter    = 0
        self._place_food()

    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        if pt.x >= self.w or pt.x < 0 or pt.y >= self.h or pt.y < 0:
            return True
        if pt in self.snake[1:]:
            return True
        return False

    def _move_by_direction(self, direction):
        x, y = self.head.x, self.head.y
        if direction == Direction.RIGHT: x += BLOCK_SIZE
        elif direction == Direction.LEFT:  x -= BLOCK_SIZE
        elif direction == Direction.DOWN:  y += BLOCK_SIZE
        elif direction == Direction.UP:    y -= BLOCK_SIZE
        self.head = Point(x, y)

    def step_human(self, direction):
        """Move human snake one step."""
        if self.game_over:
            return
        self.frame_iter += 1
        self._move_by_direction(direction)
        self.snake.insert(0, self.head)
        if self.is_collision():
            self.game_over = True
            return
        if self.head == self.food:
            self.score += 1
            self._place_food()
        else:
            self.snake.pop()

    def step_ai(self, action):
        """Move AI snake one step with relative action [straight, right, left]."""
        if self.game_over:
            return
        self.frame_iter += 1
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            new_dir = clock_wise[(idx + 1) % 4]
        else:
            new_dir = clock_wise[(idx - 1) % 4]
        self.direction = new_dir

        x, y = self.head.x, self.head.y
        if self.direction == Direction.RIGHT: x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:  x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:  y += BLOCK_SIZE
        elif self.direction == Direction.UP:    y -= BLOCK_SIZE
        self.head = Point(x, y)
        self.snake.insert(0, self.head)

        if self.is_collision() or self.frame_iter > 100 * len(self.snake):
            self.game_over = True
            return
        if self.head == self.food:
            self.score += 1
            self._place_food()
        else:
            self.snake.pop()

    def get_state(self):
        """11-feature state vector used during training."""
        head = self.snake[0]
        pl = Point(head.x - 20, head.y)
        pr = Point(head.x + 20, head.y)
        pu = Point(head.x,      head.y - 20)
        pd = Point(head.x,      head.y + 20)

        dl = self.direction == Direction.LEFT
        dr = self.direction == Direction.RIGHT
        du = self.direction == Direction.UP
        dd = self.direction == Direction.DOWN

        state = [
            (dr and self.is_collision(pr)) or (dl and self.is_collision(pl)) or
            (du and self.is_collision(pu)) or (dd and self.is_collision(pd)),

            (du and self.is_collision(pr)) or (dd and self.is_collision(pl)) or
            (dl and self.is_collision(pu)) or (dr and self.is_collision(pd)),

            (dd and self.is_collision(pr)) or (du and self.is_collision(pl)) or
            (dr and self.is_collision(pu)) or (dl and self.is_collision(pd)),

            dl, dr, du, dd,

            self.food.x < head.x,
            self.food.x > head.x,
            self.food.y < head.y,
            self.food.y > head.y,
        ]
        return np.array(state, dtype=int)

    def draw(self, surface, head_color, body_color1, body_color2, food_color):
        ox = self.x_offset

        # Draw snake
        for i, pt in enumerate(self.snake):
            c1 = head_color if i == 0 else body_color1
            pygame.draw.rect(surface, c1,
                             pygame.Rect(ox + pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(surface, body_color2,
                             pygame.Rect(ox + pt.x + 4, pt.y + 4, 12, 12))

        # Draw food (pulsing square)
        pygame.draw.rect(surface, food_color,
                         pygame.Rect(ox + self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        # Shiny highlight
        pygame.draw.rect(surface, WHITE,
                         pygame.Rect(ox + self.food.x + 4, self.food.y + 3, 5, 5))

    def draw_game_over_overlay(self, surface, label):
        ox = self.x_offset
        overlay = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        surface.blit(overlay, (ox, 0))

        go_surf = font_medium.render("GAME OVER", True, RED)
        surface.blit(go_surf, (ox + self.w // 2 - go_surf.get_width() // 2,
                                self.h // 2 - 40))
        sc_surf = font_small.render(f"{label} Score: {self.score}", True, WHITE)
        surface.blit(sc_surf, (ox + self.w // 2 - sc_surf.get_width() // 2,
                                self.h // 2 + 5))


# ── Main game controller ─────────────────────────────────────────────────────
class DualSnakeGame:
    PANEL_W = 400
    H       = 480
    DIVIDER = 6
    SCORE_BAR = 50

    def __init__(self):
        self.total_w = self.PANEL_W * 2 + self.DIVIDER
        self.display = pygame.display.set_mode(
            (self.total_w, self.H + self.SCORE_BAR))
        pygame.display.set_caption('Snake Showdown  —  Human vs AI')
        self.clock = pygame.time.Clock()

        # Load AI model
        self.ai_model = Linear_QNet(11, 256, 3)
        self.ai_loaded = False
        try:
            self.ai_model.load_state_dict(
                torch.load('./model/model.pth', map_location='cpu'))
            self.ai_model.eval()
            self.ai_loaded = True
            print("AI model loaded successfully.")
        except Exception as e:
            print(f"Could not load AI model: {e}")
            print("AI will play randomly.")

        self.human = SnakeSide(x_offset=0,              w=self.PANEL_W, h=self.H)
        self.ai    = SnakeSide(x_offset=self.PANEL_W + self.DIVIDER, w=self.PANEL_W, h=self.H)

        self.human_direction = Direction.RIGHT
        self.state = 'start'   # 'start' | 'playing' | 'human_over'

    # ── helpers ──────────────────────────────────────────────────────────────
    def _ai_action(self):
        if not self.ai_loaded:
            return [random.choice([[1,0,0],[0,1,0],[0,0,1]])]
        state = torch.tensor(self.ai.get_state(), dtype=torch.float)
        with torch.no_grad():
            pred = self.ai_model(state)
        move = torch.argmax(pred).item()
        action = [0, 0, 0]
        action[move] = 1
        return action

    def _new_game(self):
        self.human.reset()
        self.ai.reset()
        self.human_direction = Direction.RIGHT
        self.state = 'playing'

    # ── drawing ──────────────────────────────────────────────────────────────
    def _draw_start_screen(self):
        self.display.fill(DARK_GREY)
        self._draw_divider()

        title = font_large.render("SNAKE  SHOWDOWN", True, YELLOW)
        self.display.blit(title, (self.total_w // 2 - title.get_width() // 2, 80))

        sub = font_medium.render("Human  vs  AI", True, CYAN)
        self.display.blit(sub, (self.total_w // 2 - sub.get_width() // 2, 155))

        instructions = [
            ("SPACE",  "Start game"),
            ("← ↑ → ↓", "Move snake"),
            ("SHIFT",  "Restart anytime"),
        ]
        y = 230
        for key, desc in instructions:
            k_surf = font.render(key, True, ORANGE)
            d_surf = font.render(f"  —  {desc}", True, WHITE)
            total_w = k_surf.get_width() + d_surf.get_width()
            x = self.total_w // 2 - total_w // 2
            self.display.blit(k_surf, (x, y))
            self.display.blit(d_surf, (x + k_surf.get_width(), y))
            y += 38

        left_lbl  = font_medium.render("HUMAN", True, GREEN)
        right_lbl = font_medium.render("AI", True, PURPLE)
        self.display.blit(left_lbl,
            (self.PANEL_W // 2 - left_lbl.get_width() // 2, self.H - 60))
        self.display.blit(right_lbl,
            (self.PANEL_W + self.DIVIDER + self.PANEL_W // 2 - right_lbl.get_width() // 2,
             self.H - 60))

        pygame.display.flip()

    def _draw_divider(self):
        pygame.draw.rect(self.display, LIGHT_GREY,
                         pygame.Rect(self.PANEL_W, 0, self.DIVIDER, self.H + self.SCORE_BAR))

    def _draw_score_bar(self):
        bar_y = self.H
        pygame.draw.rect(self.display, GREY,
                         pygame.Rect(0, bar_y, self.total_w, self.SCORE_BAR))

        # Human score (left)
        h_txt = font_medium.render(f"👤 Human: {self.human.score}", True, GREEN)
        self.display.blit(h_txt, (20, bar_y + 10))

        # AI score (right)
        ai_txt = font_medium.render(f"🤖 AI: {self.ai.score}", True, PURPLE)
        self.display.blit(ai_txt,
            (self.total_w - ai_txt.get_width() - 20, bar_y + 10))

        # Leader indicator
        if self.human.score > self.ai.score:
            ldr = font_small.render("Human leads!", True, YELLOW)
        elif self.ai.score > self.human.score:
            ldr = font_small.render("AI leads!", True, YELLOW)
        else:
            ldr = font_small.render("Tied!", True, YELLOW)
        self.display.blit(ldr, (self.total_w // 2 - ldr.get_width() // 2, bar_y + 15))

    def _draw_playing(self):
        # Backgrounds
        pygame.draw.rect(self.display, DARK_GREY, pygame.Rect(0, 0, self.PANEL_W, self.H))
        pygame.draw.rect(self.display, DARK_GREY,
                         pygame.Rect(self.PANEL_W + self.DIVIDER, 0, self.PANEL_W, self.H))
        self._draw_divider()

        # Panel labels
        h_lbl = font_small.render("HUMAN", True, GREEN)
        a_lbl = font_small.render("AI", True, PURPLE)
        self.display.blit(h_lbl, (5, 4))
        self.display.blit(a_lbl,
            (self.PANEL_W + self.DIVIDER + self.PANEL_W - a_lbl.get_width() - 5, 4))

        # Snakes
        self.human.draw(self.display, GREEN, GREEN, DARK_GREEN, RED)
        self.ai.draw(self.display, PURPLE, BLUE1, BLUE2, ORANGE)

        # Game-over overlays
        if self.human.game_over:
            self.human.draw_game_over_overlay(self.display, "Human")
        if self.ai.game_over:
            self.ai.draw_game_over_overlay(self.display, "AI")

        self._draw_score_bar()
        pygame.display.flip()

    # ── main loop ────────────────────────────────────────────────────────────
    def run(self):
        while True:
            if self.state == 'start':
                self._run_start_screen()
            elif self.state == 'playing':
                self._run_playing()

    def _run_start_screen(self):
        self._draw_start_screen()
        while self.state == 'start':
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); quit()
                if event.type == pygame.KEYDOWN:
                    mods = pygame.key.get_mods()
                    if event.key == pygame.K_SPACE:
                        self._new_game()
                    elif mods & pygame.KMOD_SHIFT:
                        self._new_game()

    def _run_playing(self):
        while self.state == 'playing':
            # ── events ──
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); quit()
                if event.type == pygame.KEYDOWN:
                    mods = pygame.key.get_mods()
                    if mods & pygame.KMOD_SHIFT:
                        self._new_game()
                        return
                    # Human movement (no reversing)
                    if not self.human.game_over:
                        if event.key == pygame.K_LEFT \
                                and self.human_direction != Direction.RIGHT:
                            self.human_direction = Direction.LEFT
                        elif event.key == pygame.K_RIGHT \
                                and self.human_direction != Direction.LEFT:
                            self.human_direction = Direction.RIGHT
                        elif event.key == pygame.K_UP \
                                and self.human_direction != Direction.DOWN:
                            self.human_direction = Direction.UP
                        elif event.key == pygame.K_DOWN \
                                and self.human_direction != Direction.UP:
                            self.human_direction = Direction.DOWN

            # ── update snakes ──
            if not self.human.game_over:
                self.human.step_human(self.human_direction)

            if not self.ai.game_over:
                action = self._ai_action()
                self.ai.step_ai(action)

            # ── draw ──
            self._draw_playing()
            self.clock.tick(SPEED)

            # ── check state transitions ──
            # If human just died — show overlay and wait for SHIFT
            if self.human.game_over and self.state == 'playing':
                self._wait_after_human_over()
                return

    def _wait_after_human_over(self):
        """Keep rendering (AI may still be alive) until SHIFT pressed."""
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); quit()
                if event.type == pygame.KEYDOWN:
                    mods = pygame.key.get_mods()
                    if mods & pygame.KMOD_SHIFT:
                        self._new_game()
                        return

            # Keep AI alive
            if not self.ai.game_over:
                action = self._ai_action()
                self.ai.step_ai(action)

            self._draw_playing()

            # Hint for user
            hint = font_small.render("Press SHIFT for new game", True, YELLOW)
            self.display.blit(hint,
                (self.total_w // 2 - hint.get_width() // 2,
                 self.H // 2 + 50))
            pygame.display.flip()
            self.clock.tick(SPEED)


# ── Entry point ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    game = DualSnakeGame()
    game.run()


