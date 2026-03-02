import pygame
import numpy as np
import random
import gymnasium as gym
from gymnasium import spaces

WIDTH, HEIGHT = 600, 600
PADDLE_WIDTH = 100
PADDLE_HEIGHT = 10
BALL_RADIUS = 10
BALL_SPEED = 5
PADDLE_SPEED = 10

BRICK_WIDTH = 80
BRICK_HEIGHT = 20
BRICK_ROWS = 5
BRICK_COLUMNS = 8
BRICK_SPACING = 10


class BrickBreakerEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None):
        super().__init__()

        self.render_mode = render_mode
        self.action_space = spaces.Discrete(3)
        # 0 = stay, 1 = left, 2 = right

        self.observation_space = spaces.Box(
            low=0, high=1, shape=(6,), dtype=np.float32
        )

        pygame.init()
        if render_mode == "human":
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        else:
            self.screen = None

        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.ball_x = WIDTH // 2
        self.ball_y = HEIGHT // 2
        self.ball_dx = random.choice([-1, 1]) * BALL_SPEED
        self.ball_dy = -BALL_SPEED

        self.paddle_x = (WIDTH - PADDLE_WIDTH) // 2
        self.paddle_y = HEIGHT - 20

        self.bricks = []
        for r in range(BRICK_ROWS):
            for c in range(BRICK_COLUMNS):
                x = c * (BRICK_WIDTH + BRICK_SPACING)
                y = r * (BRICK_HEIGHT + BRICK_SPACING) + 50
                self.bricks.append(pygame.Rect(x, y, BRICK_WIDTH, BRICK_HEIGHT))

        return self._get_obs(), {}

    def _get_obs(self):
        return np.array([
            self.ball_x / WIDTH,
            self.ball_y / HEIGHT,
            self.ball_dx / 10,
            self.ball_dy / 10,
            self.paddle_x / WIDTH,
            len(self.bricks) / (BRICK_ROWS * BRICK_COLUMNS),
        ], dtype=np.float32)

    def step(self, action):
        reward = 0
        terminated = False

        # move paddle
        if action == 1:
            self.paddle_x -= PADDLE_SPEED
        elif action == 2:
            self.paddle_x += PADDLE_SPEED

        self.paddle_x = np.clip(self.paddle_x, 0, WIDTH - PADDLE_WIDTH)

        # move ball
        self.ball_x += self.ball_dx
        self.ball_y += self.ball_dy

        # wall bounce
        if self.ball_x <= 0 or self.ball_x >= WIDTH:
            self.ball_dx *= -1
        if self.ball_y <= 0:
            self.ball_dy *= -1

        # paddle hit
        # if (
        #     self.ball_y + BALL_RADIUS >= self.paddle_y
        #     and self.ball_x >= self.paddle_x
        #     and self.ball_x <= self.paddle_x + PADDLE_WIDTH
        # ):
        #     self.ball_dy *= -1
        #     reward += 0.1
        # paddle hit
        if (
            self.ball_y + BALL_RADIUS >= self.paddle_y
            and self.ball_x >= self.paddle_x
            and self.ball_x <= self.paddle_x + PADDLE_WIDTH
        ):
            self.ball_dy *= -1
            reward += 0.1

        # ⭐ ADD THIS BLOCK
        dist = abs(self.ball_x - (self.paddle_x + PADDLE_WIDTH / 2))
        reward += 0.01 * (1 - dist / WIDTH)

        # brick hit
        for brick in self.bricks[:]:
            if brick.collidepoint(self.ball_x, self.ball_y):
                self.bricks.remove(brick)
                self.ball_dy *= -1
                reward += 1

        # miss ball
        if self.ball_y > HEIGHT:
            terminated = True
            reward -= 5

        # win
        if len(self.bricks) == 0:
            terminated = True
            reward += 10

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, False, {}

    def render(self):
        self.screen.fill((255, 255, 255))
        pygame.draw.circle(self.screen, (255, 0, 0),
                           (int(self.ball_x), int(self.ball_y)), BALL_RADIUS)
        pygame.draw.rect(
            self.screen, (0, 0, 255),
            (self.paddle_x, self.paddle_y, PADDLE_WIDTH, PADDLE_HEIGHT)
        )
        for b in self.bricks:
            pygame.draw.rect(self.screen, (0, 255, 0), b)

        pygame.display.flip()
        self.clock.tick(60)