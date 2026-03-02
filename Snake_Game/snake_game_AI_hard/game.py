import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.Font('arial.ttf', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)
OBSTACLE_COLOR = (128, 128, 128)
BUTTON_COLOR = (0, 200, 0)
BUTTON_HOVER = (0, 255, 0)

BLOCK_SIZE = 20
SPEED = 15

class SnakeGameAI:

    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT
        # Use integer division for grid alignment.
        self.head = Point(self.w // 2, self.h // 2)
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]
        self.score = 0

        # Define obstacles as pygame.Rect objects.
        self.obstacles = []
        # Top border: only the ones visible on screen.
        for i in range(self.w // BLOCK_SIZE):
            self.obstacles.append(pygame.Rect(i * BLOCK_SIZE, 0, BLOCK_SIZE, BLOCK_SIZE))
        # Right border.
        for i in range(self.h // BLOCK_SIZE):
            self.obstacles.append(pygame.Rect(self.w - BLOCK_SIZE, i * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
 
        # Additional obstacles:
        # Obstacle 1: horizontal block at y = 100, starting at x = 100, 6 blocks long.
        for i in range(6):
            self.obstacles.append(pygame.Rect(100 + i * BLOCK_SIZE, 100, BLOCK_SIZE, BLOCK_SIZE))
        # Obstacle 2: vertical block at x = 300, starting at y = 200, 6 blocks long.
        for i in range(6):
            self.obstacles.append(pygame.Rect(300, 200 + i * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
        # Obstacle 3: horizontal block at y = 300, starting at x = 200, 6 blocks long.
        for i in range(6):
            self.obstacles.append(pygame.Rect(200 + i * BLOCK_SIZE, 300, BLOCK_SIZE, BLOCK_SIZE))
        # Obstacle 4: vertical block at x = 400, starting at y = 50, 8 blocks long.
        for i in range(8):
            self.obstacles.append(pygame.Rect(400, 50 + i * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
        # Obstacle 5: horizontal block at y = 400, starting at x = 50, 8 blocks long.
        for i in range(8):
            self.obstacles.append(pygame.Rect(50 + i * BLOCK_SIZE, 400, BLOCK_SIZE, BLOCK_SIZE))
        # Obstacle 6: a small 3x3 square block in the center.
        for i in range(3):
            for j in range(3):
                self.obstacles.append(pygame.Rect(250 + i * BLOCK_SIZE, 250 + j * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
                
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        while True:
            x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            new_food = Point(x, y)
            # Check that new_food is not on the snake and not inside any obstacle.
            if new_food not in self.snake and not any(obstacle.collidepoint(new_food.x, new_food.y) for obstacle in self.obstacles):
                self.food = new_food
                break

    def play_step(self, action):
        self.frame_iteration += 1
        # 1. Collect user input.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. Move.
        self._move(action)  # update the head
        self.snake.insert(0, self.head)

        # 3. Check if game over.
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 300 * len(self.snake):
            game_over = True
            reward = -50
            return reward, game_over, self.score

        # 4. Place new food or just move.
        if self.head == self.food:
            self.score += 1
            reward = 200
            self._place_food()
        else:
            self.snake.pop()

        # 5. Update UI and clock.
        self._update_ui()
        self.clock.tick(SPEED)
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        head_rect = pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE)
        # Check boundaries.
        if pt.x < 0 or pt.x > self.w - BLOCK_SIZE or pt.y < 0 or pt.y > self.h - BLOCK_SIZE:
            return True
        # Check collision with itself.
        for part in self.snake[1:]:
            if head_rect.collidepoint(part.x, part.y):
                return True
        # Check collision with obstacles.
        for obstacle in self.obstacles:
            if head_rect.colliderect(obstacle):
                return True
        return False

    def _update_ui(self):
        self.display.fill(BLACK)
        # Draw snake.
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))
        # Draw food.
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        # Draw obstacles.
        for obstacle in self.obstacles:
            pygame.draw.rect(self.display, OBSTACLE_COLOR, obstacle)
        score_text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(score_text, (0, 0))
        pygame.display.flip()

    def _move(self, action):
        # action: [straight, right, left]
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # no change
        elif np.array_equal(action, [0, 1, 0]):
            new_dir = clock_wise[(idx + 1) % 4]  # right turn
        else:  # [0, 0, 1]
            new_dir = clock_wise[(idx - 1) % 4]  # left turn
        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)
