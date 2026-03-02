import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.Font('arial.ttf', 25)
button_font = pygame.font.Font('arial.ttf', 30)

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

class SnakeGame:
    
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        
        self.reset_game()
        
    def reset_game(self):
        # init game state
        self.direction = Direction.RIGHT
        
        # Use integer division for grid alignment.
        self.head = Point(self.w // 2, self.h // 2)
        self.snake = [self.head, 
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]
        
        self.score = 0
        self.food = None
        
        # Define obstacles
        self.obstacles = []
        # Top border: horizontal line across the screen.
        for i in range(self.w // BLOCK_SIZE):
            self.obstacles.append(Point(i * BLOCK_SIZE, 0))
        # Right border: vertical line.
        for i in range(self.h // BLOCK_SIZE):
            self.obstacles.append(Point(self.w - BLOCK_SIZE, i * BLOCK_SIZE))
 
        # Additional obstacles:
        # Obstacle 1: horizontal block at y = 100, starting at x = 100, 6 blocks long.
        for i in range(6):
            self.obstacles.append(Point(100 + i * BLOCK_SIZE, 100))
        # Obstacle 2: vertical block at x = 300, starting at y = 200, 6 blocks long.
        for i in range(6):
            self.obstacles.append(Point(300, 200 + i * BLOCK_SIZE))
        # Obstacle 3: horizontal block at y = 300, starting at x = 200, 6 blocks long.
        for i in range(6):
            self.obstacles.append(Point(200 + i * BLOCK_SIZE, 300))
        # Obstacle 4: vertical block at x = 400, starting at y = 50, 8 blocks long.
        for i in range(8):
            self.obstacles.append(Point(400, 50 + i * BLOCK_SIZE))
        # Obstacle 5: horizontal block at y = 400, starting at x = 50, 8 blocks long.
        for i in range(8):
            self.obstacles.append(Point(50 + i * BLOCK_SIZE, 400))
        # Obstacle 6: a small 3x3 square block in the center.
        for i in range(3):
            for j in range(3):
                self.obstacles.append(Point(250 + i * BLOCK_SIZE, 250 + j * BLOCK_SIZE))
        
        self._place_food()
        
    def _place_food(self):
        # Loop until a valid position is found (not on the snake or any obstacle).
        while True:
            x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE 
            y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            new_food = Point(x, y)
            if new_food not in self.snake and new_food not in self.obstacles:
                self.food = new_food
                break
        
    def play_step(self):
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                # Prevent reverse direction:
                if event.key == pygame.K_LEFT and self.direction != Direction.RIGHT:
                    self.direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT and self.direction != Direction.LEFT:
                    self.direction = Direction.RIGHT
                elif event.key == pygame.K_UP and self.direction != Direction.DOWN:
                    self.direction = Direction.UP
                elif event.key == pygame.K_DOWN and self.direction != Direction.UP:
                    self.direction = Direction.DOWN
        
        # 2. move
        self._move(self.direction)  # update the head
        self.snake.insert(0, self.head)
        
        # 3. check if game over
        game_over = False
        if self._is_collision():
            game_over = True
            return game_over, self.score
            
        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            self._place_food()
        else:
            self.snake.pop()
        
        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return game_over, self.score
    
    def _is_collision(self):
        # Create a rectangle for the snake's head.
        head_rect = pygame.Rect(self.head.x, self.head.y, BLOCK_SIZE, BLOCK_SIZE)
        
        # Check boundaries.
        if self.head.x < 0 or self.head.x > self.w - BLOCK_SIZE or self.head.y < 0 or self.head.y > self.h - BLOCK_SIZE:
            return True
        # Check collision with itself.
        for pt in self.snake[1:]:
            if head_rect.collidepoint(pt.x, pt.y):
                return True
        # Check collision with obstacles.
        for obs in self.obstacles:
            obs_rect = pygame.Rect(obs.x, obs.y, BLOCK_SIZE, BLOCK_SIZE)
            if head_rect.colliderect(obs_rect):
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
        for pt in self.obstacles:
            pygame.draw.rect(self.display, OBSTACLE_COLOR, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
        
        score_text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(score_text, (0, 0))
        pygame.display.flip()
        
    def _move(self, direction):
        x = self.head.x
        y = self.head.y
        if direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif direction == Direction.UP:
            y -= BLOCK_SIZE
            
        self.head = Point(x, y)
    
    def show_game_over_screen(self):
        """Display a game over screen with Retry and Quit buttons and return True if retry is selected."""
        self.display.fill(BLACK)
        game_over_text = font.render("GAME OVER", True, RED)
        score_text = font.render("Final Score: " + str(self.score), True, WHITE)
        self.display.blit(game_over_text, (self.w // 2 - game_over_text.get_width() // 2, self.h // 4))
        self.display.blit(score_text, (self.w // 2 - score_text.get_width() // 2, self.h // 4 + 40))
        
        # Define buttons.
        retry_button = pygame.Rect(self.w // 2 - 100, self.h // 2, 80, 40)
        quit_button = pygame.Rect(self.w // 2 + 20, self.h // 2, 80, 40)
        
        pygame.draw.rect(self.display, BUTTON_COLOR, retry_button)
        pygame.draw.rect(self.display, BUTTON_COLOR, quit_button)
        
        retry_text = button_font.render("Retry", True, WHITE)
        quit_text = button_font.render("Quit", True, WHITE)
        self.display.blit(retry_text, (retry_button.x + (retry_button.width - retry_text.get_width()) // 2,
                                       retry_button.y + (retry_button.height - retry_text.get_height()) // 2))
        self.display.blit(quit_text, (quit_button.x + (quit_button.width - quit_text.get_width()) // 2,
                                      quit_button.y + (quit_button.height - quit_text.get_height()) // 2))
        pygame.display.flip()
        
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = pygame.mouse.get_pos()
                    if retry_button.collidepoint(mouse_pos):
                        return True
                    if quit_button.collidepoint(mouse_pos):
                        return False

if __name__ == '__main__':
    while True:
        game = SnakeGame()
        game_over = False
        # game loop
        while not game_over:
            game_over, score = game.play_step()
        # Show game over screen and check if user wants to retry.
        retry = game.show_game_over_screen()
        if not retry:
            break
        
    print('Final Score:', score)
    pygame.quit()
