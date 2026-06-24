import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
import math
import time

pygame.init()
font = pygame.font.SysFont('arial', 24)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    
Point = namedtuple('Point', 'x, y')

### RGB COLORS  ####
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)

HEAD_COLOR1 = (0, 100, 0)
HEAD_COLOR2 = (0, 150, 0)
TAIL_COLOR1 = (100, 0, 0)
TAIL_COLOR2 = (100, 60, 0)
BACKGROUND = (44, 178, 169)

### GAME SETTINGS ####
BLOCK_SIZE = 20
SPEED = 20

# Class that represents the game.
class SnakeGame:
    
    def __init__(self, w=200, h=200, gui=True):

        # Window width and height.
        self.w = w 
        self.h = h

        self.max_len = (w // BLOCK_SIZE) * (h // BLOCK_SIZE)

        # Min distance from food to get reward +1.
        self.prev_dist = 0

        # If gui is active or not.
        self.gui = gui

        # Init display.
        if self.gui:
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption('Score: 0')

        # Set clock.
        self.clock = pygame.time.Clock()

        # Init game.
        self.reset()
        
    def reset(self):
        # First direction is right.
        self.direction = Direction.RIGHT
        
        # Place the head position.
        self.head = Point(self.w/2, self.h/2)

        # Place the entire snake.
        self.snake = [
            # 3 snake.
            self.head, 
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)
        ]

        # Init score.
        self.score = 0

        # Init food.
        self.food = None
        self._place_food()

        self._dist_food_head()
        self.prev_dist = self.dist

        # Init step counter.
        self.step_counter = 0
        

    def _dist_food_head(self):
        # Distance food <-> head.
        self.dist = Point((self.head.x - self.food.x)/BLOCK_SIZE, (self.head.y - self.food.y)/BLOCK_SIZE)
        self.dist = math.sqrt(self.dist.x ** 2 + self.dist.y ** 2)
        #print("Dist =", self.dist)


    def _place_food(self):
        # Generate random coordinates.
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE 
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)

        # Loop until position is outside snake body.
        if self.food in self.snake:
            self._place_food()
        

    def play_step(self, action):
        self.step_counter += 1

        # Manage quit game.
        if self.gui:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

        # 2. Move.
        self.move(action) # Update the head.
        self.snake.insert(0, self.head)

        # 3. Check if game over.
        reward = 0
        game_over = False
        if self.is_collision() or self.step_counter > 100 * len(self.snake):
            game_over = True
            reward = -1
            self.snake.pop()
            return reward, game_over, self.score

        # 4. Place new food or just move + reward.
        if self.head == self.food:
            self.score += 1
            reward = 1
            if len(self.snake) == self.max_len:
                print("You won!")
                game_over = True
                return reward, game_over, self.score
            self._place_food()
        else:
            # Give reward for each step.
            if len(self.snake) > 10:
                reward += 0.01

            ## Give reward if head is on the border.
            #i = self.head.y // BLOCK_SIZE
            #j = self.head.x // BLOCK_SIZE
            #if i == 0 or j == 0 or i == 9 or j == 9:
            #    reward += 0.02

            self.snake.pop()
    
        # Give reward based on distance to food.
        #self.prev_dist = self.dist
        #self._dist_food_head()
        #if self.head != self.food:
        #    if  self.dist < self.prev_dist:
        #        reward = 1
        #        #print("Dist =", self.dist)
        #    else:
        #        reward = -1

        # 5. Update ui and clock.
        if self.gui:
            self._update_ui()
            self.clock.tick(SPEED)

        # 6. Return game over and score.
        return reward, game_over, self.score


    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head

        # Hits boundary.
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # Hits itself.
        if pt in self.snake[1:]:
            return True
        
        return False


    def _update_ui(self):
        self.display.fill(BACKGROUND)
        
        # Color head and tail.
        pygame.draw.rect(self.display, HEAD_COLOR1, pygame.Rect(self.head.x, self.head.y, BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(self.display, HEAD_COLOR2, pygame.Rect(self.head.x+4, self.head.y+4, BLOCK_SIZE-8, BLOCK_SIZE-8))
        pygame.draw.rect(self.display, TAIL_COLOR1, pygame.Rect(self.snake[-1].x, self.snake[-1].y, BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(self.display, TAIL_COLOR2, pygame.Rect(self.snake[-1].x+4, self.snake[-1].y+4, BLOCK_SIZE-8, BLOCK_SIZE-8))

        for pt in self.snake[1:-1]:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
            
        if len(self.snake) < self.max_len:
            pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        pygame.display.set_caption(f'S: {str(self.score)}')
        pygame.display.flip()


    def move(self, action, perform=True):
        # Action parsing. -> [straight, right, left]
        
        clock_wise = [
            Direction.RIGHT,
            Direction.DOWN,
            Direction.LEFT,
            Direction.UP,
        ]
        
        # Get index of current direction.
        idx = clock_wise.index(self.direction)


        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # No change.
        elif np.array_equal(action, [0, 1, 0]):
            new_idx = (idx + 1) % 4 # Turn right.
            new_dir = clock_wise[new_idx]
        else: # [0, 0, 1]
            new_idx = (idx - 1) % 4 # Turn left.
            new_dir = clock_wise[new_idx]

        
        x = self.head.x
        y = self.head.y
        if new_dir == Direction.RIGHT:
            x += BLOCK_SIZE
        elif new_dir == Direction.LEFT:
            x -= BLOCK_SIZE
        elif new_dir == Direction.DOWN:
            y += BLOCK_SIZE
        elif new_dir == Direction.UP:
            y -= BLOCK_SIZE
            

        # Update direction and head if necessary.
        if perform:
            self.direction = new_dir
            self.head = Point(x, y)

        return Point(x, y)


# Class that reprsent a replay game.
class ReplaySnakeGame(SnakeGame):
    def __init__(self, food_positions, w=200, h=200, gui=True):
        super().__init__(w, h, gui)
        self.food_positions = food_positions # All the food positions.
        self.food = self.food_positions[0] # Set the first one.

    def play_step(self, action):

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # Move.
        self.move(action) # Update the head.
        self.snake.insert(0, self.head)

        # Check if game over.
        game_over = False
        if self.is_collision() or self.step_counter > 100 * len(self.snake):
            game_over = True
            reward = -10
            time.sleep(5)
            return game_over

        # Update score if necessary.
        if self.head == self.food:
            self.score += 1
        else:
            self.snake.pop()

        # Update food position.
        self.food = self.food_positions.pop(0)
 
        # Update ui and clock.
        if self.gui:
            self._update_ui()
            self.clock.tick(SPEED)

        if len(self.snake) == self.max_len:
            time.sleep(3)

        # Return game over and score.
        return game_over
