import numpy as np
import torch
from Game import Point, BLOCK_SIZE


class VectorizedSnakeGame:
    """
    Vectorized Snake game that processes N environments simultaneously
    using numpy arrays. Eliminates the Python loop over environments.

    The snake body is tracked via a "body timer" grid: each cell stores
    how many steps until that segment disappears. The head always has
    value = snake_length, and each step all cells decrement by 1.
    When food is eaten, the decrement is skipped (extending the snake).
    """

    def __init__(self, num_envs, rows=10, cols=10,
                 frame_rows=12, frame_cols=12, stack_size=4):
        self.num_envs = num_envs
        self.rows = rows
        self.cols = cols
        self.frame_rows = frame_rows
        self.frame_cols = frame_cols
        self.stack_size = stack_size
        self.max_len = rows * cols

        # Game state arrays.
        self.body_grid = np.zeros((num_envs, rows, cols), dtype=np.int32)
        self.head_row = np.zeros(num_envs, dtype=np.int32)
        self.head_col = np.zeros(num_envs, dtype=np.int32)
        self.direction = np.zeros(num_envs, dtype=np.int32)  # 0=R,1=D,2=L,3=U
        self.food_row = np.zeros(num_envs, dtype=np.int32)
        self.food_col = np.zeros(num_envs, dtype=np.int32)
        self.snake_length = np.full(num_envs, 3, dtype=np.int32)
        self.score = np.zeros(num_envs, dtype=np.int32)
        self.step_counter = np.zeros(num_envs, dtype=np.int32)

        # Frame stack: (num_envs, stack_size, frame_rows, frame_cols).
        self.frame_stack = np.zeros(
            (num_envs, stack_size, frame_rows, frame_cols), dtype=np.float32
        )

        # Direction lookup tables: R, D, L, U.
        self._dr = np.array([0, 1, 0, -1], dtype=np.int32)
        self._dc = np.array([1, 0, -1, 0], dtype=np.int32)

        self.reset_all()


    def reset_all(self):
        """Reset all environments."""
        for i in range(self.num_envs):
            self.reset_env(i)


    def reset_env(self, env_idx):
        """Reset a single environment."""
        i = env_idx
        self.body_grid[i] = 0
        self.snake_length[i] = 3
        self.score[i] = 0
        self.step_counter[i] = 0
        self.direction[i] = 0  # RIGHT

        center_r = self.rows // 2
        center_c = self.cols // 2
        self.head_row[i] = center_r
        self.head_col[i] = center_c

        # Place snake: head at center, 2 body segments to the left.
        self.body_grid[i, center_r, center_c] = 3
        self.body_grid[i, center_r, center_c - 1] = 2
        self.body_grid[i, center_r, center_c - 2] = 1

        self._place_food(i)

        # Reset frame stack: fill with zeros, then push one real frame.
        # This matches the original behavior: [zero, zero, zero, real].
        self.frame_stack[i] = 0
        self._push_frame_single(i)


    def _place_food(self, env_idx):
        """Place food on a random empty cell for one environment."""
        grid = self.body_grid[env_idx]
        empty = np.argwhere(grid == 0)
        if len(empty) == 0:
            return
        choice = empty[np.random.randint(len(empty))]
        self.food_row[env_idx] = choice[0]
        self.food_col[env_idx] = choice[1]


    def _push_frame_single(self, env_idx):
        """Generate and push a single frame for one environment."""
        frame = np.zeros((self.frame_rows, self.frame_cols), dtype=np.float32)

        # Body (0.5).
        body_mask = (self.body_grid[env_idx] > 0).astype(np.float32) * 0.5
        frame[1:self.rows + 1, 1:self.cols + 1] = body_mask

        # Head (1.0).
        hr, hc = self.head_row[env_idx], self.head_col[env_idx]
        frame[hr + 1, hc + 1] = 1.0

        # Food (2.0).
        fr, fc = self.food_row[env_idx], self.food_col[env_idx]
        frame[fr + 1, fc + 1] = 2.0

        # Roll stack and replace last.
        self.frame_stack[env_idx] = np.roll(self.frame_stack[env_idx], -1, axis=0)
        self.frame_stack[env_idx, -1] = frame


    def push_frames_all(self):
        """Generate and push frames for ALL envs. Fully vectorized."""
        new_frames = np.zeros(
            (self.num_envs, self.frame_rows, self.frame_cols), dtype=np.float32
        )

        # Body (0.5) for all envs.
        body_mask = (self.body_grid > 0).astype(np.float32) * 0.5
        new_frames[:, 1:self.rows + 1, 1:self.cols + 1] = body_mask

        # Head (1.0) and food (2.0) for all envs.
        idx = np.arange(self.num_envs)
        new_frames[idx, self.head_row + 1, self.head_col + 1] = 1.0
        new_frames[idx, self.food_row + 1, self.food_col + 1] = 2.0

        # Roll stacks and replace last frame.
        self.frame_stack = np.roll(self.frame_stack, -1, axis=1)
        self.frame_stack[:, -1] = new_frames


    def get_stacked_frames(self):
        """Return stacked frames as torch tensor: (N, stack_size, H, W)."""
        return torch.from_numpy(self.frame_stack.copy())


    def get_food_point(self, env_idx):
        """Return food position as a Point (for replay compatibility)."""
        return Point(
            int(self.food_col[env_idx]) * BLOCK_SIZE,
            int(self.food_row[env_idx]) * BLOCK_SIZE
        )


    def check_collision_batch(self, actions):
        """
        Check if taking the given actions will result in a collision (wall or body).
        Does not advance the game state.
        
        Args:
            actions: (N, 3) array-like of actions.
            
        Returns:
            collisions: (N,) bool array. True if action leads to collision.
        """
        actions_arr = np.asarray(actions, dtype=np.int32)
        action_idx = np.argmax(actions_arr, axis=1)
        dir_delta = np.where(action_idx == 0, 0, np.where(action_idx == 1, 1, -1))
        test_dir = (self.direction + dir_delta) % 4
        
        new_hr = self.head_row + self._dr[test_dir]
        new_hc = self.head_col + self._dc[test_dir]
        
        wall_hit = (
            (new_hr < 0) | (new_hr >= self.rows) |
            (new_hc < 0) | (new_hc >= self.cols)
        )
        
        body_hit = np.zeros(self.num_envs, dtype=bool)
        safe = ~wall_hit
        if safe.any():
            safe_idx = np.where(safe)[0]
            body_hit[safe_idx] = (
                self.body_grid[safe_idx, new_hr[safe_idx], new_hc[safe_idx]] > 0
            )
            
        return wall_hit | body_hit


    def step(self, actions):
        """
        Step all environments simultaneously.

        Args:
            actions: list of [straight, right, left] per env, or (N, 3) array.

        Returns:
            rewards:   (N,) float32 array.
            gameovers: (N,) bool array.
            scores:    (N,) int32 array (copy).
        """
        actions_arr = np.asarray(actions, dtype=np.int32)
        self.step_counter += 1

        # 1. Convert actions to direction changes.
        #    straight=0, right=+1, left=-1.
        action_idx = np.argmax(actions_arr, axis=1)
        dir_delta = np.where(action_idx == 0, 0, np.where(action_idx == 1, 1, -1))
        self.direction = (self.direction + dir_delta) % 4

        # 2. Compute new head positions.
        new_hr = self.head_row + self._dr[self.direction]
        new_hc = self.head_col + self._dc[self.direction]

        # 3. Wall collision.
        wall_hit = (
            (new_hr < 0) | (new_hr >= self.rows) |
            (new_hc < 0) | (new_hc >= self.cols)
        )

        # 4. Body collision (only check for non-wall envs).
        body_hit = np.zeros(self.num_envs, dtype=bool)
        safe = ~wall_hit
        if safe.any():
            safe_idx = np.where(safe)[0]
            body_hit[safe_idx] = (
                self.body_grid[safe_idx, new_hr[safe_idx], new_hc[safe_idx]] > 0
            )

        # 5. Timeout: 100 steps per body segment.
        timeout = self.step_counter > 100 * self.snake_length

        # 6. Gameover.
        gameover = wall_hit | body_hit | timeout

        # 7. Rewards.
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        rewards[gameover] = -1.0

        # 8. Process alive envs.
        alive = ~gameover
        if alive.any():
            eating = alive & (new_hr == self.food_row) & (new_hc == self.food_col)
            not_eating = alive & ~eating

            # Non-eating: decrement body timers, then place head.
            if not_eating.any():
                ne_idx = np.where(not_eating)[0]
                self.body_grid[ne_idx] = np.maximum(self.body_grid[ne_idx] - 1, 0)
                self.head_row[ne_idx] = new_hr[ne_idx]
                self.head_col[ne_idx] = new_hc[ne_idx]
                self.body_grid[
                    ne_idx, new_hr[ne_idx], new_hc[ne_idx]
                ] = self.snake_length[ne_idx]

                # Survival reward for long snakes.
                long_snake = not_eating & (self.snake_length > 10)
                rewards[long_snake] += 0.01

            # Eating: do NOT decrement, place head with length+1.
            if eating.any():
                eat_idx = np.where(eating)[0]
                self.head_row[eat_idx] = new_hr[eat_idx]
                self.head_col[eat_idx] = new_hc[eat_idx]
                self.body_grid[
                    eat_idx, new_hr[eat_idx], new_hc[eat_idx]
                ] = self.snake_length[eat_idx] + 1
                self.snake_length[eat_idx] += 1
                self.score[eat_idx] += 1
                rewards[eat_idx] = 1.0

                # Place new food (per-env, can't fully vectorize).
                for idx in eat_idx:
                    if self.snake_length[idx] >= self.max_len:
                        gameover[idx] = True
                        print("You won!")
                    else:
                        self._place_food(idx)

        return rewards, gameover, self.score.copy()
