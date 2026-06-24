from collections import deque
import random
import numpy as np
import torch
from Game import SnakeGame, Direction, Point, BLOCK_SIZE
from VecGame import VectorizedSnakeGame
from Model import *
import os
from ReplayBuffer import ReplayBuffer
from abc import ABC, abstractmethod

class BaseAgent(ABC):

    def __init__(self,
            max_dataset_size     = 10,
            batch_size           = 32,
            random_steps         = 0,
            lr                   = 0.001,
            epsilon              = 0.1,
            decaying_epsilon     = 0.999,
            min_epsilon          = 0.01,
            gamma                = 0.9,
            target_sync          = 100,
            train_steps_per_step = 1,
            num_envs             = 1,
            out_model_path       = './model.pth',
            memory_path          = './memory.pth',
            out_csv_path         = None,
            device               = 'cpu',
            gui                  = False,
            checkpoint_path      = None,
            load_buffer          = False
    ):
        # Hyperparams.
        self.gamma = gamma
        self.epsilon = epsilon
        self.decaying_epsilon = decaying_epsilon
        self.min_epsilon = min_epsilon
        self.random_steps = random_steps
        self.lr = lr
        self.target_sync = target_sync
        self.max_dataset_size = max_dataset_size
        self.batch_size = batch_size
        self.train_steps_per_step = train_steps_per_step
        self.device = device

        # Paths.
        self.out_model_path = out_model_path
        self.memory_path = memory_path
        self.out_csv_path = out_csv_path

        # Game and stats.
        self.num_envs = num_envs
        # Create N game environments. Only the first gets GUI (if enabled).
        self.games = [SnakeGame(gui=(gui and i == 0)) for i in range(num_envs)]
        self.game = self.games[0]  # Backward compat.
        self.record = 0
        self.record_replay = {'actions': [], 'foods': []}
        self.num_episodes = 0
        self.num_steps = 0

        # Init all necessary components.
        self.memory = self._init_memory()
        self.model, self.target_model, self.trainer = self._init_model()

        # Used to comptue action-value function.
        self.fixed_states = None

        # Load checkpoint_path, if necesary.
        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path, load_buffer)

        # Compile models for faster execution (requires PyTorch >= 2.0).
        self.model = torch.compile(self.model)
        self.target_model = torch.compile(self.target_model)
        # Update references in the trainer as well.
        self.trainer.model = self.model
        self.trainer.target_model = self.target_model

        # Create csv file with stats, if necessary.
        if self.out_csv_path is not None:
            csv_header = "mean_score,mean_score_100,score,mean_reward_100,epsilon,max_loss,avg_q\n"
            if not os.path.exists(self.out_csv_path):
                with open(self.out_csv_path, 'w') as f:
                    f.write(csv_header)
            print("INFO: Stats will be stored in", self.out_csv_path)


    ### Abstract methods ###
    @abstractmethod
    def _init_memory(self):
        pass

    @abstractmethod
    def _init_model(self):
        pass

    @abstractmethod
    def _update_epsilon(self):
        pass

    @abstractmethod
    def get_state(self, game, env_idx=0):
        pass

    def _on_reset(self, env_idx=None):
        pass

    ### Common methods ###
    def _save_checkpoint(self):
        # Store the memory to disk.
        self.memory.store_buffer_h5(out_path=self.memory_path)

        checkpoint = {
            'episode': self.num_episodes,
            'steps': self.num_steps,
            'record': self.record,
            'epsilon': self.epsilon,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.trainer.optimizer.state_dict(),
            'record_replay': self.record_replay
        }
        torch.save(checkpoint, self.out_model_path)
    

    def _load_checkpoint(self, checkpoint_path, load_buffer):
        # Load the checkpoint file (to CPU).
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=self.device)
        # Strip the '_orig_mod.' prefix if it exists
        fixed_state_dict = {k.replace('_orig_mod.', ''): v for k, v in checkpoint['model_state_dict'].items()}
        # Load the cleaned state dict
        self.model.load_state_dict(fixed_state_dict)
        # Apply saved states.
        #self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self._target_sync()
        self.trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Restore metadata.
        self.num_episodes = checkpoint['episode']
        self.num_steps = checkpoint['steps']
        self.record = checkpoint['record']
        self.epsilon = checkpoint['epsilon']
        self.record_replay = checkpoint['record_replay']

        if load_buffer:
            self.memory.load_buffer_h5(path=self.memory_path)

        print(f"INFO: Checkpoint restored.")


    def _remember(self, state, action, reward, next_state, gameover):
        self.memory.append(state, action, reward, next_state, gameover)


    def _target_sync(self):
        self.target_model.load_state_dict(self.model.state_dict())


    def _train_batch(self):
        # Create batch.
        states, actions, rewards, next_states, gameovers = self.memory.sample_buffer(self.batch_size)
        # Train and return loss.
        loss = self.trainer.train_step(states, actions, rewards, next_states, gameovers)
        return loss 


    def get_action(self, state):
        # The action to take.
        final_move = [0,0,0]

        # Get the 3 values Q(state, a) for each action a.
        self.model.eval()
        with torch.no_grad():
            state0 = state.detach().clone()
            state0 = torch.unsqueeze(state0, 0).to(self.device)
            prediction = self.model(state0)

        # Gready choice.
        greedy_idx = torch.argmax(prediction).item()
        # Non-gready choices.
        rnd_actions = np.delete(np.array([0, 1, 2]), greedy_idx).tolist()
        
        # Choose the move.
        if self.num_steps < self.random_steps:
            action = random.randint(0, 2)
            final_move[action] = 1
        elif random.random() < self.epsilon: # Exploration.
            action = np.random.choice(rnd_actions, p=[0.5, 0.5])
            final_move[action] = 1
        else: # Exploitation.
            final_move[greedy_idx] = 1

        return final_move


    def get_action_batch(self, states_batch):
        """Get actions for all environments in a single forward pass."""
        num_envs = states_batch.shape[0]

        # Single forward pass for all envs.
        self.model.eval()
        with torch.no_grad():
            states_gpu = states_batch.to(self.device)
            predictions = self.model(states_gpu)  # (num_envs, 3)

        sorted_indices = torch.argsort(predictions, dim=1, descending=True)  # (N, 3)

        # Pre-calculate collisions for all possible actions.
        # This takes 3 vectorized calls but allows masking in O(1) per env.
        mask_0 = self._get_collision_mask_batch([[1, 0, 0]] * num_envs)
        mask_1 = self._get_collision_mask_batch([[0, 1, 0]] * num_envs)
        mask_2 = self._get_collision_mask_batch([[0, 0, 1]] * num_envs)
        env_collisions = [[mask_0[i], mask_1[i], mask_2[i]] for i in range(num_envs)]

        actions = []
        for i in range(num_envs):
            final_move = [0, 0, 0]
            
            rank_0 = sorted_indices[i, 0].item()
            rank_1 = sorted_indices[i, 1].item()
            rank_2 = sorted_indices[i, 2].item()
            rnd_actions = [rank_1, rank_2]

            if self.num_steps < self.random_steps:
                action = random.randint(0, 2)
                final_move[action] = 1
            elif random.random() < self.epsilon: # Exploration
                action = np.random.choice(rnd_actions, p=[0.5, 0.5])
                final_move[action] = 1
            else: # Exploitation: apply masking
                if not env_collisions[i][rank_0]:
                    final_move[rank_0] = 1
                elif not env_collisions[i][rank_1]:
                    final_move[rank_1] = 1
                else:
                    final_move[rank_2] = 1

            actions.append(final_move)

        return actions


    ### Hookable methods for train() — override in subclasses ###

    def _get_collision_mask_batch(self, actions):
        """Check if actions lead to collisions. Override for vectorized games."""
        collisions = []
        for i in range(self.num_envs):
            pt = self.games[i].move(actions[i], perform=False)
            collisions.append(self.games[i].is_collision(pt))
        return collisions

    def _init_env_states(self):
        """Return initial states for all environments."""
        return [self.get_state(self.games[i], env_idx=i) for i in range(self.num_envs)]

    def _step_all_envs(self, actions):
        """Step all environments. Returns (rewards, gameovers, scores, new_states)."""
        rewards, gameovers, scores, new_states = [], [], [], []
        for i in range(self.num_envs):
            r, g, s = self.games[i].play_step(actions[i])
            ns = self.get_state(self.games[i], env_idx=i)
            rewards.append(r)
            gameovers.append(g)
            scores.append(s)
            new_states.append(ns)
        return rewards, gameovers, scores, new_states

    def _reset_single_env(self, i):
        """Reset env i and return its new initial state."""
        self.games[i].reset()
        self._on_reset(env_idx=i)
        return self.get_state(self.games[i], env_idx=i)

    def _get_env_food(self, i):
        """Return food position for env i."""
        return self.games[i].food


    def train(self):
        total_score = 0
        score_last_100 = deque(maxlen=100)
        reward_last_100 = deque(maxlen=100)

        num_envs = self.num_envs

        # Per-env tracking.
        env_steps = [0] * num_envs
        env_max_loss = [0.0] * num_envs
        env_reward = [0.0] * num_envs
        env_actions_replay = [[] for _ in range(num_envs)]
        env_food_replay = [[] for _ in range(num_envs)]

        # Initial states for all envs.
        current_states = self._init_env_states()

        # To monitor action value (Q).
        if self.fixed_states is None:
            self.fixed_states = torch.zeros((250, *current_states[0].shape), device=self.device)

        while True:
            # ---- Batched action selection ----
            states_batch = torch.stack(current_states)  # (num_envs, ...)
            actions = self.get_action_batch(states_batch)

            # ---- Step all environments (hookable) ----
            rewards, gameovers, scores, new_states = self._step_all_envs(actions)

            # ---- Process results per env ----
            for i in range(num_envs):
                # Store fixed states for avg_q monitoring.
                if self.num_steps < 250:
                    self.fixed_states[self.num_steps] = current_states[i]

                # Remember transition.
                self._remember(current_states[i], actions[i], rewards[i], new_states[i], gameovers[i])

                # Update per-env tracking.
                current_states[i] = new_states[i]
                env_actions_replay[i].append(actions[i])
                env_food_replay[i].append(self._get_env_food(i))
                self.num_steps += 1
                env_steps[i] += 1
                env_reward[i] += rewards[i]

                if gameovers[i]:
                    self.num_episodes += 1

                    # Reset this env (hookable).
                    current_states[i] = self._reset_single_env(i)

                    # Save checkpoint if new record.
                    if scores[i] > self.record:
                        self.record = scores[i]
                        self.record_replay['actions'] = env_actions_replay[i]
                        self.record_replay['foods'] = env_food_replay[i]
                        print("INFO: New record! Saving checkpoint...")
                        self._save_checkpoint()

                    # Reset per-env replay.
                    env_actions_replay[i] = []
                    env_food_replay[i] = []

                    # Update global stats.
                    total_score += scores[i]
                    mean_score = total_score / self.num_episodes
                    score_last_100.append(scores[i])
                    mean_score_100 = sum(score_last_100) / len(score_last_100)

                    reward_last_100.append(env_reward[i])
                    mean_reward_100 = sum(reward_last_100) / len(reward_last_100)

                    self.model.eval()
                    with torch.no_grad():
                        avg_q = self.model(self.fixed_states)
                        avg_q = torch.max(avg_q, dim=1).values
                        avg_q = avg_q.mean().item()

                    print(
                        "INFO:\n"
                        f"\tGAME: {self.num_episodes}\n"
                        f"\tRecord: {self.record}\n"
                        f"\tSteps: {self.num_steps}\n"
                        f"\tBuffer memory size: {len(self.memory)}\n"
                        f"\tScore: {scores[i]}\n"
                        f"\tDuration (steps): {env_steps[i]}\n"
                        f"\tMean score last 100: {mean_score_100}\n"
                        f"\tMean reward last 100: {mean_reward_100}\n"
                        f"\tTotal score: {total_score}\n"
                        f"\tepsilon: {self.epsilon}"
                    )

                    csv_line = f"{mean_score},{mean_score_100},{scores[i]},{mean_reward_100},{self.epsilon},{env_max_loss[i]},{avg_q}\n"

                    if self.out_csv_path is not None:
                        with open(self.out_csv_path, 'a') as f:
                            f.write(csv_line)

                    # Reset per-env stats.
                    env_steps[i] = 0
                    env_max_loss[i] = 0
                    env_reward[i] = 0

            # ---- Train on batch (after stepping all envs) ----
            if len(self.memory) > self.batch_size:
                for _ in range(self.train_steps_per_step):
                    loss = self._train_batch()
                    for i in range(num_envs):
                        env_max_loss[i] = max(env_max_loss[i], loss)

            # ---- Sync target network ----
            if not self.num_steps % self.target_sync:
                print("Info: Syncronizing target model with main model...")
                self._target_sync()

            # ---- Update epsilon ----
            self._update_epsilon()


class AtariAgent(BaseAgent):

    def __init__(self, **kwargs):
        # Define the frame stack.
        self.frame_rows = 12
        self.frame_cols = 12
        self.stack_size = 4
        # Create per-env frame stacks (used as fallback for single env).
        num_envs = kwargs.get('num_envs', 1)
        self.env_frames = [deque(maxlen=self.stack_size) for _ in range(num_envs)]
        for i in range(num_envs):
            self._on_reset(env_idx=i)
        super().__init__(**kwargs)

        # Create vectorized game engine for fast parallel stepping.
        self.vec_game = VectorizedSnakeGame(
            num_envs=self.num_envs,
            rows=self.frame_rows - 2,   # 10 (playable area)
            cols=self.frame_cols - 2,   # 10
            frame_rows=self.frame_rows, # 12 (with wall padding)
            frame_cols=self.frame_cols, # 12
            stack_size=self.stack_size   # 4
        )


    ### Override hookable methods to use VecGame ###

    def _init_env_states(self):
        """Return initial states from the vectorized game."""
        frames = self.vec_game.get_stacked_frames()  # (N, 4, 12, 12)
        return [frames[i] for i in range(self.num_envs)]

    def _step_all_envs(self, actions):
        """Step all envs using the vectorized game engine."""
        rewards, gameovers, scores = self.vec_game.step(actions)
        self.vec_game.push_frames_all()
        frames = self.vec_game.get_stacked_frames()  # (N, 4, 12, 12)
        new_states = [frames[i] for i in range(self.num_envs)]
        return rewards.tolist(), gameovers.tolist(), scores.tolist(), new_states

    def _get_collision_mask_batch(self, actions):
        """Vectorized collision check."""
        return self.vec_game.check_collision_batch(actions).tolist()

    def _reset_single_env(self, i):
        """Reset env i in the vectorized game."""
        self.vec_game.reset_env(i)
        return self.vec_game.get_stacked_frames()[i]

    def _get_env_food(self, i):
        """Return food position from the vectorized game."""
        return self.vec_game.get_food_point(i)


    def _init_memory(self):
        return ReplayBuffer(
            state_shape=(self.stack_size, self.frame_rows, self.frame_cols),
            action_dim=3,
            max_size=self.max_dataset_size,
            device=self.device
        )
 

    def _init_model(self):
        model = CNN_QNet(
                in_chan=4,
                grid_rows=self.frame_rows,
                grid_cols=self.frame_cols
        ).to(self.device)

        target_model = CNN_QNet(
                in_chan=4,
                grid_rows=self.frame_rows,
                grid_cols=self.frame_cols
        ).to(self.device)

        trainer = QTrainer(model, target_model, self.lr, self.gamma)

        return model, target_model, trainer


    def _update_epsilon(self):
        if self.num_steps > self.memory.capacity():
            # Phase 2: Fixed epsilon
            self.epsilon = self.min_epsilon
        else:
            # Phase 1: Linear annealing
            
            # Calculate the decay factor (1.0 at start, 0.0 at end)
            decay_progress = self.num_steps / self.memory.capacity()
            
            # Calculate the current epsilon value
            self.epsilon = self.min_epsilon + (1 - self.min_epsilon) * (1 - decay_progress)


    def _on_reset(self, env_idx=None):
        """Reset frame stack. If env_idx is None, reset all envs."""
        if env_idx is not None:
            envs_to_reset = [env_idx]
        else:
            envs_to_reset = range(len(self.env_frames))
        for idx in envs_to_reset:
            self.env_frames[idx].clear()
            for _ in range(self.stack_size):
                zero_frame = torch.zeros((self.frame_rows, self.frame_cols), dtype=torch.float)
                self.env_frames[idx].append(zero_frame)


    def _get_single_frame(self, game):
        # Initialize empty grid.
        dims = (self.frame_rows, self.frame_cols)
        state = torch.zeros(dims, dtype=torch.float)

        # Draw body (value=0.5).
        for point in game.snake[1:]:
            i = int(point.y // BLOCK_SIZE) + 1
            j = int(point.x // BLOCK_SIZE) + 1
            state[i, j] = 0.5

        # Draw head (value=1.0)
        i = int(game.head.y // BLOCK_SIZE) + 1
        j = int(game.head.x // BLOCK_SIZE) + 1
        state[i, j] = 1.0

        # Draw food (value=2.0)
        i = int(game.food.y // BLOCK_SIZE) + 1
        j = int(game.food.x // BLOCK_SIZE) + 1
        state[i, j] = 2.0

        return state


    def get_state(self, game, env_idx=0):
        # Take snapshot and add to this env's stack.
        frame = self._get_single_frame(game)
        self.env_frames[env_idx].append(frame)

        # Convert deque to torch.tensor.
        return torch.from_numpy(np.array(self.env_frames[env_idx]))
