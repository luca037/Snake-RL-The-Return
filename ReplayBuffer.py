import torch
import numpy as np
import h5py

class ReplayBuffer:

    def __init__(self, state_shape, action_dim, max_size, device):

        self.mem_size = max_size
        self.mem_cntr = 0
        
        # Memory pre-allocation
        self.state_memory = torch.zeros((self.mem_size, *state_shape), dtype=torch.float, device=device)
        self.new_state_memory = torch.zeros((self.mem_size, *state_shape), dtype=torch.float, device=device)
        self.action_memory = torch.zeros((self.mem_size, action_dim), dtype=torch.int, device=device) # Store as indices.
        self.reward_memory = torch.zeros(self.mem_size, dtype=torch.float, device=device) 
        self.terminal_memory = torch.zeros(self.mem_size, dtype=torch.bool, device=device)


    def append(self, state, action, reward, new_state, gameover):
        # Use circular index.
        index = self.mem_cntr % self.mem_size
        
        # Store transition.
        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.action_memory[index] = torch.tensor(action, dtype=torch.int8)
        self.reward_memory[index] = reward
        self.terminal_memory[index] = gameover
        
        self.mem_cntr += 1


    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        
        # Generate random indices.
        batch = np.random.choice(max_mem, batch_size, replace=False)

        # Define the batch.
        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return (states, actions, rewards, new_states, dones)


    def __len__(self):
        return min(self.mem_cntr, self.mem_size)

    def capacity(self):
        return self.mem_size
    
    def store_buffer_h5(self, out_path, group_name='common_memory'):
        size = min(self.mem_cntr, self.mem_size)
        with h5py.File(out_path, 'w') as f:
            # Create a group to store all ements within a sample.
            grp = f.create_group(group_name)
            grp.create_dataset('states', data=self.state_memory[0:size].cpu(), compression='gzip')
            grp.create_dataset('next_states', data=self.new_state_memory[0:size].cpu(), compression='gzip')

            # TODO: cast to specific type to save space.
            grp.create_dataset('actions', data=self.action_memory[0:size].cpu(), compression='gzip')
            grp.create_dataset('rewards', data=self.reward_memory[0:size].cpu(), compression='gzip')
            grp.create_dataset('gameovers', data=self.terminal_memory[0:size].cpu(), compression='gzip')
            print(f"INFO: Saved {size} samples to memory disk.")


    def load_buffer_h5(self, path, group_name='common_memory'):
        with h5py.File(path, 'r') as f:
                
            size = min(self.mem_size, f[group_name]['states'].shape[0])

            # Load samples to ram (ony the one that fits).
            states = f[group_name]['states'][0:size]
            actions = f[group_name]['actions'][0:size]
            rewards = f[group_name]['rewards'][0:size]
            next_states = f[group_name]['next_states'][0:size]
            gameovers = f[group_name]['gameovers'][0:size]

            # Save to same device as self variables.
            self.state_memory[0:size] = torch.from_numpy(states)
            self.new_state_memory[0:size] = torch.from_numpy(next_states)
            self.action_memory[0:size] = torch.from_numpy(actions)
            self.reward_memory[0:size] = torch.from_numpy(rewards)
            self.terminal_memory[0:size] = torch.from_numpy(gameovers)

            self.mem_cntr = size
