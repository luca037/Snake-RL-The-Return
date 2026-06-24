import torch
import torch.nn as nn
import torch.optim as optim


class CNN_QNet(nn.Module):
    def __init__(self, in_chan, grid_rows, grid_cols):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_chan, 32, kernel_size=3, padding=1, stride=1), 
            nn.ReLU(), 

            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(), 

            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(), 
        )

        input_size = 64 * grid_rows * grid_cols
        self.linear = nn.Sequential(
            nn.Linear(input_size, 512), 
            nn.ReLU(), 
            nn.Linear(512, 3)
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)


class QTrainer:
    def __init__(self, model, target_model, lr, gamma):

        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.target_model = target_model

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        #self.criterion = nn.MSELoss()
        self.criterion = nn.SmoothL1Loss()

    def train_step(self, state, action, reward, next_state, done):
        # Q-Value forward pass.
        self.model.train() 
        pred = self.model(state) # shape: (N x 3) (3 = # actions)
        
        # Double DQN: use online model to SELECT best action,
        # but TARGET model to EVALUATE that action's Q-value.
        # This reduces Q-value overestimation.
        self.model.eval()
        self.target_model.eval() 
        with torch.no_grad():
            # Online model selects the best next action.
            online_next = self.model(next_state)                    # shape: (N x 3)
            best_actions = torch.argmax(online_next, dim=1, keepdim=True) # shape: (N x 1)
            # Target model evaluates that action.
            target_next = self.target_model(next_state)             # shape: (N x 3)
            max_next_q = target_next.gather(1, best_actions).squeeze(1) # shape: N
            
        # Compute target value (scalar).
        # Q_target := reward + gamma * max(Q(s', a'))
        Q_new = reward + self.gamma * max_next_q # shape: N
        
        # Q_target is just the reward if the episode is done (terminal state).
        Q_target = torch.where(done, reward, Q_new) # shape: N
    
        # We only update the Q-value for the action 
        # that was actually taken (action[idx]).

        # Find the index of the action taken.
        # (dim=1 means argmax over the action dimension)
        action_indices = torch.argmax(action, dim=1, keepdim=True)  # shape: (N x 1)
    
        # Efficiently update only the relevant indices
        target = pred.clone().detach() # => calculation doesn't affect gradients.
        target.scatter_(1, action_indices, Q_target.unsqueeze(1))
    
        # Optimize model.
        self.model.train() 
        self.optimizer.zero_grad()
        
        # Compute loss; backpropagate.
        loss = self.criterion(target, pred)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.optimizer.step()

        return loss.detach().item()
