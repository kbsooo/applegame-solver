import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import namedtuple, deque

# Define the structure of a single transition (experience)
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    """A cyclic buffer of bounded size that holds the transitions observed recently."""
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """Sample a batch of transitions from the memory"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    """
    Deep Q-Network model.
    It's a Convolutional Neural Network that takes the board state (17x10)
    and outputs a single value representing the 'goodness' of that state.
    We will use this network to evaluate each possible action.
    """
    def __init__(self, board_height, board_width):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        # Flatten the output of the conv layers to feed into linear layers
        def conv2d_size_out(size, kernel_size=3, stride=1, padding=1):
            return (size + 2*padding - kernel_size) // stride + 1
        
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(board_width)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(board_height)))
        linear_input_size = convw * convh * 64
        
        self.head = nn.Linear(linear_input_size, 1) # Output a single Q-value

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

class DQNAgent:
    def __init__(self, board_shape, memory_capacity=10000, batch_size=128, gamma=0.99):
        self.board_height, self.board_width = board_shape
        self.batch_size = batch_size
        self.gamma = gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create the policy network and the target network
        self.policy_net = DQN(self.board_height, self.board_width).to(self.device)
        self.target_net = DQN(self.board_height, self.board_width).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # Target network is only for evaluation

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.memory = ReplayMemory(memory_capacity)

    def select_action(self, state, possible_actions, epsilon):
        """
        Selects an action from the list of possible_actions.
        It will choose the best action based on the policy network's Q-value,
        or a random action (epsilon-greedy strategy).
        """
        if random.random() > epsilon:
            # --- Exploitation: Choose the best known action ---
            with torch.no_grad():
                best_action = None
                max_q_value = -float('inf')

                for action in possible_actions:
                    # For each possible action, create a hypothetical next state
                    next_state_tensor = self._get_state_tensor_for_action(state, action)
                    
                    # Get the Q-value for this hypothetical state from the policy network
                    q_value = self.policy_net(next_state_tensor).item()

                    if q_value > max_q_value:
                        max_q_value = q_value
                        best_action = action
                return best_action
        else:
            # --- Exploration: Choose a random action ---
            return random.choice(possible_actions)

    def _get_state_tensor_for_action(self, state, action):
        """Helper to create a hypothetical next state tensor for a given action."""
        temp_board = state.copy()
        for y, x in action:
            temp_board[y, x] = 0
        # Convert to a PyTorch tensor with the correct dimensions [Batch, Channel, Height, Width]
        state_tensor = torch.tensor(temp_board, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        return state_tensor

    def optimize_model(self):
        """Performs one step of the optimization (on the policy network)"""
        if len(self.memory) < self.batch_size:
            return # Not enough experiences in memory to learn

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # This part is complex and will be implemented in the training script.
        # It involves calculating the loss between the policy network's predictions
        # and the target network's values, and then backpropagating the loss.
        print("optimize_model function is a placeholder.")
        pass

    def update_target_net(self):
        """Update the target network with the policy network's weights."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

if __name__ == '__main__':
    # Example of how to initialize the agent
    agent = DQNAgent(board_shape=(17, 10))
    print("DQN Agent initialized successfully.")
    print("Policy Network Architecture:")
    print(agent.policy_net)
