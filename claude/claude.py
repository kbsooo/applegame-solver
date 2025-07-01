import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque, namedtuple
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt

class AppleGameEnvironment:
    def __init__(self, rows=10, cols=17):
        self.rows = rows
        self.cols = cols
        self.reset()
    
    def reset(self):
        self.board = np.random.randint(1, 10, (self.rows, self.cols))
        self.cleared = np.zeros((self.rows, self.cols), dtype=bool)
        self.score = 0
        self.moves_made = 0
        self.max_moves = 200
        return self.get_state()
    
    def get_state(self):
        visible_board = self.board.copy().astype(float)
        visible_board[self.cleared] = 0
        
        features = np.stack([
            visible_board,
            self.cleared.astype(float),
            np.ones_like(visible_board) * (self.moves_made / self.max_moves)
        ], axis=0)
        return features
    
    def get_valid_combinations(self):
        valid_moves = []
        visible_positions = np.where(~self.cleared)
        positions = list(zip(visible_positions[0], visible_positions[1]))
        
        for i in range(len(positions)):
            for j in range(i+1, min(i+20, len(positions))):
                subset = positions[i:j+1]
                if self.can_select_rectangle(subset):
                    values = [self.board[r, c] for r, c in subset]
                    if sum(values) == 10:
                        valid_moves.append(subset)
        return valid_moves
    
    def can_select_rectangle(self, positions):
        if len(positions) < 2:
            return False
        
        rows = [pos[0] for pos in positions]
        cols = [pos[1] for pos in positions]
        
        min_row, max_row = min(rows), max(rows)
        min_col, max_col = min(cols), max(cols)
        
        for r in range(min_row, max_row + 1):
            for c in range(min_col, max_col + 1):
                if not self.cleared[r, c] and (r, c) not in positions:
                    return False
        return True
    
    def step(self, action_positions):
        if not action_positions:
            return self.get_state(), -1, True, {}
        
        values = [self.board[r, c] for r, c in action_positions]
        
        if sum(values) != 10:
            return self.get_state(), -5, False, {}
        
        if not self.can_select_rectangle(action_positions):
            return self.get_state(), -3, False, {}
        
        for r, c in action_positions:
            self.cleared[r, c] = True
        
        reward = len(action_positions)
        self.score += reward
        self.moves_made += 1
        
        done = (np.sum(~self.cleared) == 0) or (self.moves_made >= self.max_moves)
        
        if done and np.sum(~self.cleared) == 0:
            reward += 50
        
        return self.get_state(), reward, done, {'score': self.score}

class DQN(nn.Module):
    def __init__(self, input_channels=3, board_height=10, board_width=17):
        super(DQN, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        conv_output_size = board_height * board_width * 128
        
        self.fc1 = nn.Linear(conv_output_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, board_height * board_width)
        
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class DQNAgent:
    def __init__(self, state_shape, action_size, lr=0.001, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_shape = state_shape
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.q_network = DQN(state_shape[0], state_shape[1], state_shape[2]).to(self.device)
        self.target_network = DQN(state_shape[0], state_shape[1], state_shape[2]).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        self.memory = deque(maxlen=10000)
        self.update_target_every = 100
        self.learn_every = 4
        self.step_count = 0
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append(Experience(state, action, reward, next_state, done))
    
    def act(self, env, state):
        valid_moves = env.get_valid_combinations()
        
        if not valid_moves:
            return []
        
        if np.random.random() <= self.epsilon:
            return random.choice(valid_moves)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        
        best_combination = None
        best_score = float('-inf')
        
        for combination in valid_moves:
            score = sum(q_values[0][r * env.cols + c].item() for r, c in combination)
            if score > best_score:
                best_score = score
                best_combination = combination
        
        return best_combination if best_combination else random.choice(valid_moves)
    
    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([e.state for e in batch]).to(self.device)
        actions = [e.action for e in batch]
        rewards = torch.FloatTensor([e.reward for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in batch]).to(self.device)
        dones = torch.BoolTensor([e.done for e in batch]).to(self.device)
        
        current_q_values = self.q_network(states)
        next_q_values = self.target_network(next_states)
        
        target_q_values = current_q_values.clone()
        
        for i, action_positions in enumerate(actions):
            if action_positions:
                target = rewards[i]
                if not dones[i]:
                    target += self.gamma * torch.max(next_q_values[i])
                
                for r, c in action_positions:
                    target_q_values[i][r * 17 + c] = target
        
        loss = F.mse_loss(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

class AppleGameSolver:
    def __init__(self):
        self.env = AppleGameEnvironment()
        self.agent = DQNAgent(state_shape=(3, 10, 17), action_size=170)
        self.scores = []
    
    def train(self, episodes=1000):
        print("Apple Game ML Solver 훈련 시작...")
        
        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0
            
            while True:
                action = self.agent.act(self.env, state)
                next_state, reward, done, info = self.env.step(action)
                
                self.agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                
                if done:
                    break
            
            self.scores.append(info['score'])
            
            if len(self.agent.memory) > 32:
                self.agent.replay()
            
            self.agent.step_count += 1
            if self.agent.step_count % self.agent.update_target_every == 0:
                self.agent.update_target_network()
            
            if episode % 100 == 0:
                avg_score = np.mean(self.scores[-100:]) if len(self.scores) >= 100 else np.mean(self.scores)
                print(f"Episode {episode}, Average Score: {avg_score:.2f}, Epsilon: {self.agent.epsilon:.3f}")
    
    def solve(self, display_board=True):
        state = self.env.reset()
        
        if display_board:
            print("초기 보드:")
            self.display_board()
        
        moves = []
        
        while True:
            action = self.agent.act(self.env, state)
            
            if not action:
                break
            
            next_state, reward, done, info = self.env.step(action)
            moves.append((action, reward))
            
            if display_board:
                print(f"\n이동: {action}, 보상: {reward}")
                self.display_board()
            
            state = next_state
            
            if done:
                break
        
        print(f"\n최종 점수: {self.env.score}")
        print(f"총 이동 횟수: {len(moves)}")
        return self.env.score, moves
    
    def display_board(self):
        display = self.env.board.copy()
        display[self.env.cleared] = 0
        
        for i in range(self.env.rows):
            row_str = ""
            for j in range(self.env.cols):
                if self.env.cleared[i, j]:
                    row_str += " . "
                else:
                    row_str += f" {display[i, j]} "
            print(row_str)
    
    def plot_training_progress(self):
        if not self.scores:
            print("훈련 데이터가 없습니다.")
            return
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.scores)
        plt.title('Training Scores')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        
        plt.subplot(1, 2, 2)
        window = 100
        if len(self.scores) >= window:
            moving_avg = [np.mean(self.scores[i:i+window]) for i in range(len(self.scores)-window+1)]
            plt.plot(moving_avg)
            plt.title(f'Moving Average ({window} episodes)')
            plt.xlabel('Episode')
            plt.ylabel('Average Score')
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    solver = AppleGameSolver()
    
    print("=== Apple Game ML Solver ===")
    print("1. 훈련 시작")
    solver.train(episodes=500)
    
    print("\n2. 훈련 완료, 게임 해결 시도")
    score, moves = solver.solve(display_board=True)
    
    print("\n3. 훈련 진행상황 시각화")
    solver.plot_training_progress()