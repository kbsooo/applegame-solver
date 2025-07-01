import torch
import numpy as np
import random
import math
from collections import namedtuple

from rl import AppleGameEnv
from agent import DQNAgent, Transition

# --- Hyperparameters ---
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10 # How often to update the target network (in episodes)
NUM_EPISODES = 500 # Total number of episodes to train for
MEMORY_CAPACITY = 10000

def optimize_model(agent):
    """Performs one step of the optimization on the policy network."""
    if len(agent.memory) < BATCH_SIZE:
        return

    transitions = agent.memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # Create batches of states, actions, rewards, etc.
    # We need to handle the format of states and actions carefully.
    state_batch = torch.cat([agent._get_state_tensor_for_action(s, []) for s in batch.state])
    next_state_batch = torch.cat([agent._get_state_tensor_for_action(s, []) for s in batch.next_state if s is not None])
    reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=agent.device)

    # We need to compute the Q-values for the actions that were actually taken.
    # This is tricky because our network evaluates states, not state-action pairs directly.
    # We model Q(s,a) as the value of the state *after* the action is taken.
    
    # 1. Calculate Q(s, a) for the actions we took
    # The Q-value of the state s and action a is the output of the policy_net
    # for the state that results from taking action a in state s.
    q_values = []
    for i in range(BATCH_SIZE):
        state = batch.state[i]
        action = batch.action[i]
        state_after_action = agent._get_state_tensor_for_action(state, action)
        q_values.append(agent.policy_net(state_after_action))
    state_action_values = torch.cat(q_values)

    # 2. Calculate V(s_{t+1}) for all next states.
    # This is the maximum Q-value for the next state, predicted by the target network.
    next_state_values = torch.zeros(BATCH_SIZE, device=agent.device)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=agent.device, dtype=torch.bool)
    
    # For non-final next states, calculate the max Q-value among all possible actions
    non_final_next_states = [s for s in batch.next_state if s is not None]
    if len(non_final_next_states) > 0:
        max_q_next = []
        for next_s in non_final_next_states:
            possible_actions = env.get_possible_actions() # This should be based on next_s
            if not possible_actions:
                max_q_next.append(torch.tensor([[0.0]], device=agent.device))
                continue
            
            q_vals = [agent.target_net(agent._get_state_tensor_for_action(next_s, a)) for a in possible_actions]
            max_q_next.append(torch.max(torch.cat(q_vals)))

        next_state_values[non_final_mask] = torch.tensor(max_q_next, device=agent.device)

    # 3. Compute the expected Q values (Bellman equation)
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # 4. Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # 5. Optimize the model
    agent.optimizer.zero_grad()
    loss.backward()
    for param in agent.policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    agent.optimizer.step()

# --- Main Training Loop ---
if __name__ == "__main__":
    env = AppleGameEnv()
    agent = DQNAgent(board_shape=env.board_shape, memory_capacity=MEMORY_CAPACITY, batch_size=BATCH_SIZE, gamma=GAMMA)
    
    # Monkey-patch the agent's optimize_model function
    # This is a simple way to connect the training script's logic to the agent class
    DQNAgent.optimize_model = optimize_model

    episode_scores = []

    for i_episode in range(NUM_EPISODES):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            # Determine epsilon for epsilon-greedy action selection
            epsilon = EPS_END + (EPS_START - EPS_END) * \
                      math.exp(-1. * i_episode / EPS_DECAY)

            possible_actions = env.get_possible_actions()
            if not possible_actions:
                break

            # Select and perform an action
            action = agent.select_action(state, possible_actions, epsilon)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            # Store the transition in memory
            # If the game is done, the next_state is None
            agent.memory.push(state, action, next_state if not done else None, reward)

            state = next_state

            # Perform one step of the optimization (on the policy network)
            agent.optimize_model(agent)

        episode_scores.append(total_reward)
        print(f"Episode {i_episode}: Total Score = {total_reward}")

        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            agent.update_target_net()

    print("\n--- Training Complete ---")
    # Here you could add code to save the trained model and plot the scores
    # torch.save(agent.policy_net.state_dict(), 'apple_solver_dqn.pth')
