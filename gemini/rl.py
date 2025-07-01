import numpy as np

class AppleGameEnv:
    """
    A class that implements the Apple Game environment for Reinforcement Learning,
    following a structure similar to OpenAI Gym environments.
    """
    def __init__(self):
        """
        Initializes the game environment.
        """
        self.board_shape = (17, 10)
        # The state is the board itself, represented as a 2D numpy array.
        # 0 represents an empty space.
        self.board = np.zeros(self.board_shape, dtype=int)

    def reset(self):
        """
        Resets the environment to a new game.

        - Creates a new random 17x10 board with numbers from 1 to 9.
        - Returns the initial state (the new board).
        """
        self.board = np.random.randint(1, 10, size=self.board_shape)
        return self.board.copy()

    def step(self, action):
        """
        Executes one time step in the environment based on the given action.

        Args:
            action: A list of coordinates for the apples to be cleared.
                    Example: [(y1, x1), (y2, x2)]

        Returns:
            A tuple (observation, reward, done, info):
            - observation (np.array): The board state after the action.
            - reward (float): The reward from the action.
            - done (bool): True if the game is over, False otherwise.
            - info (dict): Auxiliary diagnostic information.
        """
        # 1. Validate the action
        current_sum = sum(self.board[y, x] for y, x in action)
        if current_sum != 10:
            # Punish invalid moves. The agent should learn not to do this.
            return self.board.copy(), -10.0, True, {"error": "Invalid action sum"}

        # 2. Update the board and calculate reward
        reward = len(action)
        for y, x in action:
            self.board[y, x] = 0

        # 3. Check if the game is done
        possible_actions = self.get_possible_actions()
        done = not possible_actions  # Game is over if no more moves are possible

        # Optional: Add a bonus for clearing the whole board
        if done and np.all(self.board == 0):
            reward += 100 # Bonus for a perfect clear

        return self.board.copy(), float(reward), done, {}

    def get_possible_actions(self):
        """
        Finds all possible combinations of apples that sum to 10.

        Returns:
            A list of all valid actions. An action is a list of coordinates.
            Example: [[(y1, x1), (y2, x2)], [(y3, x3), (y4, x4), (y5, x5)]]
        """
        apples = []
        for y in range(self.board_shape[0]):
            for x in range(self.board_shape[1]):
                if self.board[y, x] > 0:
                    apples.append(((y, x), self.board[y, x]))

        possible_actions = []
        
        # Helper function for recursive search (subset sum problem)
        def find_combos(start_index, current_combo, current_sum):
            if current_sum == 10:
                # Found a valid combination
                possible_actions.append([pos for pos, val in current_combo])
                return # Don't search further down this path

            if current_sum > 10 or start_index == len(apples):
                # Backtrack if sum is over 10 or no more apples left
                return

            # Recursive step:
            for i in range(start_index, len(apples)):
                # Include the apple at index i and search further
                find_combos(i + 1, current_combo + [apples[i]], current_sum + apples[i][1])

        find_combos(0, [], 0)
        return possible_actions

    def render(self, mode='human'):
        """
        Renders the environment.
        """
        if mode == 'human':
            for row in self.board:
                print(' '.join(f'{num:2}' for num in row))

if __name__ == '__main__':
    env = AppleGameEnv()
    state = env.reset()
    done = False
    total_reward = 0
    turn = 0

    print(f"--- Starting a new game ---")
    env.render()

    while not done:
        turn += 1
        print(f"\n--- Turn {turn} ---")

        # Get all possible moves
        possible_actions = env.get_possible_actions()

        if not possible_actions:
            print("No more possible moves. Game Over.")
            done = True
            continue

        # --- Agent's Logic would go here ---
        # For now, we just pick the first possible action as a demo
        action_to_take = possible_actions[0]
        # ------------------------------------

        print(f"Action: Clearing {len(action_to_take)} apples at {action_to_take}")

        # Perform the action
        state, reward, done, info = env.step(action_to_take)
        total_reward += reward

        print(f"Reward: {reward}, Total Reward: {total_reward}")
        print("Board after action:")
        env.render()

    print(f"\n--- Final Score: {total_reward} ---")

