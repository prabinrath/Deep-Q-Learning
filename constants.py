class constants:
    def __init__(self, env):
        if env == "PongDeterministic-v4":
            self.GAMMA = 0.99 # Discount factor
            self.POLICY_UPDATE_INTERVAL = 4 # Interval for policy update
            self.TARGET_UPDATE_INTERVAL = 1000 # Interval for target update
            self.LR = 0.00025 # Adam learning rate
            self.EPSILON_START = 1 # Annealing start
            self.EPSILON_END = 0.1 # Annealing end
            self.EXPLORATION_FRAMES = 100000 # Annealing frames
            self.BATCH_SIZE = 32 # Sampling size from memory
            self.MEMORY_BUFFER = 50000 # Replay buffer size
            self.EPISODES = 10000 # Number of episodes for training
            self.VALIDATE_FREQ = 100 # Episodes
            self.max_possible_reward = 20
            self.reward_increment = self.max_possible_reward/10
            self.max_valid_reward = -21
        elif env == "BreakoutDeterministic-v4":
            self.GAMMA = 0.99 # Discount factor
            self.POLICY_UPDATE_INTERVAL = 4 # Interval for policy update
            self.TARGET_UPDATE_INTERVAL = 10000 # Interval for target update
            self.LR = 0.00001 # Adam learning rate
            self.EPSILON_START = 1 # Annealing start
            self.EPSILON_END = 0.1 # Annealing end
            self.EXPLORATION_FRAMES = 1000000 # Annealing frames
            self.BATCH_SIZE = 32 # Sampling size from memory
            self.MEMORY_BUFFER = 1000000 # Replay buffer size
            self.EPISODES = 50000 # Number of episodes for training
            self.VALIDATE_FREQ = 100 # Episodes
            self.max_possible_reward = 350
            self.reward_increment = self.max_possible_reward/50
            self.max_valid_reward = 0