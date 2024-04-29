import gym
import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

# Create the environment from OpenAI's Gym
env = gym.make('CartPole-v1')

# Define the state and action size from the environment
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
batch_size = 32  # Size of the batch used in the replay method

# Set hyperparameters for the learning algorithm
gamma = 0.95    # Discount factor for past rewards
epsilon = 1.0   # Exploration rate: how much to act randomly
epsilon_min = 0.01  # Minimum exploration probability
epsilon_decay = 0.995  # Exponential decay rate for exploration prob
learning_rate = 0.001  # Learning rate for the neural network optimizer

# Neural Network for Deep Q Learning
class DQNNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNNetwork, self).__init__()
        self.dense1 = nn.Linear(state_size, 24)  # First fully-connected layer
        self.relu = nn.ReLU()  # ReLU activation function
        self.dense2 = nn.Linear(24, 24)  # Second fully-connected layer
        self.output = nn.Linear(24, action_size)  # Output layer

    def forward(self, x):
        x = self.relu(self.dense1(x))  # Activation function for the first layer
        x = self.relu(self.dense2(x))  # Activation function for the second layer
        x = self.output(x)  # Output layer
        return x

# Instantiate the model and the optimizer
model = DQNNetwork(state_size, action_size)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Memory to store experiences for replay
memory = deque(maxlen=2000)

# Function to decide the next action
def choose_action(state, epsilon):
    if random.uniform(0, 1) < epsilon:  # Check if we should act randomly
        return random.randrange(action_size)
    else:
        state = torch.FloatTensor(state).unsqueeze(0)  # Convert state to tensor
        with torch.no_grad():  # Disable gradient calculation
            action_values = model(state)  # Get action values from model
        return torch.argmax(action_values).item()  # Choose the best action

# Function to train the model with experiences sampled from memory
def replay():
    global epsilon  # Ensure that epsilon is treated as global
    if len(memory) < batch_size:
        return
    minibatch = random.sample(memory, batch_size)
    for state, action, reward, next_state, done in minibatch:
        state = torch.FloatTensor(state).unsqueeze(0)  # Reshape and convert to tensor
        next_state = torch.FloatTensor(next_state).unsqueeze(0)
        reward = torch.tensor(reward, dtype=torch.float32)
        action = torch.tensor([action], dtype=torch.int64)
        
        # Predict Q-values with the target network for next states
        if done:
            target = reward
        else:
            target = reward + gamma * torch.max(model(next_state).detach())
        
        # Get current Q-value predictions for each state and the taken action
        current = model(state).gather(1, action.unsqueeze(-1)).squeeze(-1)
        
        # Calculate loss
        loss = nn.functional.mse_loss(current, target)
        
        # Optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay


# Main function to train the agent
def train_agent(episodes=1000):
    global epsilon
    for e in range(episodes):
        state = env.reset()  # Reset the environment for a new episode
        state = np.reshape(state, [1, state_size])  # Reshape state
        for time in range(500):  # Start a new episode
            action = choose_action(state, epsilon)  # Select an action
            next_state, reward, done, _ = env.step(action)  # Execute the action
            next_state = np.reshape(next_state, [1, state_size])
            memory.append((state, action, reward, next_state, done))  # Store the experience
            state = next_state
            if done:
                print(f"episode: {e+1}/{episodes}, score: {time}")  # Print the score of the episode
                break
        replay()  # Replay experiences

if __name__ == "__main__":
    train_agent()
