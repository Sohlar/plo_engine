import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda
import random
from collections import deque
from metrics import q_value, epsilon, action_taken, bet_size_metric


class PLONetwork(nn.Module):
    def __init__(self, state_size):
        super(PLONetwork, self).__init_()
        self.share = nn.Sequential(
            nn.Linear(state_size, 512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
        )

        self.policy_head = nn.Sequential(
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,4),
            nn.Softmax(dim=1)
        )

        self.value_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        shared_features = self.shared(x)
        policy = self.policy_head(shared_features)
        value = self.value_head(shared_features)
        return policy, value


class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = {} # Maps actions to child nodes
        self.visit_count = 0
        self.value_sum = 0
        self.prior_policy = None # Probability from policy network for this action

    def select_child(self):
        best_score = float('-inf')
        best_child = None

        for action, child in self.children.items():
            score = child.get_puct_score()
            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def get_puct_score(self):
        if self.visit_count == 0:
            return float('inf') # Unvisited nodes get prio
        
        # Exploitation: Average value from sims
        q_value = self.value_sum / self.visit_count
        # Higher value = better pre sim results

        # Exploration: Based on prior poliocy and visit counts
        exploration = (
            2.0 *                               # Exploration Constanct
            self.prior_policy *                 # Network confidence in action
            math.sqrt(self.parent.visit_count)  # Parent visits
            / (1 + self.visit_count)            # Reduced exploration for visited nodes
        )
        # Higher when:
        # - Network thinks action is promising
        # - Parent node heavily visited
        # - Current node rarely visited
        return q_value + exploration


# fmt: off
class DQNAgent:
    def __init__(self, state_size, action_size):
        """
        Initialize the DQN Agent.

        Args:
            state_size (int): The size of the state space.
            action_size (int): The number of possible actions.
        """
        self.name = None
        self.state_size = state_size  # Dimension of poker game state (cards, pot, etc.)
        self.action_size = action_size  # Number of possible actions (fold, call, raise)
        self.memory = deque(maxlen=10000)  # Experience replay buffer, crucial for stable learning in poker
        self.batch_size = 128
        self.gamma = 0.95    # Discount rate for future rewards, important for long-term strategy
        self.epsilon = 1.0   # Start with 100% exploration to learn diverse poker situations
        self.epsilon_min = 0.01  # Minimum exploration to always adapt to opponent's strategy
        self.epsilon_decay = 0.995  # Gradually reduce exploration to exploit learned poker knowledge
        self.learning_rate = 0.001  # Small learning rate for stable improvement of poker strategy
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU acceleration for faster learning

        # Initialize network
        self.model = DQN(state_size, action_size).to(self.device)  # Main network for action selection
        self.target_model = DQN(state_size, action_size).to(self.device)  # Target network for stable Q-learning
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)  # Adam optimizer works well for poker's noisy rewards

        # MCTS parameters
        self.mcts_simulations = 100 # Number of searches
        self.min_bet = 2


    def remember(self, state, action, reward, next_state, done):
        """
        Store a transition in the replay memory.

        Args:
            state: The current state.
            action: The action taken.
            reward: The reward received.
            next_state: The resulting state.
            done: Whether the episode has ended.
        """
        state = torch.FloatTensor(state).to(self.device) if not isinstance(state, torch.Tensor) else state.to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device) if not isinstance(next_state, torch.Tensor) else next_state.to(self.device)
        action = torch.LongTensor([action]).to(self.device)
        reward = torch.FloatTensor([reward]).to(self.device)
        done = torch.FloatTensor([done]).to(self.device)

        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, valid_actions, max_bet, min_bet):
        street = self._get_street(state)

        if street <= 1:
            return self._get_nash_action(state, valid_actions, max_bet, min_bet)
        else:
            return self._get_mcts_action(state, valid_actions, max_bet, min_bet)
        
    def _get_nash_action(self, state, valid_actions, max_bet, min_bet):
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            policy, value = self.model(state_tensor)

        # Convert policy to action probabilities
        action_map = {"fold": 0, "check": 1, "call": 2, "bet": 3}
        valid_action_indeces = [action_map[action] for action in valid_actions]

        # Filter and normalize probabilities for valid actions
        valid_probs = policy[0][valid_action_indeces]
        valid_probs = valid_probs / valid_probs.sum()

        # Record metrics
        q_value.labels(player='oop' if self.name == 'OOP' else 'ip').set(value.item())
        epsilon.labels(player='oop' if self.name == 'OOP' else 'ip').set(value.item())

        if np.random.rand() <= self.epsilon:
            action_idx = np.random.choice(len(valid_actions))
            action = valid_actions[action_idx]
        else:
            action_idx = valid_probs.argmax().item()
            action = valid_actions[action_idx]
        
        if action == "bet":
            bet_size = self._get_bet_size(value.item(), valid_probs[action_idx], max_bet, min_bet)
            return action, bet_size

        return action
    
    def _get_mcts_action(self, state, valid_actions, max_bet, min_bet):
        
        root = MCTSNode(state)

        for _ in range(self.mcts_simulations):
            node = root
            search_path = [node]

            # Selection
            while node.children and not self._is_terminal(node.state):
                node = node.select_child()
                search_path.append(node)

            # Expansion
            if not self._is_terminal(node.state):
                state_tensor = torch.FloatTensor(node.state).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    policy, value = self.model(state_tensor)
                node.expand(policy[0])
            
            # Backprop
            if self._is_terminal(node.state):
                value = self._get_terminal_value(node.state)
            else:
                _, value = self.model(torch.FloatTensor(node.state).unsqueeze(0).to(self.device))
                value = value.item()
            
            for node in reversed(search_path):
                node.visit_count += 1
                node.value_sum += value
        
        best_action = max(root.children.items(), key=lambda x: x[1].visit_count) [0]

        if best_action == "bet":
            bet_size = self._get_bet_size(
                root.children[best_action].value_sum / root.children[best_action].visit_count,
                root.children[best_action].prior_policy,
                max_bet,
                min_bet
            )
            return best_action, bet_size
        
        return best_action

    def _get_bet_size(self, value, action_prob, max_bet, min_bet):
        """Calculate bet size based on hand strength and action probability"""
        # Scale bet size based on value and action probability
        sizing_factor = (value+1) / 2 # Convert from [-1,1] to [0,1]
        confidence = action_prob

        # Combine factors for final sizing
        bet_size = min_bet + (max_bet-min_bet) * sizing_factor * confidence
        bet_size = max(min_bet, min(max_bet, round(bet_size, 0)))

        # Record metric
        bet_size_metric.labels(
            player='oop' if self.name == 'OOP' else 'ip',
            street="agent_decision"
        ).observe(bet_size)

        return bet_size


    def replay(self, batch_size):
        """
        Train the model using experiences from the replay memory.

        Args:
            batch_size (int): The number of samples to use for training.
        """

        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)  
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.stack(states)
        actions = torch.cat(actions)
        rewards = torch.cat(rewards)
        next_states = torch.stack(next_states)
        dones = torch.cat(dones)

        # Get current predictions
        current_policy, current_value = self.model(states)
        
        # Get next state predicitions from target network
        with torch.no_grad():
            next_policy, next_value = self.target_model(next_states)

        # Calculate targets
        value_target = rewards + self.gamma * next_value * (1-dones)
        policy_target = current_policy.clone()

        # Update policy target based on advantages
        advantages = rewards + self.gamma * next_value * (1-dones) - current_value
        for i in range(batch_size):
            policy_target[i, actions[i]] += advantages[i]

        # Calculate losses
        value_loss = nn.MSELoss()(current_value, value_target)
        policy_loss = nn.CrossEntropyLoss()(current_policy, policy_target)

        # Combined loss
        loss = value_loss + policy_loss

        # Optimize
        self.optimizer.zer_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsioln
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()


    def update_target_model(self):
        """
        Update the target model with the weights of the main model.
        """
        # Periodically update target network to stabilize training
        self.target_model.load_state_dict(self.model.state_dict())

    def _get_street(self, state):
        pass
    
    def _is_terminal(self, state):
        pass
    
    def _get_terminal_value(self, state):
        pass
