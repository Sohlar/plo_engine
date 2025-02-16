import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda
import random
from collections import deque
from metrics import q_value, epsilon, action_taken, bet_size_metric
from constants import ACTION_MAP

# Add near top of file, after imports
# action_to_index = {
#     'fold': 0,
#     'check': 1,
#     'call': 2,
#     'bet': 3
# }

# Use ACTION_MAP instead of action_to_index
index_to_action = {v: k for k, v in ACTION_MAP.items()}
torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.capture_scalar_outputs = True

class PLONetwork(nn.Module):
    def __init__(self, state_size):
        super(PLONetwork, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )

        self.policy_head = nn.Sequential(
            nn.Linear(512, 256), 
            nn.ReLU(), 
            nn.Linear(256, 4)  # Remove Softmax from Sequential
        )

        self.value_head = nn.Sequential(
            nn.Linear(512, 256), 
            nn.ReLU(), 
            nn.Linear(256, 1), 
            nn.Tanh()
        )

    @torch.compile
    def forward(self, x):
        shared_features = self.shared(x)
        policy_logits = self.policy_head(shared_features)
        # Apply softmax with proper reshaping
        policy = torch.softmax(policy_logits, dim=-1)  # Use -1 for last dimension
        value = self.value_head(shared_features)
        return policy, value


class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}  # Maps actions to child nodes
        self.visit_count = 0
        self.value_sum = 0
        self.prior_policy = None  # Probability from policy network for this action
        self.game = None  # Reference to game instance for valid actions/simulation

    def get_valid_actions(self, state):
        """Get valid actions for the current state"""
        # For preflop
        if len(state[7:17]) // 2 == 0:  # No community cards
            if state[3] == 0:  # No current bet
                return ["check", "bet"]
            else:
                return ["fold", "call", "bet"]
        # For postflop
        else:
            if state[3] == 0:  # No current bet
                return ["check", "bet"]
            else:
                return ["fold", "call", "bet"]

    def simulate_action(self, state, action):
        """Simulate taking an action"""
        # Convert tensor to numpy, copy, then back to tensor
        if torch.is_tensor(state):
            new_state = state.clone().detach()
        else:
            new_state = state.copy()
        
        # Update pot and chips based on action
        if action == "fold":
            new_state[0] = 0  # Clear pot
        elif action == "call":
            call_amount = new_state[2]  # Current bet
            new_state[0] += call_amount  # Add to pot
        elif action == "bet":
            bet_amount = min(new_state[3], new_state[0])  # Min of chips or pot
            new_state[0] += bet_amount  # Add to pot
            new_state[2] = bet_amount  # Set current bet
            
        return new_state

    def select_child(self):
        best_score = float("-inf")
        best_child = None

        for action, child in self.children.items():
            score = child.get_puct_score()
            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def get_puct_score(self):
        if self.visit_count == 0:
            return float("inf")

        q_value = self.value_sum / self.visit_count
        exploration = (
            2.0
            * self.prior_policy
            * math.sqrt(self.parent.visit_count)
            / (1 + self.visit_count)
        )
        return q_value + exploration

    def expand(self, policy_probs):
        """Expands node with children for all valid actions"""
        valid_actions = self.get_valid_actions(self.state)
        
        for action in valid_actions:
            new_state = self.simulate_action(self.state, action)
            child = MCTSNode(new_state, parent=self)
            child.prior_policy = policy_probs[ACTION_MAP[action]]
            self.children[action] = child


# fmt: off
class DQNAgent:
    def __init__(self, state_size, action_size, batch_size):
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
        self.batch_size = batch_size
        self.gamma = 0.95    # Discount rate for future rewards, important for long-term strategy
        self.epsilon = 1.0   # Start with 100% exploration to learn diverse poker situations
        self.epsilon_min = 0.01  # Minimum exploration to always adapt to opponent's strategy
        self.epsilon_decay = 0.995  # Gradually reduce exploration to exploit learned poker knowledge
        self.learning_rate = 0.001  # Small learning rate for stable improvement of poker strategy
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU acceleration for faster learning

        # Initialize network
        self.model = PLONetwork(state_size).to(self.device)  # Main network for action selection
        self.target_model = PLONetwork(state_size).to(self.device)  # Target network for stable Q-learning
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)  # Adam optimizer works well for poker's noisy rewards

        # MCTS parameters
        self.mcts_simulations = 25  # Reduce from 100 to 25
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
        #print('\n')
        #print(state)
        #print("\n")
        street = self._get_street(state)
        pot_size = state[0]  # First value in state tensor is pot size
        
        # Changed logic: Use Nash for preflop and small pots
        if street == 0 or pot_size <= 50:  # Preflop or small pot
            return self._get_nash_action(state, valid_actions, max_bet, min_bet)
        else:
            return self._get_mcts_action(state, valid_actions, max_bet, min_bet)
        
    def _get_nash_action(self, state, valid_actions, max_bet, min_bet):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            policy, value = self.model(state_tensor)

        # Convert policy to action probabilities
        valid_action_indeces = [ACTION_MAP[action] for action in valid_actions]

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
        
        # Only return bet size if bet is a valid action and was chosen
        if action == "bet" and "bet" in valid_actions:
            bet_size = self._get_bet_size(value.item(), valid_probs[action_idx], max_bet, min_bet)
            return action, bet_size

        # Return tuple with None for bet_size for non-bet actions
        return action, None
    
    def _get_mcts_action(self, state, valid_actions, max_bet, min_bet):
        root = MCTSNode(state)
        max_depth = 3  # Limit tree depth
        
        for _ in range(self.mcts_simulations):
            node = root
            search_path = [node]
            depth = 0
            
            # Add depth check
            while node.children and not self._is_terminal(node.state) and depth < max_depth:
                node = node.select_child()
                search_path.append(node)
                depth += 1

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
        
        # Filter children to only include valid actions
        valid_children = {action: node for action, node in root.children.items() 
                         if action in valid_actions}
        
        if not valid_children:
            # If no valid children (shouldn't happen), pick random valid action
            action = random.choice(valid_actions)
            return action, None
        
        best_action = max(valid_children.items(), key=lambda x: x[1].visit_count)[0]

        if best_action == "bet" and "bet" in valid_actions:
            bet_size = self._get_bet_size(
                root.children[best_action].value_sum / root.children[best_action].visit_count,
                root.children[best_action].prior_policy,
                max_bet,
                min_bet
            )
            return best_action, bet_size
        
        # Return tuple with None for bet_size for non-bet actions
        return best_action, None

    def _get_bet_size(self, value, action_prob, max_bet, min_bet):
        """Calculate bet size based on hand strength and action probability"""
        # Scale bet size based on value and action probability
        sizing_factor = (value+1) / 2  # Convert from [-1,1] to [0,1]
        confidence = action_prob if torch.is_tensor(action_prob) else action_prob  # Convert tensor to float

        # Combine factors for final sizing
        bet_size = min_bet + (max_bet-min_bet) * sizing_factor * confidence
        bet_size = float(bet_size) if torch.is_tensor(bet_size) else bet_size  # Convert to float if tensor
        bet_size = max(min_bet, min(max_bet, round(bet_size, 0)))

        # Record metric
        bet_size_metric.labels(
            player='oop' if self.name == 'OOP' else 'ip',
            street="agent_decision"
        ).observe(bet_size)

        return bet_size


    def replay(self, batch_size):
        """Train the model using experiences from the replay memory."""
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        # Convert to tensors for batch processing
        states = torch.stack(states)
        actions = torch.cat(actions)
        rewards = torch.cat(rewards)
        next_states = torch.stack(next_states)
        dones = torch.cat(dones)

        # Get current predictions
        current_policy, current_value = self.model(states)
        
        # Get next state predictions from target network
        with torch.no_grad():
            next_policy, next_value = self.target_model(next_states)

        # Calculate value target
        value_target = rewards + self.gamma * next_value.squeeze(-1) * (1-dones)
        value_loss = nn.MSELoss()(current_value.squeeze(-1), value_target)

        # Calculate policy target
        advantages = rewards + self.gamma * next_value.squeeze(-1) * (1-dones) - current_value.squeeze(-1)
        policy_target = current_policy.clone()
        
        # Update policy based on advantages
        for i in range(batch_size):
            policy_target[i, actions[i]] = policy_target[i, actions[i]] + advantages[i]

        # Policy loss with cross entropy
        policy_loss = nn.CrossEntropyLoss()(current_policy, policy_target)

        # Combined loss
        loss = value_loss + policy_loss

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
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
        """Determine street based on number of community cards"""
        # Current version tries to access state.street which doesn't exist
        num_cards = len(state[6:16]) // 2  # Extract community cards from state tensor
        if num_cards == 0:
            return 0  # preflop
        elif num_cards == 3:
            return 1  # flop
        elif num_cards == 4:
            return 2  # turn
        else:
            return 3  # river
    
    def _is_terminal(self, state):
        """Check if state is terminal"""
        # For MCTS simulation
        if isinstance(state, dict):
            return state.get('hand_over', False) or state.get('num_active_players', 2) == 1
        return False
    
    def _get_terminal_value(self, state):
        """Get value of terminal state"""
        # For MCTS simulation
        if isinstance(state, dict):
            if state.get('num_active_players', 2) == 1:
                # Win/loss based on if current player won
                return 1.0 if state['current_player'] == self.name else -1.0
            
            # For showdown, calculate based on chips
            player_key = 'oop_player' if self.name == 'OOP' else 'ip_player'
            chips = state[player_key]['chips']
            value = (chips - 200) / 200  # Normalize around starting stack
            return max(min(value, 1.0), -1.0)
        return 0.0
