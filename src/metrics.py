from prometheus_client import Gauge, Counter, Histogram
import psutil

# AI Performance Metrics
episode_reward = Gauge('episode_reward', 'Reward for the current episode', ['player'])
cumulative_reward = Gauge('cumulative_reward', 'Total reward across all episodes', ['player'])
q_value = Gauge('q_value', 'Current Q-value', ['player'])
loss = Gauge('loss', 'Current loss value', ['player'])
epsilon = Gauge('epsilon', 'Current epsilon value', ['player'])
winrate = Gauge('winrate', 'Current BB/100', ['player'])

# Game State Metrics
pot_size = Gauge('pot_size', 'Current pot size')
player_chips = Gauge('player_chips', 'Current player chips', ['player'])
community_cards = Gauge('community_cards', 'Number of community cards')

# Action Metrics
action_taken = Counter('action_taken', 'Number of times each action was taken', ['player_action'])
bet_size_pct = Histogram('bet_size_pct', 'Bet sizes as percentage of pot', ['player'], buckets=(10, 20, 30, 40, 50, 60, 70, 80, 90, 100))
bet_size_metric = Histogram('bet_size', 'Bet sizes in chips', ['player', 'street'], buckets=(2, 5, 10, 20, 50, 100, 200, 500, 1000))


# System Resource Metrics
cpu_usage = Gauge('cpu_usage', 'Current CPU usage percentage')
memory_usage = Gauge('memory_usage', 'Current memory usage percentage')
gpu_usage = Gauge('gpu_usage', 'Current GPU usage percentage')

# Training Progress Metrics
episodes_completed = Counter('episodes_completed', 'Number of completed episodes')
training_time = Histogram('training_time', 'Time taken for training')

def update_system_metrics():
    cpu_usage.set(psutil.cpu_percent())
    memory_usage.set(psutil.virtual_memory().percent)
    # Note: GPU usage requires additional setup with libraries like pynvml
    # For simplicity, we'll leave it at 0 for now
    gpu_usage.set(0)


def update_bet_size(player, bet_amount, pot_size):
    bet_pct = (bet_amount / pot_size) * 100
    bet_size_pct.labels(player=player).observe(bet_pct)
