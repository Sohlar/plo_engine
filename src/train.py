import torch
import sys
import os
from ai_trainer import PokerGame, HumanPlayer
from agent2 import DQNAgent, PLONetwork
import time
from logging_config import setup_logging
import logging
import torch.cuda
import datetime
import argparse

from prometheus_client import start_http_server
from metrics import loss as loss_metric, winrate, episode_reward, cumulative_reward, player_chips, pot_size, community_cards, episodes_completed, action_taken, q_value, epsilon, update_system_metrics
from constants import STATE_SIZE, ACTION_SIZE, STARTING_STACK, MINIMUM_BET_INCREMENT

setup_logging()

BATCH_SIZE = 512

def load_model(model_path):
    agent = DQNAgent(STATE_SIZE, ACTION_SIZE, BATCH_SIZE)
    
    try:
        # Load the state dict
        state_dict = torch.load(model_path, weights_only=True)
        
        # Create new PLONetwork model
        agent.model = PLONetwork(STATE_SIZE).to(agent.device)
        agent.model.load_state_dict(state_dict)
        agent.model.eval()
        
    except RuntimeError as e:
        print(f"Error loading model: {e}")
        print("\nModel architecture mismatch. Available options:")
        print("1. Train a new model with current architecture")
        print("2. Use a compatible model")
        sys.exit(1)
        
    return agent

def list_available_models():
    models_dir = "./models"
    models = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
    return models

def train_dqn_poker(game, episodes, batch_size=BATCH_SIZE, train_ip=True, train_oop=True):
    logging.info("Starting DQN training for PLO...")

    # Initialize agents if they don't exist
    if train_oop and game.oop_agent is None:
        game.oop_agent = DQNAgent(STATE_SIZE, ACTION_SIZE)
    if train_ip and game.ip_agent is None:
        game.ip_agent = DQNAgent(STATE_SIZE, ACTION_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if train_oop:
        game.oop_agent.model.to(device)
        game.oop_agent.target_model.to(device)
        game.oop_agent.device = device
    if train_ip:
        game.ip_agent.model.to(device)
        game.ip_agent.target_model.to(device)
        game.ip_agent.device = device

    oop_cumulative_reward = 0
    ip_cumulative_reward = 0


    for e in range(episodes):
        print(f"Hand #{e}")
        # Can view at localhost:9090 {episodes_completed_total} metric
        game_state, oop_reward, ip_reward = game.play_hand()

        oop_cumulative_reward += oop_reward
        ip_cumulative_reward += ip_reward

        # Train every hand
        if train_oop and len(game.oop_agent.memory) > batch_size:
            oop_loss = game.oop_agent.replay(batch_size)
            if oop_loss is not None:
                game.oop_loss = oop_loss
                loss_metric.labels(player='oop').set(oop_loss)
        else:
            game.oop_loss = None

        if train_ip and len(game.ip_agent.memory) > batch_size:
            ip_loss = game.ip_agent.replay(batch_size)
            if ip_loss is not None:
                game.ip_loss = ip_loss
                loss_metric.labels(player='ip').set(ip_loss)
        else:
            game.ip_loss = None

        # Update target models
        if e % 10 == 0:
            if train_oop:
                game.oop_agent.update_target_model()
            if train_ip:
                game.ip_agent.update_target_model()

        # Update metrics
        cumulative_reward.labels(player='oop').set(oop_cumulative_reward)
        cumulative_reward.labels(player='ip').set(ip_cumulative_reward)
        player_chips.labels(player='oop').set(game_state['oop_player']['chips'])
        player_chips.labels(player='ip').set(game_state['ip_player']['chips'])
        pot_size.set(game_state['pot'])
        community_cards.set(len(game_state['community_cards']))

        # Progress Report
        if e % 100 == 0:
            episode_count = max(e/100, 1)
            oop_winrate = oop_cumulative_reward/episode_count
            ip_winrate = ip_cumulative_reward/episode_count
            winrate.labels(player='oop').set(oop_winrate)
            winrate.labels(player='ip').set(ip_winrate)
            logging.info(
                f"Episode: {e}/{episodes}"
            )
            if train_oop and game.oop_loss is not None:
                logging.info(f"OOP Loss: {game.oop_loss:.4f}")
            if train_ip and game.ip_loss is not None:
                logging.info(f"IP Loss: {game.ip_loss:.4f}")

            update_system_metrics()

    print("\nTraining Complete!")

    if train_oop:
        save_model(game.oop_agent, "oop")
    if train_ip:
        save_model(game.ip_agent, "ip")


def main(args):
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()

    while True:
        #print(args)
        if args.interactive == False and args.numhands != 0: choice = "4"
        else:
            print("\nPLO AI Training and Game Menu")
            print("1. Train AI")
            print("2. Play Against AI")
            print("3. Exit")
            print("4. Train AI with no debugging, assumes both positions")
            
            choice = input("\nEnter your choice (1-3): ")

        print(args.interactive)
        print(choice)

        if choice == "1" or choice == "4":

            if choice == "1":
                print("\nTraining Configuration:")
                print("1. Train both positions")
                print("2. Train OOP only")
                print("3. Train IP only")
                print("4. Train against existing model")
                print("5. Train existing model")
                train_choice = input("\nEnter training choice (1-5): ")

            else:
                train_choice = "1"
            
            if args.numhands != 0:
                num_hands = args.numhands
            else:
                num_hands = int(input("Enter number of hands to train: "))

            train_oop = train_choice in ["1", "2", "5"]
            train_ip = train_choice in ["1", "3", "5"]
            
            oop_agent = None
            ip_agent = None

            if train_choice == "4":
                position = input("Which position to train (oop/ip)? ").lower()
                train_oop = position == "oop"
                train_ip = position == "ip"
                
                print("\nAvailable Models:")
                models = list_available_models()
                for i, model in enumerate(models):
                    print(f"{i+1}. {model}")
                    
                model_choice = int(input("\nEnter the number of the model to use: "))
                model_path = f"./models/{models[model_choice-1]}"
                
                if not train_oop:
                    oop_agent = load_model(model_path)
                    oop_agent.model.eval()
                if not train_ip:
                    ip_agent = load_model(model_path)
                    ip_agent.model.eval()

            if train_choice == "5":
                print("\nAvailable Models:")
                models = list_available_models()
                for i, model in enumerate(models):
                    print(f"{i+1}. {model}")
                    
                ip_model_choice = int(input("\nEnter the number of the IP model to use: "))
                oop_model_choice = int(input("\nEnter the number of the OOP model to use: "))
                ip_model_path = f"./models/{models[ip_model_choice-1]}"
                oop_model_path = f"./models/{models[oop_model_choice-1]}"

                oop_agent = load_model(oop_model_path)
                ip_agent = load_model(ip_model_path)

            start_time = time.time()
            
            game = PokerGame(
                oop_agent=oop_agent,
                ip_agent=ip_agent,
                state_size=STATE_SIZE,
                batch_size=BATCH_SIZE
            )
            train_dqn_poker(game, num_hands, batch_size=BATCH_SIZE, train_ip=train_ip, train_oop=train_oop)
            end_time = time.time()
            print(f"Total Training Time: {end_time - start_time:.2f} seconds")
            if args.interactive == False:
                return(True)
                #sys.exit(0)

        elif choice == "2":
            position = input("Enter your position (oop/ip): ").lower()
            while position not in ['oop', 'ip']:
                position = input("Invalid position. Please enter 'oop' or 'ip': ").lower()

            print("\nAvailable Models:")
            models = list_available_models()
            for i, model in enumerate(models):
                print(f"{i+1}. {model}")

            model_choice = int(input("\nEnter the number of the model: "))
            chosen_model = f"./models/{models[model_choice-1]}"

            ai_agent = load_model(chosen_model)
            game = PokerGame(
                human_position=position,
                oop_agent=ai_agent if position == 'ip' else None,
                ip_agent=ai_agent if position == 'oop' else None,
                state_size=STATE_SIZE
            )

            play_against_ai(game,args)

        elif choice == "3":
            print("Exiting...")
            sys.exit(0)
        
        else:
            #print("Invalid choice. Please try again.")
            print("Invalid choice closing...")
            sys.exit(0)

def play_against_ai(game,args):
    while True:
        game_state, oop_reward, ip_reward = game.play_hand()  # Unpack all three values
        print("\nFinal Chip Counts:")
        print(f"OOP Player chips: {game_state['oop_player']['chips']}")
        print(f"IP Player chips: {game_state['ip_player']['chips']}")
        
        # Optionally show rewards
        print(f"\nHand Results:")
        print(f"OOP Reward: {oop_reward}")
        print(f"IP Reward: {ip_reward}")
         
        play_again = input("\nDo you want to play again (y/n)? ")
        if play_again.lower() != 'y':
            break
        
def save_model(agent, position):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Determine model type
    model_type = "plo" if isinstance(agent.model, PLONetwork) else "dqn"
    
    # Create models directory if it doesn't exist
    os.makedirs("./models", exist_ok=True)
    
    # Save with architecture type in filename
    filename = f"./models/{position}_{model_type}_model_{timestamp}.pth"
    torch.save(agent.model.state_dict(), filename)
    print(f"Saved {model_type.upper()} {position} model: {filename}")

if __name__ == "__main__":
    start_http_server(8000)

    parser = argparse.ArgumentParser(description="PLO Training Simulator",prog="python3 train.py")
    parser.add_argument("-i","--interactive", help="Set to 1 for interactive mode, must set hands if this flag used",action="store_false")
    parser.add_argument("-n","--numhands", help="Set number of hands to run", type=int, default=0)
    args = parser.parse_args()
    main(args)

