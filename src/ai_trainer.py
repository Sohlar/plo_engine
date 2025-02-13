import secrets
import logging
import sys
import numpy as np
from agent2 import DQNAgent, PLONetwork
from phevaluator import evaluate_omaha_cards
from logging_config import setup_logging
import torch
from metrics import (
    episode_reward,
    cumulative_reward,
    player_chips,
    pot_size,
    community_cards,
    episodes_completed,
    action_taken,
    q_value,
    epsilon,
    loss,
    bet_size_metric,
    update_bet_size,
)
from constants import (
    STARTING_STACK, MINIMUM_BET_INCREMENT, SMALL_BLIND, 
    BIG_BLIND, INITIAL_POT, ACTION_MAP
)

setup_logging()

# Add near top of file
action_to_index = {
    'fold': 0,
    'check': 1,
    'call': 2,
    'bet': 3
}

index_to_action = {v: k for k, v in action_to_index.items()}

def print_if_human(*args, **kwargs):
    """Only print if there's a human player in the game"""
    if hasattr(print_if_human, 'has_human') and print_if_human.has_human:
        print(*args, **kwargs)

#### Player Class ####
class Player:
    players = []

    def __init__(self, name: str, chips: int) -> None:
        Player.players.append(self)
        self.name: str = name
        self.chips: int = chips
        self.hand = []

    @classmethod
    def get_players(cls):
        return cls.players

    def reset_hands():
        [player.hand.clear() for player in Player.get_players()]

    def reset_chips():
        for player in Player.get_players():
            player.chips = STARTING_STACK


class HumanPlayer(Player):
    def get_action(self, valid_actions, max_bet, min_bet):
        while True:
            action = input(f"Choose an action {valid_actions}: ")
            if action in valid_actions:
                if action == "bet":
                    while True:
                        try:
                            bet_size = int(input(f"Enter the bet size (min {min_bet}, max {max_bet}): "))
                            if bet_size <= max_bet and bet_size > 0:
                                return action, bet_size
                            print(f"Invalid bet size. Please enter a value between 1 and {max_bet}")
                        except ValueError:
                            print("Please enter a valid number")
                return action
            print("Invalid action. Please try again.")


class Deck:
    def __init__(self):
        # fmt: off
        self.cards = [
            "2c", "3c", "4c", "5c", "6c", "7c", "8c", "9c", "Tc", "Jc", "Qc", "Kc", "Ac", 
            "2d", "3d", "4d", "5d", "6d", "7d", "8d", "9d", "Td", "Jd", "Qd", "Kd", "Ad", 
            "2h", "3h", "4h", "5h", "6h", "7h", "8h", "9h", "Th", "Jh", "Qh", "Kh", "Ah", 
            "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s", "Ts", "Js", "Qs", "Ks", "As",
        ]
        # fmt: on

    def shuffle(self):
        secrets.SystemRandom().shuffle(self.cards)


class PokerGame:
    def __init__( self, human_position=None, oop_agent=None, ip_agent=None, state_size=None, batch_size=None):
        self.state_size = state_size or (
            7 + (5 * 2) + 2 * 4 * 2
        )  # Use provided state_size or calculate default
        self.action_size = 4

        self.human_position = human_position
        
        # Initialize device first
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize agents
        if oop_agent is None:
            self.oop_agent = DQNAgent(self.state_size, self.action_size, batch_size=batch_size)
        else:
            self.oop_agent = oop_agent

        if ip_agent is None:
            self.ip_agent = DQNAgent(self.state_size, self.action_size, batch_size=batch_size)
        else:
            self.ip_agent = ip_agent

        # Set agent names
        if self.oop_agent:
            self.oop_agent.name = "OOP"
        if self.ip_agent:
            self.ip_agent.name = "IP"

        # Initialize rest of game state
        self.deck = Deck()

        if human_position == "oop":
            self.oop_player = HumanPlayer(name="OOP", chips=STARTING_STACK)
            self.ip_player = Player(name="IP", chips=STARTING_STACK)
        elif human_position == "ip":
            self.oop_player = Player(name="OOP", chips=STARTING_STACK)
            self.ip_player = HumanPlayer(name="IP", chips=STARTING_STACK)
        else:
            self.oop_player = Player(name="OOP", chips=STARTING_STACK)
            self.ip_player = Player(name="IP", chips=STARTING_STACK)

        self._initialize_game_state()

        self.oop_loss = None
        self.ip_loss = None
        self.oop_cumulative_reward = 0
        self.ip_cumulative_reward = 0

        logging.info("PokerGame initialized")

        # Set has_human flag based on human_position
        print_if_human.has_human = human_position is not None

    def _calculate_state_size(self):
        num_float_values = 7  # pot, current_bet, length of community_cards == street, chips * 2, and committed * 2
        num_community_cards = 5
        num_hand_cards = 4
        num_players = 2
        card_encoding_size = 2

        state_size = (
            num_float_values
            + (num_community_cards * card_encoding_size)
            + (num_hand_cards * num_players * card_encoding_size)
        )

    def _get_state_representation(self, state=None, current_player=None):
        # Convert the game state to a numerical representation for the DQN
        if state is None:
            state = self._get_game_state()
        representation = [
            float(state["pot"]),
            float(len(state["community_cards"])),
            float(state["previous_bet"]),
            float(state["current_bet"]),
            float(state["oop_player"]["chips"]),
            float(state["ip_player"]["chips"]),
            float(state["oop_player"]["committed"]),
            float(state["ip_player"]["committed"]),
        ]
        # Add encoded representations of community cards and player hands
        community_cards = state["community_cards"] + [""] * (
            5 - len(state["community_cards"])
        )
        for card in community_cards:
            representation.extend(self._encode_card(card))

        if current_player == self.oop_player:
            for card in state["oop_player"]["hand"]:
                representation.extend(self._encode_card(card))
            representation.extend([0, 0] * 4)
        else:
            for card in state["ip_player"]["hand"]:
                representation.extend(self._encode_card(card))
            representation.extend([0, 0] * 4)

        assert (
            len(representation) == self.state_size
        ), f"State size mismatch: expected { self.state_size }"

        return torch.FloatTensor(representation)

    def _encode_card(self, card):
        if not card:
            return [0, 0]
        # Simple encoding: rank (2-14) and suit (0-3)
        ranks = "23456789TJQKA"
        suits = "cdhs"
        return (ranks.index(card[0]) + 2, suits.index(card[1]))

    def _initialize_game_state(self):
        self.community_cards = []
        self.pot = INITIAL_POT
        self.hand_over = False
        self.previous_bet = 0
        self.current_bet = SMALL_BLIND
        self.num_actions = 0
        self.last_action = "bet"
        self.current_player = self.ip_player
        self.num_active_players = 2
        self.ip_committed = 1
        self.oop_committed = BIG_BLIND
        self.street = "preflop"
        self.waiting_for_action = False
        self.is_allin = False

    def _deal_cards(self):
        for player in [self.oop_player, self.ip_player]:
            player.hand = [self.deck.cards.pop() for _ in range(4)]

    def start_new_hand(self):
        self._initialize_game_state()
        self._reset_hands()
        self.deck.shuffle()
        self.pot = INITIAL_POT
        self.oop_player.chips = STARTING_STACK - BIG_BLIND
        self.ip_player.chips = STARTING_STACK - SMALL_BLIND
        self._deal_cards()
        return self._get_game_state()

    def play_hand(self):
        logging.info("Starting new hand")
        game_state = self.start_new_hand()

        print_if_human(f"Your hand: {self.get_player_hand()}")

        oop_experiences = []
        ip_experiences = []

        print_if_human(r"""
 ________  ________  _______   ________ ___       ________  ________ 
|\   __  \|\   __  \|\  ___ \ |\  _____\\  \     |\   __  \|\   __  \
\ \  \|\  \ \  \|\  \ \   __/|\ \  \__/\ \  \    \ \  \|\  \ \  \|\  \
 \ \   ____\ \   _  _\ \  \_|/_\ \   __\\ \  \    \ \  \\\  \ \   ____\
  \ \  \___|\ \  \\  \\ \  \_|\ \ \  \_| \ \  \____\ \  \\\  \ \  \___|
   \ \__\    \ \__\\ _\\ \_______\ \__\   \ \_______\ \_______\ \__\
    \|__|     \|__|\|__|\|_______|\|__|    \|_______|\|_______|\|__|""")


        game_state, round_experiences = self.preflop_betting()
        if game_state["is_allin"]:
            self._deal_community_cards(5)
            game_state = self.determine_showdown_winner()
            game_state["hand_over"] = True

        oop_experiences.extend(round_experiences[0])
        ip_experiences.extend(round_experiences[1])

        if not game_state["hand_over"]:
            print_if_human(r"""
 ________ ___       ________  ________   
|\  _____\\  \     |\   __  \|\   __  \  
\ \  \__/\ \  \    \ \  \|\  \ \  \|\  \ 
 \ \   __\\ \  \    \ \  \\\  \ \   ____\
  \ \  \_| \ \  \____\ \  \\\  \ \  \___|
   \ \__\   \ \_______\ \_______\ \__\   
    \|__|    \|_______|\|_______|\|__|   
            """)
            print_if_human("\nDEALING FLOP\n")
            game_state, round_experiences = self.deal_flop()
            oop_experiences.extend(round_experiences[0])

            if game_state["is_allin"]:
                self._deal_community_cards(2)
                game_state = self.determine_showdown_winner()
                game_state["hand_over"] = True

            if not game_state["hand_over"]:
                print_if_human(r"""
 _________  ___  ___  ________  ________      
|\___   ___\\  \|\  \|\   __  \|\   ___  \    
\|___ \  \_\ \  \\\  \ \  \|\  \ \  \\ \  \   
     \ \  \ \ \  \\\  \ \   _  _\ \  \\ \  \  
      \ \  \ \ \  \\\  \ \  \\  \\ \  \\ \  \ 
       \ \__\ \ \_______\ \__\\ _\\ \__\\ \__\
        \|__|  \|_______|\|__|\|__|\|__| \|__|
                                              
                """)
                print_if_human("\nDEALING TURN\n")
                game_state, round_experiences = self.deal_turn()
                oop_experiences.extend(round_experiences[0])

                if game_state["is_allin"]:
                    self._deal_community_cards(1)
                    game_state = self.determine_showdown_winner()
                    game_state["hand_over"] = True

                if not game_state["hand_over"]:
                    print_if_human(r"""
 ________  ___  ___      ___ _______   ________     
|\   __  \|\  \|\  \    /  /|\  ___ \ |\   __  \    
\ \  \|\  \ \  \ \  \  /  / | \   __/|\ \  \|\  \   
 \ \   _  _\ \  \ \  \/  / / \ \  \_|/_\ \   _  _\  
  \ \  \\  \\ \  \ \    / /   \ \  \_|\ \ \  \\  \| 
   \ \__\\ _\\ \__\ \__/ /     \ \_______\ \__\\ _\ 
    \|__|\|__|\|__|\|__|/       \|_______|\|__|\|__|
                    """)
                    print_if_human("\nDEALING RIVER\n")
                    game_state, round_experiences = self.deal_river()
                    oop_experiences.extend(round_experiences[0])

                    if not game_state["hand_over"]:
                        game_state = self.determine_showdown_winner()

        final_state = self._get_state_representation()
        oop_reward, ip_reward = self.calculate_rewards(game_state)
        logging.info(f"OOP Reward: {oop_reward}")
        logging.info(f"IP Reward: {ip_reward}")

        for state, action, valid_actions, bet_size, max_bet in oop_experiences:
            self.oop_agent.remember(state, action, oop_reward, final_state, True)
        for state, action, valid_actions, bet_size, max_bet in ip_experiences:
            self.ip_agent.remember(state, action, ip_reward, final_state, True)

        loss.labels(player="oop").set(self.oop_loss if self.oop_loss is not None else 0)
        loss.labels(player="ip").set(self.ip_loss if self.ip_loss is not None else 0)

        episode_reward.labels(player="oop").set(oop_reward)
        episode_reward.labels(player="ip").set(ip_reward)

        player_chips.labels(player="oop").set(self.oop_player.chips)
        player_chips.labels(player="ip").set(self.ip_player.chips)
        pot_size.set(self.pot)
        community_cards.set(len(self.community_cards))
        episodes_completed.inc()

        return game_state, oop_reward, ip_reward

    def get_player_hand(self):
        if isinstance(self.oop_player, HumanPlayer):
            return self.oop_player.hand
        elif isinstance(self.ip_player, HumanPlayer):
            return self.ip_player.hand
        else:
            return None

    def determine_showdown_winner(self):
        logging.info("Determining Showdown Winner")
        print_if_human(f"\nCommunity cards: {self.community_cards}")
        print_if_human(f"\nIP Tables: {self.ip_player.hand}")
        print_if_human(f"\nOOP Tables: {self.oop_player.hand}\n")
        updated_state = self._get_game_state()
        # fmt: off
        ip_rank = evaluate_omaha_cards(
            self.community_cards[0], self.community_cards[1], self.community_cards[2], self.community_cards[3], self.community_cards[4],
            self.ip_player.hand[0], self.ip_player.hand[1], self.ip_player.hand[2], self.ip_player.hand[3],
        )
        oop_rank = evaluate_omaha_cards(
            self.community_cards[0], self.community_cards[1], self.community_cards[2], self.community_cards[3], self.community_cards[4],
            self.oop_player.hand[0], self.oop_player.hand[1], self.oop_player.hand[2], self.oop_player.hand[3],
        )
        # fmt: on

        if ip_rank < oop_rank:
            updated_state["ip_player"]["chips"] += self.pot
        elif oop_rank < ip_rank:
            updated_state["oop_player"]["chips"] += self.pot
        else:
            updated_state["ip_player"]["chips"] += self.pot / 2
            updated_state["oop_player"]["chips"] += self.pot / 2


        updated_state["hand_over"] = True

        return updated_state

    def calculate_rewards(self, game_state):
        # Calculate net change from starting stack
        oop_net = self.oop_player.chips - STARTING_STACK
        ip_net = self.ip_player.chips - STARTING_STACK
        
        # Verify the zero-sum property
        total_chips = self.oop_player.chips + self.ip_player.chips
        if total_chips != 2 * STARTING_STACK:
            logging.error(f"Invalid total chips: {total_chips}, expected {2 * STARTING_STACK}")
            # Adjust chips to maintain proper total
            excess = total_chips - (2 * STARTING_STACK)
            if self.oop_player.chips > STARTING_STACK:
                self.oop_player.chips -= excess
                oop_net = self.oop_player.chips - STARTING_STACK
            else:
                self.ip_player.chips -= excess
                ip_net = self.ip_player.chips - STARTING_STACK

        # Rewards should be zero-sum and capped at starting stack
        oop_reward = max(min(oop_net, STARTING_STACK), -STARTING_STACK)
        ip_reward = max(min(ip_net, STARTING_STACK), -STARTING_STACK)

        return oop_reward, ip_reward

    def deal_flop(self):
        logging.info("Dealing flop")
        self._deal_community_cards(3)
        return self.postflop_betting(street="flop")

    def deal_turn(self):
        logging.info("Dealing Turn")
        self._deal_community_cards(1)
        return self.postflop_betting(street="turn")

    def deal_river(self):
        logging.info("Dealing River")
        self._deal_community_cards(1)
        return self.postflop_betting(street="river")

    def _deal_community_cards(self, num_cards):
        self.community_cards.extend([self.deck.cards.pop() for _ in range(num_cards)])

    def _reset_hands(self):
        logging.info("Resetting Hands")
        self.deck = Deck()
        self.community_cards = []
        self.pot = INITIAL_POT
        self.oop_player.hand = []
        self.ip_player.hand = []
        self.current_bet = 2
        self.num_actions = 0
        self.last_action = None
        self.hand_over = False
        self.is_allin = False

    def _get_game_state(self):
        return {
            "pot": self.pot,
            "community_cards": self.community_cards,
            "current_player": self.current_player.name,
            "previous_bet": self.previous_bet,
            "current_bet": self.current_bet,
            "last_action": self.last_action,
            "num_actions": self.num_actions,
            "hand_over": self.hand_over,
            "waiting_for_action": self.waiting_for_action,
            "is_allin": self.is_allin,
            "oop_player": {
                "name": self.oop_player.name,
                "chips": self.oop_player.chips,
                "hand": self.oop_player.hand,
                "committed": self.oop_committed,
            },
            "ip_player": {
                "name": self.ip_player.name,
                "chips": self.ip_player.chips,
                "hand": self.ip_player.hand,
                "committed": self.ip_committed,
            },
        }

    def _get_public_game_state(self):
        return {
            "pot": self.pot,
            "community_cards": self.community_cards,
            "current_player": self.current_player.name,
            "previous_bet": self.previous_bet,
            "current_bet": self.current_bet,
            "last_action": self.last_action,
            "num_actions": self.num_actions,
            "hand_over": self.hand_over,
            "waiting_for_action": self.waiting_for_action,
            "oop_player": {
                "name": self.oop_player.name,
                "chips": self.oop_player.chips,
                "committed": self.oop_committed,
            },
            "ip_player": {
                "name": self.ip_player.name,
                "chips": self.ip_player.chips,
                "committed": self.ip_committed,
            },
        }

    def _get_private_game_state(self):
        return {
            "oop_player": {
                "hand": self.oop_player.hand,
            },
            "ip_player": {
                "hand": self.ip_player.hand,
            },
        }

    def _get_player_action(self, valid_actions, max_bet, min_bet):
        logging.info(f"Getting {self.current_player.name}'s Action: ({valid_actions})")
        if isinstance(self.current_player, HumanPlayer):
            return self.current_player.get_action(valid_actions, max_bet, min_bet)
        else:
            state = self._get_state_representation(current_player=self.current_player)
            if self.current_player == self.oop_player:
                action, bet_size = self.oop_agent.act(
                    state, valid_actions, max_bet, min_bet
                )
                player = "oop"
            else:
                action, bet_size = self.ip_agent.act(
                    state, valid_actions, max_bet, min_bet
                )
                player = "ip"


            # Ensure bet size for bet actions
            if action == 3 or action == "bet":  # Handle both int and string actions
                if bet_size is None:
                    bet_size = min_bet  # Use minimum bet if none specified
                bet_size = min(bet_size, max_bet)
                return action, bet_size
            
            return action, None

    def preflop_betting(self):
        self.current_bet, self.previous_bet, self.num_actions = 2, 0, 0
        self.current_player = self.ip_player  # IP acts first preflop
        self.oop_committed = 2  # OOP posts big blind
        self.ip_committed = 1   # IP posts small blind
        
        logging.info("Starting Preflop Betting")
        oop_experiences = []
        ip_experiences = []

        while True:
            game_state = self._get_game_state()
            state_representation = self._get_state_representation()
            print_if_human(f"\nCommunity Cards: {game_state['community_cards']}")
            print_if_human(f"Your Hand: {self.get_player_hand()}")
            print_if_human(f"IP Chips: {game_state[self.ip_player.name.lower() + '_player']['chips']}")
            print_if_human(f"OOP Chips: {game_state[self.oop_player.name.lower() + '_player']['chips']}")

            if self.hand_over or self.num_active_players == 1:
                self.current_player.chips += self.pot
                game_state = self._get_game_state()
                game_state["message"] = f"{self.current_player.name} wins ${self.pot}"
                game_state["hand_over"] = True
                return game_state, (oop_experiences, ip_experiences)

            if self.num_actions >= 2:
                all_bets_settled = self.oop_committed == self.ip_committed
                if all_bets_settled:
                    game_state = self._get_game_state()
                    game_state["message"] = "Preflop betting complete"
                    return game_state, (oop_experiences, ip_experiences)

            valid_actions, max_bet, min_bet = self._get_valid_preflop_actions()
            action = self._get_player_action(valid_actions, max_bet, min_bet)

            if isinstance(action, tuple):
                action, bet_size = action
            else:
                bet_size = None
            
            # Convert int action to string if needed
            if isinstance(action, int):
                action_int = action
                action = {0: "fold", 1: "check", 2: "call", 3: "bet"}[action]
            else:
                action_int = self._action_to_int(action)
            
            
            # Store experience with integer action
            if self.current_player == self.oop_player:
                oop_experiences.append(
                    (state_representation, action_int, valid_actions, bet_size, max_bet)
                )
            else:
                ip_experiences.append(
                    (state_representation, action_int, valid_actions, bet_size, max_bet)
                )

            # Process action using string representation
            self._process_preflop_action(action, bet_size)
            self.num_actions += 1
            self._switch_players()
        return game_state, (oop_experiences, ip_experiences)

    def _action_to_int(self, action):
        """Convert action string or int to integer"""
        action_map = {
            "fold": 0, "check": 1, "call": 2, "bet": 3,
            0: 0, 1: 1, 2: 2, 3: 3  # Allow direct integer inputs
        }
        return action_map[action]

    def _int_to_action(self, action_int):
        """Convert integer to action string"""
        action_map = {0: "fold", 1: "check", 2: "call", 3: "bet"}
        return action_map[action_int]

    def _process_preflop_action(self, action, bet_size=None):
        logging.info(f"Processing preflop action: {action}")
        # This method will be called from the server to process each action
        print_if_human(f"{self.current_player.name} {action}s")

        if action == "bet":
            self._handle_preflop_bet(bet_size)
        if action == "call":
            self._handle_preflop_call()
        if action == "check":
            self._handle_preflop_check()
        if action == "fold":
            self._handle_fold()


    def _get_valid_preflop_actions(self):
        # fmt: off
        valid_actions =[]
        if self.oop_player.chips == 0 or self.ip_player.chips == 0: #Facing All in
            valid_actions.extend(["call", "fold"])
        elif self.current_player == self.ip_player and self.num_actions == 0: #Preflop Open
            valid_actions.extend(["call", "bet", "fold"])
        elif self.current_player == self.oop_player and self.last_action == "call": #Facing Open limp
            valid_actions.extend(["check", "bet"])
        else:
            valid_actions.extend(["call", "bet", "fold"])
        # fmt: on
        max_bet = self._calculate_max_preflop_bet_size()

        if self.num_actions == 0:
            min_bet = max(MINIMUM_BET_INCREMENT, self.current_bet * 2)
        else:
            min_bet = max(MINIMUM_BET_INCREMENT, (self.current_bet - self.previous_bet) + self.current_bet)
        max_bet = max(max_bet, min_bet)

        max_bet = min(max_bet, self.current_player.chips)
        min_bet = min(min_bet, self.current_player.chips)

        return valid_actions, max_bet, min_bet

    def _calculate_max_preflop_bet_size(self):
        max_bet = 0
        state = self._get_game_state()
        pot = state["pot"]
        player_chips = self.current_player.chips
        player_committed = (
            self.oop_committed
            if self.current_player == self.oop_player
            else self.ip_committed
        )
        current_bet = state["current_bet"] 
        previous_bet = state["previous_bet"]

        if current_bet == 0:
            max_bet = pot
        else:
            max_raise = (3 * current_bet) 

            max_bet = min(max_raise, player_chips)

        return max_bet

    def _calculate_max_postflop_bet_size(self, initial_pot):
        max_bet = 0
        state = self._get_game_state()
        pot = state["pot"]
        current_bet = state["current_bet"]
        player_chips = self.current_player.chips
        player_committed = (
            self.oop_committed
            if self.current_player == self.oop_player
            else self.ip_committed
        )

        opponent_committed = (
            self.oop_committed
            if self.current_player == self.ip_player
            else self.ip_committed
        )

        to_call = opponent_committed - player_committed

        if current_bet == 0:
            max_bet = min(initial_pot, player_chips)
        else:
            max_raise = 3 * current_bet + initial_pot
            max_bet = min(max_raise, player_chips)

        # Could include the min bet to be set here as well
        return min(max_bet, player_chips)

    def _handle_preflop_bet(self, bet_size=None):
        logging.info(f"Handling preflop bet for {self.current_player.name}")
        
        is_allin, final_bet_size = self._calculate_preflop_bet_size(bet_size)
        print_if_human(f"Allin {is_allin}")
        
        if not is_allin:
            if self.current_player.name == self.ip_player.name:
                self.current_player.chips -= final_bet_size - self.ip_committed
                self.pot += final_bet_size - self.ip_committed
                self.ip_committed = final_bet_size
                self.previous_bet = self.current_bet
                self.current_bet = final_bet_size
                update_bet_size("ip", final_bet_size, self.pot)
            else:
                self.current_player.chips -= final_bet_size - self.oop_committed
                self.pot += final_bet_size - self.oop_committed
                self.oop_committed = final_bet_size
                self.previous_bet = self.current_bet
                self.current_bet = final_bet_size
                update_bet_size("ip", final_bet_size, self.pot)
        else:
            self.is_allin = True
            if self.current_player.name == self.ip_player.name:
                self.ip_committed += self.current_player.chips
            else:
                self.oop_committed += self.current_player.chips
            self.pot += self.current_player.chips
            self.current_player.chips = 0
            update_bet_size(
                "ip" if self.current_player.name == self.ip_player.name else "oop",
                self.current_player.chips,
                self.pot,
            )
        self.last_action = "bet"

    def _handle_preflop_call(self):
        logging.info(f"Handling preflop call for {self.current_player.name}")
        if self.current_player == self.ip_player and self.num_actions == 0:
            self.current_player.chips -= 1
            self.ip_committed += 1
            self.pot += 1
        else:
            if self.current_player.name == self.oop_player.name:
                diff = self.ip_committed - self.oop_committed
                self.oop_committed += diff
            else:
                diff = self.oop_committed - self.ip_committed
                self.ip_committed += diff
            self.current_player.chips -= diff
            self.pot += diff
        self.last_action = "call"

    def _handle_preflop_check(self):
        logging.info(f"Handling preflop check from {self.current_player.name}")
        if self.current_player == self.oop_player and self.last_action == "call":
            self.last_action = "check"

    def _calculate_preflop_bet_size(self, bet_size=None):
        state = self._get_game_state()
        all_in = False
        print_if_human(f"{self.current_player.name}: {self.current_player.chips} bets {bet_size}")
        if self.current_player.chips <= bet_size:
            bet_size = self.current_player.chips
            all_in = True

        return all_in, bet_size

    def postflop_betting(self, street):
        logging.info("Starting Postflop Betting")
        state = self._get_game_state()
        initial_pot = state["pot"]
        self.current_bet = 0
        self.num_actions = 0
        self.current_player = self.oop_player

        oop_experiences = []
        ip_experiences = []
        oop_committed = 0
        ip_committed = 0

        while True:
            state = self._get_game_state()
            state_representation = self._get_state_representation()

            logging.info(f"Current State: {state}")
            print_if_human(f"\nCommunity Cards: {state['community_cards']}")
            print_if_human(f"Pot: {self.pot}")
            print_if_human(f"IP Chips: {state[self.ip_player.name.lower() + '_player']['chips']}")
            print_if_human(f"OOP Chips: {state[self.oop_player.name.lower() + '_player']['chips']}")
            print_if_human(f"Your Hand: {self.get_player_hand()}")

            if self.hand_over or self.num_active_players == 1:
                self.current_player.chips += self.pot
                game_state = self._get_game_state()
                game_state["message"] = f"{self.current_player.name} wins ${self.pot}"
                game_state["hand_over"] = True
                return game_state, (oop_experiences, ip_experiences)

            all_players_acted = self.num_actions >= self.num_active_players
            all_bets_settled = self.oop_committed == self.ip_committed
            if all_players_acted and all_bets_settled:
                game_state = self._get_game_state()
                game_state["message"] = f"{street.capitalize()} betting complete"
                return game_state, (oop_experiences, ip_experiences)

            valid_actions, min_bet = self._get_valid_postflop_actions()
            max_bet = self._calculate_max_postflop_bet_size(initial_pot)
            action = self._get_player_action(valid_actions, max_bet, min_bet)

            if isinstance(action, tuple):
                action, bet_size = action
                logging.info(f"Received bet size: {bet_size}")
            else:
                bet_size = None

            # Convert int action to string if needed
            if isinstance(action, int):
                action_int = action
                action = self._int_to_action(action)
            else:
                action_int = self._action_to_int(action)

            print_if_human(f"Action: {action}, Bet Size: {bet_size}")

            # Store experience with integer action
            if self.current_player == self.oop_player:
                oop_experiences.append(
                    (state_representation, action_int, valid_actions, bet_size, max_bet)
                )
            else:
                ip_experiences.append(
                    (state_representation, action_int, valid_actions, bet_size, max_bet)
                )

            self._process_postflop_action(action, bet_size)

            if self.last_action == "call" and self.oop_committed == self.ip_committed:
                game_state = self._get_game_state()
                game_state["message"] = f"{street.capitalize()} betting complete"
                return game_state, (oop_experiences, ip_experiences)
        return game_state, (oop_experiences, ip_experiences)

    def _process_postflop_action(self, action, bet_size=None):
        logging.info(f"Processing Postflop Action: {action}")
        print_if_human(f"{self.current_player.name} {action}s")

        if action == "check":
            self._handle_postflop_check()
        elif action == "bet":
            self._handle_postflop_bet(bet_size)
        elif action == "call":
            self._handle_postflop_call()
        elif action == "fold":
            self._handle_fold()

        self.num_actions += 1
        self._switch_players()


    def _get_valid_postflop_actions(self):
        valid_actions = []
        min_bet = max(
            MINIMUM_BET_INCREMENT, min(self.current_bet * 2, self.current_player.chips)
        )

        if self.oop_player.chips == 0 or self.ip_player.chips == 0:
            valid_actions = ["call", "fold"]
        elif self.current_bet == 0:
            valid_actions = ["check", "bet"]
        else:
            valid_actions = ["call", "bet", "fold"]

        return valid_actions, min_bet

    def _handle_postflop_bet(self, bet_size):
        if bet_size is None:
            bet_size = MINIMUM_BET_INCREMENT
        
        # Validate bet size against player's chips
        max_bet = min(self.current_player.chips, self._calculate_max_postflop_bet_size(self.pot))
        bet_size = min(bet_size, max_bet)
        
        # Track chips and pot
        self.current_player.chips -= bet_size
        self.pot += bet_size
        self.current_bet = bet_size
        
        if self.current_player == self.oop_player:
            self.oop_committed += bet_size
        else:
            self.ip_committed += bet_size

        if self.current_player.chips == 0:
            self.is_allin = True
        
        self.last_action = "bet"
        
        # Validate total chips haven't changed
        total_chips = (self.oop_player.chips + self.ip_player.chips + self.pot)
        assert total_chips == 2 * STARTING_STACK, f"Invalid chip total: {total_chips}"

    def _handle_postflop_call(self):
        logging.info("Handling postflop CALL for {self.current_player.name}")
        if self.current_player.name == self.oop_player.name:
            call_amount = self.ip_committed - self.oop_committed
        else:
            call_amount = self.oop_committed - self.ip_committed
        if call_amount <= self.current_player.chips:
            self.current_player.chips -= call_amount
            self.pot += call_amount
            if self.current_player == self.oop_player:
                self.oop_committed = self.ip_committed
            else:
                self.ip_committed = self.oop_committed
        else:
            # This is prone to bug. Only works under equal starting stacks.
            all_in_amount = self.current_player.chips
            self.current_player.chips = 0
            self.pot += all_in_amount
            if self.current_player == self.oop_player:
                self.oop_committed += all_in_amount
            else:
                self.ip_committed += all_in_amount
        self.last_action = "call"

    def _handle_postflop_check(self):
        logging.info(f"Handling postflop CHECK for {self.current_player.name}")
        self.last_action = "check"

    def _handle_fold(self):
        logging.info(f"Handling FOLD for {self.current_player.name}")
        self.num_active_players -= 1
        self.num_actions += 1
        self.hand_over = True
        
        # Give pot to the other player immediately
        winning_player = self.ip_player if self.current_player == self.oop_player else self.oop_player
        winning_player.chips += self.pot
        self.pot = 0

    def _calculate_postflop_bet_size(self):
        all_in = False
        max_bet = self._calculate_max_postflop_bet_size()
        
        # Get current state for pot-based calculations
        state = self._get_game_state()
        
        # Initialize bet_size based on game situation
        if self.last_action == "bet":
            # If it's a raise, size is 3x last raise plus pot
            bet_size = min(2 * self.current_bet + state['pot'], max_bet)
        else:
            # If it's a standard bet, size is pot-sized
            bet_size = min(state['pot'], max_bet)

        if self.current_player.chips < bet_size:
            bet_size = self.current_player.chips
            all_in = True

        return all_in, bet_size

    def _switch_players(self):
        self.current_player = (
            self.oop_player
            if self.current_player.name == self.ip_player.name
            else self.ip_player
        )

    def _is_terminal(self, state):
        """Check if the game state is terminal"""
        if state['hand_over']:
            return True
        if state['num_active_players'] == 1:
            return True
        return False

    def _get_terminal_value(self, state):
        """Get the value of a terminal state"""
        if state['num_active_players'] == 1:
            # If only one player remains, return win/loss value
            return 1.0 if state['current_player'] == self.current_player.name else -1.0
        
        # For showdown, calculate based on pot equity
        pot_share = state['pot'] / 2  # Simplified equity calculation
        value = (state['current_player']['chips'] - STARTING_STACK + pot_share) / STARTING_STACK
        return max(min(value, 1.0), -1.0)

    def get_valid_actions(self, state):
        """Get valid actions for MCTS simulation"""
        if state['street'] == 'preflop':
            valid_actions, max_bet, min_bet = self._get_valid_preflop_actions()
        else:
            valid_actions, min_bet = self._get_valid_postflop_actions()
            max_bet = self._calculate_max_postflop_bet_size(state['pot'])
        
        return {
            'valid_actions': valid_actions,
            'max_bet': max_bet,
            'min_bet': min_bet
        }

    def simulate_action(self, state, action):
        """Simulate taking an action in MCTS"""
        new_state = state.copy()
        
        if action == 'fold':
            new_state['num_active_players'] -= 1
            new_state['hand_over'] = True
        elif action == 'call':
            # Simulate call action
            if new_state['current_player'] == 'OOP':
                new_state['oop_committed'] = new_state['ip_committed']
            else:
                new_state['ip_committed'] = new_state['oop_committed']
        elif action == 'bet':
            # Simulate bet action
            _, bet_size = self._calculate_postflop_bet_size()
            if new_state['current_player'] == 'OOP':
                new_state['oop_committed'] += bet_size
            else:
                new_state['ip_committed'] += bet_size
            new_state['pot'] += bet_size
            
        return new_state

    def _get_street(self, state):
        """Determine the current street based on community cards"""
        num_cards = len(state['community_cards'])
        if num_cards == 0:
            return 0  # preflop
        elif num_cards == 3:
            return 1  # flop
        elif num_cards == 4:
            return 2  # turn
        else:
            return 3  # river
