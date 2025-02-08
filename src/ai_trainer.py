import secrets
import logging
import sys
import numpy as np
from agent2 import DQNAgent
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

CONST_100bb = 200
CONST_200bb = 400
CONST_300bb = 600
MINIMUM_BET_INCREMENT = 2

setup_logging()

# Add near top of file
action_to_index = {
    'fold': 0,
    'check': 1,
    'call': 2,
    'bet': 3
}

index_to_action = {v: k for k, v in action_to_index.items()}

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
            player.chips = CONST_100bb


class HumanPlayer(Player):
    def get_action(self, valid_actions, max_bet):
        # print(f"Valid actions: {valid_actions}")
        # print(f"Maximum Bet: {max_bet}")
        while True:
            action = input(f"Choose an action {valid_actions}: ")
            if action in valid_actions:
                if action == "bet":
                    bet_size = int(input("Enter the bet size: "))
                    return action, bet_size
                return action
            # print("Invalid action. Please try again.")


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
    def __init__(
        self, human_position=None, oop_agent=None, ip_agent=None, state_size=None
    ):
        self.state_size = state_size or (
            7 + (5 * 2) + 2 * 4 * 2
        )  # Use provided state_size or calculate default
        self.action_size = 4

        self.human_position = human_position
        self.oop_agent = oop_agent or DQNAgent(self.state_size, self.action_size)
        self.ip_agent = ip_agent or DQNAgent(self.state_size, self.action_size)

        self.deck = Deck()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if human_position == "oop":
            self.oop_player = HumanPlayer(name="OOP", chips=200)
            self.ip_player = Player(name="IP", chips=200)
        if human_position == "ip":
            self.oop_player = Player(name="OOP", chips=200)
            self.ip_player = HumanPlayer(name="IP", chips=200)
        else:
            self.oop_player = Player(name="OOP", chips=200)
            self.ip_player = Player(name="IP", chips=200)

        self.initialize_game_state()

        self.oop_loss = None
        self.ip_loss = None
        self.oop_cumulative_reward = 0
        self.ip_cumulative_reward = 0

        logging.info("PokerGame initialized")

        if self.oop_agent:
            self.oop_agent.name = "OOP"
        if self.ip_agent:
            self.ip_agent.name = "IP"

    def calculate_state_size(self):
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

    def get_state_representation(self, state=None, current_player=None):
        # Convert the game state to a numerical representation for the DQN
        if state is None:
            state = self.get_game_state()
        representation = [
            float(state["pot"]),
            float(len(state["community_cards"])),
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
            representation.extend(self.encode_card(card))

        if current_player == self.oop_player:
            for card in state["oop_player"]["hand"]:
                representation.extend(self.encode_card(card))
            representation.extend([0, 0] * 4)
        else:
            for card in state["ip_player"]["hand"]:
                representation.extend(self.encode_card(card))
            representation.extend([0, 0] * 4)

        assert (
            len(representation) == self.state_size
        ), f"State size mismatch: expected { self.state_size }"

        return torch.FloatTensor(representation)

    def encode_card(self, card):
        if not card:
            return [0, 0]
        # Simple encoding: rank (2-14) and suit (0-3)
        ranks = "23456789TJQKA"
        suits = "cdhs"
        return (ranks.index(card[0]) + 2, suits.index(card[1]))

    def initialize_game_state(self):
        self.community_cards = []
        self.pot = 3
        self.hand_over = False
        self.current_bet = 2
        self.num_actions = 0
        self.last_action = "bet"
        self.current_player = self.ip_player
        self.num_active_players = 2
        self.ip_committed = 1
        self.oop_committed = 2
        self.street = "preflop"
        self.waiting_for_action = False
        self.is_allin = False

    def deal_cards(self):
        for player in [self.oop_player, self.ip_player]:
            player.hand = [self.deck.cards.pop() for _ in range(4)]

    def start_new_hand(self):
        self.initialize_game_state()
        self.reset_hands()
        self.deck.shuffle()
        self.pot = 3
        self.oop_player.chips = 198
        self.ip_player.chips = 199
        self.deal_cards()
        return self.get_game_state()

    def play_hand(self):
        logging.info("Starting new hand")
        game_state = self.start_new_hand()

        # print(f"Your hand: {self.get_player_hand()}")

        oop_experiences = []
        ip_experiences = []

        game_state, round_experiences = self.preflop_betting()
        if game_state["is_allin"]:
            self.deal_community_cards(5)
            game_state = self.determine_showdown_winner()
            game_state["hand_over"] = True

        oop_experiences.extend(round_experiences[0])
        ip_experiences.extend(round_experiences[1])

        if not game_state["hand_over"]:
            # print("\nDEALING FLOP\n")
            game_state, round_experiences = self.deal_flop()
            oop_experiences.extend(round_experiences[0])
            ip_experiences.extend(round_experiences[1])

            if game_state["is_allin"]:
                self.deal_community_cards(2)
                game_state = self.determine_showdown_winner()
                game_state["hand_over"] = True

            if not game_state["hand_over"]:
                # print("\nDEALING TURN\n")
                game_state, round_experiences = self.deal_turn()
                oop_experiences.extend(round_experiences[0])
                ip_experiences.extend(round_experiences[1])

                if game_state["is_allin"]:
                    self.deal_community_cards(1)
                    game_state = self.determine_showdown_winner()
                    game_state["hand_over"] = True

                if not game_state["hand_over"]:
                    # print("\nDEALING RIVER\n")
                    game_state, round_experiences = self.deal_river()
                    oop_experiences.extend(round_experiences[0])
                    ip_experiences.extend(round_experiences[1])

                    if not game_state["hand_over"]:
                        game_state = self.determine_showdown_winner()

        final_state = self.get_state_representation()
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
        # print(f"\nIP Tables: {self.ip_player.hand}")
        # print(f"\nOOP Tables: {self.oop_player.hand}\n")
        updated_state = self.get_game_state()
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
            # print(f"{updated_state['ip_player']['name']} wins {self.pot}")
            # print(f"{self.ip_player.hand}")
            updated_state["ip_player"]["chips"] += self.pot
        elif oop_rank < ip_rank:
            # print(f"{updated_state['oop_player']['name']} wins {self.pot}")
            updated_state["oop_player"]["chips"] += self.pot
        else:
            updated_state["ip_player"]["chips"] += self.pot / 2
            updated_state["oop_player"]["chips"] += self.pot / 2

        # print(f"\nCommunity Cards: {self.community_cards}")

        updated_state["hand_over"] = True

        return updated_state

    def calculate_rewards(self, game_state):
        oop_reward = game_state["oop_player"]["chips"] - CONST_100bb
        ip_reward = game_state["ip_player"]["chips"] - CONST_100bb

        total_reward = oop_reward + ip_reward
        oop_reward -= total_reward / 2
        ip_reward -= total_reward / 2

        return oop_reward, ip_reward

    def deal_flop(self):
        logging.info("Dealing flop")
        self.deal_community_cards(3)
        return self.postflop_betting(street="flop")

    def deal_turn(self):
        logging.info("Dealing Turn")
        self.deal_community_cards(1)
        return self.postflop_betting(street="turn")

    def deal_river(self):
        logging.info("Dealing River")
        self.deal_community_cards(1)
        return self.postflop_betting(street="river")

    def deal_community_cards(self, num_cards):
        self.community_cards.extend([self.deck.cards.pop() for _ in range(num_cards)])

    def reset_hands(self):
        logging.info("Resetting Hands")
        self.deck = Deck()
        self.community_cards = []
        self.pot = 0
        self.oop_player.hand = []
        self.ip_player.hand = []
        self.current_bet = 2
        self.num_actions = 0
        self.last_action = None
        self.hand_over = False
        self.is_allin = False

    def get_game_state(self):
        return {
            "pot": self.pot,
            "community_cards": self.community_cards,
            "current_player": self.current_player.name,
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

    def get_public_game_state(self):
        return {
            "pot": self.pot,
            "community_cards": self.community_cards,
            "current_player": self.current_player.name,
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

    def get_private_game_state(self):
        return {
            "oop_player": {
                "hand": self.oop_player.hand,
            },
            "ip_player": {
                "hand": self.ip_player.hand,
            },
        }

    def get_player_action(self, valid_actions, max_bet, min_bet):
        logging.info(f"Getting {self.current_player.name}'s Action: ({valid_actions})")
        if isinstance(self.current_player, HumanPlayer):
            return self.current_player.get_action(valid_actions, max_bet)
        else:
            state = self.get_state_representation(current_player=self.current_player)
            if self.current_player == self.oop_player:
                action, bet_size = self.oop_agent.act(
                    state, valid_actions, max_bet, min_bet
                )
                player = "oop"
                agent = self.oop_agent
            else:
                action, bet_size = self.ip_agent.act(
                    state, valid_actions, max_bet, min_bet
                )
                player = "ip"
                agent = self.ip_agent

            # Get policy and value from model
            with torch.no_grad():
                policy, value = agent.model(state.to(agent.device))
                q_val = value.cpu().data.numpy().max()
                q_value.labels(player).set(q_val)
                
            epsilon.labels(player).set(agent.epsilon)
            action_taken.labels(player_action=f"{player}_{action}").inc()

            if action == "bet":
                logging.info(f"Bet size: {bet_size}")
                logging.info(f"Bet Max: {max_bet}")
                return action, min(bet_size, max_bet)
            return action, None

    def preflop_betting(self):
        # self.num_active_players = len(
        # [p for p in [self.oop_player, self.ip_player] if p.chips > 0]
        # )
        # self.ip_committed, self.oop_committed = 1, 2
        self.current_bet, self.num_actions = 2, 0
        # self.last_action = "bet"
        # self.current_player = self.ip_player
        # self.hand_over = False

        logging.info("Starting Preflop Betting")
        oop_experiences = []
        ip_experiences = []

        while True:
            game_state = self.get_game_state()
            state_representation = self.get_state_representation()

            if self.hand_over or self.num_active_players == 1:
                self.current_player.chips += self.pot
                game_state = self.get_game_state()
                game_state["message"] = f"{self.current_player.name} wins ${self.pot}"
                game_state["hand_over"] = True
                return game_state, (oop_experiences, ip_experiences)

            all_players_acted = self.num_actions >= self.num_active_players
            all_bets_settled = self.oop_player.chips == self.ip_player.chips
            if all_players_acted and all_bets_settled:
                game_state = self.get_game_state()
                game_state["message"] = "Preflop betting complete"
                return game_state, (oop_experiences, ip_experiences)

            # print(f"All players acted: {all_players_acted}")
            # print(f"All bets settled: {all_bets_settled}")

            logging.info(f"Current State: {game_state}")
            # print(f"\nCommunity Cards: {game_state['community_cards']}")
            # print(f"Pot: {game_state['pot']}")
            # print(f"Your Hand: {self.get_player_hand()}")
            # print( f"IP Chips: {game_state[self.ip_player.name.lower() + '_player']['chips']}")
            # print( f"OOP Chips: {game_state[self.oop_player.name.lower() + '_player']['chips']}")

            # Instead of prompting for input, we'll return the game state
            valid_actions, max_bet, min_bet = self.get_valid_preflop_actions()
            # print(f"Finished getting valid actions: {valid_actions}")
            action = self.get_player_action(valid_actions, max_bet, min_bet)

            # print(f"RETURNED TUPLE: {action}")
            if isinstance(action, tuple):
                action, bet_size = action
            else:
                bet_size = None
            # print(f"Finished getting action: {action}")
            # print(f"Bet Size: {bet_size}")

            action_int = self.action_to_int(action)

            # print("About to append experience")
            if self.current_player == self.oop_player:
                oop_experiences.append(
                    (state_representation, action_int, valid_actions, bet_size, max_bet)
                )
            else:
                ip_experiences.append(
                    (state_representation, action_int, valid_actions, bet_size, max_bet)
                )

            # print("About to process action")

            self.process_preflop_action(action, bet_size)
        return game_state, (oop_experiences, ip_experiences)

    def action_to_int(self, action):
        action_map = {"fold": 0, "check": 1, "call": 2, "bet": 3}
        return action_map[action]

    def process_preflop_action(self, action, bet_size=None):
        logging.info(f"Processing preflop action: {action}")
        # This method will be called from the server to process each action
        # print(f"Processing action: {action}")

        if action == "bet":
            # print("Handling Bet")
            self.handle_preflop_bet(bet_size)
        if action == "call":
            # print("Handling Call")
            self.handle_preflop_call()
        if action == "check":
            self.handle_preflop_check()
            # print("Handling Check")
        if action == "fold":
            # print("handling Fold")
            self.handle_fold()

        self.switch_players()

    def get_valid_preflop_actions(self):
        # fmt: off
        valid_actions =[]
        if self.oop_player.chips == 0 or self.ip_player.chips == 0: #Facing All in
            valid_actions.extend(["call", "fold"])
        if self.current_player == self.ip_player and self.num_actions == 0: #Preflop Open
            valid_actions.extend(["call", "bet", "fold"])
        elif self.current_player == self.oop_player and self.last_action == "call": #Facing Open limp
            valid_actions.extend(["check", "bet"])
        else:
            valid_actions.extend(["call", "bet", "fold"])
        # fmt: on
        max_bet = self.calculate_max_preflop_bet_size()
        min_bet = max(
            MINIMUM_BET_INCREMENT, min(self.current_bet * 2, self.current_player.chips)
        )
        return valid_actions, max_bet, min_bet

    def calculate_max_preflop_bet_size(self):
        max_bet = 0
        state = self.get_game_state()
        pot = state["pot"]
        current_bet = state["current_bet"]
        player_chips = self.current_player.chips
        player_committed = (
            self.oop_committed
            if self.current_player == self.oop_player
            else self.ip_committed
        )

        # print(f"{type(current_bet)}\n{type(pot)}\n{type(player_committed)}")

        if current_bet == 0:
            max_bet = pot
        else:
            max_raise = 3 * current_bet + pot - player_committed

            max_bet = min(max_raise, player_chips)

        max_bet = (max_bet // MINIMUM_BET_INCREMENT) * MINIMUM_BET_INCREMENT
        max_bet = min(self.current_player.chips, 3 * state["current_bet"])
        return max_bet

    def calculate_max_postflop_bet_size(self, initial_pot):
        max_bet = 0
        state = self.get_game_state()
        pot = state["pot"]
        print(f"POT: {pot}")
        current_bet = state["current_bet"]
        print(f"CURRENT_BET: {current_bet}")
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

        # print(f"{type(current_bet)}\n{type(pot)}\n{type(player_committed)}")
        to_call = opponent_committed - player_committed

        if current_bet == 0:
            max_bet = min(initial_pot, player_chips)
        else:
            max_raise = 3 * current_bet + initial_pot
            max_bet = min(max_raise, player_chips)

        # Could include the min bet to be set here as well
        return min(max_bet, player_chips)

    def handle_preflop_bet(self, bet_size=None):
        logging.info(f"Handling preflop bet for {self.current_player.name}")
        is_allin, bet_amount = self.calculate_preflop_bet_size()
        # logging.info(f"\n\nis_allin: {is_allin}  bet_amount: {bet_amount}\n\n")
        if not is_allin:
            if self.current_player.name == self.ip_player.name:
                self.current_player.chips -= bet_amount - self.ip_committed
                self.pot += bet_amount - self.ip_committed
                self.ip_committed = bet_amount
                self.current_bet = bet_amount
                update_bet_size("ip", bet_amount, self.pot)
            else:
                self.current_player.chips -= bet_amount - self.oop_committed
                self.pot += bet_amount - self.oop_committed
                self.oop_committed = bet_amount
                self.current_bet = bet_amount
                update_bet_size("ip", bet_amount, self.pot)
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
        self.num_actions += 1
        self.last_action = "bet"

    def handle_preflop_call(self):
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
        self.num_actions += 1
        self.last_action = "call"

    def handle_preflop_check(self):
        logging.info(f"Handling preflop check from {self.current_player.name}")
        if self.current_player == self.oop_player and self.last_action == "call":
            self.num_actions += 1
            self.last_action = "check"

    def calculate_preflop_bet_size(self, bet_size=None):
        state = self.get_game_state()
        all_in = False
        is_raise = True
        if is_raise:
            # If it's a raise, the bet size is 3 times the last raise plus the current pot size
            bet_size = 3 * state["current_bet"]
        if self.current_player.chips < bet_size:
            bet_size = self.current_player.chips
            all_in = True

        return all_in, bet_size

    def postflop_betting(self, street):
        logging.info("Starting Postflop Betting")
        state = self.get_game_state()
        initial_pot = state["pot"]
        self.current_bet = 0
        self.num_actions = 0
        self.current_player = self.oop_player

        oop_experiences = []
        ip_experiences = []
        oop_committed = 0
        ip_committed = 0

        while True:
            state = self.get_game_state()
            state_representation = self.get_state_representation()

            logging.info(f"Current State: {state}")
            # print(f"\nCommunity Cards: {state['community_cards']}")
            # print(f"Your Hand: {self.get_player_hand()}")
            # print( f"IP Chips: {state[self.ip_player.name.lower() + '_player']['chips']}")
            # print( f"OOP Chips: {state[self.oop_player.name.lower() + '_player']['chips']}")

            if self.hand_over or self.num_active_players == 1:
                self.current_player.chips += self.pot
                game_state = self.get_game_state()
                game_state["message"] = f"{self.current_player.name} wins ${self.pot}"
                game_state["hand_over"] = True
                return game_state, (oop_experiences, ip_experiences)

            all_players_acted = self.num_actions >= self.num_active_players
            all_bets_settled = self.oop_committed == self.ip_committed
            if all_players_acted and all_bets_settled:
                game_state = self.get_game_state()
                game_state["message"] = f"{street.capitalize()} betting complete"
                return game_state, (oop_experiences, ip_experiences)

            valid_actions, min_bet = self.get_valid_postflop_actions()
            max_bet = self.calculate_max_postflop_bet_size(initial_pot)
            # print(f"Finished getting valid actions: {valid_actions}")
            # print(f"MAX BET: {max_bet}")
            action = self.get_player_action(valid_actions, max_bet, min_bet)

            if isinstance(action, tuple):
                # print("CORRECT INSTANCE TUPLE")
                action, bet_size = action
                logging.info(f"Received bet size: {bet_size}")
            else:
                bet_size = None
            # print(f"Finished getting action: {action}")
            action_int = self.action_to_int(action)

            if self.current_player == self.oop_player:
                oop_experiences.append(
                    (state_representation, action_int, valid_actions, bet_size, max_bet)
                )
            else:
                ip_experiences.append(
                    (state_representation, action_int, valid_actions, bet_size, max_bet)
                )

            # print( f"Before process_postflop_action, current player: {self.current_player.name}")
            self.process_postflop_action(action, bet_size)
            # print( f"After process_postflop_action, current player: {self.current_player.name}")

            if self.last_action == "call" and self.oop_committed == self.ip_committed:
                game_state = self.get_game_state()
                game_state["message"] = f"{street.capitalize()} betting complete"
                return game_state, (oop_experiences, ip_experiences)
        return game_state, (oop_experiences, ip_experiences)

    def process_postflop_action(self, action, bet_size=None):
        logging.info(f"Processing Postflop Action: {action}")
        # print(f"Bet Size: {bet_size}")

        if action == "check":
            self.handle_postflop_check()
        elif action == "bet":
            self.handle_postflop_bet(bet_size)
        elif action == "call":
            self.handle_postflop_call()
        elif action == "fold":
            self.handle_fold()

        self.switch_players()

    def get_valid_postflop_actions(self):
        valid_actions = []
        min_bet = max(
            MINIMUM_BET_INCREMENT, min(self.current_bet * 2, self.current_player.chips)
        )
        # print(f"GET VALID POSTFLOP ACTIONS max_bet {max_bet}")

        if self.oop_player.chips == 0 or self.ip_player.chips == 0:
            valid_actions = ["call", "fold"]
        if self.current_bet == 0:
            valid_actions = ["check", "bet"]
        else:
            valid_actions = ["call", "bet", "fold"]

        return valid_actions, min_bet

    def handle_postflop_bet(self, bet_size=MINIMUM_BET_INCREMENT):
        logging.info(
            f"Handling postflop bet of {bet_size} for {self.current_player.name}"
        )

        if self.current_player.name == self.ip_player.name:
            self.current_player.chips -= bet_size
            self.pot += bet_size
            self.ip_committed += bet_size
            self.current_bet = bet_size
            bet_size_metric.labels(player="ip", street="postflop").observe(bet_size)
            update_bet_size("ip", bet_size, self.pot)
        else:
            self.current_player.chips -= bet_size
            self.pot += bet_size
            self.oop_committed += bet_size
            self.current_bet = bet_size
            bet_size_metric.labels(player="oop", street="postflop").observe(bet_size)
            update_bet_size("oop", bet_size, self.pot)

    def handle_postflop_call(self):
        logging.info("Handling postflop CALL for {self.current_player.name}")
        if self.current_player.name == self.oop_player.name:
            call_amount = self.oop_committed - self.ip_committed
        else:
            call_amount = self.ip_committed - self.oop_committed
        if call_amount <= self.current_player.chips:
            self.current_player.chips += call_amount
            self.pot -= call_amount
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
            """
            other_player = (
                self.ip_player
                if self.current_player == self.oop_player
                else self.oop_player
            )
            other_player.chips += difference
            self.pot -= difference
            """
        self.num_actions += 1
        self.last_action = "call"

    def handle_postflop_check(self):
        logging.info(f"Handling postflop CHECK for {self.current_player.name}")
        self.num_actions += 1
        self.last_action = "check"

    def handle_fold(self):
        logging.info(f"Handling postflop FOLD for {self.current_player.name}")
        self.num_active_players -= 1
        self.hand_over = True

    def calculate_postflop_bet_size(self):
        all_in = False
        max_bet = self.calculate_max_postflop_bet_size()
        
        # Get current state for pot-based calculations
        state = self.get_game_state()
        
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

    def switch_players(self):
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
        value = (state['current_player']['chips'] - CONST_100bb + pot_share) / CONST_100bb
        return max(min(value, 1.0), -1.0)

    def get_valid_actions(self, state):
        """Get valid actions for MCTS simulation"""
        if state['street'] == 'preflop':
            valid_actions, max_bet, min_bet = self.get_valid_preflop_actions()
        else:
            valid_actions, min_bet = self.get_valid_postflop_actions()
            max_bet = self.calculate_max_postflop_bet_size(state['pot'])
        
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
            _, bet_size = self.calculate_postflop_bet_size()
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
