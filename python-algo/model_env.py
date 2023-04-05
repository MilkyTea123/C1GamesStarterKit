import gym
from gym import Env
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete

import numpy as np
import math
import random
import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

import gamelib
import random
import math
import warnings
from sys import maxsize
import json

from gamelib.game_state import GameState
from gamelib.util import get_command, debug_write, BANNER_TEXT, send_command


class GameEnv(Env):

    def __init__(self):
        self.NUMUNITS = 6
        self.action_space = MultiDiscrete(np.ones((14, 28))*self.NUMUNITS)
        self.observation_space = Dict({
            'board': MultiDiscrete(np.stack([np.ones((28, 28))*self.NUMUNITS*10, np.ones((28, 28))*1000], axis=2)),
            'p1Stats': MultiDiscrete([5000, 5000, 40]),  # [SP,MP,Health] (SP and MP are x10)
            'p2Stats': MultiDiscrete([5000, 5000, 40])
        })
        seed = random.randrange(maxsize)
        random.seed(seed)
        gamelib.debug_write('Random seed: {}'.format(seed))

        self.state = {
            'board': -np.ones((28, 28, 2)),
            'p1Stats': [40, 5, 30],
            'p2Stats': [40, 5, 30]
        }

    def dict_to_space(self):
        obs_space = Dict({
            'board': MultiDiscrete(self.state['board']),
            'p1Stats': MultiDiscrete(self.state['p1Stats']),
            'p2Stats': MultiDiscrete(self.state['p2Stats'])
        })

    def reset(self):
        self.__init__()

    def on_game_start(self, config):
        """ 
        Read in config and perform any initial setup here 
        """
        gamelib.debug_write('Configuring your custom algo strategy...')
        self.config = config
        global WALL, SUPPORT, TURRET, SCOUT, DEMOLISHER, INTERCEPTOR, MP, SP
        global UNIT_TYPES, WALL_UP_HP
        global S_HEALTH
        WALL = config["unitInformation"][0]["shorthand"]
        SUPPORT = config["unitInformation"][1]["shorthand"]
        TURRET = config["unitInformation"][2]["shorthand"]
        SCOUT = config["unitInformation"][3]["shorthand"]
        DEMOLISHER = config["unitInformation"][4]["shorthand"]
        INTERCEPTOR = config["unitInformation"][5]["shorthand"]
        UNIT_TYPES = [WALL, SUPPORT, TURRET, SCOUT, DEMOLISHER, INTERCEPTOR]
        S_HEALTH = [60,30,75]
        WALL_UP_HP = 120
        MP = 1
        SP = 0
        # This is a good place to do initial setup
        self.scored_on_locations = []

    def on_turn(self, turn_state, structures=None, mobiles=None):
        """
        This function is called every turn with the game state wrapper as
        an argument. The wrapper stores the state of the arena and has methods
        for querying its state, allocating your current resources as planned
        unit deployments, and transmitting your intended deployments to the
        game engine.
        """

        game_state = gamelib.GameState(self.config, turn_state)

        game_state.attempt_spawn(DEMOLISHER, [24, 10], 3)

        gamelib.debug_write('Performing turn {} of your custom algo strategy'.format(
            game_state.turn_number))
        # Comment or remove this line to enable warnings.
        game_state.suppress_warnings(True)

        # self.starter_strategy(game_state)
        if structures != None:
            for structure in structures:
                if structure[3]:
                    upped = game_state.attempt_upgrade([structure[1], structure[2]])
                    if UNIT_TYPES[structure[0]] == WALL and upped > 0:
                        self.state['board'][structure[1], structure[2], 1] += WALL_UP_HP
                else:
                    spawned = game_state.attempt_spawn(UNIT_TYPES[structure[0]], [structure[1], structure[2]])
                    if spawned > 0:
                        self.state['board'][structure[1], structure[2]] = [structure[0],
                                                                           S_HEALTH[structure[0]]]

        if mobiles != None:
            for mobile in mobiles:
                game_state.attempt_spawn(UNIT_TYPES[mobile[0]], [mobile[1], mobile[2]])

        game_state.submit_turn()

    def on_action_frame(self, turn_string):
        """
        This is the action frame of the game. This function could be called 
        hundreds of times per turn and could slow the algo down so avoid putting slow code here.
        Processing the action frames is complicated so we only suggest it if you have time and experience.
        Full doc on format of a game frame at in json-docs.html in the root of the Starterkit.
        """
        # Let's record at what position we get scored on
        state = json.loads(turn_string)
        events = state["events"]
        breaches = events["breach"]
        for breach in breaches:
            location = breach[0]
            unit_owner_self = True if breach[4] == 1 else False
            # When parsing the frame data directly,
            # 1 is integer for yourself, 2 is opponent (StarterKit code uses 0, 1 as player_index instead)
            if not unit_owner_self:
                gamelib.debug_write("Got scored on at: {}".format(location))
                self.scored_on_locations.append(location)
                gamelib.debug_write(
                    "All locations: {}".format(self.scored_on_locations))

    def start(self):
        log_path = os.path.join('Training', 'Logs')
        model = PPO('MultiInputPolicy', env, verbose=1, tensorboard_log=log_path)
        model.learn(total_timesteps=20000)

        save_path = os.path.join('Training', 'Saved Models', 'Save_PPO')
        model.save(save_path)
        # """
        # Start the parsing loop.
        # After starting the algo, it will wait until it recieves information from the game 
        # engine, proccess this information, and respond if needed to take it's turn. 
        # The algo continues this loop until it recieves the "End" turn message from the game.
        # """
        # debug_write(BANNER_TEXT)

        # done = False
        # reward = 2
        # obs_space = self.state

        # while True:
        #     # Note: Python blocks and hangs on stdin. Can cause issues if connections aren't setup properly and may need to
        #     # manually kill this Python program.
        #     game_state_string = get_command()
        #     if "replaySave" in game_state_string:
        #         """
        #         This means this must be the config file. So, load in the config file as a json and add it to your AlgoStrategy class.
        #         """
        #         parsed_config = json.loads(game_state_string)
        #         self.on_game_start(parsed_config)
        #     elif "turnInfo" in game_state_string:
        #         state = json.loads(game_state_string)
        #         stateType = int(state.get("turnInfo")[0])
        #         if stateType == 0:

        #             """
        #             This is the game turn game state message. Algo must now print to stdout 2 lines, one for build phase one for
        #             deploy phase. Printing is handled by the provided functions.
        #             """

        #             game_state = GameState(self.config, game_state_string)

        #             action = np.array(self.action_space.sample())
        #             debug_write(action)
        #             def gen_plan(action): 
        #                 invalid = 0
        #                 valid = 0
        #                 struct_queue = []
        #                 mobile_queue = []

        #                 game_state.suppress_warnings(True)
        #                 for y in range(game_state.game_map.HALF_ARENA): # goes through every coordinate point
        #                     for x in range(game_state.game_map.ARENA_SIZE):
        #                         if action[y,x] >= 0:
        #                             if action[y,x] == game_state.get_positions()[y,x,0] and action[y,x] < 3: # if structure exists
        #                                 struct_queue.append((action[y,x],x,y,True))
        #                             elif not game_state.can_spawn(UNIT_TYPES[action[y,x] % self.NUMUNITS],
        #                                                           [x,y],
        #                                                           math.ceil(float(action[y,x])/self.NUMUNITS)):
        #                                 invalid += 1
        #                             else:
        #                                 if action[y,x] < 3: # if structure
        #                                     struct_queue.append((action[y,x],x,y,False))
        #                                 else: # if mobile
        #                                     val = action[y,x]
        #                                     for _ in range(math.ceil(float(val)/self.NUMUNITS)):
        #                                         mobile_queue.append((val%self.NUMUNITS,x,y))
        #                 game_state.suppress_warnings(False)

        #                 random.shuffle(struct_queue)
        #                 random.shuffle(mobile_queue)
                        
        #                 sp = np.array([game_state.type_cost(UNIT_TYPES[unit[0]],unit[3])[0]
        #                                     for unit in struct_queue]).sum() # Structure Points
        #                 mp = np.array([game_state.type_cost(UNIT_TYPES[unit[0]])[1]
        #                                     for unit in struct_queue]).sum() # Mobile Points

        #                 budget = game_state.get_resources()

        #                 while (len(struct_queue) > 0 and sp > budget[0]):
        #                     sp -= game_state.type_cost(UNIT_TYPES[struct_queue.pop(0)[0]])[0]
        #                     invalid += 1
        #                 while len(mobile_queue) > 0 and mp > budget[1]:
        #                     mp -= game_state.type_cost(UNIT_TYPES[mobile_queue.pop(0)[0]])[1]
        #                     invalid += 1

        #                 valid = len(struct_queue) + len(mobile_queue)

        #                 return valid, invalid, struct_queue, mobile_queue

        #             valid, invalid, structures, mobiles = gen_plan(action)

        #             self.on_turn(game_state_string, structures, mobiles)

        #         elif stateType == 1:
        #             """
        #             If stateType == 1, this game_state_string string represents a single frame of an action phase
        #             """
        #             self.on_action_frame(game_state_string)
        #         elif stateType == 2:
        #             """
        #             This is the end game message. This means the game is over so break and finish the program.
        #             """
        #             debug_write("Got end state, game over. Stopping algo.")
        #             done = True
        #             break
        #         else:
        #             """
        #             Something is wrong? Received an incorrect or improperly formatted string.
        #             """
        #             debug_write("Got unexpected string with turnInfo: {}".format(
        #                 game_state_string))
        #     else:
        #         """
        #         Something is wrong? Received an incorrect or improperly formatted string.
        #         """
        #         debug_write("Got unexpected string : {}".format(game_state_string))

        # return obs_space, reward, done, None    

    def step(self, action=None, model=None):
        """ 
        Start the parsing loop.
        After starting the algo, it will wait until it recieves information from the game 
        engine, proccess this information, and respond if needed to take it's turn. 
        The algo continues this loop until it recieves the "End" turn message from the game.
        """
        debug_write(BANNER_TEXT)

        done = False
        reward = 2
        obs_space = self.state

        # Note: Python blocks and hangs on stdin. Can cause issues if connections aren't setup properly and may need to
        # manually kill this Python program.
        game_state_string = get_command()
        if "replaySave" in game_state_string:
            """
            This means this must be the config file. So, load in the config file as a json and add it to your AlgoStrategy class.
            """
            parsed_config = json.loads(game_state_string)
            self.on_game_start(parsed_config)
        elif "turnInfo" in game_state_string:
            state = json.loads(game_state_string)
            stateType = int(state.get("turnInfo")[0])
            if stateType == 0:
                """
                This is the game turn game state message. Algo must now print to stdout 2 lines, one for build phase one for
                deploy phase. Printing is handled by the provided functions.
                """

                game_state = GameState(self.config, game_state_string)
                self.state['board'] = game_state.get_positions()
                self.state['p1Stats'] = [game_state.get_resource(0,0),
                                         game_state.get_resource(1,0),
                                         game_state.my_health]
                self.state['p2Stats'] = [game_state.get_resource(0,1),
                                         game_state.get_resource(1,1),
                                         game_state.enemy_health]

                action = np.array(self.action_space.sample())
                def gen_plan(action): 
                    invalid = 0
                    valid = 0
                    struct_queue = []
                    mobile_queue = []

                    game_state.suppress_warnings(True)
                    for y in range(action.shape[0]): # goes through every coordinate point
                        for x in range(action.shape[1]):
                            if action[x,y] >= 0:
                                if action[x,y] == game_state.get_positions()[x,y,0] and action[x,y] < 3: # if structure exists
                                    struct_queue.append((action[x,y],x,y,True))
                                elif not game_state.can_spawn(UNIT_TYPES[action[x,y] % self.NUMUNITS],
                                                                [x,y],
                                                                math.ceil(float(action[x,y])/self.NUMUNITS)):
                                    invalid += 1
                                else:
                                    if action[x,y] < 3: # if structure
                                        struct_queue.append((action[x,y],x,y,False))
                                    else: # if mobile
                                        val = action[x,y]
                                        for _ in range(math.ceil(float(val)/self.NUMUNITS)):
                                            mobile_queue.append((val%self.NUMUNITS,x,y))
                    game_state.suppress_warnings(False)

                    random.shuffle(struct_queue)
                    random.shuffle(mobile_queue)
                    
                    sp = np.array([game_state.type_cost(UNIT_TYPES[unit[0]],unit[3])[0]
                                        for unit in struct_queue]).sum() # Structure Points
                    mp = np.array([game_state.type_cost(UNIT_TYPES[unit[0]])[1]
                                        for unit in struct_queue]).sum() # Mobile Points

                    budget = game_state.get_resources()

                    while (len(struct_queue) > 0 and sp > budget[0]):
                        sp -= game_state.type_cost(UNIT_TYPES[struct_queue.pop(0)[0]])[0]
                        invalid += 1
                    while len(mobile_queue) > 0 and mp > budget[1]:
                        mp -= game_state.type_cost(UNIT_TYPES[mobile_queue.pop(0)[0]])[1]
                        invalid += 1

                    valid = len(struct_queue) + len(mobile_queue)

                    return valid, invalid, struct_queue, mobile_queue

                valid, invalid, structures, mobiles = gen_plan(action)

                self.on_turn(game_state_string, structures, mobiles)

                obs_space = self.dict_to_space()
                reward = valid - invalid + 2*game_state.turn_number + 5*(game_state.my_health - game_state.enemy_health)

            elif stateType == 1:
                """
                If stateType == 1, this game_state_string string represents a single frame of an action phase
                """
                self.on_action_frame(game_state_string)
            elif stateType == 2:
                """
                This is the end game message. This means the game is over so break and finish the program.
                """
                debug_write("Got end state, game over. Stopping algo.")
                done = True
            else:
                """
                Something is wrong? Received an incorrect or improperly formatted string.
                """
                debug_write("Got unexpected string with turnInfo: {}".format(
                    game_state_string))
        else:
            """
            Something is wrong? Received an incorrect or improperly formatted string.
            """
            debug_write("Got unexpected string : {}".format(game_state_string)) 

        return obs_space, reward, done, None

    def starter_strategy(self, game_state):
        """
        For defense we will use a spread out layout and some interceptors early on.
        We will place turrets near locations the opponent managed to score on.
        For offense we will use long range demolishers if they place stationary units near the enemy's front.
        If there are no stationary units to attack in the front, we will send Scouts to try and score quickly.
        """
        # First, place basic defenses
        self.build_defences(game_state)
        # Now build reactive defenses based on where the enemy scored
        self.build_reactive_defense(game_state)

        if game_state.turn_number >= 2:
            gamelib.debug_write('pre', game_state.get_positions()[8, 8])
            if (game_state.attempt_spawn(WALL, [8, 8], 1) > 0):
                gamelib.debug_write('spawned wall (8,8)')
                gamelib.debug_write('post', game_state.get_positions()[8, 8])

        # If the turn is less than 5, stall with interceptors and wait to see enemy's base
        if game_state.turn_number < 5:
            self.stall_with_interceptors(game_state)
        else:
            # Now let's analyze the enemy base to see where their defenses are concentrated.
            # If they have many units in the front we can build a line for our demolishers to attack them at long range.
            if self.detect_enemy_unit(game_state, unit_type=None, valid_x=None, valid_y=[14, 15]) > 10:
                self.demolisher_line_strategy(game_state)
            else:
                # They don't have many units in the front so lets figure out their least defended area and send Scouts there.

                # Only spawn Scouts every other turn
                # Sending more at once is better since attacks can only hit a single scout at a time
                if game_state.turn_number % 2 == 1:
                    # To simplify we will just check sending them from back left and right
                    scout_spawn_location_options = [[13, 0], [14, 0]]
                    best_location = self.least_damage_spawn_location(
                        game_state, scout_spawn_location_options)
                    game_state.attempt_spawn(SCOUT, best_location, 1000)

                # Lastly, if we have spare SP, let's build some supports
                support_locations = [[13, 2], [14, 2], [13, 3], [14, 3]]
                game_state.attempt_spawn(SUPPORT, support_locations)

    def build_defences(self, game_state):
        """
        Build basic defenses using hardcoded locations.
        Remember to defend corners and avoid placing units in the front where enemy demolishers can attack them.
        """
        # Useful tool for setting up your base locations: https://www.kevinbai.design/terminal-map-maker
        # More community tools available at: https://terminal.c1games.com/rules#Download

        # Place turrets that attack enemy units
        turret_locations = [[0, 13], [27, 13], [
            8, 11], [19, 11], [13, 11], [14, 11]]
        # attempt_spawn will try to spawn units if we have resources, and will check if a blocking unit is already there
        game_state.attempt_spawn(TURRET, turret_locations)

        # Place walls in front of turrets to soak up damage for them
        wall_locations = [[8, 12], [19, 12]]
        game_state.attempt_spawn(WALL, wall_locations)
        # upgrade walls so they soak more damage
        game_state.attempt_upgrade(wall_locations)

    def build_reactive_defense(self, game_state):
        """
        This function builds reactive defenses based on where the enemy scored on us from.
        We can track where the opponent scored by looking at events in action frames 
        as shown in the on_action_frame function
        """
        for location in self.scored_on_locations:
            # Build turret one space above so that it doesn't block our own edge spawn locations
            build_location = [location[0], location[1]+1]
            game_state.attempt_spawn(TURRET, build_location)

    def stall_with_interceptors(self, game_state):
        """
        Send out interceptors at random locations to defend our base from enemy moving units.
        """
        # We can spawn moving units on our edges so a list of all our edge locations
        friendly_edges = game_state.game_map.get_edge_locations(
            game_state.game_map.BOTTOM_LEFT) + game_state.game_map.get_edge_locations(game_state.game_map.BOTTOM_RIGHT)

        # Remove locations that are blocked by our own structures
        # since we can't deploy units there.
        deploy_locations = self.filter_blocked_locations(
            friendly_edges, game_state)

        # While we have remaining MP to spend lets send out interceptors randomly.
        while game_state.get_resource(MP) >= game_state.type_cost(INTERCEPTOR)[MP] and len(deploy_locations) > 0:
            # Choose a random deploy location.
            deploy_index = random.randint(0, len(deploy_locations) - 1)
            deploy_location = deploy_locations[deploy_index]

            game_state.attempt_spawn(INTERCEPTOR, deploy_location)
            """
            We don't have to remove the location since multiple mobile 
            units can occupy the same space.
            """

    def demolisher_line_strategy(self, game_state):
        """
        Build a line of the cheapest stationary unit so our demolisher can attack from long range.
        """
        # First let's figure out the cheapest unit
        # We could just check the game rules, but this demonstrates how to use the GameUnit class
        stationary_units = [WALL, TURRET, SUPPORT]
        cheapest_unit = WALL
        for unit in stationary_units:
            unit_class = gamelib.GameUnit(unit, game_state.config)
            if unit_class.cost[game_state.MP] < gamelib.GameUnit(cheapest_unit, game_state.config).cost[game_state.MP]:
                cheapest_unit = unit

        # Now let's build out a line of stationary units. This will prevent our demolisher from running into the enemy base.
        # Instead they will stay at the perfect distance to attack the front two rows of the enemy base.
        for x in range(27, 5, -1):
            game_state.attempt_spawn(cheapest_unit, [x, 11])

        # Now spawn demolishers next to the line
        # By asking attempt_spawn to spawn 1000 units, it will essentially spawn as many as we have resources for
        game_state.attempt_spawn(DEMOLISHER, [24, 10], 1000)

    def least_damage_spawn_location(self, game_state, location_options):
        """
        This function will help us guess which location is the safest to spawn moving units from.
        It gets the path the unit will take then checks locations on that path to 
        estimate the path's damage risk.
        """
        damages = []
        # Get the damage estimate each path will take
        for location in location_options:
            path = game_state.find_path_to_edge(location)
            damage = 0
            for path_location in path:
                # Get number of enemy turrets that can attack each location and multiply by turret damage
                damage += len(game_state.get_attackers(path_location, 0)) * \
                    gamelib.GameUnit(TURRET, game_state.config).damage_i
            damages.append(damage)

        # Now just return the location that takes the least damage
        return location_options[damages.index(min(damages))]

    def detect_enemy_unit(self, game_state, unit_type=None, valid_x=None, valid_y=None):
        total_units = 0
        for location in game_state.game_map:
            if game_state.contains_stationary_unit(location):
                for unit in game_state.game_map[location]:
                    if unit.player_index == 1 and (unit_type is None or unit.unit_type == unit_type) and (valid_x is None or location[0] in valid_x) and (valid_y is None or location[1] in valid_y):
                        total_units += 1
        return total_units

    def filter_blocked_locations(self, locations, game_state):
        filtered = []
        for location in locations:
            if not game_state.contains_stationary_unit(location):
                filtered.append(location)
        return filtered

env = GameEnv()
# env.start()
# log_path = os.path.join('Training', 'Logs')
# model = PPO('MultiInputPolicy', env, verbose=1, tensorboard_log=log_path)
# model.learn(total_timesteps=20000)

# save_path = os.path.join('Training', 'Saved Models', 'Save_PPO')
# model.save(save_path)
env.start()