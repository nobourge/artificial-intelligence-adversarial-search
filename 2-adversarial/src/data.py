from cmath import log
from math import sqrt
from lle import World
from adversarial_search import minimax, alpha_beta, expectimax
from utils import print_items
from world_mdp import BetterValueFunction, WorldMDP
import datetime
from map_generator import MapGenerator


def print_in_file(*args):
    """Prints in file"""
    with open("output.txt", "a") as file:
        print(*args, file=file)

# map on which minimax and alpha_beta differ the least
map1 = """
        S0 X
        """
map2 = """
        S0 G  .  X
        .  .  .  .
        X L1N S1 .
        """
# map on which minimax and alpha_beta differ the most
map3 = """
        S0 G  X  .  .  .  .
        X L1N S1 .  .  .  .
        .  .  .  .  .  .  .
        """        

map4 = """
        S0 G  S1
        G  @  X
        G  G  X
        """

map5 = """
        S0 X  .
        G  @  .
        X  .  .
        """
# execute the 3 adversarial search algorithms on the 3 maps, 
# , and for each map, compare the number of nodes extended during the search for 
# minimax
# , minimax with better value function
#  alpha_beta 
# alpha_beta with better value function
# and expectimax     

    
def compare_adversarial_search_algorithms():
    """
    # execute the 3 adversarial search algorithms on the 3 maps, 
        # , and for each map, compare the number of nodes extended during the search for 
        # minimax
        # , minimax with better value function
        #  alpha_beta 
        # alpha_beta with better value function
        # and expectimax """



    # date time
    now = datetime.datetime.now()
    print_in_file("date time: ", now.strftime("%Y-%m-%d %H:%M:%S"))

    generator = MapGenerator()

    for map in [map1
                # , map2
                # , map3
                , generator.generate_map_str()
                ]:
        now = datetime.datetime.now()
        date_time = now.strftime("%Y_%m_%d_%H_%M_%S")
        world = World(map)
        # print("map: \n", map)
        print("map: \n")
        print(map)
        print_in_file("map"+ date_time + " = "+ "\"\"\""+map+"\"\"\"")
        depth = 1+ int(sqrt(world.width + world.height))
        # depth = 1+ log(world.width + world.height)
        # depth = (world.width + world.height)//3
        print_in_file("results"+ date_time + " = {")
        print("depth: ", depth)
        print_in_file("\"depth\": ", depth,",")

        world.reset()
        mdp = WorldMDP(world)
        action_minimax = minimax(mdp, mdp.reset(), depth)
        print("minimax: ", mdp.n_expanded_states)
        print_in_file("\"minimax\": ", mdp.n_expanded_states,",")
        world.reset()
        mdp = BetterValueFunction(world)
        action_minimax_with_better_value_function = minimax(mdp, mdp.reset(), depth)
        print("minimax_with_better_value_function: ", mdp.n_expanded_states)
        print_in_file("\"minimax_with_better_value_function\": ", mdp.n_expanded_states,",")
        world.reset()
        mdp = WorldMDP(world)
        action_alpha_beta = alpha_beta(mdp, mdp.reset(), depth)  
        print("alpha_beta: ", mdp.n_expanded_states)
        print_in_file("\"alpha_beta\": ", mdp.n_expanded_states,",")
        world.reset()
        mdp = BetterValueFunction(world)
        action_alpha_beta_better_value_function = alpha_beta(mdp, mdp.reset(), depth)
        print("alpha_beta with better value function: ", mdp.n_expanded_states)
        print_in_file("\"alpha_beta_with_better_value_function\": ", mdp.n_expanded_states,",")
        world.reset()
        mdp = WorldMDP(world)
        action_expectimax = expectimax(mdp, mdp.reset(), depth)
        print("expectimax: ", mdp.n_expanded_states)
        print_in_file("\"expectimax\": ", mdp.n_expanded_states,",")
        print("action_minimax: ", action_minimax)
        print_in_file("\"action_minimax\": ", "\"",str(action_minimax),"\"",",")
        print("action_minimax_with_better_value_function: ", action_minimax_with_better_value_function)
        print_in_file("\"action_minimax_with_better_value_function\": ", "\"",str(action_minimax_with_better_value_function),"\"",",")
        print("action_alpha_beta: ", action_alpha_beta)
        print_in_file("\"action_alpha_beta\": ", "\"",str(action_alpha_beta),"\"",",")
        print("action_alpha_beta_better_value_function: ", action_alpha_beta_better_value_function)
        print_in_file("\"action_alpha_beta_better_value_function\": ", "\"",str(action_alpha_beta_better_value_function),"\"",",")
        print("action_expectimax: ", action_expectimax)
        print_in_file("\"action_expectimax\": ", "\"",str(action_expectimax),"\"",",")
        print_in_file("}")

compare_adversarial_search_algorithms()

