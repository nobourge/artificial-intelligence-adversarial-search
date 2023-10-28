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
        world = World(map)
        print("map: ", map)
        print_in_file("map: ", map)
        depth = (world.width + world.height)//3
        world.reset()
        mdp = WorldMDP(world)
        minimax(mdp, mdp.reset(), depth)
        print("minimax: ", mdp.n_expanded_states)
        print_in_file("minimax: ", mdp.n_expanded_states)
        # world.reset()
        # mdp = BetterValueFunction(world)
        # minimax(mdp, mdp.reset(), depth)
        # print("minimax with better value function: ", mdp.n_expanded_states)
        # print_in_file("minimax with better value function: ", mdp.n_expanded_states)
        world.reset()
        mdp = WorldMDP(world)
        alpha_beta(mdp, mdp.reset(), depth)  
        print("alpha_beta: ", mdp.n_expanded_states)
        print_in_file("alpha_beta: ", mdp.n_expanded_states)
        world.reset()
        mdp = BetterValueFunction(world)
        alpha_beta(mdp, mdp.reset(), depth)
        print("alpha_beta with better value function: ", mdp.n_expanded_states)
        print_in_file("alpha_beta with better value function: ", mdp.n_expanded_states)
        world.reset()
        mdp = WorldMDP(world)
        expectimax(mdp, mdp.reset(), depth)
        print("expectimax: ", mdp.n_expanded_states)
        print_in_file("expectimax: ", mdp.n_expanded_states)

compare_adversarial_search_algorithms()

