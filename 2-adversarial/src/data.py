from lle import World
from adversarial_search import minimax, alpha_beta, expectimax
from utils import print_items
from world_mdp import BetterValueFunction, WorldMDP

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
        S0 G  .  X
        .  .  .  .
        X L1N S1 .
        .  G  .  .
        @  @  @  .
        S2 .  .  X
        """        
# execute the 3 adversarial search algorithms on the 3 maps, 
# , and for each map, compare the number of nodes extended during the search for 
# minimax
# , minimax with better value function
#  alpha_beta 
# alpha_beta with better value function
# and expectimax 

def compare_adversarial_search_algorithms():
    """# execute the 3 adversarial search algorithms on the 3 maps, 
        # , and for each map, compare the number of nodes extended during the search for 
        # minimax
        # , minimax with better value function
        #  alpha_beta 
        # alpha_beta with better value function
        # and expectimax """

    for map in [map1
                , map2
                , map3
                ]:
        world = World(map)
        print("map: ", map)
        depth = (world.width + world.height)//2
        world.reset()
        mdp = WorldMDP(world)
        minimax(mdp, mdp.reset(), depth)
        print("minimax: ", mdp.n_expanded_states)
        world.reset()
        mdp = BetterValueFunction(world)
        minimax(mdp, mdp.reset(), depth)
        print("minimax with better value function: ", mdp.n_expanded_states)
        world.reset()
        mdp = WorldMDP(world)
        alpha_beta(mdp, mdp.reset(), depth)  
        print("alpha_beta: ", mdp.n_expanded_states)
        world.reset()
        mdp = BetterValueFunction(world)
        alpha_beta(mdp, mdp.reset(), depth)
        print("alpha_beta with better value function: ", mdp.n_expanded_states)
        world.reset()
        mdp = WorldMDP(world)
        expectimax(mdp, mdp.reset(), depth)
        print("expectimax: ", mdp.n_expanded_states)

compare_adversarial_search_algorithms()

