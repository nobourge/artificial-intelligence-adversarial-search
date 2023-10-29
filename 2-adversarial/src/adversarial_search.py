import copy
import datetime
import os
import sys
from typing import List, Tuple

# from loguru import logger
from lle import Action, World
from mdp import MDP, S, A

import auto_indent
# from tests.graph_mdp import GraphMDP
# from ..tests.graph_mdp import GraphMDP
from utils import print_items
from world_mdp import BetterValueFunction, WorldMDP
from anytree import Node, RenderTree
from anytree.exporter import UniqueDotExporter
import cairosvg

# sys.stdout.reconfigure(encoding='utf-8')

sys.stdout = auto_indent.AutoIndent(sys.stdout)


def stock_tree(mdp: MDP[A, S]
               , algorithm: str
                ) -> None:
    """Stocks the tree in a png file"""
    # print("stock_tree()")
    if isinstance(mdp, WorldMDP):
        if not os.path.exists('tree/current/'+algorithm):
            os.makedirs('tree/current/'+algorithm)
        UniqueDotExporter(mdp.root).to_picture("tree/current/"+algorithm+".png")
        print("tree stocked in tree/current/"+algorithm+".png")

        date_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        if not os.path.exists('tree/'+algorithm):
            os.makedirs('tree/'+algorithm)
        UniqueDotExporter(mdp.root).to_picture("tree/"+algorithm+"/"+date_time+".png")
        print("tree stocked in tree/"+algorithm+"/"+date_time+".png")
    # picture to svg:
    # png_path = "mdp_root.png"
    # svg_path = "mdp_root.svg"
    # cairosvg.png2svg(png_path, svg_path)

    # UniqueDotExporter(mdp.root).to_picture("mdp_root.svg") #todo render

def transition(mdp: MDP[A, S]
                , state: S
                , action: A
                , depth: int = 0
                ) -> S:
    """Returns the state reached by performing the given action in the given state."""
    # if isinstance(mdp, WorldMDP):
    if isinstance(mdp, BetterValueFunction):
        new_state = mdp.transition(state
                                , action
                                , depth
                                )
    else:
        new_state = mdp.transition(state
                                , action
                                )
    if isinstance(mdp, WorldMDP):
        if mdp.was_visited(new_state):
            # print("was visited")
            # print("visited states:")
            # print_items(mdp.visited)
            # continue
            raise ValueError("was visited")
        mdp.add_to_visited(new_state)
        new_state_string = new_state.to_string()
        mdp.nodes[new_state_string] = Node(new_state_string, parent=mdp.nodes[state.to_string()])
    
    return new_state
            
        
def _max(mdp: MDP[A, S]
         , state: S
         , max_depth: int
         , depth: int = 0 
         ) -> Tuple[float, A]:
    """Returns the value of the state and the action that maximizes it."""
    # print("_max()")
    # print(f"max_depth: {max_depth}")
    # print(f"current_agent: {state.current_agent}")

    if mdp.is_final(state) or depth == max_depth :
        return state.value, None
    best_value = float('-inf')
    best_action = None
    # input_state = copy.deepcopy(state) # TypeError: cannot pickle 'builtins.Action' object
    mdp_available_actions = mdp.available_actions(state)
    # mdp_available_actions = get_available_actions_ordered(mdp, state) # test_minimax_two_agents - assert North in [South, Stay]
    # print(f"mdp_available_actions: {mdp_available_actions}")
    for action in mdp_available_actions: #todo reverse order
        # print(f"action: {action}")
        # print(f"state.current_agent: {state.current_agent}")
        # if isinstance(mdp, WorldMDP):
        #     print(f"state.world.agents_positions[state.current_agent]: {state.world.agents_positions[state.current_agent]}")
        # print(f"state.agents_positions: {state.agents_positions}")
        try:
            new_state = transition(mdp
                                    , state
                                    , action
                                    , depth
                                    )
        except ValueError:
            continue
        if new_state.current_agent == 0:
            value, _ = _max(mdp, new_state, max_depth, depth + 1)
        else:
            value = _min(mdp, new_state, max_depth, depth + 1)
        # print(f"depth: {depth}")
        # print(f"best_value: {best_value}")
        # print(f"value: {value}")
        if value > best_value:
            best_value = value
            # print(f"best_value between {best_value} and {value}: {best_value}")
            best_action = action
            # print(f"best_action: {best_action}")
        if isinstance(mdp, WorldMDP):
            mdp.add_value_to_node(new_state
                                  , value
                                  , "best"
                                  )
            mdp.remove_from_visited(new_state)
    return best_value, best_action

def _min(mdp: MDP[A, S]
         , state: S
         , max_depth: int
         , depth: int = 0
         ) -> float:
    """Returns the worst value of the state."""
    # print("_min()")
    # print(f"max_depth: {max_depth}")
    # print(f"current_agent: {state.current_agent}")
    if mdp.is_final(state) or depth == max_depth:
        return state.value
    worst_value = float('inf')
    
    for action in mdp.available_actions(state):
    # for action in get_available_actions_ordered(mdp, state):
        # print(f"action: {action}")
        # print(f"state.current_agent: {state.current_agent}")
        # new_state = mdp.transition(copy.deepcopy(state), action)
        try:
            new_state = transition(mdp
                                    , state
                                    , action
                                    , depth
                                    )
        except ValueError:
            continue
        if new_state.current_agent == 0:
            value, _ = _max(mdp, new_state, max_depth, depth + 1)
        else:
            value = _min(mdp, new_state, max_depth, depth)
        # print(f"depth: {depth}")
        # print(f"worst_value: {worst_value}")
        # print(f"value: {value}")
        worst_value = min(worst_value, value)
        # print(f"worst_value between {worst_value} and {value}: {worst_value}")
        if isinstance(mdp, WorldMDP):
            mdp.add_value_to_node(new_state
                                    , value
                                    , "worst"
                                    )
            mdp.remove_from_visited(new_state)
    return worst_value

def minimax(mdp: MDP[A, S]
            , state: S
            , max_depth: int) -> A:
    """Returns the action to be performed by Agent 0 in the given state. 
    This function only accepts 
    states where it's Agent 0's turn to play 
    and raises a ValueError otherwise. 
    if 2 agents, 
        minimax() is used.
        It is suggested to divide this algorithm into 
        a _min function and 
        a _max function. 
    if 3 agents or more,
        max_n() is used.
    Don't forget that there may be more than one opponent"""

    if state.current_agent != 0:
        raise ValueError("It's not Agent 0's turn to play")
    # if isinstance(mdp, GraphMDP):
    #     print("GraphMDP")
    if isinstance(mdp, WorldMDP):
        # print("WorldMDP")
        new_state_string = state.to_string()
        mdp.root = Node(new_state_string)
        mdp.nodes[new_state_string] = mdp.root
        # print_items(mdp.nodes)
    value, action = _max(mdp, state, max_depth, 0)
    # print(f"action: {action}")
    # print(f"value: {value}")
    stock_tree(mdp, "minimax")
    

    return action




def _alpha_beta_max(mdp: MDP[A, S]
                    , state: S
                    , alpha: float
                    , beta: float
                    , max_depth: int
                    , depth: int = 0
                    # ) -> Tuple[float, A]:
                    ) -> Tuple[float, A, float, float]:
    if mdp.is_final(state) or depth == max_depth:
        return state.value, None
    best_value = float('-inf')
    best_action = None
    # available_actions = get_available_actions_ordered(mdp, state)
        # print(f"available_actions_ordered: {available_actions_ordered}")
    available_actions = mdp.available_actions(state)
    for action in available_actions:
    # for action in list(reversed(mdp.available_actions(state))): #todo FAILED tests/test_alpha_beta.py::test_alpha_beta_graph_mdp - assert 10 == 9
        try:
            new_state = transition(mdp
                                    , state
                                    , action
                                    , depth
                                    )
        except ValueError:
            continue

        if new_state.current_agent == 0:
            # value, _, _, _ = _alpha_beta_max(mdp
            value, action = _alpha_beta_max(mdp
                                       , new_state
                                       , alpha
                                       , beta
                                       , max_depth
                                       , depth + 1
                                       )
        else:
            value = _alpha_beta_min(mdp
                                    , new_state
                                    , alpha
                                    , beta
                                    , max_depth
                                    , depth + 1
                                    )
        if value > best_value:
            best_value = value
            # best_value_vector = value_vector
            best_action = action
        if isinstance(mdp, WorldMDP):
            mdp.add_value_to_node(new_state
                                    , value
                                    , "best"
                                    , alpha
                                    , beta
                                    )
            mdp.remove_from_visited(new_state)
        # alpha = max(alpha, best_value)  # Update alpha before cutoff: fail soft #todo tests\test_alpha_beta.py:131: AssertionError
        if beta <= best_value:  # Beta cutoff
        # if beta <= value:  # Beta cutoff
            if isinstance(mdp, WorldMDP):
                print("beta cutoff from state: ", new_state.to_string())
            return best_value, best_action
        alpha = max(alpha, best_value)  # Update alpha after cutoff: fail hard

    print(f"best_value: {best_value}")
    print(f"best_action: {best_action}")
    print(f"alpha: {alpha}")
    print(f"beta: {beta}")

    # return best_value, best_action, alpha, beta
    return best_value, best_action

def _alpha_beta_min(mdp: MDP[A, S]
                    , state: S
                    , alpha: float
                    , beta: float
                    , max_depth: int
                    , depth: int = 0
                    # ) -> float:
                    ) -> Tuple[float, A, float, float]:
    if mdp.is_final(state) or depth == max_depth:
        # if isinstance(mdp, WorldMDP):
        #     mdp.add_value_to_node(state
        #                             , state.value
        #                             , "worst"
        #                             , alpha
        #                             , beta
        #                             , "FINAL"
        #                             )
        return state.value
    worst_value = float('inf')
    # available_actions = get_available_actions_ordered(mdp, state)
        # print(f"available_actions_ordered: {available_actions_ordered}")
    available_actions = mdp.available_actions(state)
    for action in available_actions:
    # for action in list(reversed(mdp.available_actions(state))): #todo FAILED tests/test_alpha_beta.py::test_alpha_beta_graph_mdp - assert 10 == 9
    # FAILED tests/test_alpha_beta.py::test_alpha_beta_two_agents - assert 44 <= 30
    # FAILED tests/test_alpha_beta.py::test_three_agents2 - assert West == South
        try:
            new_state = transition(mdp
                                    , state
                                    , action
                                    , depth
                                    )
        except ValueError:
            continue
        
        if new_state.current_agent == 0:
            # value, _, _, _ = _alpha_beta_max(mdp
            value, _ = _alpha_beta_max(mdp
                                       , new_state
                                       , alpha
                                       , beta
                                       , max_depth
                                       , depth + 1
                                       )
        else:
            # value, _, _, _ = _alpha_beta_min(mdp
            value = _alpha_beta_min(mdp
                                    , new_state
                                    , alpha
                                    , beta
                                    , max_depth
                                    , depth)
        worst_value = min(worst_value, value)
        if isinstance(mdp, WorldMDP):
            mdp.add_value_to_node(new_state
                                    , value
                                    , "worst"
                                    , alpha
                                    , beta
                                    )
            mdp.remove_from_visited(new_state)
        # beta = min(beta, worst_value)  # Update beta before cutoff: fail soft #todo tests\test_alpha_beta.py:131: AssertionError
        if worst_value <= alpha:  # Alpha cutoff
        # if value <= alpha:  # Alpha cutoff
            if isinstance(mdp, WorldMDP):
                print("alpha cutoff from state: ", new_state.to_string())
            return worst_value
        beta = min(beta, worst_value)  # Update beta after cutoff: fail hard

    # return worst_value, worst_action, alpha, beta
    return worst_value

def alpha_beta(mdp: MDP[A, S]
               , state: S
               , max_depth: int) -> A: # todo good node ordering reduces time complexity to O(b^m/2)
    """The alpha-beta pruning algorithm 
    is an improvement over 
    minimax 
    that allows for pruning of the search tree."""
    # todo In maxn (Luckhardt and Irani, 1986), 
    # the extension of minimax to multi-player games
    # , pruning is not as successful.
    if state.current_agent != 0:
        raise ValueError("It's not Agent 0's turn to play")
    alpha = float('-inf')
    beta = float('inf')
    if isinstance(mdp, WorldMDP):
        new_state_string = state.to_string()
        mdp.root = Node(new_state_string)
        mdp.nodes[new_state_string] = mdp.root
    
    # value, action, _, _ = _alpha_beta_max(mdp
    value, action = _alpha_beta_max(mdp
                                , state
                                , alpha
                                , beta
                                , max_depth
                                , 0
                                )
    if isinstance(mdp, WorldMDP):
        mdp.root.value = value
        mdp.add_value_to_node(state
                                , value
                                , "best"
                                , alpha
                                , beta
                                )
    if isinstance(mdp, BetterValueFunction):
        stock_tree(mdp, "alpha_beta/better_value_function")
    elif isinstance(mdp, WorldMDP) and not isinstance(mdp, BetterValueFunction):
        stock_tree(mdp, "alpha_beta")

    return action

def _expectimax_max(mdp: MDP[A, S]
                    , state: S
                    , max_depth: int
                    , depth: int = 0
                    ) -> Tuple[float, A]:
    if mdp.is_final(state) or depth == max_depth:
        return state.value, None
    
    best_value = float('-inf')
    best_action = None
    
    for action in mdp.available_actions(state):
        
        try:
            new_state = transition(mdp
                                    , state
                                    , action
                                    , depth
                                    )
        except ValueError:
            continue
        
        value = _expectimax_exp(mdp
                                , new_state
                                , max_depth 
                                , depth + 1
                                )
        if value > best_value:
            best_value = value
            best_action = action
        if isinstance(mdp, WorldMDP):
            mdp.add_value_to_node(new_state
                                    , value
                                    , "max"
                                    )
            mdp.remove_from_visited(new_state)
    
    return best_value, best_action

def _expectimax_exp(mdp: MDP[A, S]
                    , state: S
                    , max_depth: int
                    , depth: int = 0
                    ) -> float:
    """Returns the expected value of the state.
    The expected value of a state is
    the average value of the state
    after all possible actions are performed.
    """
    if mdp.is_final(state) or depth == max_depth:
        return state.value
    
    total_value = 0
    num_actions = len(mdp.available_actions(state))
    
    for action in mdp.available_actions(state):
        try:
            new_state = transition(mdp
                                    , state
                                    , action
                                    , depth
                                    )
        except ValueError:
            continue
        value, _ = _expectimax_max(mdp
                                   , new_state
                                   , max_depth
                                   , depth + 1
                                   )
        total_value += value
        if isinstance(mdp, WorldMDP):
            mdp.add_value_to_node(new_state
                                    , value
                                    , "exp"
                                    )
            mdp.remove_from_visited(new_state)
    
    expected_value = total_value / num_actions if num_actions != 0 else 0
    return expected_value

def expectimax(mdp: MDP[A, S]
               , state: S
               , max_depth: int
               ) -> Action:
    """ The 'expectimax' algorithm allows for 
    modeling the probabilistic behavior of humans 
    who might make suboptimal choices. 
    The nature of expectimax requires that we know 
    the probability that the opponent will take each action. 
    Here, we will assume that 
    the other agents take actions that are uniformly random."""
    if state.current_agent != 0:
        raise ValueError("It's not Agent 0's turn to play")
    
    if isinstance(mdp, WorldMDP):
        new_state_string = state.to_string()
        mdp.root = Node(new_state_string)
        mdp.nodes[new_state_string] = mdp.root

    _, action = _expectimax_max(mdp
                                , state
                                , max_depth
                                , 0
                                )
    stock_tree(mdp, "expectimax")
    return action

#main
if __name__ == "__main__":

    # # world = WorldMDP(
    # world = BetterValueFunction(
    #     World(
    #         """
    #     S0 X
    #     """
    #     )
    # )
    # # action = alpha_beta(world, world.reset(), 1)
    # action = minimax(world, world.reset(), 3)
    # print(f"action: {action}")
    # # n_expanded_states
    # print(f"n_expanded_states: {world.n_expanded_states}")

    # world = WorldMDP(
    world = BetterValueFunction(
            # World(
            #     """
            # .  L1W
            # S0 X
            # """
            # )
            #  World(
            #     """
            # .  L1W
            # S0 S1
            # X  X"""
            # )
               World(
            """
          S0 X  .
        G  @  .
        G  X  .
        """
            )
        )
    # action = minimax(world, world.reset(), 5)
    action = alpha_beta(world, world.reset(), 2)
    # action = alpha_beta(world, world.reset(), 1)
    # assert action in [Action.SOUTH, Action.STAY]
    # assert action in [Action.SOUTH]

    world = WorldMDP(
            # World(
            #     """
            # .  L1W
            # S0 X
            # """
            # )
            #  World(
            #     """
            # .  L1W
            # S0 S1
            # X  X"""
            # )
               World(
              """
        S0 X  .
        G  @  .
        G  X  .
        """
            )
        )
    # action = minimax(world, world.reset(), 5)
    # action = alpha_beta(world, world.reset(), 1)
    action = alpha_beta(world, world.reset(), 2)
    # assert action in [Action.SOUTH, Action.STAY]
    # assert action in [Action.SOUTH]
