import copy
import datetime
import sys
from typing import List, Tuple

# from loguru import logger
from lle import Action, World
from mdp import MDP, S, A

import auto_indent
# from ..tests.graph_mdp import GraphMDP
from utils import print_items
from world_mdp import WorldMDP
from anytree import Node, RenderTree
from anytree.exporter import UniqueDotExporter
import cairosvg

# sys.stdout.reconfigure(encoding='utf-8')

sys.stdout = auto_indent.AutoIndent(sys.stdout)

def _max_n(mdp: MDP[A, S]
          , state: S
          , max_depth: int
          ) -> Tuple[List[float], A]:
    if mdp.is_final(state) or max_depth == 0:
        return state.value, None  # Assume state.value_vector is a list of values for each agent
    num_agents = mdp.world.n_agents
    print(f"num_agents: {num_agents}")
    # best_value_vector = [float('-inf') for _ in range(num_agents)]
    best_value = float('-inf')
    best_action = None
    for action in mdp.available_actions(state):
        new_state = mdp.transition(state, action)
        value = _max_n(mdp
                             , new_state
                             , max_depth - 1)[0]
        if value > best_value:
            best_value = value
            best_action = action
    return best_value, best_action

def max_n(mdp: MDP[A, S]
         , state: S
         , max_depth: int) -> A:
    if state.current_agent != 0:
        raise ValueError("It's not Agent 0's turn to play")
    # num_agents = mdp.world.n_agents #todo
    _, action = _max_n(mdp
                       , state
                       , max_depth
                    #    , num_agents
                       )
    return action

def _max(mdp: MDP[A, S]
         , state: S
         , max_depth: int) -> Tuple[float, A]:
    """Returns the value of the state and the action that maximizes it."""
    # print("_max()")
    # print(f"max_depth: {max_depth}")
    # print(f"current_agent: {state.current_agent}")
    # if isinstance(mdp, WorldMDP):
    #     print(f"current_agent position: {state.world.agents_positions[state.current_agent]}")
    if mdp.is_final(state) or max_depth == 0:
        return state.value, None
    best_value = float('-inf')
    best_action = None
    # input_state = copy.deepcopy(state) # TypeError: cannot pickle 'builtins.Action' object
    mdp_available_actions = mdp.available_actions(state)
    # print(f"mdp_available_actions: {mdp_available_actions}")
    for action in mdp_available_actions: #todo reverse order
        # print(f"action: {action}")
        # print(f"state.current_agent: {state.current_agent}")
        # if isinstance(mdp, WorldMDP):
        #     print(f"state.world.agents_positions[state.current_agent]: {state.world.agents_positions[state.current_agent]}")
        # print(f"state.agents_positions: {state.agents_positions}")
        # if isinstance(mdp, WorldMDP):
        #     if state.last_action:
        #         print(f"state.last_action: {state.last_action}")
        #     else:
        #         print(f"state.last_action: None")
        new_state = mdp.transition(state, action)
        if mdp.was_visited(new_state):
            continue
        # if isinstance(mdp, WorldMDP):
        #     #add state to tree
        #     new_state_string = new_state.to_string()
        #     # print_items(mdp.nodes)
        #     # print(f"state.to_string(): {state.to_string()}")
            
        #     mdp.nodes[new_state_string] = Node(new_state_string, parent=mdp.nodes[state.to_string()])        
        
        value = _min(mdp, new_state, max_depth - 1)
        mdp.add_to_visited(new_state)
        if value > best_value:
            best_value = value
            best_action = action
    return best_value, best_action

def _min(mdp: MDP[A, S]
         , state: S
         , max_depth: int) -> float:
    """Returns the worst value of the state."""
    # print("_min()")
    # print(f"max_depth: {max_depth}")
    # print(f"current_agent: {state.current_agent}")
    if mdp.is_final(state) or max_depth == 0:
        return state.value
    best_value = float('inf')
    
    for action in mdp.available_actions(state):
        # print(f"action: {action}")
        # print(f"state.current_agent: {state.current_agent}")
        # new_state = mdp.transition(copy.deepcopy(state), action)
        new_state = mdp.transition(state, action)
        if mdp.was_visited(new_state):
            continue
        # print(f"new_state.current_agent: {new_state.current_agent}")
        # new_state.last_action = action #todo
        # if isinstance(mdp, WorldMDP):
        #     #add state to tree
        #     new_state_string = new_state.to_string()
        #     print(f"new_state_string: {new_state_string}")
        #     print_items(mdp.nodes)
        #     mdp.nodes[new_state_string] = Node(new_state_string, parent=mdp.nodes[state.to_string()])
        
        value, _ = _max(mdp, new_state, max_depth - 1)
        mdp.add_to_visited(new_state)

        # print(f"new_state.current_agent: {new_state.current_agent}")
        # if state.current_agent == 0:
        if new_state.current_agent == 0:
            value, _ = _max(mdp, new_state, max_depth - 1)
        else:
            value = _min(mdp, new_state, max_depth)
        best_value = min(best_value, value)
    return best_value

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
    # if isinstance(mdp, WorldMDP):
    #     print("WorldMDP")
    #     new_state_string = state.to_string()
    #     mdp.root = Node(new_state_string)
    #     mdp.nodes[new_state_string] = mdp.root
    #     print_items(mdp.nodes)
    _, action = _max(mdp, state, max_depth)
    print(f"action: {action}")
    # if isinstance(mdp, WorldMDP):
    #     UniqueDotExporter(mdp.root).to_picture("mdp_root.png")
    # picture to svg:
    # png_path = "mdp_root.png"
    # svg_path = "mdp_root.svg"
    # cairosvg.png2svg(png_path, svg_path)

    # UniqueDotExporter(mdp.root).to_picture("mdp_root.svg") #todo render

    return action

def _alpha_beta_max(mdp: MDP[A, S]
                    , state: S
                    , alpha: float
                    # , alpha_vector: List[float]
                    , beta: float
                    # , beta_vector: List[float]
                    , max_depth: int
                    ) -> Tuple[float, A]:
    if mdp.is_final(state) or max_depth == 0:
        return state.value, None
    best_value = float('-inf')
    # best_value_vector = [float('-inf') for _ in range(len(alpha_vector))]
    best_action = None
    for action in mdp.available_actions(state):
    # for action in list(reversed(mdp.available_actions(state))): #todo FAILED tests/test_alpha_beta.py::test_alpha_beta_graph_mdp - assert 10 == 9
        new_state = mdp.transition(state, action)
        if isinstance(mdp, WorldMDP):
            new_state_string = new_state.to_string()
            mdp.nodes[new_state_string] = Node(new_state_string, parent=mdp.nodes[state.to_string()])

        value = _alpha_beta_min(mdp
        # value_vector = _alpha_beta_min(mdp
                                , new_state
                                , alpha
                                # , alpha_vector
                                , beta
                                # , beta_vector
                                , max_depth - 1)
        if value > best_value:
        # if value_vector > best_value_vector:
            best_value = value
            # best_value_vector = value_vector
            best_action = action
        # alpha = max(alpha, best_value)  # Update alpha before cutoff: fail soft
        # if beta <= alpha:  # Beta cutoff
        if beta <= best_value:  # Beta cutoff
            return best_value, best_action
        alpha = max(alpha, best_value)  # Update alpha after cutoff: fail hard

    return best_value, best_action

    #     for i in range(len(best_value_vector)):
    #         if value_vector[i] > best_value_vector[i]:
    #             best_value_vector[i] = value_vector[i]
    #             best_action = action
                
    #     for i in range(len(alpha_vector)):
    #         alpha_vector[i] = max(alpha_vector[i], best_value_vector[i])
        
    #     if all(beta <= alpha for alpha, beta in zip(alpha_vector, beta_vector)):
    #         break
    # return best_value_vector, best_action

def _alpha_beta_min(mdp: MDP[A, S]
                    , state: S
                    , alpha: float
                    # , alpha_vector: list[float]
                    , beta: float
                    # , beta_vector: list[float]
                    , max_depth: int) -> float:
    if mdp.is_final(state) or max_depth == 0:
        return state.value
        # return state.value_vector
    worst_value = float('inf')
    for action in mdp.available_actions(state):
    # for action in list(reversed(mdp.available_actions(state))): #todo FAILED tests/test_alpha_beta.py::test_alpha_beta_graph_mdp - assert 10 == 9
    # FAILED tests/test_alpha_beta.py::test_alpha_beta_two_agents - assert 44 <= 30
    # FAILED tests/test_alpha_beta.py::test_three_agents2 - assert West == South
        new_state = mdp.transition(state, action)
        if isinstance(mdp, WorldMDP):
            new_state_string = new_state.to_string()
            mdp.nodes[new_state_string] = Node(new_state_string, parent=mdp.nodes[state.to_string()])
        
        if new_state.current_agent == 0:
            value, _ = _alpha_beta_max(mdp
                                       , new_state
                                       , alpha
                                       , beta
                                       , max_depth - 1)
        else:
            value = _alpha_beta_min(mdp
                                    , new_state
                                    , alpha
                                    , beta
                                    # , max_depth - 1)
                                    , max_depth)
        worst_value = min(worst_value, value)
        # beta = min(beta, best_value)  # Update beta before cutoff: fail soft
        # if beta <= alpha:  # Alpha cutoff
        if worst_value <= alpha:  # Alpha cutoff
            return worst_value
        beta = min(beta, worst_value)  # Update beta after cutoff: fail hard
    return worst_value

    # best_value_vector = [float('inf') for _ in range(len(beta_vector))]
    # for action in mdp.available_actions(state):
    #     new_state = mdp.transition(state, action)
    #     if new_state.current_agent == 0:
    #         value_vector, _ = _alpha_beta_max(mdp, new_state, alpha_vector, beta_vector, max_depth - 1)
    #     else:
    #         value_vector = _alpha_beta_min(mdp, new_state, alpha_vector, beta_vector, max_depth - 1)
    #     for i in range(len(best_value_vector)):
    #         if value_vector[i] < best_value_vector[i]:
    #             best_value_vector[i] = value_vector[i]
                
    #     for i in range(len(beta_vector)):
    #         beta_vector[i] = min(beta_vector[i], best_value_vector[i])
        
    #     if all(beta <= alpha for alpha, beta in zip(alpha_vector, beta_vector)):
    #         break
            
    # # return best_value_vector[0]
    # return best_value_vector

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
    if isinstance(mdp, WorldMDP):
        new_state_string = state.to_string()
        mdp.root = Node(new_state_string)
        mdp.nodes[new_state_string] = mdp.root
    
    # alpha_vector = [float('-inf') for _ in range(mdp.n_agents)]
    # beta_vector = [float('inf') for _ in range(mdp.n_agents)]

    _, action = _alpha_beta_max(mdp
                                , state
                                , float('-inf')
                                # , alpha_vector
                                , float('inf')
                                # , beta_vector
                                , max_depth
                                )
    if isinstance(mdp, WorldMDP):
        date_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        UniqueDotExporter(mdp.root).to_picture("alpha_beta_tree"+date_time+".png")

    return action

def _expectimax_max(mdp: MDP[A, S], state: S, max_depth: int) -> Tuple[float, A]:
    if mdp.is_final(state) or max_depth == 0:
        return state.value, None
    
    best_value = float('-inf')
    best_action = None
    
    for action in mdp.available_actions(state):
        new_state = mdp.transition(state, action)
        value = _expectimax_exp(mdp, new_state, max_depth - 1)
        if value > best_value:
            best_value = value
            best_action = action
    
    return best_value, best_action

def _expectimax_exp(mdp: MDP[A, S], state: S, max_depth: int) -> float:
    if mdp.is_final(state) or max_depth == 0:
        return state.value
    
    total_value = 0
    num_actions = len(mdp.available_actions(state))
    
    for action in mdp.available_actions(state):
        new_state = mdp.transition(state, action)
        value, _ = _expectimax_max(mdp, new_state, max_depth - 1)
        total_value += value
    
    expected_value = total_value / num_actions if num_actions != 0 else 0
    return expected_value

def expectimax(mdp: MDP[A, S]
               , state: S
               , max_depth: int) -> Action:
    """ The 'expectimax' algorithm allows for 
    modeling the probabilistic behavior of humans 
    who might make suboptimal choices. 
    The nature of expectimax requires that we know 
    the probability that the opponent will take each action. 
    Here, we will assume that 
    the other agents take actions that are uniformly random."""
    if state.current_agent != 0:
        raise ValueError("It's not Agent 0's turn to play")
    
    _, action = _expectimax_max(mdp, state, max_depth)
    return action

#main
if __name__ == "__main__":
 
    # def test_three_agents2():
    """In this test, Agent 2 should take the gem on top of him
    in order to prevent Agent 0 from getting it, even if Agent 2
    could deny two gems by going left."""
    world = WorldMDP(
        World(
            """
        .  . . . G G S0
        .  . . @ @ @ G
        S2 . . X X X G
        .  . . . G G S1
        """
        )
    )
    # action = alpha_beta(world, world.reset(), 1)
    # print(f"action: {action}")
    # assert action == Action.SOUTH
    # print(f"world.n_expanded_states: {world.n_expanded_states}")
    # assert world.n_expanded_states <= 4

    world.reset()
    action = alpha_beta(world, world.reset(), 3)
    print(f"action: {action}")
    print(f"world.n_expanded_states: {world.n_expanded_states}")
    assert action == Action.WEST
    assert world.n_expanded_states <= 116
