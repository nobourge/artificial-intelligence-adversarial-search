import copy
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
    print("_max()")
    print(f"max_depth: {max_depth}")
    print(f"current_agent: {state.current_agent}")
    print(f"current_agent position: {state.world.agents_positions[state.current_agent]}")
    if mdp.is_final(state) or max_depth == 0:
        return state.value, None
    best_value = float('-inf')
    best_action = None
    # input_state = copy.deepcopy(state) # TypeError: cannot pickle 'builtins.Action' object


    mdp_available_actions = mdp.available_actions(state)
    print(f"mdp_available_actions: {mdp_available_actions}")
    for action in mdp_available_actions: #todo reverse order
        print(f"action: {action}")
        print(f"state.current_agent: {state.current_agent}")
        print(f"state.world.agents_positions[state.current_agent]: {state.world.agents_positions[state.current_agent]}")
        # print(f"state.agents_positions: {state.agents_positions}")
        if state.last_action:
            print(f"state.last_action: {state.last_action}")
        else:
            print(f"state.last_action: None")
        # state_deepcopy = copy.deepcopy(state)
        # new_state = mdp.transition(state_deepcopy, action)
        new_state = mdp.transition(state, action)
        #add state to tree
        new_state_string = new_state.to_string()
        print_items(mdp.nodes)
        print(f"state.to_string(): {state.to_string()}")
        if state.last_action:
            print(f"state.last_action: {state.last_action}")
        else:
            print(f"state.last_action: None")
        mdp.nodes[new_state_string] = Node(new_state_string, parent=mdp.nodes[state.to_string()])        
       
        value = _min(mdp, new_state, max_depth - 1)
        if value > best_value:
            best_value = value
            best_action = action
    return best_value, best_action

def _min(mdp: MDP[A, S]
         , state: S
         , max_depth: int) -> float:
    """Returns the worst value of the state."""
    print("_min()")
    print(f"max_depth: {max_depth}")
    print(f"current_agent: {state.current_agent}")
    if mdp.is_final(state) or max_depth == 0:
        return state.value
    best_value = float('inf')
    
    for action in mdp.available_actions(state):
        print(f"action: {action}")
        print(f"state.current_agent: {state.current_agent}")
        # new_state = mdp.transition(copy.deepcopy(state), action)
        new_state = mdp.transition(state, action)
        print(f"new_state.current_agent: {new_state.current_agent}")
        # new_state.last_action = action #todo
        #add state to tree
        new_state_string = new_state.to_string()
        print(f"new_state_string: {new_state_string}")
        print_items(mdp.nodes)
        mdp.nodes[new_state_string] = Node(new_state_string, parent=mdp.nodes[state.to_string()])
     
        value, _ = _max(mdp, new_state, max_depth - 1)

        print(f"new_state.current_agent: {new_state.current_agent}")
        # if state.current_agent == 0:
        if new_state.current_agent == 0:
            value, _ = _max(mdp, new_state, max_depth - 1)
        else:
            value = _min(mdp, new_state, max_depth - 1)
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
    new_state_string = state.to_string()
    mdp.root = Node(new_state_string)
    mdp.nodes[new_state_string] = mdp.root
    print_items(mdp.nodes)
    _, action = _max(mdp, state, max_depth)
    print(f"action: {action}")
    UniqueDotExporter(mdp.root).to_picture("mdp_root.png")
    # picture to svg:
    # png_path = "mdp_root.png"
    # svg_path = "mdp_root.svg"
    # cairosvg.png2svg(png_path, svg_path)

    # UniqueDotExporter(mdp.root).to_picture("mdp_root.svg") #todo render

    return action


def alpha_beta(mdp: MDP[A, S]
               , state: S
               , max_depth: int) -> A:
    """The alpha-beta pruning algorithm 
    is an improvement over 
    minimax 
    that allows for pruning of the search tree."""

    ...


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
    ...

#main
if __name__ == "__main__":
    
    # world = WorldMDP(
    #     World(
    #         """
    #     .  . . . G G S0
    #     .  . . @ @ @ G
    #     S2 . . X X X G
    #     .  . . . G G S1
    #         """
    #     )
    # )

    # action = minimax(world, world.reset(), 3)
    # assert action == Action.WEST

    # action = minimax(world, world.reset(), 7)
    # assert action == Action.SOUTH

    # def test_two_agents_laser():
    world = WorldMDP(
            World(
                """
            S0 G  .  X
            .  .  .  .
            X L1N S1 .
    """
            )
        )
    # # step
    # # world.world.step([Action.EAST, Action.NORTH])
    # world.transition(world.reset(), Action.EAST)
    # print(world.world.get_state.world_string)
    # # print world
    # print(world.state.world_string)
    # action = minimax(world, world.reset(), 1)
    action = minimax(world, world.reset(), 2)
    # action = minimax(world, world.reset(), 3)


#  mdp = GraphMDP.parse("tests/graphs/vary-depth.graph")
#  assert minimax(mdp, mdp.reset(), 1) == "Right"
#  assert minimax(mdp, mdp.reset(), 2) == "Left"
#  assert minimax(mdp, mdp.reset(), 3) == "Right"
