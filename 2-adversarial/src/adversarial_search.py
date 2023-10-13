import sys
from typing import Tuple

# from loguru import logger
from lle import Action, World
from mdp import MDP, S, A

import auto_indent
from utils import print_items
from world_mdp import WorldMDP

sys.stdout = auto_indent.AutoIndent(sys.stdout)

def _max(mdp: MDP[A, S]
         , state: S
         , max_depth: int) -> Tuple[float, A]:
    """Returns the value of the state and the action that maximizes it."""
    print("_max()")
    print(f"max_depth: {max_depth}")
    if mdp.is_final(state) or max_depth == 0:
        return state.value, None
    best_value = float('-inf')
    best_action = None
    for action in mdp.available_actions(state):
        new_state = mdp.transition(state, action)
        # value, _ = _min(mdp, new_state, max_depth - 1)
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
    if mdp.is_final(state) or max_depth == 0:
        return state.value
    best_value = float('inf')
    for action in mdp.available_actions(state):
        new_state = mdp.transition(state, action)
        value, _ = _max(mdp, new_state, max_depth - 1)

        # if state.current_agent == 0:
        #     value, _ = _max(mdp, new_state, max_depth - 1)
        # else:
        #     value = _min(mdp, new_state, max_depth - 1)
        best_value = min(best_value, value)
    return best_value


def minimax(mdp: MDP[A, S]
            , state: S
            , max_depth: int) -> A:
    """Returns the action to be performed by Agent 0 in the given state. 
    This function only accepts 
    states where it's Agent 0's turn to play 
    and raises a ValueError otherwise. 
    It is suggested to divide this algorithm into 
    a _min function and 
    a _max function. 
    Don't forget that there may be more than one opponent"""
    if state.current_agent != 0:
        raise ValueError("It's not Agent 0's turn to play")
    _, action = _max(mdp, state, max_depth)
    print(f"action: {action}")
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
    The nature of expectimax requires that we know the probability that the opponent will take each action. 
    Here, we will assume that 
    the other agents take actions that are uniformly random."""
    ...

#main
if __name__ == "__main__":
    
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

    action = minimax(world, world.reset(), 3)
    assert action == Action.WEST

    action = minimax(world, world.reset(), 7)
    assert action == Action.SOUTH
