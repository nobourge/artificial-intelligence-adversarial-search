from typing import List, Tuple
from lle import Action
from mdp import MDP, S, A
from world_mdp import BetterValueFunction, WorldMDP

WORLD_STEP_NOT_POSSIBLE_ERROR = "There is no more step to take"

def transition(mdp: MDP[A, S], state: S, action: A, depth: int = 0) -> S:
    """Returns the state reached by performing the given action in the given state."""
    if isinstance(mdp, BetterValueFunction):
        new_state = mdp.transition(state, action, depth)
    else:
        new_state = mdp.transition(state, action)
    if isinstance(mdp, WorldMDP):
        if mdp.was_visited(new_state):
            raise ValueError("was visited")
        mdp.add_to_visited(new_state)
    return new_state
            
        
def _max(mdp: MDP[A, S], state: S, max_depth: int, depth: int = 0) -> Tuple[float, A]:
    """Returns the value of the state and the action that maximizes it."""
    if mdp.is_final(state) or depth == max_depth :
        return state.value, None
    best_value = float('-inf')
    best_action = None
    mdp_available_actions = mdp.available_actions(state)
    for action in mdp_available_actions:
        try:
            new_state = transition(mdp, state, action, depth)
      
            if new_state.current_agent == 0:
                value, _ = _max(mdp, new_state, max_depth, depth + 1)
            else:
                value = _min(mdp, new_state, max_depth, depth + 1)
            if value > best_value:
                best_value = value
                best_action = action
            if isinstance(mdp, WorldMDP):
                mdp.remove_from_visited(new_state)
        except ValueError:
            # print(WORLD_STEP_NOT_POSSIBLE_ERROR)
            pass
    return best_value, best_action

def _min(mdp: MDP[A, S], state: S, max_depth: int, depth: int = 0) -> float:
    """Returns the worst value of the state."""
    if mdp.is_final(state) or depth == max_depth:
        return state.value
    worst_value = float('inf')
    for action in mdp.available_actions(state):
        try:
            new_state = transition(mdp, state, action, depth)
        
            if new_state.current_agent == 0:
                value, _ = _max(mdp, new_state, max_depth, depth + 1)
            else:
                value = _min(mdp, new_state, max_depth, depth)
            worst_value = min(worst_value, value)
            if isinstance(mdp, WorldMDP):
                mdp.remove_from_visited(new_state)
        except ValueError:
            # print(WORLD_STEP_NOT_POSSIBLE_ERROR)
            pass
        
    return worst_value

def minimax(mdp: MDP[A, S], state: S, max_depth: int) -> A:
    """Returns the action to be performed by Agent 0 in the given state. 
    This function only accepts 
    states where it's Agent 0's turn to play 
    and raises a ValueError otherwise. 
    Don't forget that there may be more than one opponent"""
    if state.current_agent != 0:
        raise ValueError("It's not Agent 0's turn to play")
    value, action = _max(mdp, state, max_depth, 0)
    return action

def _alpha_beta_max(mdp: MDP[A, S], state: S, alpha: float, beta: float, max_depth: int, depth: int = 0) -> Tuple[float, A, float, float]:
    if mdp.is_final(state) or depth == max_depth:
        return state.value, None
    best_value = float('-inf')
    best_action = None
    available_actions = mdp.available_actions(state)
    for action in available_actions:
        try:
            new_state = transition(mdp, state, action, depth)
    
            if new_state.current_agent == 0:
                value, _ = _alpha_beta_max(mdp, new_state, alpha, beta, max_depth, depth + 1)
            else:
                value = _alpha_beta_min(mdp, new_state, alpha, beta, max_depth, depth + 1)
            if value > best_value:
                best_value = value
                best_action = action
            if isinstance(mdp, WorldMDP):
                mdp.remove_from_visited(new_state)
            if beta <= best_value:  # Beta cutoff
                return best_value, best_action
            alpha = max(alpha, best_value)  # Update alpha after cutoff: fail hard
        except ValueError:
            # print(WORLD_STEP_NOT_POSSIBLE_ERROR)
            pass

    return best_value, best_action

def _alpha_beta_min(mdp: MDP[A, S], state: S, alpha: float, beta: float, max_depth: int, depth: int = 0) -> Tuple[float, A, float, float]:
    if mdp.is_final(state) or depth == max_depth:
        return state.value
    worst_value = float('inf')
    available_actions = mdp.available_actions(state)
    for action in available_actions:
        try:
            new_state = transition(mdp, state, action, depth)
    
            if new_state.current_agent == 0:
                value, _ = _alpha_beta_max(mdp, new_state, alpha, beta, max_depth, depth + 1)
            else:
                value = _alpha_beta_min(mdp, new_state, alpha, beta, max_depth, depth)
            worst_value = min(worst_value, value)
            if isinstance(mdp, WorldMDP):
                mdp.remove_from_visited(new_state)
            if worst_value <= alpha:  # Alpha cutoff
                return worst_value
            beta = min(beta, worst_value)  # Update beta after cutoff: fail hard
        except ValueError:
            # print(WORLD_STEP_NOT_POSSIBLE_ERROR)
            pass
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
    value, action = _alpha_beta_max(mdp, 
                                    state, 
                                    alpha, 
                                    beta, 
                                    max_depth, 
                                    0)
    return action

def _expectimax_max(mdp: MDP[A, S], 
                    state: S, 
                    max_depth: int, 
                    depth: int = 0
                    ) -> Tuple[float, A]:
    if mdp.is_final(state) or depth == max_depth:
        return state.value, None
    best_value = float('-inf')
    best_action = None
    for action in mdp.available_actions(state):
        try:
            new_state = transition(mdp, 
                                   state, 
                                   action, 
                                   depth
                                    )

            if new_state.current_agent == 0:
                value, _ = _expectimax_max(mdp, 
                                        new_state, 
                                        max_depth, 
                                        depth + 1
                                        )
            else:
                value = _expectimax_exp(mdp, new_state, max_depth, depth + 1)
            if value > best_value:
                best_value = value
                best_action = action
            if isinstance(mdp, WorldMDP):
                mdp.remove_from_visited(new_state)
        except ValueError:
            # print(WORLD_STEP_NOT_POSSIBLE_ERROR)
            pass
    return best_value, best_action

def _expectimax_exp(mdp: MDP[A, S],
                     state: S, 
                     max_depth: int, 
                     depth: int = 0
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
            new_state = transition(mdp, state, action, depth)
    
            if new_state.current_agent == 0:
                value, _ = _expectimax_max(mdp, new_state, max_depth, depth + 1)
            else:
                value = _expectimax_exp(mdp, new_state, max_depth, depth)
            total_value += value
            if isinstance(mdp, WorldMDP):
                mdp.remove_from_visited(new_state)
        except ValueError:
            # print(WORLD_STEP_NOT_POSSIBLE_ERROR)
            pass
    expected_value = total_value / num_actions if num_actions != 0 else 0
    return expected_value

def expectimax(mdp: MDP[A, S], 
               state: S, 
               max_depth: int
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
    _, action = _expectimax_max(mdp, state, max_depth, 0)
    return action
