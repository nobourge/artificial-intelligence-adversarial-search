from lle import Action
from mdp import MDP, S, A


def minimax(mdp: MDP[A, S], state: S, max_depth: int) -> A:
    """Returns the action to be performed by Agent 0 in the given state. 
    This function only accepts 
    states where it's Agent 0's turn to play 
    and raises a ValueError otherwise. 
    It is suggested to divide this algorithm into 
    a _min function and 
    a _max function. 
    Don't forget that there may be more than one opponent"""
    ...


def alpha_beta(mdp: MDP[A, S], state: S, max_depth: int) -> A:
    """The alpha-beta pruning algorithm 
    is an improvement over 
    minimax 
    that allows for pruning of the search tree."""

    ...


def expectimax(mdp: MDP[A, S], state: S, max_depth: int) -> Action:
    """ The 'expectimax' algorithm allows for 
    modeling the probabilistic behavior of humans 
    who might make suboptimal choices. 
    The nature of expectimax requires that we know the probability that the opponent will take each action. 
    Here, we will assume that 
    the other agents take actions that are uniformly random."""
    ...
