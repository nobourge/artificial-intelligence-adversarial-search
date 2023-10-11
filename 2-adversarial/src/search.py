from dataclasses import dataclass
from typing import Optional
from lle import Action, World, WorldState

from problem import CornerSearchProblem, GemSearchProblem, SearchProblem, SimpleSearchProblem, serialize, was
from priority_queue import PriorityQueue
# import sys
# import auto_indent
# from utils import print_items

# sys.stdout = auto_indent.AutoIndent(sys.stdout)

@dataclass
class Solution:
    actions: list[tuple[Action]]

    @property
    def n_steps(self) -> int:
        return len(self.actions)

    ...

def is_empty(data_structure) -> bool:
    """Returns True if data_structure is empty, False otherwise"""
    if isinstance(data_structure, list):
        return len(data_structure) == 0
    elif isinstance(data_structure, set):
        return len(data_structure) == 0
    elif isinstance(data_structure, PriorityQueue):
        return data_structure.is_empty()

def check_goal_state(problem: SearchProblem
                     , current_state: WorldState
                     , actions: list[tuple[Action]]
                     , objectives_reached = None
                     ) -> bool:
    # Check if the current state is the goal state
    if isinstance(problem, CornerSearchProblem):
        current_state_is_goal_state = problem.is_goal_state(current_state, objectives_reached)
    else:
        current_state_is_goal_state = problem.is_goal_state(current_state)
        
    if current_state_is_goal_state:
        # print("Solution found!")
        print("nodes expanded: ", problem.nodes_expanded)
        # print("actions: ", actions)
        problem.path_size = len(actions)
        print( "n_steps: ", len(actions))
        return Solution(actions)

def get_initial_objectives_reached(problem: SearchProblem
                                        , initial_state: WorldState
                                        ) -> list[tuple[int, int]]:
    objectives_reached = []
    if isinstance(problem, CornerSearchProblem):
        for corner in problem.corners:
            if corner in initial_state.agents_positions:
                objectives_reached.append(corner)
    return objectives_reached
    
def tree_search(problem: SearchProblem, mode: str) -> Optional[Solution]:
    """Tree search algorithm.
    Args:
        problem: the problem to solve.
        mode: the search mode to use:
            - "dfs": Depth-First Search
            - "bfs": Breadth-First Search
            - "astar": A* Search
    Returns:
        A solution to the problem, or None if no solution exists.
    """
    initial_state = problem.initial_state
    actions = []
    cost = 0
    objectives_reached = get_initial_objectives_reached(problem, initial_state)
    if mode == "astar":
        data_structure = PriorityQueue() 
        if isinstance(problem, CornerSearchProblem) or isinstance(problem, GemSearchProblem):
            data_structure.push((initial_state
                                 , actions
                                 , objectives_reached)
                                , cost)
        else:
            data_structure.push((initial_state
                                 , actions
                                 )
                                , cost)
    else:
        data_structure = [(initial_state
                           , actions)]  #  to keep track of states
    visited = set()  # Set to keep track of visited states

    while not is_empty(data_structure):
        # Pop the top state from the data_structure
        if mode == "bfs":
            current_state, actions = data_structure.pop(0)
        else:
            if isinstance(problem, CornerSearchProblem) or isinstance(problem, GemSearchProblem):
                current_state, actions, objectives_reached = data_structure.pop()
            else:
                current_state, actions = data_structure.pop()

        # Check if the current state is in the visited set
        if was(current_state
               , objectives_reached
               , visited):
            continue
        solution = None
        if isinstance(problem, CornerSearchProblem):
            solution = check_goal_state(problem
                            , current_state
                            , actions
                            , objectives_reached)
        else:
            solution = check_goal_state(problem
                            , current_state
                            , actions
                            , None)
        if solution:
            return solution
        current_state_hashable = serialize(current_state, objectives_reached)
        visited.add(current_state_hashable)
        # Add successors to data_structure
        if isinstance(problem, CornerSearchProblem) or isinstance(problem, GemSearchProblem):
            successors = problem.get_successors(current_state
                                                ,visited
                                                ,objectives_reached)
        else:
            successors = problem.get_successors(current_state
                                            ,visited)
        for successor_tuple in successors:  
            if isinstance(problem, CornerSearchProblem) or isinstance(problem, GemSearchProblem):
                successor, successor_actions, cost, objectives_reached = successor_tuple
            else:
                successor, successor_actions, cost = successor_tuple
                objectives_reached = None
            new_actions = actions + [successor_actions]
            if mode == "astar":
                
                if isinstance(problem, CornerSearchProblem) or isinstance(problem, GemSearchProblem):
                    successor_cost = problem.heuristic(successor, objectives_reached)
                    total_cost = cost + successor_cost
                    data_structure.push((successor
                                         , new_actions
                                         , objectives_reached)
                                        , total_cost)
                else:
                    successor_cost = problem.heuristic(successor, successor_actions)
                    total_cost = cost + successor_cost
                    data_structure.push((successor
                                         , new_actions)
                                        , total_cost)
            else:
                data_structure.append((successor, new_actions))
    return None

def dfs(problem: SearchProblem) -> Optional[Solution]:
    """Depth-First Search"""
    return tree_search(problem, "dfs")

def bfs(problem: SearchProblem) -> Optional[Solution]:
    """Breadth-First Search"""
    return tree_search(problem, "bfs")

def astar(problem: SearchProblem) -> Optional[Solution]:
    """A* Search"""
    return tree_search(problem, "astar")

if __name__ == "__main__":

    # world = World.from_file("cartes/1_agent/simplest")
    # world = World.from_file("cartes/1_agent/impossible_simplest")
    # world = World.from_file("cartes/2_agents/impossible")
    # world = World.from_file("cartes/1_agent/zigzag")
    # world = World.from_file("cartes/2_agents/zigzag")
    # world = World.from_file("cartes/2_agents/zigzag_simpler")
    # world = World.from_file("cartes/corners_simplest")

    # world = World.from_file("level3")
    # world.reset()

    # problem = SimpleSearchProblem(world)
    # solution = dfs(problem)
    # solution = astar(problem)
    # print("solution: ", solution)

    # world = World.from_file("cartes/gems_simplest")
    # world = World.from_file("cartes/2_agents/zigzag")
    # world = World.from_file("cartes/2_agents/zigzag_gems")
    # world = World.from_file("cartes/gems")
    # problem = GemSearchProblem(world)
    # solution = astar(problem)
    # print("solution: ", solution)
    # check_world_done(problem, solution)
    # if world.n_gems != world.gems_collected:
    #     raise AssertionError("Your is_goal_state method is likely erroneous beacuse some gems have not been collected")



    
    # world = World(
    #         """
    #         S0 . X
    #         . . ."""
    #     )
    # problem = SimpleSearchProblem(world)
    # successors = list(problem.get_successors(problem.initial_state))
    # assert len(successors) == 3
    # world.reset()
    # available = world.available_actions()[0]
    # agent_pos = ((0, 0), (0, 1), (1, 0))
    # for state, action, cost in successors:
    #     assert action[0] in available
    #     assert state.agents_positions[0] in agent_pos


    world = World.from_file("cartes/corners")
    # world = World.from_file("cartes/corners5x5")
    problem = CornerSearchProblem(world)
    solution = astar(problem)
    world.reset()
    corners = set([(0, 0), (0, world.width - 1), (world.height - 1, 0), (world.height - 1, world.width - 1)])
    for action in solution.actions:
        world.step(action)
        agent_pos = world.agents_positions[0]
        if agent_pos in corners:
            corners.remove(agent_pos)
    assert len(corners) == 0, f"The agent did not reach these corners: {corners}"