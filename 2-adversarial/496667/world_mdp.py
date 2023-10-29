import copy
from dataclasses import dataclass
import random
import sys
from typing import List, Optional, Tuple#, override
import lle
from lle import Position, World, Action
from mdp import A, MDP, State

# import auto_indent

from anytree import Node, RenderTree
from loguru import logger
import numpy as np
from scipy.optimize import linear_sum_assignment

# sys.stdout = auto_indent.AutoIndent(sys.stdout)

def get_distance(coord1, coord2):
    """Returns the distance between two coordinates"""
    x1, y1 = coord1
    x2, y2 = coord2
    return abs(x1 - x2) + abs(y1 - y2)

def min_distance_position(position : Tuple[int, int]
                          , positions: list[Tuple[int, int]] 
                            ) -> Tuple[Tuple[int, int], float]:
    """Returns the position in positions that is closest to position"""
    min_distance = float("inf")
    min_position = None
    for pos in positions:
        distance = 0
        distance = get_distance(position, pos)
        if distance < min_distance:
            min_distance = distance
            min_position = pos
    return min_position, min_distance

def min_distance_pairing(list_1
                             , list_2):
        # Create a cost matrix
        cost_matrix = np.zeros((len(list_1), len(list_2)))
        for i, point1 in enumerate(list_1):
            for j, point2 in enumerate(list_2):
                cost_matrix[i, j] = ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5
        # Hungarian algorithm:
        # from cost_matrix, it does the pairing by minimizing the total distance
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        # Extract the paired points, their distances, and the minimum total distance
        paired_points = []
        distances = []
        min_total_distance = 0
        for i, j in zip(row_ind, col_ind):
            paired_points.append((list_1[i], list_2[j]))
            distances.append(cost_matrix[i, j])
            min_total_distance += cost_matrix[i, j]
        return paired_points, distances, min_total_distance


@dataclass
class MyWorldState(State):
    """Comme il s’agit d’un MDP à plusieurs agents et à tour par tour, 
    chaque état doit retenir à quel agent
    c’est le tour d’effectuer une action.
    """
    # la valeur d’un état correspond à la somme des rewards obtenues par les actions de l’agent 0 
    # (c’est-à-dire les gemmes collectées + arriver sur une case de ﬁn)
    value: float 
    current_agent: int
    last_action: Action
    agents_positions: list
    gems_collected: list[bool]
    value_vector: List[float]
    alpha: Optional[float] = None
    beta: Optional[float] = None
    # Add more attributes here if needed.
    def __init__(self
                 , value: float
                 , value_vector: List[float]
                 , current_agent: int
                 , world: World
                    , world_string: str = None
                    , last_action: Action = None
                 ):
        super().__init__(value, current_agent)
        self.world = world
        if world_string:
            self.world_string = world_string
        else:
            self.world_string = world.world_string
        self.agents_positions = world.agents_positions
        self.gems_collected = world.get_state().gems_collected
        self.value_vector = value_vector
        self.node = None
        if last_action:
            self.last_action = last_action
        else:
            self.last_action = None

    def get_agents_positions(self) -> list:
        # return self.agents_positions
        return self.world.agents_positions
    
    def layout_to_matrix(self
                         , layout):
        """
        Convert a given layout into a matrix where each first row of each line
        contains the (group of) character of the layout line.
        Parameters:
        layout (str): A multi-line string where each line represents a row in the layout.
        Returns:
        list of list of str: A matrix representing the layout.
        """
        # Split the layout into lines
        lines = layout.strip().split('\n')
        matrix = []
        max_cols = 0  # Keep track of the maximum number of columns
        # Convert each line into a row in the matrix
        for line in lines:
            row = [char for char in line.split() if char != ' ']
            matrix.append(row)
            max_cols = max(max_cols, len(row))
        # Fill in missing columns with '.'
        for row in matrix:
            while len(row) < max_cols:
                row.append('.')
        return matrix
    
    def matrix_to_layout(self
                         ,matrix):
        """
        Convert a given matrix into a layout.
        
        Parameters:
        matrix (list of list of str): A matrix representing the layout.

        Returns:
        list of str: Each string represents a row in the layout.
        """
        # Determine the maximum length of any element in the matrix for alignment
        max_len = max(len(str(item)) for row in matrix for item in row)
        layout = ""
        for row in matrix:
            # Align the elements by padding with spaces
            aligned_row = " ".join(str(item).ljust(max_len) for item in row)
            layout += aligned_row + "\n"
            
        return layout

    
    def update_world_string(self
                            ,current_agent: int
                            ,current_agent_previous_position: Position
                            ,action) -> None:
        """Updates world_string attribute with current world state:
        current agent position, gems collected, etc."""
        matrix = self.layout_to_matrix(self.world_string)
        if action != Action.STAY:
            agent_string = "S"+str(current_agent)
            matrix[current_agent_previous_position[0]][current_agent_previous_position[1]] = "."
            matrix[self.agents_positions[current_agent][0]][self.agents_positions[current_agent][1]] = agent_string
            matrix_after_action = matrix
            layout_after_action = self.matrix_to_layout(matrix_after_action)
            self.world_string = layout_after_action
            
    def to_string(self) -> str:
        """Returns a string representation of the state.
        with each state attribute on a new line."""
        # return f"current_agent: {self.current_agent}, value: {self.value}, value_vector: {self.value_vector}, agents_positions: {self.agents_positions}, gems_collected: {self.gems_collected}"
        state_attributes = f"current_agent: {self.current_agent}\n"
        
        if self.last_action :
            state_attributes += f"last_action: {self.last_action}\n"
        state_attributes += f"value: {self.value}\n"
        state_attributes += f"value_vector: {self.value_vector}\n"
        state_attributes += f"agents_positions: {self.agents_positions}\n"
        state_attributes += f"gems_collected: {self.gems_collected}\n"
        state_attributes += f"world: \n{self.world_string}\n"
        return state_attributes
    
    def serialize(self) -> tuple:
        """Serialize the given world state.
        Args:
            world_state: the world state to serialize.
        Returns:
            A tuple that represents the given world state.
        """
        return (tuple(self.agents_positions), tuple(self.gems_collected), self.current_agent)
    

class WorldMDP(MDP[Action, MyWorldState]):
    def __init__(self
                 , world: World):
        self.world = world
        world.reset()
        self.n_agents = world.n_agents

        self.initial_state = world.get_state()
        self.root = None

        self.visited = set() # visited states
        # nodes dict
        self.nodes = {} # key: state, value: node
        self.n_expanded_states = 0
        self.lasers_dangerous_for_agents = self.get_lasers_dangerous_for_agents()

    def get_lasers_dangerous_for_agents(self) -> list[list[Position]]:
        """Returns a list of lists
        , each corresponding to the agent of same index
        , containing positions of the lasers of a different agent_id (color)."""

        lasers_dangerous_for_agents = [[] for _ in range(self.world.n_agents)]
        laser_sources = self.world.laser_sources

        for laser_source in laser_sources:
            laser_source_position = laser_source[0]
            laser_source_agent_id = laser_source[1].agent_id
            #add the laser source position to the list of lasers dangerous for the agents of index different from laser_source_agent_id
            for agent_id in range(self.world.n_agents):
                if agent_id != laser_source_agent_id:
                    lasers_dangerous_for_agents[agent_id].append(laser_source_position)

        return lasers_dangerous_for_agents

    def reset(self):
        """The world.reset() method returns an initial state of the game. 
        After performing reset(), 
        it's Agent 0's turn to take an action. 
        Thus, world.transition(Action.NORTH) 
        will only move Agent 0 to the north, 
        while all other agents will remain in place. 
        Then, it's Agent 1's turn to move, and so on"""
        self.n_expanded_states = 0
        self.world.reset()
        return MyWorldState(0.0
                            , [0.0 for _ in range(self.world.n_agents)]
                            , 0
                            , self.world)

    def available_actions(self, state: MyWorldState) -> list[Action]:
        """returns the actions available to the current agent."""
        world_available_actions = state.world.available_actions()
        current_agent = state.current_agent
        current_agent_available_actions = world_available_actions[current_agent]
        return current_agent_available_actions
      
    def is_final(self, state: MyWorldState) -> bool:
        """returns True if the state is final, False otherwise."""
        return state.world.done and not [gem[0] for gem in state.world.gems if not gem[1].is_collected]

    def get_actions(self
                , current_agent: int
                , action: Action) -> list[Action]:
        """from current agent action, returns list with action at agent index and STAY at others's ."""
        actions = [Action.STAY for _ in range(self.world.n_agents)]
        actions[current_agent] = action
        return actions

    def convert_to_WorldState(self, state: MyWorldState) -> lle.WorldState:
        """Converts MyWorldState to lle.WorldState"""
        return lle.WorldState(state.agents_positions, state.gems_collected)
    
    def agents_each_on_different_exit_pos(self
                                          , state: MyWorldState) -> bool:
        """Whether each agent is on a different exit position."""
        agent_positions = set(state.world.agents_positions)  

        exit_positions = set(self.world.exit_pos)  
        # Intersect the sets to find agents that are on exit positions
        agents_on_exits = agent_positions.intersection(exit_positions)
        # Check if the number of agents on exits is equal to the total number of agents
        # and if each agent is on a different exit
        return len(agents_on_exits) == len(agent_positions) # and len(agents_on_exits) == len(exit_positions)

    def current_agent_on_exit(self
                              , state: MyWorldState
                              , current_agent: int
                              ) -> bool:
        """Whether the current agent is on an exit position."""
        current_agent_position = state.agents_positions[current_agent]
        return current_agent_position in self.world.exit_pos

    def add_to_visited(self
                          , state: MyWorldState) -> None:
        """Adds state to visited states."""
        self.visited.add(state.serialize())

    def remove_from_visited(self
                            , state: MyWorldState) -> None:
        """Removes state from visited states."""
        self.visited.remove(state.serialize())

    def was_visited(self,
                    state: MyWorldState) -> bool:
        return state.serialize() in self.visited
    
    def add_value_to_node(self
                          , state
                          , value: float
                          , discriminator: str
                          , alpha: float = None
                            , beta: float = None
                          ) -> None:
        """Adds value to node"""
        #add best_value to the node name
        new_state_string = state.to_string()
        new_state_string_with_best_value = new_state_string + "\n "+discriminator+" value : " + str(value)
        if alpha != None:
            new_state_string_with_best_value += "\n alpha : " + str(alpha)
        if beta != None:
            new_state_string_with_best_value += "\n beta : " + str(beta)
        if self.is_final(state):
            new_state_string_with_best_value += "\n FINAL"
        # replace
        self.nodes[new_state_string].name = new_state_string_with_best_value

    def transition(self
                   , state: MyWorldState
                   , action: Action
                   ) -> MyWorldState:
        """Returns the next state and the reward.
        If Agent 0 dies during a transition, 
        the state value immediately drops to 
        lle.REWARD_AGENT_DIED (-1), 
        without taking into account any gems already collected
        """
        self.n_expanded_states += 1
        simulation_world = copy.deepcopy(state.world)
        world_string = copy.deepcopy(state.world_string)
        simulation_world.set_state(self.convert_to_WorldState(state))

        simulation_state = simulation_world.get_state()
        simulation_state_current_agent = state.current_agent
        current_agent_previous_position = simulation_state.agents_positions[simulation_state_current_agent]
        actions = self.get_actions(simulation_state_current_agent, action)
        next_state_value_vector = copy.deepcopy(state.value_vector)
        reward = 0.0
        reward = simulation_world.step(actions)
        next_state_value_vector[simulation_state_current_agent] += reward
        if simulation_state_current_agent == 0:
            if reward == -1:
                next_state_value_vector[0] = -1.0 #lle.REWARD_AGENT_DIED
        next_state_current_agent = (simulation_state_current_agent+1)%simulation_world.n_agents
        my_world_state_transitioned = MyWorldState(next_state_value_vector[0]
                                                   , next_state_value_vector
                                                   , next_state_current_agent
                                                   , simulation_world
                                                   , world_string
                                                   , action
                                                   )
        my_world_state_transitioned.update_world_string(simulation_state_current_agent
                                                        , current_agent_previous_position
                                                        , actions)
        return my_world_state_transitioned
    

def balanced_multi_salesmen_greedy_tsp(remaining_cities: list[Tuple[int, int]]
                                       , num_salesmen: int
                                       , start_cities: list[Tuple[int, int]]
                                       , finish_cities: list[Tuple[int, int]]
                                       ) -> Tuple[dict[str, list[Tuple[int, int]]], dict[str, float], float]: 
    #todo: calculate the distance between the last city and the finish city one time at problem creation
    """Given a list of cities coordinates, returns a list of cities visited by each agent
    in the order that minimizes the total distance traveled.
    """
    routes = {f"agent_{i+1}": [start_cities[i]] for i in range(num_salesmen)}
    distances = {f"agent_{i+1}": 0.0 for i in range(num_salesmen)}
    while remaining_cities:
        for agent in routes.keys():
            if not remaining_cities:
                break
            nearest_city, nearest_distance = min_distance_position(routes[agent][-1], remaining_cities)
            distances[agent] += nearest_distance
            routes[agent].append(nearest_city)
            remaining_cities.remove(nearest_city)
    for agent in routes.keys():
        current_city = routes[agent][-1]
        finish_city, final_distance = min_distance_position(current_city, finish_cities)
        distances[agent] += final_distance
        routes[agent].append(finish_city)
    total_distance = sum(distances.values())
    return routes, distances, total_distance


class BetterValueFunction(WorldMDP):
    """Subclass of WorldMDP
    in which the state value
      is calculated more intelligently than simply considering Agent 0's score. 
     
        Improvements:

        If Agent 0 dies during a transition, 
            the state value is reduced by #todo
            , but the gems already collected are taken into account.
        The value of a state is increased by 
        the average of the score differences between Agent 0 and the other agents.."""
    def get_position_after_action(self
                                  , agent_pos: Tuple[int, int]
                                    , action: Action
                                    ) -> Tuple[int, int]:
        """Returns the position of the agent after performing the given action in the given state."""
        agent_pos_after_action = None
        # Apply the action to the agent's position
        if action == Action.NORTH:
            agent_pos_after_action = (agent_pos[0] - 1, agent_pos[1])
        elif action == Action.SOUTH:
            agent_pos_after_action = (agent_pos[0] + 1, agent_pos[1])
        elif action == Action.WEST:
            agent_pos_after_action = (agent_pos[0], agent_pos[1] - 1)
        elif action == Action.EAST:
            agent_pos_after_action = (agent_pos[0], agent_pos[1] + 1)
        elif action == Action.STAY:
            agent_pos_after_action = (agent_pos[0], agent_pos[1])
        else:
            raise ValueError("Invalid action")
        return agent_pos_after_action
    
    def get_available_actions_ordered(self
                                    , state: MyWorldState
                                    ) -> List[A]:
        """Returns the available actions ordered by heuristic value"""
        available_actions = super().available_actions(state)
        current_agent = state.current_agent
        # move STAY to the end of the list
        available_actions_ordered = [action for action in available_actions if action != Action.STAY]
        available_actions_ordered.append(Action.STAY)
        for action in available_actions:
            position_after_action = self.get_position_after_action(state.agents_positions[current_agent]
                                                                    , action
                                                                    )
            # if not all gems are collected,
            # not all (not gem for gem in state.gems_collected):
            gems_to_collect = [gem[0] for gem in state.world.gems if not gem[1].is_collected]

            if gems_to_collect:
                # if action leads to a gem, move it to the top of the list
                if position_after_action in [gem[0] for gem in state.world.gems]:
                    available_actions_ordered.remove(action)
                    available_actions_ordered.insert(0, action)
            # if a laser sources has not the same color as the agent, 
            if self.lasers_dangerous_for_agents[state.current_agent]:
                if position_after_action in [laser[0] for laser in state.world.lasers if laser[1].is_on]:
                    available_actions_ordered.remove(action)
                    available_actions_ordered.append(action)

        return available_actions_ordered

    def available_actions(self, state: MyWorldState) -> list[Action]:
        return self.get_available_actions_ordered(state)
    
    def transition(self
                   , state: MyWorldState
                   , action: Action
                   , depth: int = 0
                   ) -> MyWorldState:
        """Returns the next state and the reward.
        """
        # Change the value of the state here.
        state = super().transition(state
                                   , action
                                   )
        n_agents = self.world.n_agents
        previous_agent = (state.current_agent-1)%n_agents
        current_agent = previous_agent
        state_agents_positions = state.agents_positions
        value = state.value
        if value == -1 or value == 0:
            return state
        world_gems = state.world.gems
        gems_to_collect = [gem[0] for gem in world_gems if not gem[1].is_collected]
        _, distances, total_distance = balanced_multi_salesmen_greedy_tsp(copy.deepcopy(gems_to_collect)
                                                       , n_agents
                                                       , state_agents_positions
                                                       , self.world.exit_pos)
        current_agent_distance = distances[f"agent_{current_agent+1}"] # +1 because agent_0 is agent_1 #todo
        if current_agent_distance == 1:
            current_agent_distance = 1.5
        other_agents_distances = [distances[f"agent_{i+1}"] for i in range(n_agents) if i != current_agent]
        other_agents_average_distance_length = len(other_agents_distances)
        if other_agents_average_distance_length == 0:
            other_agents_average_distance_length = 1
        if gems_to_collect:
            # prefer current agent to be closer to the nearest gem
            # and other agents to be far from the nearest gem
            if current_agent_distance != 0:
                value = value + (len(gems_to_collect) / current_agent_distance)
        else:
            # prefer agent to be closer to the exit
            if current_agent_distance != 0:
                value = value + (lle.REWARD_AGENT_JUST_ARRIVED / current_agent_distance)
                # prefer all agents to be closer to the exit
                average_distance_to_exit = total_distance/n_agents
                # add reward for each agent being on exit lle.REWARD_AGENT_ON_EXIT/their distance to exit
                if other_agents_distances:
                    if all(distance == 0 for distance in other_agents_distances):
                        value = value + lle.REWARD_END_GAME/average_distance_to_exit
                else:
                    value = value + lle.REWARD_END_GAME*10/current_agent_distance
        state.value = value
        state.value_vector[current_agent] = value
        return state
    
