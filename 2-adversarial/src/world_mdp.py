import copy
from dataclasses import dataclass
import sys
from typing import List, Tuple
import lle
from lle import Position, World, Action
from mdp import MDP, State

import auto_indent
from utils import print_items

from anytree import Node, RenderTree
from loguru import logger

sys.stdout = auto_indent.AutoIndent(sys.stdout)

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
    gems_collected: list
    value_vector: List[float]
    # alpha_beta_vector: List[Tuple[float, float]] #todo?
    # gems_collected_by_agents: list[list[Position]]
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
        # self.alpha_beta_vector = [] #todo?
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
        print(f"update_world_string()")
        print(f"current_agent: {current_agent}")
        print(f"current_agent_previous_position: {current_agent_previous_position}")
        print(f"action: {action}")
        # self.world_string = self.world.world_string
        matrix = self.layout_to_matrix(self.world_string)
        print(f"matrix: {matrix}")
        if action != Action.STAY:
            agent_string = "S"+str(current_agent)
            matrix[current_agent_previous_position[0]][current_agent_previous_position[1]] = "."
            matrix[self.agents_positions[current_agent][0]][self.agents_positions[current_agent][1]] = agent_string
            matrix_after_action = matrix
            print(f"matrix_after_action: {matrix_after_action}")
            layout_after_action = self.matrix_to_layout(matrix_after_action)
            print(f"layout_after_action: {layout_after_action}")
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
        # state_attributes += f"world: {self.world.world_string}\n"
        state_attributes += f"world: \n{self.world_string}\n"
        return state_attributes
class WorldMDP(MDP[Action, MyWorldState]):
    def __init__(self
                 , world: World):
        self.world = world
        world.reset()
        self.n_agents = world.n_agents

        self.initial_state = world.get_state()
        self.root = None

        # nodes dict
        self.nodes = {}
        self.n_expanded_states = 0

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
        print("available_actions()")
        world_available_actions = state.world.available_actions()
        print(f"world_available_actions: {world_available_actions}")
        current_agent = state.current_agent
        print(f"current_agent: {current_agent}")
        current_agent_available_actions = world_available_actions[current_agent]
        print(f"current_agent_available_actions: {current_agent_available_actions}")
        return current_agent_available_actions
        # alpha beta pruning optimization
        # action stay is in general augmenting the number of expanded states
        #                           and not augmenting the state value
        # so stay must be the last action to be considered
        # reverse the list of actions 
        # reversed_current_agent_available_actions = list(reversed(current_agent_available_actions))
        # return reversed_current_agent_available_actions #todo FAILED tests/test_alpha_beta.py::test_alpha_beta_two_agents - assert 44 <= 30
        #todo FAILED tests/test_expectimax.py::test_two_agents2 - assert West == South

    def is_final(self, state: MyWorldState) -> bool:
        """returns True if the state is final, False otherwise."""
        return state.world.done

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
        # agent_positions = set(state.agents_positions)  
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

    def transition(self
                   , state: MyWorldState
                   , action: Action) -> MyWorldState:
        """Returns the next state and the reward.
        If Agent 0 dies during a transition, 
        the state value immediately drops to 
        lle.REWARD_AGENT_DIED (-1), 
        without taking into account any gems already collected
        """
        print(f"transition()")
        print(f"state: {state}")
        print(f"state.world.agents_positions: {state.world.agents_positions}")
        print(f"state.world_string: {state.world_string}")
        print(f"state.agents_positions: {state.agents_positions}")
        
        print(f"state.current_agent: {state.current_agent}")
        print(f"state.current_agent position: {state.agents_positions[state.current_agent]}")
        print(f"action: {action}")

        self.n_expanded_states += 1
        # real_state = self.world.get_state()
        # simulation_world = copy.deepcopy(self.world)
        simulation_world = copy.deepcopy(state.world)
        world_string = copy.deepcopy(state.world_string)
        print(f"world_string: {world_string}")
        # simulation_world = self.world
        # self.world.set_state(self.convert_to_WorldState(state))
        simulation_world.set_state(self.convert_to_WorldState(state))


        simulation_state = simulation_world.get_state()
        simulation_state_current_agent = state.current_agent
        current_agent_previous_position = simulation_state.agents_positions[simulation_state_current_agent]
        actions = self.get_actions(simulation_state_current_agent, action)
        # actions = self.get_actions(simulation_state.current_agent, action)
        # next_state_value_vector = copy.deepcopy(state.value_vector)
        print(f"state.value_vector: {state.value_vector}")
        next_state_value_vector = copy.deepcopy(state.value_vector)
        reward = 0.0
        print(f"actions: {actions}")
        # reward = self.world.step(actions)
        reward = simulation_world.step(actions)

        world_state_after_action = simulation_world.get_state()
        # world_state_after_action = state.world.get_state()

        next_state_value_vector[simulation_state_current_agent] += reward
        if simulation_state_current_agent == 0:
            print(f"simulation_state.current_agent: {simulation_state_current_agent}")
            agents_positions = world_state_after_action.agents_positions
            print(f"agents_positions: {agents_positions}")
            print(f"reward: {reward}")
            if reward == -1:
                next_state_value_vector[0] = -1.0 #lle.REWARD_AGENT_DIED
                
        print(f"next_state_value_vector: {next_state_value_vector}")
        print(f"transitioned state: {world_state_after_action}")
        print(f"transitioned state value: {next_state_value_vector[simulation_state_current_agent]}")
        
        next_state_current_agent = (simulation_state_current_agent+1)%simulation_world.n_agents

        print(f"state.value_vector: {state.value_vector}")
        print(f"next_state_value_vector: {next_state_value_vector}")
        # self.world.set_state(current_state)
        # my_world_state_transitioned = MyWorldState(next_state_value_vector[state.current_agent]
        my_world_state_transitioned = MyWorldState(next_state_value_vector[0]
                                                   , next_state_value_vector
                                                   , next_state_current_agent
                                                #    , self.world
                                                   , simulation_world
                                                   , world_string
                                                   , action
                                                   )
        # my_world_state_transitioned.last_action = action
        my_world_state_transitioned.update_world_string(simulation_state_current_agent
                                                        , current_agent_previous_position
                                                        , actions)
       
        print(f"my_world_state_transitioned: {my_world_state_transitioned.to_string()}")
        # self.world.set_state(real_state)

        return my_world_state_transitioned

class BetterValueFunction(WorldMDP):
    def transition(self, state: MyWorldState, action: Action) -> MyWorldState:
        """Subclass of WorldMDP 
        in which the state value
          is calculated more intelligently than simply considering Agent 0's score. 
          Write this subclass and verify that for the same map, 
          the number of expanded nodes is indeed lower 
          than with the basic evaluation function. 
          There is no test for this exercise.

            Improvements:

            If Agent 0 dies during a transition, 
                the state value is reduced by #todo
                , but the gems already collected are taken into account.
            The value of a state is increased by 
            the average of the score differences between Agent 0 and the other agents.."""
        # Change the value of the state here.
        ...
    
    # def evaluate
