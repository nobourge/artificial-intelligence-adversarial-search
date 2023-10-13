from dataclasses import dataclass
import lle
from lle import Position, World, Action
from mdp import MDP, State


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
    agents_positions: list
    gems_collected: list
    # gems_collected_by_agents: list[list[Position]]
    # Add more attributes here if needed.
    def __init__(self
                 , value: float
                 , current_agent: int
                 , world: World
                #  , gems_collected_by_agents: list(list(Position)) = None
                 ):
        super().__init__(value, current_agent)
        self.world = world
        self.agents_positions = world.agents_positions
        self.gems_collected = world.get_state().gems_collected
        # self.gems_collected_by_agents = gems_collected_by_agents
    
class WorldMDP(MDP[Action, MyWorldState]):
    def __init__(self, world: World):
        self.world = world

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
        return MyWorldState(0, 0, self.world)
        ...

    def available_actions(self, state: MyWorldState) -> list[Action]:
        """returns the actions available to the current agent."""
        world_available_actions = state.world.available_actions()
        print(f"world_available_actions: {world_available_actions}")
        current_agent = state.current_agent
        print(f"current_agent: {current_agent}")
        return world_available_actions[current_agent]

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
        agent_positions = set(state.agents_positions)  
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

    def transition(self, state: MyWorldState, action: Action) -> MyWorldState:
        """Returns the next state and the reward.
        If Agent 0 dies during a transition, 
        the state value immediately drops to 
        lle.REWARD_AGENT_DIED (-1), 
        without taking into account any gems already collected
        """
        self.n_expanded_states += 1
        # current_state = self.world.get_state()
        self.world.set_state(self.convert_to_WorldState(state))

        actions = self.get_actions(state.current_agent, action)
        next_state_value = state.value
        reward = 0
        print(f"actions: {actions}")
        reward = self.world.step(actions)
        world_state_after_action = self.world.get_state()
        if state.current_agent == 0:
            agents_positions = world_state_after_action.agents_positions
            print(f"agents_positions: {agents_positions}")
            print(f"reward: {reward}")
            if reward == -1:
                next_state_value = -1
            else:
                next_state_value += reward
        print(f"next_state_value: {next_state_value}")
        next_state_current_agent = (state.current_agent+1)%self.world.n_agents

        # self.world.set_state(current_state)
        my_world_state_transitioned = MyWorldState(next_state_value, next_state_current_agent, self.world)
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
