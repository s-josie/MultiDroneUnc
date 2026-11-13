from multi_drone import MultiDroneUnc
import numpy as np
from typing import Optional, Tuple, Dict, List
import math
from multi_drone import MultiDroneUnc
from time import perf_counter
from dataclasses import dataclass, field

#online policy: MCTS
#simulation step already implemented and provided for usage
#expand using step from multi_drone.py

@dataclass
class Action_Stats:
    """This class allows us to reason about the non-determinisitic effects of actions, by recording
    the observed outcomes of each action (from a given state), and allowing a mean Q-value (over all possible
    stochastic outcomes) to be calculated.
    """
    
    action: int 
    visit_count: int = 0 #initialise with 0 visits
    total_reward: float = 0.0 #initialise
    possible_outcomes: Dict[Tuple[int, ...], "State_Node"] = field(default_factory = dict) #initialise dictionary containing possible successor states from the action (given a start state)

    @property
    def mean_q_value(self) -> float: #mean return over possible stochastic actions of outcomes
        return self.total_reward/self.visit_count if self.visit_count else 0.0 #avoid division by zero if action, state pair hasn't been visited

@dataclass 
class State_Node:
    state: np.ndarray #current state
    done: bool #if state is terminal or not
    actions: Dict[int, Action_Stats] #store all avaliable actions, and information about their outcomes (in Action_Stats)
    visit_count: int = 0 #initial number of visits is 0 


class MyPlanner: #MCTS style planner
    def __init__(self, env: MultiDroneUnc, c: float = 1.4, max_rollout_steps: int = 25, max_actions_per_state: int = 64, max_tree_depth: int = 50):
        self.env = env
        self.c = c #exploration v. exploitation constant for UCB
        self.max_rollout_steps = max_rollout_steps
        self.max_actions_per_state = max_actions_per_state #limits joint actions assessed per state to limit tree branching
        self.max_tree_depth = max_tree_depth #limits tree depth to prevent excessively large trees
        self.discount_factor = env.get_config().discount_factor
        seed = self.env.cfg.seed #fixed seed for experiments
        self.rng = np.random.default_rng(seed=seed)
        self.last_stats = {} #stores information about the last run for debugging and evaluation
        

    def plan(self, current_state: np.ndarray, planning_time_per_step: float):
        """
        Use the implemented MCTS with UCB for planning.
        """

        start_time = perf_counter()
        iterations = 0 

        root_node = self.make_state_node(current_state, done = self.is_terminal(current_state)) #initialise start node

        if root_node.done: #edge case: root node is terminal
            return 0 #return dummy action (that won't ever be executed)

        while perf_counter() - start_time < planning_time_per_step: #run MCTS while there is time, MCTS is an anytime algorithm so we are guaranteed a solution when time runs out
            self.MCTS_iteration(root_node)
            iterations += 1

        #choose action with most visits, if there is a tie choose the higher q value
        selected_action = max(root_node.actions.values(), key=lambda a: (a.visit_count, a.mean_q_value))

        self.last_stats = {"elapsed_time": perf_counter() - start_time,
                           "iterations": iterations} #record information about the last iteration of the planner

        return selected_action.action #return selected action for execution 

        
    def MCTS_iteration(self, root: State_Node) -> None:
        """
        Performs one iteration of MCTS with random rollout and UCB selection.
        """
       
        node = root #initialise current node as the root

        root.visit_count += 1 #each MCTS iteration means we visit the root once more
        path = [] #initialise path explored

        for _ in range(self.max_tree_depth): #limit depth of search for computational efficiency

            if node.done: #node is terminal state
                leaf_reward = 0.0 #return value of terminal state 
                break

            #1. Selection: Select an action to perform
            action_stats = self.action_selection(node)

            #2. Expansion:
            next_state, reward, done, info = self.env.simulate(node.state, action = action_stats.action)
            path.append((node, action_stats, reward))

            outcome_key = (self.assign_state_key(next_state), done) #get dictionary key for next_state, include done to distinguish between states that were reverted due to a drone collision
            child = action_stats.possible_outcomes.get(outcome_key) #check if next_state has been observed as a possible outcome of the selected action before

            if child is None: #the observed outcome from the selected action has not been seen before
                child = self.make_state_node(state=next_state, done = done) #make child node for child state
                child.visit_count += 1 #increment visit count
                action_stats.possible_outcomes[outcome_key] = child #add child to observed outcomes for next_state

                #3. Simulation: estimate value of next state with rollout
                leaf_reward = self.simulate(child.state, child.done) #rollout 
                break

            #action/outcome pair has been seen before so keep expanding tree
            child.visit_count += 1 #count visit to child
            node = child #update current node to expand from
        else: #loop finishes (i.e. tree is expanded to max depth), so simluate from current node
            #3. Simulation: estimate value of next state with rollout
            leaf_reward = self.simulate(node.state, node.done) #rollout 

        #4. Backpropagation 
        self.back_propagate(traversed_path=path, leaf_reward=leaf_reward)
        
        return None

    def is_terminal(self, state: np.ndarray) -> bool:
        """
        Checks if a state is terminal.
        """

        #return true if state is terminal, False otherwise
        drone_positions = state[:, :3]

        #check if state is in bounds
        max_X, max_Y, max_Z = self.env.cfg.grid_size

        in_bounds = ((drone_positions[:, 0] >= 0) & (drone_positions[:, 0] < max_X) & 
                     (drone_positions[:, 1] >= 0) & (drone_positions[:, 1] < max_Y) &
                     (drone_positions[:, 2] >= 0) & (drone_positions[:, 2] < max_Z))

        if not np.all(in_bounds): #if any drone is out of the bounds of the env
            return True #state is terminal

        #check if any drones have collided with obstacles
        if np.any(self.env.obstacles[drone_positions[:,0], drone_positions[:,1], drone_positions[:,2]]):
            return True #state is terminal

        #check if drones have collided with each other
        _, counts = np.unique(drone_positions, axis=0, return_counts=True)
        num_drone_collisions = int(counts[counts > 1].sum())
        if num_drone_collisions > 0: #at least one collision between drones
            return True #state is terminal

        #check if all goals are reached
        reached = state[:, 3].astype(bool)
        if np.all(reached): #all goals are reached
            return True #state is terminal

        return False #passed all terminal state checks so state isn't terminal

    def get_candidate_joint_actions(self, state: np.ndarray) -> List[int]:
        """
        Generate a subset of all possible actions to limit tree branching. 
        """
        total_possible_actions = self.env.num_actions #full set of all possible actions
    
        if total_possible_actions <= self.max_actions_per_state: #number of actions is below limit on number of actions to consider
            return list(range(total_possible_actions)) #return all possible actions

        reached = state[:,3].astype(bool) #boolean for each drone, indicating if it has reached the goal or not

        #the number of actions is impacted by how many drones have reached their goals (since those drones don't move)
        num_active_drones = len(np.flatnonzero(~reached)) #count drones that haven't reached their goals yet, and are thus still moveable
        num_individual_actions = self.env._action_vectors.shape[0]
        effective_action_count = num_active_drones * num_individual_actions #number of valid actions, given the number of drones at their goal already

        candidates = set() #use list so duplicates aren't accidentally added
        while len(candidates) < min(self.max_actions_per_state, effective_action_count): #to avoid endless loop, aim to get minimum of avaliable actions (given drones at goals) and the target number
            individual_actions = self.rng.integers(low=0, high= self.env._action_vectors.shape[0], size= self.env.N) #get random, per-drone actions

            individual_actions[reached] = 0 #set 0 action for drones that have already reached their goals. This prevents effective duplicate actions from being added to candidate set.

            joint_action = self.env._encode_action(individual_actions) #make joint action for all drones

            candidates.add(joint_action) 

        return list(candidates) #return list of candidate actions

    def make_state_node(self, state:np.ndarray, done:bool) -> State_Node:
        """
        Constructs a state node from a state, and its terminal status.
        """
        candidate_actions = [] if done else self.get_candidate_joint_actions(state) #randomly sample a subset of actions as candidates
        actions = {a: Action_Stats(action=a) for a in candidate_actions} #make dictionary storing action, action statistics pairs for candidate actions
        return State_Node(state=state.copy(), done = done, actions = actions)
    
    def assign_state_key(self, state: np.ndarray) -> Tuple:
        """
        Flatten np.ndarray for state to a tuple so it can be used as a
        dictionary key. Then we can use dictionary lookup (O(1)) instead of checking
        for list containment (O(n)) when checking if a state is a previouosly observed action outcome
        or not."""
        return tuple(state.flatten().tolist()) 

    def action_selection(self, node: State_Node) -> Action_Stats:
        """Select an unvisited/unused action. If no unvisited nodes use UCB."""

        unvisited_actions = [action_stats for action_stats in node.actions.values() if action_stats.visit_count == 0]

        if unvisited_actions: #still haven't explored all actions
            return unvisited_actions[self.rng.integers(len(unvisited_actions))] #randomly select unvisited action
        else: 
            return self.best_ucb_action(node) #use UCB for action selection


    def best_ucb_action(self, node:State_Node) -> Action_Stats:     
        """Select an action using UCB to balance exploration and exploitation."""

        best_ucb_score = -np.inf #initialise score
        best_action = None #initialise best_action

        parent_node_visit_count = max(1, node.visit_count) #take max to prevent log(0)=undefined errors

        for action_stat in node.actions.values(): #for each action
                exploitation = action_stat.mean_q_value #averaged q-value over possible successor states stemming from this action
                exploration = self.c * math.sqrt((math.log(parent_node_visit_count))/action_stat.visit_count) #increase weight of more unexplored actions
                ucb_score = exploration + exploitation

                if ucb_score > best_ucb_score: #new best is found
                    best_ucb_score = ucb_score #update best score
                    best_action = action_stat #update best child node 

        if best_action is None:
            raise RuntimeError("No avaliable actions for UCB Selection.")
        return best_action


    def simulate(self, state: np.ndarray, done: bool) -> float:
        """
        Simulate from the given node, to estimate its value. 
        """
        
        if done: #state is terminal
            return 0.0 #the future discounted reward from a terminal state is 0


        current_state = state.copy() #copy so we don't edit the current state
        total_discounted_reward = 0.0 #initialise 
        discount_factor = 1.0 #simulated reward of immediate next state is not discounted

    
        for _ in range(self.max_rollout_steps): #rollout to max steps forward, limit computational time per simulation

            action_int = self.random_joint_action() #select action randomly
            
            next_state, reward, terminal, info = self.env.simulate(current_state, action_int)

            total_discounted_reward += discount_factor * reward #add discounted reward from step forward

            discount_factor *= self.discount_factor #discount_factor gets decreased as we go deeper in the tree, so states more in the future contribute less to estimated reward (since they are less certain)

            current_state = next_state #update current_state for next iteration

            if terminal: #next_state is terminal so no need for further rollout, return current reward val found
                break
                
        return total_discounted_reward #estimated value of input node based on simulated future states

    def random_joint_action(self) -> int:
        """
        Gives a random joint action."""

        num_actions = self.env._action_vectors.shape[0] #get the number of actions
        action_int_array = self.rng.integers(low = 0, high =num_actions, size = self.env.N) #randomly select an action integer for each drone

        return int(self.env._encode_action(action_int_array)) #gives joint action


    def back_propagate(self, traversed_path: List[Tuple[State_Node, Action_Stats, float]], leaf_reward: float):
        """
        Update Q values and the number of times the current state and state, action pair have been visited.
        """
        downstream_reward = leaf_reward #initialise

        for _, action_stat, immediate_reward in reversed(traversed_path): #moving up the traversed path

            downstream_reward = immediate_reward + self.discount_factor * downstream_reward
            
            action_stat.visit_count += 1 #increment (action, state) pair visit count
            action_stat.total_reward += downstream_reward #update total reward
            
        return None

    

    