from multi_drone import MultiDroneUnc, _make_action_vectors
import numpy as np
from typing import Optional, Tuple, Dict, List
import math
from multi_drone import MultiDroneUnc
import time
import random
from copy import deepcopy
import itertools
from dataclasses import dataclass, field

#online policy: MCTS
#simulation step already implemented and provided for usage
#expand using step from multi_drone.py

class Action_Stats:
    #allows us to reason about the non-determinisitic effects of actions
    def __init__(self, action: int):
        self.action = action
        self.visit_count: int = 0 #initialise with 0 visits
        self.total_reward: float = 0.0 #initialise
        self.possible_outcomes: Dict[Tuple[int, ...]], "State_Node"] = field(default_factory = dict)

    @property
    def mean_q_value(self) -> float: #mean return over possible stochastic actions of outcomes
        return self.total_reward/self.visits if self.visits else 0.0 #avoid division by zero if action, state pair hasn't been visited
    
class State_Node:
        def __init__(self, state: np.ndarray, done: bool):

            self.state = state #current state
            self.done: bool #if state is terminal or not
            self.visit_count: int = 0 #initial number of visits is 0 
            self.actions: Dict[int, Action_Stats]


class MyPlanner: #MCTS style planner
    def __init__(self, env: MultiDroneUnc, c: float = 1.4, max_rollout_steps: int = 25, max_actions_per_state: int = 64, max_tree_depth: int = 50): #TODO tune hyper-parameters
        self.env = env
        self.c = c #exploration v. exploitation constant for UCB
        self.max_rollout_steps = max_rollout_steps
        self.max_actions_per_state = max_actions_per_state #limits joint actions assessed per state to limit tree branching
        self.max_tree_depth = max_tree_depth
        self.discount_factor = env.get_config().discount_factor
        seed = self.env.cfg.seed #fixed seed for experiments
        self.rng = np.random.default_rng(seed=seed)

        
        
    def plan(self, current_state: np.ndarray, planning_time_per_step: float):
        start_time = time.time()
                iterations = 0 

                current_node = State_Node(current_state)
        
                while time.time() - start_time < planning_time_per_step: #enforcing max planning time per step

                MCTS_iteration(current_node, )
        return action
        
    def MCTS_iteration(self, root: State_Node, planning_time_per_step: float) -> None:
       
        node = root #initialise current node as the root

        root.visit_count += 1 #each MCTS iteration means we visit the root once more
        path = [] #initialise path explored
        depth = 0 #record depth so we can limit depth of search

        for _ in range(self.max_tree_depth):

            if node.done: #node is terminal state
                leaf_return = 0.0 #return value of terminal state 
                break

            #1. Selection: Select an action to perform
            action_stats = self.action_selection(node)

            #2. Expansion:
            next_state, reward, done, info = self.env.simulate(node.state, action = action_stats.action)
            path.append((node, action_stats, reward))
            depth += 1

            outcome_key = self.assign_state_key(next_state) #get dictionary key for next_state
            child = action_stats.possible_outcomes.get(outcome_key) #check if next_state has been observed as a possible outcome of the selected action before

            if child is None: #the observed outcome from the selected action has not been seen before
                child = State_Node(state=next_state, done = done) #
                action_stats.possible_outcomes[outcome_key] = child #add child to observed outcomes for next_state
                child.visit_count += 1

                #3. Simulation: estimate value of next state with rollout
                leaf_reward = self.simulate(child.state, child.done) #rollout 
                break

            #action/outcome pair has been seen before so keep expanding tree
            child.visits += 1 #count visit to child
            node = child #update current node to expand from
        else: #if loop finishes (tree is expanded to max depth), so simluate from current node
            #3. Simulation: estimate value of next state with rollout
            leaf_reward = self.simulate(node.state, node.done) #rollout 

        #4. Backpropagation 
        self.back_propagate(traversed_path=path, leaf_reward=leaf_reward)
        
        return None

    def assign_state_key(self, state: np.ndarray) -> Tuple:
        """
        Flatten np.ndarray for state to a tuple so it can be used as a
        dictionary key. Then we can use dictionary lookup (O(1)) instead of checking
        for list containment (O(n)) when checking if a state is a previouosly observed action outcome
        or not."""
        return tuple(state.flatten().tolist()) 

    def action_selection(self, node: State_Node) -> Action_Stats:
        """Select an unvisited/unused action. If no unvisited nodes use UCB."""

        unvisited_actions = [Action_Stats(a) for a in node.actions if Action_Stats(a).visit_count == 0]

        if unvisited_actions: #still haven't explored all actions
            return unvisited_actions.pop(self.rng.integers(len(unvisited_actions))) #random unvisited action
        else: 
            return best_ucb_action(node)


    def best_ucb_action(self, node:State_Node) -> Action_Stats:     
        #select action using UCB 

        best_ucb_score = -np.inf #initialise score
        best_action = None #initialise best_action

        parent_node_visit_count = max(1, node.visit_count) #take max to prevent log(0)=undefined errors

        for action_stat in node.actions.values():
                exploitation = action_stat.mean_q_value #averaged over possible states stemming from this action
                exploration = self.c * math.sqrt((math.log(parent_node_visit_count))/action_stat.visit_count)
                ucb_score = exploration + exploitation

                if ucb_score > best_ucb_score:
                    best_ucb_score = ucb_score #update best score
                    best_action = action_stat #update best child node 

        return best_action if best_action != None else Action_Stats(1) #give fallback action if optimisation with ucb fails


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


    def random_joint_action(self) -> int: #TODO

        print(f"action vec shape: {self.env._action_vectors.shape}")
        num_actions = int(self.env._action_vectors.shape[0])
        action_int_array = self.rng.integers(low = 0, high =num_actions, size = self.env.N)

        return int(self.env._encode_action(action_int_array)) #gives joint action


 
    def back_propagate(self, traversed_path: List[Tuple[State_Node, Action_Stats, float]], leaf_reward: float):
        """
        Update Q values and the number of times the current state and state, action pair have been visited.
        """
        downstream_reward = leaf_reward #initialise

        for _, action_stat, immediate_reward in reversed(traversed_path): #moving up the traversed path

            action_stat.visit_count += 1
            action_stat.total_reward += immediate_reward + self.discount_factor * downstream_reward

        return None

    

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Add a child node to the tree."""

        if node._untried_actions == []: #no more actions to try, so cannot expand
            return node #return node since we can't expand it
        
        current_state = node.state

        action = node._untried_actions.pop() #pop action to try from shuffled list (so actions are tried in a random order)
        next_state, reward, done, info = self.env.simulate(current_state, action) #child is the result of applying the selected action to the current_state

        #create a child node from the information provided from the simulation
        child = MCTSNode(env=self.env, state=next_state, parent = node, parent_action=action, done=done, info=info)

        child.total_reward = 0.0 #initialise
        child.visit_count = 0 #initialise
        node.children[action] = child #get child from action

        return child