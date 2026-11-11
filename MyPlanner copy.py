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
        state: np.ndarray #current state
        done: bool = #if state is terminal or not
        visit_count: int = 0 #initial number of visits is 0 
        actions: Dict[int, Action_Stats]


class MyPlanner: #MCTS style planner
    def __init__(self, env: MultiDroneUnc, c: float = 1.4, max_rollout_steps: int = 25, max_actions_per_state: int = 64, max_tree_depth: int = 50): #TODO tune hyper-parameters
        self.env = env
        self.c = c #exploration v. exploitation constant for UCB
        self.max_rollout_steps = max_rollout_steps
        self.max_actions_per_state = max_actions_per_state #limits joint actions assessed per state to limit tree branching
        self.max_tree_depth = max_tree_depth
        self.discount_factor = env.get_config().discount_factor
        seed = 1 #fixed seed for experiments
        self.rng = np.random.default_rng(seed=seed)
        

    def plan(self, current_state: np.ndarray, planning_time_per_step: float) -> int:
       
        #check if initial state is terminal (end case):
        position = current_state[:, :3].astype(np.int32) #current positions of drones

        reached = current_state[:, 3].astype(bool) #terminal state where goal is reached

        #drone + drone collisions (still include reached drones as solid)
        _, counts = np.unique(position, axis=0, return_counts=True)
        veh_collisions = int(counts[counts > 1].sum())

        #drone + obstacle collisions
        obstacles_hit = self.env.obstacles[position[:, 0], position[:, 1], position[:, 2]]
        obstacle_collisions = int(obstacles_hit.sum())
        
        done = reached.all() or (obstacle_collisions > 0) or (veh_collisions > 0)

        node = MCTSNode(env = self.env, state= current_state, done= bool(done))

        #initialise start time and number of iterations
        start_time = time.time()
        iterations = 0 

        while time.time() - start_time < planning_time_per_step: #enforcing max planning time per step

            #selection
            node = self._node_selection(node)

            if node.done: #node selected is a terminla state
                break

            #simulation
            reward_sum = self._simulate(node)

            #backpropagation
            self._back_propagate(node, reward_sum)

            iterations += 1 #count iteration just performed

        if not root.children: #if root has no children (no possible next states) do action 0
            return 0 
        
        best_action = max(root.children.items(), key=lambda item: item[1].visit_count)[0] #select the action int corresponding to the transition to the most visited child (we use the most visited since we have lots of info about it, and if it is visited often it will likely have a better Q-value)

        return best_action


    def _node_selection(self, node: State_Node) -> Action_Stats:
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

    P
    def _simulate(self, node: MCTSNode) -> float:
        """
        Simulate from the given node, to estimate its value. 
        """
        
        if node.done: #state is terminal
            return 0.0 #the future discounted reward from a terminal state is 0


        current_state = node.state.copy() #copy so we don't edit the current state
        total_discounted_reward = 0.0 #initialise 
        discount_factor = 1.0 #simulated reward of immediate next state is not discounted

        
        for step in range(self.max_rollout_steps): #rollout to max steps forward, limit computational time per simulation
            
            action = self.heuristic_action_selection(current_state) #select action with heuristic
            next_state, reward, done, _ = self.env.simulate(current_state, action)

            total_discounted_reward += discount_factor * reward #add discounted reward from step forward

            discount_factor *= self.discount_factor #discount_factor gets decreased as we go deeper in the tree, so states more in the future contribute less to estimated reward (since they are less certain)

            current_state = next_state #update current_state for next iteration

            if done: #next_state is terminal so no need for further rollout, return current reward val found
                return total_discounted_reward
                


        return total_discounted_reward #estimated value of input node based on simulated future states



    def heuristic_action_selection(self, state: np.ndarray):
        """
        Selects actions that move drones towards their goals/ greedy action choice.
        Small penalty if drone is close to obstacels (since this increases the chance of an action failing, and a collision occuring.)
        """

        #bounds of env
        max_X, max_Y, max_Z = self.env.cfg.grid_size
        #print(f"untried: {node._untried_actions}")
        drone_positions = state[:, :3]
        goal_positions = self.env.goals
        reached = state[:, 3].astype(bool)

        action_idx_vectors = [(action_int, self.env._action_vectors[self.env._decode_action(action_int)]) for action_int in node._untried_actions]
        
        chosen_moves = np.zeros(self.env.N, dtype=np.int32) #initialise actions for all drones

        for i in range(self.env.N): #for each drone
            if reached[i]: #drone has already reached its goal
                chosen_moves[i] = 0 #arbitrary move since it won't actually be applied
                continue #check next drone's action

            best_action = 0 #initialise
            best_score = np.inf #initialise

            for action_index, joint_action_vec in action_idx_vectors: #for all multi-drone actions

                next_joint_pos = drone_positions + joint_action_vec

                next_pos = next_joint_pos[i] #position of drone i after given action
                #check if action puts drone in bounds
                if 0 <= next_pos[0] < max_X and 0 <= next_pos[1] < max_Y and 0 <= next_pos[2] < max_Z: #next_pos moves drone to a state within the bounds:

                    #check if action causes a collision with obstacles:
                    if self.env.obstacles[tuple(next_pos)]:
                        continue #skip this action since it causes a collision with 1+ obstacle(s)

                    #check drone + drone collisions
                    _, counts = np.unique(next_joint_pos, axis=0, return_counts=True)
                    veh_collisions = int(counts[counts > 1].sum())

                    if veh_collisions > 0:
                        continue #skip this action since it causes a collision between drones
                            
                    #find distance to goal, distance to obstacle
                    goal_dist = np.linalg.norm(next_pos - goal_positions[i]) #distance from next_pos and the relevant goal

                    obs_dist = self.env.dist_field[next_pos[0], next_pos[1], next_pos[2]]
                    
                    """obs_proximity_penalty = 0.0
                    if obs_dist < 1.0: #if a drone is close to an obstacle
                        obs_proximity_penalty = 1.0 #grows rapidly if very close to obstacle
"""
                    obs_proximity_penalty = (1 - obs_dist / (obs_dist + self.env.get_config().alpha)) #failure probability of action
                                    
                    score = goal_dist + 0.5*obs_proximity_penalty #want to minimise this
                    #print(f"score {score}; best score: {best_score}")
                    if score < best_score: #new best dist is found
                        best_score = score
                        best_action = action_index



            chosen_moves[i] = best_action

        return self.env._encode_action(chosen_moves)




    def _back_propagate(self, node: Optional[MCTSNode], reward: float):
        """
        Update Q values and the number of times the current state and state, action pair have been visited.
        """
        current_node = node

        while current_node is not None: #iterate up the tree through parents of the current node, until the root is reached (which has node.parent = None)

            current_node.visit_count += 1 #node has been visited one more time
            current_node.total_reward += reward #add reward found via. simulation
            current_node = current_node.parent #update current_node for next iteration, so we propagate reward values up the tree


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