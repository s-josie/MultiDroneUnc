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

class MCTSNode:
    def __init__(self, env: MultiDroneUnc, state: np.ndarray, parent: Optional["MCTSNode"] = None, parent_action: Optional[int] = None, done: bool = False, info: Optional[Dict]=None):
        self.env = env
        self.state = state #current state
        self.parent = parent #parent of current state (if not the root node)
        self.parent_action = parent_action #action taken from parent to current state (if not the root node)

        self.done = done #if state is terminal or not
        self.info = info if info is not None else {} #store info about the num collisions with obstacles, vehicles or if the goal state is reached 
        self.children: Dict[int, MCTSNode] = {} #no children initially, int represents the action int used to reach the child from the current node, while MCTSNode is the child node itself
        self.visit_count = 0 #initial number of visits = 0 
        self.total_reward = 0.0 #initial total reward is 0

        self._untried_actions: list = [] #initialise, this is populated with the top-k ranked actions later
    @property
    def q_value(self) -> float: #average total reward 
        if self.visit_count == 0: #no information about this node yet
            return 0.0 #assign 0 average total reward
        return self.total_reward/self.visit_count #return average total reward
    

    
class MyPlanner: #MCTS style planner
    def __init__(self, env: MultiDroneUnc, c: float = 1.0, max_rollout_steps: int = 25, top_k: int = 8): #TODO tune hyper-parameters
        self.env = env
        self.c = c #exploration v. exploitation constant for UCB
        self.max_rollout_steps = max_rollout_steps
        self.discount_factor = env.get_config().discount_factor
        self.num_actions = env.num_actions #number of possible actions
        self.top_k = top_k

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


    def _node_selection(self, node: MCTSNode):

            while not node.done: #node not terminal, so we can traverse down the tree

                if len(node._untried_actions) == self.num_actions: #true to initialised nodes
                    if len(node.children) == 0: #no children added, but not all actions tried so we populate actions to try
                        node._untried_actions = self.select_top_k_actions(node) #keep only top k actions to reduce A space per node
                        random.shuffle(node._untried_actions) #randomly order the actions so they are popped in a random order"""
                
                if len(node._untried_actions) > 0:
                    return self._expand(node) #still have actions to try so we expand the node
                else:
                    node = self.best_child_ucb(node) #all children added, so we can choose the best child
    
            return node #node is terminal so we don't select a child or expand further



    def select_top_k_actions(self, node:MCTSNode) -> List[int]:

        state = node.state

        max_X, max_Y, max_Z = self.env.cfg.grid_size
        
        positions = state[:, :3]
        reached = state[:, 3].astype(bool)

        action_idx_vectors = [(action_int, self.env._action_vectors[self.env._decode_action(action_int)]) for action_int in node._untried_actions]

        scored_joint_actions = [] #initialise

        
        for action_index, joint_action_vec in action_idx_vectors: #for all multi-drone actions
        
            next_joint_pos = positions + joint_action_vec
            
            #check if action causes a collision between drones
            _, counts = np.unique(next_joint_pos, axis=0, return_counts=True)
            veh_collisions = int(counts[counts > 1].sum())

            if veh_collisions > 0:
                continue #skip this action since it causes a collision between drones

            #initialise info about this action before we assess it per drone
            joint_score = 0.0
            action_is_valid = True

            for i in range(self.env.N): #for each drone

                if reached[i]:
                    continue #drone is already at goal

                next_pos = next_joint_pos[i] #position of drone after action
            
                #check if action puts drone outside bounds
                if not (0 <= next_pos[0] < max_X and 0 <= next_pos[1] < max_Y and 0 <= next_pos[2] < max_Z):
                    action_is_valid = False
                    break #evaluate next action
    
                #check if action causes a collision with obstacles:
                if self.env.obstacles[tuple(next_pos)]:
                    action_is_valid = False
                    break #skip this action since it causes a collision with 1+ obstacle(s)
    
                obs_dist = self.env.dist_field[next_pos[0], next_pos[1], next_pos[2]]
                
                obs_proximity_penalty = (1 - obs_dist / (obs_dist + self.env.get_config().alpha)) #failure probability of action
                
                #find distance to goal, distance to obstacle
                goal_dist = np.linalg.norm(next_pos - self.env.goals[i]) #distance from next_pos and the relevant goal

                #accumulate score of this action across all drones
                joint_score += goal_dist + obs_proximity_penalty #want to minimise this

            if action_is_valid: #only add moves that dont cause failure
                scored_joint_actions.append((action_index, joint_score))
                
            scored_joint_actions.sort(key=lambda x : x[1]) #sort by score, ascending order (best to worst)

            top_joint_actions = [item[0] for item in scored_joint_actions[:self.top_k]]

            if top_joint_actions == []: #all actions are pruned, so return dummy action
                return [0]

        return top_joint_actions
    
    
    def best_child_ucb(self, node:MCTSNode) -> MCTSNode:
                    
        #select child using UCB 
        best_ucb_score = -np.inf #initialise score
        best_child = None #initialise best_child

        parent_node_visit_count = max(1, node.visit_count) #take max to prevent log(0)=undefined errors

        for child in node.children.values(): 
            if child.visit_count == 0: # no info about child, so set score high so its explored
                ucb_score = np.inf
                best_child = child
            else:
                exploitation = child.q_value
                exploration = self.c * math.sqrt((math.log(parent_node_visit_count))/child.visit_count) #c*\sqrt(\frac{N(s)}{N(s,a)})
                ucb_score = exploration + exploitation

                if ucb_score > best_ucb_score:
                    best_ucb_score = ucb_score #update best score
                    best_child = child #update best child node 

        return best_child if best_child != None else node
    
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