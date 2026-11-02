from multi_drone import MultiDroneUnc
import numpy as np
from typing import Optional, Tuple, Dict, List
import math
from multi_drone import MultiDroneUnc

#online policy: MCTS
#simulation step already implemented and provided for usage
#expand using step from multi_drone.py

class MyPlanner:
    def __init__(self, env: MultiDroneUnc, a_param: float = 1.0, b_param: int = 1.0):
        self._env = env
        self._a_param = a_param
        self._b_param = b_param

    def plan(self, current_state: np.ndarray, planning_time_per_step: float) -> int:
        # This doesn't do anything useful. It simply returns the action 
        # representen by integer 0.
        return 0

    def back_propagation(self, Qsa: float, Ns: int, Nsa: int, q: float) -> Tuple[float, int, int]:
        """
        Update Q values and the number of times the current state and state, action pair have been visited.

        Args:
            Qsa: Average total discounted reward over the simulations that start from s and perform action a 
                as its first action.
            Ns: Number of times state s has been visited.
            Nsa: Number of times the state action pair (s, a) has been visited.
            q: Total discounted return obtained from the current simulation.

        """
        Qsa_new = (Qsa*Nsa+q)/(Nsa+1)
        Ns += 1
        Nsa += 1

        return Qsa_new, Ns, Nsa
    

    def node_selection(self, state, actions: List[int], c: float):
        #UCB 

        best_action_idx = np.argmax([(MultiDroneUnc.simulate(state, a)+c*math.sqrt(math.log*(Ns)/Nsa)) for a in actions])

        return actions[best_action_idx]