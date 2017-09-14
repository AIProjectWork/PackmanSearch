# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  ["South", s, w, s, w, w, s, w]


"""
****** DFS
"""
result_success = "Success"
result_fail = "failed"


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.
    """
    stack = util.Stack()
    visited = []

    current_state = problem.getStartState()
    starting_node = DfsNode(current_state)
    result = dfs_explore(problem, stack, starting_node, visited)
    if result == result_success:
        return directions_from_stack(stack)
    elif result == result_fail:
        print "No solution found"


def dfs_explore(problem, stack, dfs_node, visited):
    visited.append(dfs_node.get_state())
    stack.push(dfs_node)
    if problem.isGoalState(dfs_node.get_state()):
        return result_success
    successors = problem.getSuccessors(dfs_node.get_state())
    for successor_state, successor_action, successor_cost in reversed(successors):
        if successor_state not in visited:
            result = dfs_explore(problem, stack, DfsNode(successor_state, successor_action), visited)
            if result == result_success:
                return result_success
    stack.pop()
    return result_fail


def directions_from_stack(stack):
    directions = []
    for dfs_node in stack.list:
        if dfs_node.get_action() is not None:
            directions.append(dfs_node.get_action())
    return directions


class DfsNode:

    def __init__(self, state, action=None):
        self.state = state
        self.action = action

    def get_state(self):
        return self.state

    def get_action(self):
        return self.action


'''
*** BFS
'''


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    queue = util.Queue()
    visited = []
    goal_node = None
    path = None
#     add starting Node as the first node of queue
    queue.push(BfsNode(problem.getStartState()))

    while not queue.isEmpty():
        result = bfs_explore(problem, queue, visited)
        if result is not None:
            goal_node = result
            break

    if goal_node is not None:
        # goal node found successfully.
        print "Found solution"
        path = directions_using_bfs_goal_node(goal_node)
    else:
        print "No solution found"
    return path


def bfs_explore(problem, queue, visited):

    # make first queue node as current one to explore it
    current_node = queue.pop()
    visited.append(current_node.get_state())

    # check if current node is the goal node
    if problem.isGoalState(current_node.get_state()):
        return current_node
    else: #else add all unexplored successor nodes into the queue
        #fetch all successors
        successors = problem.getSuccessors(current_node.get_state())
        for successor_state, successor_action, successor_cost in successors:
            if successor_state not in visited:
                successor_node = BfsNode(successor_state, successor_action, current_node)
                queue.push(successor_node)
        return None


def directions_using_bfs_goal_node(goal_node):
    directions = []
    current_node = goal_node
    while current_node.get_action() is not None:
        directions.insert(0, current_node.get_action())
        current_node = current_node.get_parent_node()
    return directions


class BfsNode:

    def __init__(self, state, action=None, parent_node=None):
        self.state = state
        self.action = action
        self.parent_node = parent_node

    def get_parent_node(self):
        return self.parent_node

    def get_state(self):
        return self.state

    def get_action(self):
        return self.action


'''
*** Uniform Cost Search
'''


def ucsExplore(problem, queue, visited):

    # make first queue node as current one to explore it
    current_node = queue.pop()
    visited.append(current_node.get_state())

    # check if current node is the goal node
    if problem.isGoalState(current_node.get_state()):
        return current_node
    else: #else add all unexplored successor nodes into the queue
        #fetch all successors
        successors = problem.getSuccessors(current_node.get_state())
        for successor_state, successor_action, successor_cost in successors:
            if successor_state not in visited:
                successor_node = UcsNode(successor_state, successor_action, current_node, successor_cost)
                queue.push(successor_node, successor_node.get_cost_till_here())
        return None



def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    queue = util.PriorityQueue()
    visited = []
    goal_node = None
    path = None
    #     add starting Node as the first node of queue
    queue.push(UcsNode(problem.getStartState()), 0)

    while not queue.isEmpty():
        result = ucsExplore(problem, queue, visited)
        if result is not None:
            goal_node = result
            break

    if goal_node is not None:
        # goal node found successfully.
        print "Found solution"
        path = directions_using_ucs_goal_node(goal_node)
    else:
        print "No solution found"
    return path


def directions_using_ucs_goal_node(goal_node):
    directions = []
    current_node = goal_node
    while current_node.get_action() is not None:
        directions.insert(0, current_node.get_action())
        current_node = current_node.get_parent_node()
    return directions


class UcsNode:

    def __init__(self, state, action=None, parent_node=None, cost=0):
        self.state = state
        self.action = action
        self.parent_node = parent_node
        self.cost = cost
        if parent_node is not None:
            self.cost_till_here = parent_node.cost_till_here + cost
        else:
            self.cost_till_here = cost

    def get_parent_node(self):
        return self.parent_node

    def get_action(self):
        return self.action

    def get_state(self):
        return self.state

    def get_cost_till_here(self):
        return self.cost_till_here


'''
*** A-Star
'''


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    queue = util.PriorityQueue()
    visited = []
    goal_node = None
    path = None
    #     add starting Node as the first node of queue
    starting_node = AStarNode(problem.getStartState(),None,None,0,heuristic(problem.getStartState(),problem))
    queue.push(starting_node, starting_node.get_total_predicted_cost())

    while not queue.isEmpty():
        result = a_star_explore(problem, queue, visited, heuristic)
        if result is not None:
            goal_node = result
            break

    if goal_node is not None:
        # goal node found successfully.
        print "Found solution"
        path = directions_using_ucs_goal_node(goal_node)
    else:
        print "No solution found"
    return path


def a_star_explore(problem, queue, visited, heuristic):

    # make first queue node as current one to explore it
    current_node = queue.pop()
    visited.append(current_node.get_state())

    # check if current node is the goal node
    if problem.isGoalState(current_node.get_state()):
        return current_node
    else: #else add all unexplored successor nodes into the queue
        #fetch all successors
        successors = problem.getSuccessors(current_node.get_state())
        for successor_state, successor_action, successor_cost in successors:
            if successor_state not in visited:
                heuristic_value = heuristic(successor_state, problem)
                successor_node = AStarNode(successor_state, successor_action,
                                           current_node, successor_cost, heuristic_value)
                queue.push(successor_node, successor_node.get_total_predicted_cost())
        return None


def directions_using_astar_goal_node(goal_node):
    directions = []
    current_node = goal_node
    while current_node.get_action() is not None:
        directions.insert(0, current_node.get_action())
        current_node = current_node.get_parent_node()
    return directions


class AStarNode:

    def __init__(self, state, action=None, parent_node=None, cost=0, heuristic_cost=0):
        self.state = state
        self.action = action
        self.parent_node = parent_node
        self.cost = cost
        if parent_node is not None:
            self.cost_till_here = parent_node.cost_till_here + cost
        else:
            self.cost_till_here = cost
        self.total_predicted_cost = self.cost_till_here + heuristic_cost

    def get_parent_node(self):
        return self.parent_node

    def get_action(self):
        return self.action

    def get_state(self):
        return self.state

    def get_cost_till_here(self):
        return self.cost_till_here

    def get_total_predicted_cost(self):
        return self.total_predicted_cost



# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
