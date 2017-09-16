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
****** Depth First Search (DFS)
"""
result_success = "Success"
result_fail = "failed"


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    """
    # this will hold all the nodes we need to visit. It follows Last In First Out.
    # so whatever node is inserted last, will be picked first to explore.
    stack = util.Stack()

    # visited list is a list of all visited node.
    # In graph, it is possible that same node is connected by multiple branches. This may cause cyclic path.
    # To avoid cyclic path of graph, we will check if node is never visited before inserting into the stack
    visited = []

    # Prepare first node.
    # A DfsNode contains
    #   state (position in terms of X,Y)
    #   action (direction that requires to reach this node from previous)
    starting_state = problem.getStartState()  # initial X,Y position
    starting_node = DfsNode(starting_state)  # starting node with initial position. Action is None for this
    result = dfs_explore(problem, stack, starting_node, visited)  # starts exploring from start node
    if result == result_success:  # when goal is found at some depth
        return directions_from_stack(stack)  # returns list of direction from stack
    elif result == result_fail:  # when goal is not found after exploring entire graph
        print "No solution found"


def dfs_explore(problem, stack, dfs_node, visited):
    """
    This function will explore the given dfs_node
    if dfs_node is the goal, then it will return result_success
    else it will call recursive dfs_explore function for all of it's successor to find goal
    :param problem: problem object
    :param stack: main stack used to store nodes
    :param dfs_node: current node to explore
    :param visited: list of visited states
    :return: result_success if current node or it's  Descendant is goal
             result_fail otherwise
    """
    # mark current state as visited
    visited.append(dfs_node.get_state())

    # add node to the stack
    stack.push(dfs_node)

    # check if current state is goal
    if problem.isGoalState(dfs_node.get_state()):
        return result_success  # return result_success if current node is goal

    # if current node is not the goal, go for it's successors' exploration
    successors = problem.getSuccessors(dfs_node.get_state())
    for successor_state, successor_action, successor_cost in successors:
        if successor_state not in visited:
            # recursive call for successor
            result = dfs_explore(problem, stack, DfsNode(successor_state, successor_action), visited)
            if result == result_success:
                return result_success
    # if no descendant is goal, pop out the item from stack and return result_fail
    stack.pop()
    return result_fail


def directions_from_stack(stack):
    """
    Program expects a list of actions that leads to the goal
    For DFS, it can be derived from stack.
    This function will prepare list of directions from the nodes of stack
    :param stack: it is a stack of nodes [{state:((1,2), action:None},{state:((1,3), action:North},{state:((2,3),
                    action:East}]
    :return: list of directions [North, East]
    """
    directions = []
    for dfs_node in stack.list:
        if dfs_node.get_action() is not None:
            directions.append(dfs_node.get_action())
    return directions


class DfsNode:

    """
    This is a node structure used for DFS problem. It contains state and action to reach that state.
    """

    def __init__(self, state, action=None):
        self.state = state
        self.action = action

    def get_state(self):
        return self.state

    def get_action(self):
        return self.action


"""
*** Breadth First Search (BFS)
"""


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    queue = util.Queue()
    visited = []
    goal_node = None
    path = None
#     add starting Node as the first node of queue
    visited.append(problem.getStartState())
    queue.push(BfsNode(problem.getStartState()))

    while not queue.isEmpty():
        result = bfs_explore(problem, queue, visited)
        if result is not None:
            goal_node = result
            break

    if goal_node is not None:
        # goal node found successfully.
        path = directions_using_bfs_goal_node(goal_node)
    else:
        print "No solution found"
    return path


def bfs_explore(problem, queue, visited):
    # make first queue node as current one to explore it
    current_node = queue.pop()

    # check if current node is the goal node
    if problem.isGoalState(current_node.get_state()):
        return current_node
    else:
        # else add all unexplored successor nodes into the queue
        # fetch all successors
        successors = problem.getSuccessors(current_node.get_state())
        for successor_state, successor_action, successor_cost in successors:
            if successor_state not in visited:
                visited.append(successor_state)
                successor_node = BfsNode(successor_state, successor_action, current_node)
                # todo: discuss pros and cons of commented code
                # if problem.isGoalState(successor_node.get_state()):
                #     return successor_node
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


"""
*** Uniform Cost Search
"""



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
    visited.append([problem.getStartState(), 0])
    queue.push(UcsNode(problem.getStartState()), 0)

    while not queue.isEmpty():
        result = ucsExplore(problem, queue, visited)
        if result is not None:
            goal_node = result
            break

    if goal_node is not None:
        # goal node found successfully.
        path = directions_using_ucs_goal_node(goal_node)
    else:
        print "No solution found"
    return path


def ucsExplore(problem, queue, visited):

    # make first queue node as current one to explore it
    current_node = queue.pop()

    # check if current node is the goal node
    if problem.isGoalState(current_node.get_state()):
        return current_node
    else:
        # else add all unexplored successor nodes into the queue
        # fetch all successors
        successors = problem.getSuccessors(current_node.get_state())
        for successor_state, successor_action, successor_cost in successors:
            successor_node = UcsNode(successor_state, successor_action, current_node, successor_cost)
            if is_an_acceptable_ucs_node(successor_node, visited):
                visited.append([successor_state, successor_node.get_cost_till_here()])
                queue.push(successor_node, successor_node.get_cost_till_here())
        return None


def is_an_acceptable_ucs_node(ucs_node, visited):
    for visited_state, visited_cost in visited:
        if ucs_node.get_state() == visited_state:
            return visited_cost > ucs_node.get_cost_till_here()
    return True


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


"""
*** A-Star
"""


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
    starting_node = AStarNode(problem.getStartState(), None, None, 0, heuristic(problem.getStartState(), problem))
    visited.append([starting_node.get_state(), starting_node.get_cost_till_here()])
    queue.push(starting_node, starting_node.get_total_predicted_cost())

    while not queue.isEmpty():
        result = a_star_explore(problem, queue, visited, heuristic)
        if result is not None:
            goal_node = result
            break

    if goal_node is not None:
        # goal node found successfully.
        path = directions_using_ucs_goal_node(goal_node)
    else:
        print "No solution found"
    return path


def a_star_explore(problem, queue, visited, heuristic):

    # make first queue node as current one to explore it
    current_node = queue.pop()

    # check if current node is the goal node
    if problem.isGoalState(current_node.get_state()):
        return current_node
    else:
        # else add all unexplored successor nodes into the queue
        # fetch all successors
        successors = problem.getSuccessors(current_node.get_state())
        for successor_state, successor_action, successor_cost in successors:
            heuristic_value = heuristic(successor_state, problem)
            successor_node = AStarNode(successor_state, successor_action,
                                       current_node, successor_cost, heuristic_value)
            if is_an_acceptable_a_star_node(successor_node, visited):
                visited.append([successor_node.get_state(), successor_node.get_cost_till_here()])
                queue.push(successor_node, successor_node.get_total_predicted_cost())
        return None


def directions_using_astar_goal_node(goal_node):
    directions = []
    current_node = goal_node
    while current_node.get_action() is not None:
        directions.insert(0, current_node.get_action())
        current_node = current_node.get_parent_node()
    return directions


def is_an_acceptable_a_star_node(a_star_node, visited):
    for visited_state, visited_cost in visited:
        if a_star_node.get_state() == visited_state:
            return visited_cost > a_star_node.get_cost_till_here()
    return True


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

# =====================================================================================================================#


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
