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

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    stack = util.Stack()
    visited = []

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())

    current_state = problem.getStartState()

    result = dfsExplore(problem, stack, current_state, None, visited)
    if result == "goal":
        printStack(stack)
        return directions_from_stack(stack)
    elif result == "end":
        print "No solution found"


def directions_from_stack(stack):
    directions = []
    for tuple in stack.list:
        if tuple[1] is not None:
            directions.append(tuple[1])

    return directions


def dfsExplore(problem, stack, current_state, action, visited):
    visited.append(current_state)
    stack.push([current_state, action])
    print "exploring", current_state
    if problem.isGoalState(current_state):
        return "goal"

    successors = problem.getSuccessors(current_state)

    for nextState, action, cost in reversed(successors):
        if nextState not in visited:
            result = dfsExplore(problem, stack, nextState, action, visited)
            if result == "goal":
                return "goal"

    stack.pop()
    return "end"

def printStack(stack):
    print "Final stack is", stack.list

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    queue = util.Queue()
    visited = []
    goalNode = None
    path = None
#     add starting Node as the first node of queue
    queue.push(BfsNode(problem.getStartState()))

    while not queue.isEmpty():
        result = bfsExplore(problem, queue, visited)
        if result is not None:
            goalNode = result
            break

    if goalNode is not None:
        # goal node found successfully.
        print "Found solution"
        path = directions_using_bfs_goal_node(goalNode)
    else:
        print "No solution found"
    return path


def bfsExplore(problem, queue, visited):

    # make first queue node as current one to explore it
    currentNode = queue.pop()
    visited.append(currentNode.getState())

    # check if current node is the goal node
    if problem.isGoalState(currentNode.getState()):
        return currentNode
    else: #else add all unexplored successor nodes into the queue
        #fetch all successors
        successors = problem.getSuccessors(currentNode.get_state())
        for successorState, successorAction, successorCost in successors:
            if successorState not in visited:
                successorNode = BfsNode(successorState, successorAction, currentNode)
                queue.push(successorNode)
        return None

def ucsExplore(problem, queue, visited):

    # make first queue node as current one to explore it
    currentNode = queue.pop()
    visited.append(currentNode.get_state())

    # check if current node is the goal node
    if problem.isGoalState(currentNode.get_state()):
        return currentNode
    else: #else add all unexplored successor nodes into the queue
        #fetch all successors
        successors = problem.getSuccessors(currentNode.get_state())
        for successorState, successorAction, successorCost in successors:
            if successorState not in visited:
                successorNode = UcsNode(successorState, successorAction, currentNode, successorCost)
                queue.push(successorNode, successorNode.get_cost_till_here())
        return None


def directions_using_bfs_goal_node(goalNode):
    directions = []
    currentNode = goalNode
    while currentNode.getAction() is not None:
        directions.insert(0, currentNode.getAction())
        currentNode = currentNode.getParentNode()
    return directions

def directions_using_ucs_goal_node(goalNode):
    directions = []
    currentNode = goalNode
    while currentNode.get_action() is not None:
        directions.insert(0, currentNode.get_action())
        currentNode = currentNode.get_parent_node()
    return directions

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    queue = util.PriorityQueue()
    visited = []
    goalNode = None
    path = None
    #     add starting Node as the first node of queue
    queue.push(UcsNode(problem.getStartState()), 0)

    while not queue.isEmpty():
        result = ucsExplore(problem, queue, visited)
        if result is not None:
            goalNode = result
            break

    if goalNode is not None:
        # goal node found successfully.
        print "Found solution"
        path = directions_using_ucs_goal_node(goalNode)
    else:
        print "No solution found"
    return path


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
    goalNode = None
    path = None
    #     add starting Node as the first node of queue
    starting_node = AStarNode(problem.getStartState(),None,None,0,heuristic(problem.getStartState(),problem))
    queue.push(starting_node, starting_node.get_total_predicted_cost())

    while not queue.isEmpty():
        result = aStarExplore(problem, queue, visited, heuristic)
        if result is not None:
            goalNode = result
            break

    if goalNode is not None:
        # goal node found successfully.
        print "Found solution"
        path = directions_using_ucs_goal_node(goalNode)
    else:
        print "No solution found"
    return path

def aStarExplore(problem, queue, visited, heuristic):

    # make first queue node as current one to explore it
    currentNode = queue.pop()
    visited.append(currentNode.get_state())

    # check if current node is the goal node
    if problem.isGoalState(currentNode.get_state()):
        return currentNode
    else: #else add all unexplored successor nodes into the queue
        #fetch all successors
        successors = problem.getSuccessors(currentNode.get_state())
        for successorState, successorAction, successorCost in successors:
            if successorState not in visited:
                heuristic_value = heuristic(successorState, problem)
                successorNode = AStarNode(successorState, successorAction, currentNode, successorCost, heuristic_value)
                queue.push(successorNode, successorNode.get_total_predicted_cost())
        return None

def directions_using_astar_goal_node(goalNode):
    directions = []
    currentNode = goalNode
    while currentNode.get_action() is not None:
        directions.insert(0, currentNode.get_action())
        currentNode = currentNode.get_parent_node()
    return directions



class BfsNode:

    def __init__(self, state, action=None, parent_node=None):
        self.state = state
        self.action = action
        self.parentNode = parent_node

    def getParentNode(self):
        return self.parentNode

    def getState(self):
        return self.state

    def getAction(self):
        return self.action


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
