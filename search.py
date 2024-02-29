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
import sys
import copy

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

    def goalTest(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getActions(self, state):
        """
        Given a state, returns available actions.
        Returns a list of actions
        """        
        util.raiseNotDefined()

    def getResult(self, state, action):
        """
        Given a state and an action, returns resulting state.
        """
        util.raiseNotDefined()

    def getCost(self, state, action):
        """
        Given a state and an action, returns step cost, which is the incremental cost 
        of moving to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()

class Node:
    """
    Search node object for your convenience.

    This object uses the state of the node to compare equality and for its hash function,
    so you can use it in things like sets and priority queues if you want those structures
    to use the state for comparison.

    Example usage:
    >>> S = Node("Start", None, None, 0)
    >>> A1 = Node("A", S, "North", 4)
    >>> B1 = Node("B", S, "South", 3)
    >>> B2 = Node("B", A1, "West", 6)
    >>> B1 == B2
    True
    >>> A1 == B2
    False
    >>> node_list1 = [B1, B2]
    >>> B1 in node_list1
    True
    >>> A1 in node_list1
    False
    """
    def __init__(self, state, parent, action, path_cost):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost

    def __hash__(self):
        return hash(self.state)

    def __eq__(self, other):
        return self.state == other.state

    def __ne__(self, other):
        return self.state != other.state


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.

    You are not required to implement this.
    """
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def iterativeDeepeningSearch(problem):
    """
    Perform DFS with increasingly larger depth. Begin with a depth of 1 and increment depth by 1 at every step.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.goalTest(problem.getStartState()))
    print("Actions from start state:", problem.getActions(problem.getStartState()))

    Then try to print the resulting state for one of those actions
    by calling problem.getResult(problem.getStartState(), one_of_the_actions)
    or the resulting cost for one of these actions
    by calling problem.getCost(problem.getStartState(), one_of_the_actions)

    """
    "*** YOUR CODE HERE ***"
    
    # Start at depth 1 and increment from there
    _DEPTH = 1
    
    stack = util.Stack()
    visited = set()
    
    start = Node(problem.getStartState(), None, None, 0)
    
    # Keep incrementing the depth until a solution is found
    while True:
        # print("root::Testing depth: ", _DEPTH)
        for action in problem.getActions(problem.getStartState()):
            node = Node(problem.getResult(problem.getStartState(), action), start, action, problem.getCost(problem.getStartState(), action))
            stack.push(node)
            visited.add(node.state)
        
        while not stack.isEmpty():
            node = stack.pop()
            # print("root::Testing action: ", node.action)
            result = depthLimitedSearch([node], problem, _DEPTH - 1, visited)
            if result is None:
                continue
            else:
                solution = []
                for node in result:
                    solution.append(node.action)
                return solution
        visited.clear()
        _DEPTH += 1 
    
def depthLimitedSearch(nodePath, problem, limit, visited):
    """Helper function for IDS to perform DFS with a depth limit.
    Will return the solution path if one is found within the depth limit.

    Args:
        path (list): The current path of actions to the current state
        problem (_type_): The overarching graph and search problem/functions
        limit (_type_): The depth limit for the search
    Returns:
        list: The solution path to the goal state, None if no solution is found
    """
    # Get the current action from the path
    currentNode = nodePath[-1]

    
    # Check if the current state is the goal state
    if problem.goalTest(currentNode.state):
        return nodePath
    
    # Check if the depth limit has been reached
    if limit == 0:
        return None
    
    # Try the actions from the current state
    stack = util.Stack()
    for action in problem.getActions(currentNode.state):
        # print("dfs::Adding action: ", action)
        node = Node(problem.getResult(currentNode.state, action), currentNode, action, problem.getCost(currentNode.state, action))
        if node.state in visited:
            continue
        stack.push(node)
        visited.add(node.state)
    
    # print("dfs::Visited Nodes: ", visited)
    while not stack.isEmpty():
        node = stack.pop()
        # print("dfs::Testing action: ", node.action)
        # Create a new path possibility
        newNodePath = list(nodePath)
        newNodePath.append(node)
        result = depthLimitedSearch(newNodePath, problem, limit - 1, visited)
        if result is None:
            continue
        else:
            return result
    
    return None
        
    
    
    
        

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviations
bfs = breadthFirstSearch
astar = aStarSearch
ids = iterativeDeepeningSearch
