# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent & AlphaBetaPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)
    
    def _populateTree(self, gameState):
        # Check the depth
        if self.depth == 0:
            raise Exception("MinimaxAgent::getAction - Depth is 0")
        
        # Set the current agent to be processed
        currentAgentIndex = 0
        currentDepth = 0
        
        # Create the minmax tree
        startNode = TreeNode(gameState, None, currentDepth, currentAgentIndex)
        tree = Tree(startNode)
        
        _totalDepth = self.depth * gameState.getNumAgents()
        
        while currentDepth < _totalDepth:
            # Get the leafs of the current tree 
            leafs = tree.leafs()
            
            # Get the agent for the children
            # currentAgentIndex = (currentAgentIndex + 1) % gameState.getNumAgents() # Agent to move at this given depth
            
            # For each leaf, get the legal actions of the current agent and create a child for each action
            for leaf in leafs:
                actions = leaf.data.getLegalActions(currentAgentIndex)
                for action in actions:
                    child = TreeNode(leaf.data.generateSuccessor(currentAgentIndex, action), action, currentDepth, (currentAgentIndex + 1) % gameState.getNumAgents())
                    tree.insert(leaf, child)
                        
            currentDepth+= 1
            currentAgentIndex = (currentAgentIndex + 1) % gameState.getNumAgents()
        
        return tree

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        
        
        tree = self._populateTree(gameState) 
        
        # Get the leaves of the tree
        leafs = tree.leafs()
        
        # Score the leaves
        toBeCalculated = util.PriorityQueue()
        for leaf in leafs:
            leaf.score = self.evaluationFunction(leaf.data)
            toBeCalculated.push(leaf, self.depth - leaf.depth)
            
        # Calculate the parents of the calculated children one depth at a time using the queue
        while not toBeCalculated.isEmpty():
            
            # Get a node that is already calculated from the front of the queue
            node = toBeCalculated.pop()
            
            # Find its parent
            parent = node.parent
            
            # Check if the parent is the root
            if tree.isRoot(parent):
                continue # do be delt with seperately  
        
            # check if there is a parent
            if parent.parent is None:
                raise Exception("The parent of the current node is none even though this should not happen. The root of the tree is:", tree.root, "and the current node is", parent)
            
            # Double check that all the children of the parent have been calculated
            if not all([child.score != None for child in parent.children]):
                raise Exception("MinimaxAgent::getAction - Not all children of the parent have been calculated:", parent.children)
            
            # Calculate the score of the parent
            if parent.agent == 0: # Max
                parent.score = max(parent.children, key=lambda node: node.score).score
            else: # Min
                parent.score = min(parent.children, key=lambda node: node.score).score
            
            # Add the parent to the queue
            toBeCalculated.push(parent, self.depth - parent.depth)
        
        # Double check that all root children have been calculated
        if not all([child.score != None for child in tree.root.children]):
            raise Exception("MinimaxAgent::getAction - Not all children of the root have been calculated", [child.score for child in tree.root.children])
        
        # Return the action that has the highest score
        bestChild = max(tree.root.children, key=lambda node: node.score)
        return bestChild.action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        
        tree = self._populateTree(gameState) 
        
        # Get the leaves of the tree
        leafs = tree.leafs()
        
        # Score the leaves
        toBeCalculated = util.PriorityQueue()
        for leaf in leafs:
            leaf.score = self.evaluationFunction(leaf.data)
            toBeCalculated.push(leaf, self.depth - leaf.depth)
            
        # Calculate the parents of the calculated children one depth at a time using the queue
        while not toBeCalculated.isEmpty():
            
            # Get a node that is already calculated from the front of the queue
            node = toBeCalculated.pop()
            
            # Find its parent
            parent = node.parent
            
            # Check if the parent is the root
            if tree.isRoot(parent):
                continue # do be delt with seperately  
        
            # check if there is a parent
            if parent.parent is None:
                raise Exception("The parent of the current node is none even though this should not happen. The root of the tree is:", tree.root, "and the current node is", parent)
            
            # Double check that all the children of the parent have been calculated
            if not all([child.score != None for child in parent.children]):
                raise Exception("MinimaxAgent::getAction - Not all children of the parent have been calculated:", parent.children)
            
            # Calculate the score of the parent
            if parent.agent == 0: # Max
                parent.score = max(parent.children, key=lambda node: node.score).score
            else: # Min
                parent.score = self._expected(parent.children, key=lambda node: node.score)
            
            # Add the parent to the queue
            toBeCalculated.push(parent, self.depth - parent.depth)
        
        # Double check that all root children have been calculated
        if not all([child.score != None for child in tree.root.children]):
            raise Exception("MinimaxAgent::getAction - Not all children of the root have been calculated", [child.score for child in tree.root.children])
        
        # Return the action that has the highest score
        bestChild = max(tree.root.children, key=lambda node: node.score)
        return bestChild.action
    
    def _expected(self, iterable, key=None) -> float:
        if key is None:
            key = lambda x: x
        
        return float(sum(key(x) for x in iterable)) / float(len(iterable))

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 9).

    DESCRIPTION: This is a linear combination of four game data weighted to
    determine the score of a given game state. Starting from the most weighted
    to least:
    
    Game Score - the game score of a given state should be at least one of the more
    important ones. We are trying to maxamize this parameter afterall.
    
    Ghost Distance - to survive the longest, we want to keep away from the ghosts as
    much as possible
    
    Food Amount - the less food there is on the board, the closer we are to winning
    
    Capsule Amount - the less capsules we have the closer we are to completing the board
    
    Valid Moves - in case of two boards that have close to equal score, the state in which
    PacMan has more possibilities is the better option
    """
    "*** YOUR CODE HERE ***"
    
    numOfFood = currentGameState.getNumFood()
    
    remainingCapsules = currentGameState.getCapsules()
    numberOfCapsules = len(remainingCapsules)
    
    currentGameStateScore = currentGameState.getScore()
    
    pacManPos = currentGameState.getPacmanPosition()
    ghostsPos = currentGameState.getGhostPositions()
    
    pacManActions = currentGameState.getLegalActions(0)
    numOfPacManActions = len(pacManActions)
    
    
    score = 0.00
    weights = [1.75, 0.75, 2.75, 2.25, 0.1]
    
    # Having less food is better (that means we've eaten more)
    score += weights[0] * float(1.0 / float(numOfFood + 1))
    
    # Having less capsules is also better (that means we've completed more of the board)
    score += weights[1] * float(1.0 / float(numberOfCapsules + 1))
    
    # The current score is also important
    if currentGameStateScore != 0:
        score += weights[2] * float( 1.0 / float(currentGameStateScore))
    
    # Keep away from the ghosts
    for ghost in ghostsPos:
        distanceBetweenPacManAndGhost = util.manhattanDistance(pacManPos, ghost)
        if distanceBetweenPacManAndGhost != 0:
            score += weights[3] * float(1.0 / float(distanceBetweenPacManAndGhost))
        else:
            score -= weights[3]
    
    # In close calls, the move with more moves is better
    score += weights[4] * float( 1.0 / float(numOfPacManActions + 1))
    
    
    return score

# Abbreviation
better = betterEvaluationFunction


########################
#    Helper Classes    #
########################
class TreeNode:
    
    def __init__(self, data, action, depth, agent):
        self.data = data
        self.action = action
        self.children = []
        self.score = None
        self.parent = None
        self.depth = depth
        self.agent = agent
    
    def _noscore(self) -> bool:
        return self.score == None
    
    def __hash__(self):
        return hash(self.data)
    
    def __eq__(self, other):
        return self.data is other.data
    
    def __ne__(self, other):
        return self.data is not other.data

class Tree:
    
    def __init__(self, root):
        self.root = root
        self.tree = []
        self.tree.append(root)
    
    def insert(self, parent, child):
        parent.children.append(child)
        child.parent = parent
        self.tree.append(child)
    
    def leafs(self):
        return [node for node in self.tree if node.children == []]

    def isRoot(self, node):
        return node is self.root

    def _print(self, node, depth) -> str:
        string = ""
        for _ in range(depth):
            string += "| "
        string += str(node.action) + " - " + str(node.score) + " - " + str(node.agent) + "\n"
        for child in node.children:
            string += self._print(child, depth + 1)
        return string
    
    def __repr__(self) -> str:
        print()
        return self._print(self.root, 0)
            
    