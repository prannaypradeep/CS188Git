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

        "*** YOUR CODE HERE ***"
        ghostLoc = successorGameState.getGhostPosition(-1)

        myScore = successorGameState.getScore()
        foodList = newFood.asList()

        if action == Directions.STOP:
            myScore -= 2
        elif newPos in currentGameState.getCapsules():
            myScore += 5
        
        minDist = 5
        for fod in foodList:
            manhattanDist = util.manhattanDistance(newPos, fod)
            if manhattanDist < minDist:
                minDist = manhattanDist
 
        myScore += 1/minDist**0.5
        return myScore        

        

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
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.
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

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
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
        gameState.isWin():
        Returns whether or not the game state is a winning state
        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        dep = self.depth
        if(gameState.isLose() or gameState.isWin() or self.depth == 0):
            return Directions.STOP
        else:
            return self.maxAgentHelper(gameState, dep, 0)[1]

        "*** YOUR CODE HERE ***"
        

    def maxAgentHelper(self, gameState, dep, pacIndex):
        if(gameState.isLose() or gameState.isWin() or dep == 0):
            return self.evaluationFunction(gameState), Directions.STOP
        best_act = Directions.STOP
        score = -999999
        for act in gameState.getLegalActions(0):
            new_score = score
            score = max(self.minAgentHelper(gameState.generateSuccessor(0,act), dep, 1)[0], score)
            if(new_score != score):
                best_act = act

        return score, best_act

    def minAgentHelper(self, gameState, dep, ghostIndex):
        if(gameState.isLose() or gameState.isWin() or dep == 0):
            return self.evaluationFunction(gameState), Directions.STOP
        best_act = Directions.STOP
        score = 999999
        for act in gameState.getLegalActions(ghostIndex):
            new_score = score
            if(ghostIndex == gameState.getNumAgents() - 1):
                score = min(self.maxAgentHelper(gameState.generateSuccessor(ghostIndex,act), dep - 1, 0)[0], score)
            else:
                score = min(self.minAgentHelper(gameState.generateSuccessor(ghostIndex,act), dep, ghostIndex + 1)[0], score)
            if(new_score != score):
                best_act = act               
        return score, best_act

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        alpha = float("-inf")
        beta = float("inf")

        dep = self.depth
        if(gameState.isLose() or gameState.isWin() or self.depth == 0):
            return Directions.STOP
        else:
            return self.alphaHelper(gameState, dep, 0, alpha, beta)[1]
        util.raiseNotDefined()

    def alphaHelper(self, gameState, dep, pacIndex, alpha, beta):
        if(gameState.isLose() or gameState.isWin() or dep == 0):
            return self.evaluationFunction(gameState), Directions.STOP
        best_act = Directions.STOP
        score = -999999
        for act in gameState.getLegalActions(0):
            new_score = score
            score = max(self.betaHelper(gameState.generateSuccessor(0,act), dep, 1, alpha, beta)[0], score)
            if(new_score != score):
                best_act = act
            if(score > beta):
                return score, best_act
            alpha = max(alpha, score)
        return score, best_act

    def betaHelper(self, gameState, dep, ghostIndex, alpha, beta):
        if(gameState.isLose() or gameState.isWin() or dep == 0):
            return self.evaluationFunction(gameState), Directions.STOP
        best_act = Directions.STOP
        score = 999999
        for act in gameState.getLegalActions(ghostIndex):
            new_score = score
            if(ghostIndex == gameState.getNumAgents() - 1):
                score = min(self.alphaHelper(gameState.generateSuccessor(ghostIndex,act), dep - 1, 0, alpha, beta)[0], score)
            else:
                score = min(self.betaHelper(gameState.generateSuccessor(ghostIndex,act), dep, ghostIndex + 1, alpha, beta)[0], score)
            if(new_score != score):
                best_act = act
            if(score < alpha):
                return score, best_act  
            beta = min(score,beta)             
        return score, best_act

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction
        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        dep = self.depth
        if(gameState.isLose() or gameState.isWin() or self.depth == 0):
            return Directions.STOP
        else:
            return self.maxAgentHelper(gameState, dep, 0)[1]

    def maxAgentHelper(self, gameState, dep, pacIndex):
        if(gameState.isLose() or gameState.isWin() or dep == 0):
            return self.evaluationFunction(gameState), Directions.STOP
        best_act = Directions.STOP
        score = -999999
        for act in gameState.getLegalActions(0):
            new_score = score
            score = max(self.minAgentHelper(gameState.generateSuccessor(0,act), dep, 1)[0], score)
            if(new_score != score):
                best_act = act

        return score, best_act

    def minAgentHelper(self, gameState, dep, ghostIndex):
        if(gameState.isLose() or gameState.isWin() or dep == 0):
            return self.evaluationFunction(gameState), Directions.STOP
        best_act = Directions.STOP
        score = 0
        len = 0
        for act in gameState.getLegalActions(ghostIndex):
            new_score = score
            if(ghostIndex == gameState.getNumAgents() - 1):
                score += self.maxAgentHelper(gameState.generateSuccessor(ghostIndex,act), dep - 1, 0)[0]
            else:
                score += self.minAgentHelper(gameState.generateSuccessor(ghostIndex,act), dep, ghostIndex + 1)[0]
            len+=1
        final_score = score/len                
        return final_score, best_act
def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).
    DESCRIPTION: This is a cleaner, more efficient evaluation function. Instead of tracking ghost distances
    individuaklly and storing these values, as well as the various states of the ghpsts, this
    simply calculates and returns a score based on the relevant details: the number of food pellets the Pacman
    ate, the minimum distance at all times, and of course, the current score of the Pacman.
    """
    "*** YOUR CODE HERE ***"
    foodList = currentGameState.getFood().asList()
    distList = [manhattanDistance(currentGameState.getPacmanPosition(), fod) for fod in foodList]

    if len(distList) != 0:
        minDist = min(distList)
    else:
        minDist = 0

    myScore = currentGameState.getScore() - minDist - 10 * currentGameState.getNumFood() 
    return myScore
# Abbreviation
better = betterEvaluationFunction