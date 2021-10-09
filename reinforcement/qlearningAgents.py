# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        
        self.values = util.Counter()


    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        return self.values[(state, action)]


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        possibleActions = self.getLegalActions(state)
        preferred = -9e999

        if not possibleActions:
            return 0
        
        bestMove = ''

        for possibleAction in possibleActions:
            val = self.getQValue(state, possibleAction)
            if preferred < val:
                bestMove = possibleActions
                preferred = val
        

        return preferred

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"

        #create a loop and traverse thru
        possibleActions = self.getLegalActions(state)
        if not possibleActions:
            return None
        idealMove = ''
        preferred = -9e999
        for possibleAction in possibleActions:
            val = self.getQValue(state, possibleAction)
            if preferred < val:
                preferred = val
                idealMove = possibleAction
        return idealMove

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        validChoices = self.getLegalActions(state)
        x = None
        "*** YOUR CODE HERE ***"
        if util.flipCoin(self.epsilon):
            x = random.choice(validChoices)
        else:
            x = self.computeActionFromQValues(state)
        return x

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        prev = self.getQValue(state, action)
        self.values[(state, action)] = prev*(1 - self.alpha)  + self.alpha*(self.discount * self.computeValueFromQValues(theNext) + reward)

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):

        action = QLearningAgent.getAction(self,state)
        self.doAction(state,   action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):

        "*** YOUR CODE HERE ***"
        val = 0
        weightsArray = self.getWeights()
        
        features = self.featExtractor.getFeatures(state, action)
        for feature in features:
            val += features[feature] * weightsArray[feature]
        return val


    #Vivek, i finished this part but need you to update the loop to handle the edge cases
    def update(self, state, action, nextState, reward):
 
        "*** YOUR CODE HERE ***"
        
        updated = self.discount * self.computeValueFromQValues(nextState) 

        updated += reward
        allFeatures = self.featExtractor.getFeatures(state, action)
        for singleFeature in allFeatures:
            self.weights[singleFeature] += (updated   - self.getQValue(state, action)) * allFeatures[singleFeature] * self.alpha

    def final(self, state):
        PacmanQAgent.final(self, state)

        if self.episodesSoFar == self.numTraining:
            "*** YOUR CODE HERE ***"
            print(self.weights)
