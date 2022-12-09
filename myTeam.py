# baselineTeam.py
# ---------------
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


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import util

from captureAgents import CaptureAgent
from game import Directions
from util import nearestPoint

from contest.capture import AgentRules
from contest.game import Actions


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        currentCapsules = game_state.get_capsules()
        currentWalls = game_state.get_walls()
        isWallXY = game_state.has_wall(1, 2)

        foodToEat = self.get_food(game_state)
        foodToDefend = self.get_food_you_are_defending(game_state)

        capsulesToEat = self.get_capsules(game_state)
        capsulesToDefend = self.get_capsules_you_are_defending(game_state)

        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        #If we carry more than 5 point we will go back to our side to sum it to our score (less risk of losing them by a ghost)
        myState = game_state.get_agent_state(self.index)
        if food_left < 2 or (myState.is_pacman and myState.num_carrying > 5):
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}

    # to avoid the routes that have not exit since are more risk that once who have an exit to avoid the ghost
    def is_road_without_exit(self, game_state, conf):
        actions = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]
        x, y = (int(conf.pos[0]), int(conf.pos[1]))

        if conf.direction == Directions.NORTH:
            y -= 1
        if conf.direction == Directions.SOUTH:
            y += 1
        if conf.direction == Directions.EAST:
            x += 1
        if conf.direction == Directions.WEST:
            x -= 1

        directionsToTake = 0
        for action in actions:
            if action == Directions.NORTH:
                if not game_state.has_wall(x, y-1):
                    directionsToTake += 1
            if action == Directions.SOUTH:
                if not game_state.has_wall(x, y+1):
                    directionsToTake += 1
            if action == Directions.EAST:
                if not game_state.has_wall(x+1, y):
                    directionsToTake += 1
            if action == Directions.WEST:
                if not game_state.has_wall(x-1, y):
                    directionsToTake += 1

        if directionsToTake == 1:
            return True

        return False


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)  # self.getScore(successor)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(successor)]
        enemyGhosts = [a for a in enemies if not a.is_pacman and a.get_position() is not None and a.scared_timer == 0]
        if len(enemyGhosts) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in enemyGhosts]
            features['ghost_distance'] = min(dists)
            if features['ghost_distance'] <= 2:
                features['ghost_close'] = 1

                if self.is_road_without_exit(game_state, my_state.configuration):
                    features['danger_closed_road'] = 1

        if len(food_list) > 0:
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance

        if action == Directions.STOP:
            features['stop'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'successor_score': 10000, 'ghost_distance': -1, 'ghost_close': -100, 'distance_to_food': -10, 'danger_closed_road': -5000, 'stop': -6000}


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        #features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            features['has_invaders'] = 1
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'has_invaders': 10000, 'on_defense': 10000, 'invader_distance': -10, 'stop': -100, 'reverse': -2}
