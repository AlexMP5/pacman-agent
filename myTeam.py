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
import time

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
        # current_capsules = game_state.get_capsules()
        # current_walls = game_state.get_walls()
        # food_to_eat = self.get_food(game_state)
        # food_to_defend = self.get_food_you_are_defending(game_state)
        # capsules_to_eat = self.get_capsules(game_state)
        # capsules_to_defend = self.get_capsules_you_are_defending(game_state)

        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print('eval time for agent %d: %.4f' % (self.index, time.time() - start))

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        my_state = game_state.get_agent_state(self.index)

        loosing_when_finish = game_state.data.timeleft < 400 and ((self.red and game_state.data.score <= 0) or
                                                                  (not self.red and game_state.data.score >= 0))
        if food_left <= 2 or (my_state.is_pacman and loosing_when_finish
                              and my_state.num_carrying > abs(game_state.data.score)):
            best_dist = 9999
            best_action = Directions.STOP

            enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
            enemy_ghosts = [a for a in enemies if not a.is_pacman and a.get_position() is not None and
                            a.scared_timer == 0]

            enemy_ghosts_min_dist = None
            if len(enemy_ghosts) > 0:
                enemy_ghosts_dists = [self.get_maze_distance(my_state.get_position(),
                                                             a.get_position()) for a in enemy_ghosts]
                enemy_ghosts_min_dist = min(enemy_ghosts_dists)

            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)

                enemy_ghosts_future_min_dist = None
                if len(enemy_ghosts) > 0:
                    enemy_ghosts_future_dists = [self.get_maze_distance(pos2, a.get_position()) for a in enemy_ghosts]
                    enemy_ghosts_future_min_dist = min(enemy_ghosts_future_dists)

                if dist < best_dist and (enemy_ghosts_future_min_dist is None or enemy_ghosts_future_min_dist > 2 or
                                         enemy_ghosts_future_min_dist > enemy_ghosts_min_dist):
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

    def is_road_without_exit(self, game_state, conf, max_steps):
        actions = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]
        x, y = (int(conf.pos[0]), int(conf.pos[1]))
        is_road_without_exit = self.is_road_without_exit_i(game_state, conf.direction, x, y, actions, 0, max_steps)
        return is_road_without_exit

    def is_road_without_exit_i(self, game_state, current_direction, pos_x, pos_y, actions, steps, max_steps):
        if steps != 0:
            if current_direction == Directions.NORTH:
                pos_y += 1
            if current_direction == Directions.SOUTH:
                pos_y -= 1
            if current_direction == Directions.EAST:
                pos_x += 1
            if current_direction == Directions.WEST:
                pos_x -= 1

        directions_to_take = []
        for action in actions:
            if not action == Directions.REVERSE[current_direction]:
                if action == Directions.NORTH:
                    if not game_state.has_wall(pos_x, pos_y + 1):
                        directions_to_take.append(action)
                if action == Directions.SOUTH:
                    if not game_state.has_wall(pos_x, pos_y - 1):
                        directions_to_take.append(action)
                if action == Directions.EAST:
                    if not game_state.has_wall(pos_x + 1, pos_y):
                        directions_to_take.append(action)
                if action == Directions.WEST:
                    if not game_state.has_wall(pos_x - 1, pos_y):
                        directions_to_take.append(action)

        if len(directions_to_take) == 0:
            return True

        if steps < max_steps:
            if len(directions_to_take) == 1:
                return self.is_road_without_exit_i(game_state, directions_to_take[0], pos_x, pos_y,
                                                   actions, steps + 1, max_steps)
            if len(directions_to_take) == 2:
                return self.is_road_without_exit_i(game_state, directions_to_take[0], pos_x, pos_y, actions, steps + 1,
                                                   max_steps) and \
                    self.is_road_without_exit_i(game_state, directions_to_take[1], pos_x, pos_y, actions, steps + 1,
                                                max_steps)
            if len(directions_to_take) == 3:
                return self.is_road_without_exit_i(
                    game_state, directions_to_take[0], pos_x, pos_y, actions, steps + 1, max_steps) and \
                    self.is_road_without_exit_i(game_state, directions_to_take[1], pos_x, pos_y, actions, steps + 1,
                                                max_steps) \
                    and self.is_road_without_exit_i(game_state, directions_to_take[2], pos_x, pos_y, actions, steps + 1,
                                                    max_steps)
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

        capsules_to_eat = self.get_capsules(successor)
        features['successor_score'] -= len(capsules_to_eat) * 2

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        enemy_ghosts = [a for a in enemies if not a.is_pacman and a.get_position() is not None and a.scared_timer == 0]
        if len(enemy_ghosts) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in enemy_ghosts]
            features['ghost_distance'] = min(min(dists), 5)
            if features['ghost_distance'] <= 3:
                features['ghost_close'] = 1

                if self.is_road_without_exit(successor, my_state.configuration, 20):
                    features['danger_closed_road'] = 1

        if len(food_list) > 0:
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance

            if len(capsules_to_eat) > 0:
                min_distance_to_capsule = min([self.get_maze_distance(my_pos, capsule) for capsule in capsules_to_eat])
                features['distance_to_capsule'] = min(min_distance_to_capsule, 5)

        if action == Directions.STOP:
            features['stop'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'successor_score': 10000, 'ghost_distance': -2, 'ghost_close': -200, 'distance_to_food': -10,
                'distance_to_capsule': -30, 'danger_closed_road': -5000, 'stop': -6000}


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
        if not my_state.is_pacman:
            features['on_defense'] = 1

        enemies_before = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        invaders_before = [a for a in enemies_before if a.is_pacman and a.get_position() is not None]

        enemies_after = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders_after = [a for a in enemies_after if a.is_pacman and a.get_position() is not None]

        enemies_killed = 0
        if len(invaders_before) > 0:
            enemies_killed = [a for a in enemies_after if not a.is_pacman and a.get_position() == a.start.pos]
            features['enemies_killed'] = len(enemies_killed)

        if len(invaders_after) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders_after]
            features['invader_distance'] = min(min(dists), 5)
        else:
            features['invader_distance'] = 6

            if enemies_killed == 0 and self.is_road_without_exit(successor, my_state.configuration, 5):
                features['empty_closed_road'] = 1

        if action == Directions.STOP:
            features['stop'] = 1

        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'on_defense': 10000, 'enemies_killed': 9000, 'invader_distance': -10, 'empty_closed_road': -1000,
                'stop': -100, 'reverse': -2}
