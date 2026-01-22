from base import Agent, Action, Perception
from representation import GridRelativeOrientation, GridOrientation, GridPosition
from pathfinding import PathfindingEnvironment, PathfindingAgentData, PathfindingAgent
from enum import Enum

import sys
import time, random
from heapq import heappop, heappush


class MyAction(Action, Enum):
    """
    Physical actions for wildlife agents.
    """

    # The agent must move north (up)
    NORTH = 0

    # The agent must move north-east (up-right)
    NORTH_EAST = 1

    # The agent must move north-west (up-left)
    NORTH_WEST = 2

    # The agent must move east (right).
    EAST = 3

    # The agent must move south (down).
    SOUTH = 4

    # The agent must move south-east (down-right).
    SOUTH_EAST = 5

    # The agent must move south-west (down-left).
    SOUTH_WEST = 6

    # The agent must move west (left).
    WEST = 7

    # The agent must hold its position
    WAIT = 8


class MyAgentPerception(Perception):
    """
    The perceptions of a wildlife agent.
    """

    def __init__(self, agent_position, absolute_orientation, obstacles, goal, moves):
        """
        Default constructor
        :param agent_position: agents's position.
        :param obstacles: visible obstacles
        :param messages: incoming messages, may be None
        """
        self.agent_position = agent_position
        self.absolute_orientation = absolute_orientation
        self.obstacles = obstacles
        self.goal = goal
        self.moves = moves


def print_path(path):
    for pos in path:
        print(pos)


def process_relative_pos(relative_pos):
    if relative_pos == GridRelativeOrientation.FRONT:
        return MyAction.NORTH
    elif relative_pos == GridRelativeOrientation.FRONT_LEFT:
        return MyAction.NORTH_WEST
    elif relative_pos == GridRelativeOrientation.FRONT_RIGHT:
        return MyAction.NORTH_EAST
    elif relative_pos == GridRelativeOrientation.LEFT:
        return MyAction.WEST
    elif relative_pos == GridRelativeOrientation.RIGHT:
        return MyAction.EAST
    elif relative_pos == GridRelativeOrientation.BACK:
        return MyAction.SOUTH
    elif relative_pos == GridRelativeOrientation.BACK_LEFT:
        return MyAction.SOUTH_WEST
    elif relative_pos == GridRelativeOrientation.BACK_RIGHT:
        return MyAction.SOUTH_EAST


def check_diagonal_conflict(pre_move1, post_move1, pre_move2, post_move2):
    pre_x = pre_move1.get_x() - pre_move2.get_x()
    pre_y = pre_move1.get_y() - pre_move2.get_y()
    post_x = post_move1.get_x() - post_move2.get_x()
    post_y = post_move1.get_y() - post_move2.get_y()

    # only nearby
    if abs(pre_x) > 1 or abs(pre_y) > 1 or abs(post_x) > 1 or abs(post_y) > 1:
        return False

    pre_diff = pre_x + pre_y
    post_diff = post_x + post_y

    # when parallel diagonals
    if pre_diff == 0 and post_diff == 0:
        return False

    if abs(pre_diff) > 1 or abs(post_diff) > 1:
        return False

    if pre_diff + post_diff == 0:
        return True
    else:
        return False


def a_star(start, end, absolute_orientation, obstacles, moves, agent_name):
    frontier = []
    heappush(frontier, (0 + start.get_euclidean_distance_to(end), start))
    discovered = {start: (None, 0)}
    curr_node = None

    expanded_node = False
    while frontier:
        if expanded_node:
            break

        (_, curr_node) = heappop(frontier)

        if curr_node == end:
            break

        # get available cells
        neighbour_cells = curr_node.get_neighbours(absolute_orientation)
        free_neighbour_cells = [cell_pos for cell_pos in neighbour_cells if cell_pos not in obstacles]

        agent_pre_move, agent_post_move = moves[agent_name]

        # filter the moves that do not due to conflict with other agents
        invalid_neighbour_cells = []
        for pre_move, post_move in moves.values():
            if post_move != None:
                for new_post_move in free_neighbour_cells:
                    ok = True

                    # check constraints between moves
                    # when 2 agents go through each other
                    if new_post_move == pre_move and agent_pre_move == post_move:
                        ok = False
                    # when 2 agents go on the same cell
                    elif new_post_move == post_move:
                        ok = False
                    # when 2 agents intersect on diagonal
                    elif check_diagonal_conflict(agent_pre_move, new_post_move, pre_move, post_move):
                        ok = False

                    # keep that move, to remove it later
                    if not ok:
                        invalid_neighbour_cells.append(new_post_move)

        # keep only the good moves
        free_neighbour_cells = [cell for cell in free_neighbour_cells if cell not in invalid_neighbour_cells]

        # expand current node
        # I used chebyshev distance, because the agents can move on diagonal, too (Wikipedia)
        for neighbour in free_neighbour_cells:
            new_cost = discovered[curr_node][1] + neighbour.get_euclidean_distance_to(curr_node)

            if neighbour not in discovered:
                discovered[neighbour] = (curr_node, new_cost)
                heappush(frontier, (discovered[neighbour][1] + neighbour.get_euclidean_distance_to(end), neighbour))

                expanded_node = True  # expand once only
            else:
                if new_cost >= discovered[neighbour][1]:
                    continue

                discovered[neighbour] = (curr_node, new_cost)
                heappush(frontier, (discovered[neighbour][1] + neighbour.get_euclidean_distance_to(end), neighbour))

                expanded_node = True

    # return admissible moves
    return frontier


class AStarPFAgent(PathfindingAgent):

    def __init__(self):
        super(AStarPFAgent, self).__init__(PathfindingAgentData.PATHFINDER)

    def response(self, perceptions):
        agent_pos = perceptions.agent_position
        absolute_orientation = perceptions.absolute_orientation
        obstacles = perceptions.obstacles
        goal_name, goal_pos = perceptions.goal
        moves = perceptions.moves

        # expand current node by running A*
        admissible_moves = a_star(agent_pos, goal_pos, absolute_orientation, obstacles, moves, self.agent_name)

        return admissible_moves


class MyEnvironment(PathfindingEnvironment):
    PF_AGENT_RANGE = 2

    def __init__(self, w, h, num_pf_agents, random_seed):
        """
        Default constructor. This should call the initialize methods offered by the super class.
        """
        rand_seed = random_seed

        print("Seed = %i" % rand_seed)

        super(MyEnvironment, self).__init__()

        pf_agents = []

        for i in range(num_pf_agents):
            agent = AStarPFAgent()
            pf_agents.append(agent)

        # initialize the environment
        self.initialize_world(w=w, h=h, pathfinding_agents=pf_agents, rand_seed=rand_seed)

    def step(self):
        """
        This method should iterate through all agents, provide them with perceptions, and apply the action they return.
        """
        """
        STAGE 1: generate perceptions for all agents, based on the state of the environment at the beginning of this
        turn
        """
        # keep the moves of all agents as tuples of (pre_move, post_move)
        # initially all have post_move = None
        moves = {}
        for pf_agent_data in self._pathfinding_agents:
            moves[pf_agent_data.linked_agent.agent_name] = (pf_agent_data.grid_position, None)

        agent_perceptions = {}
        # create perceptions for pathfinding agents
        for pf_agent_data in self._pathfinding_agents:
            if not pf_agent_data.linked_agent.finished:
                # get obstacles and goal for this agent
                nearby_obstacles = self.get_nearby_obstacles(pf_agent_data.grid_position, MyEnvironment.PF_AGENT_RANGE)
                pf_agent_goal = self.agent_goal_map[pf_agent_data.linked_agent.agent_name]

                agent_perceptions[pf_agent_data] = MyAgentPerception(agent_position=pf_agent_data.grid_position,
                                                                     absolute_orientation=pf_agent_data.current_orientation,
                                                                     obstacles=nearby_obstacles, goal=pf_agent_goal,
                                                                     moves=moves)

        """
        STAGE 2: call response for each agent to obtain desired actions
        """
        agent_actions = {}
        # get actions for all agents
        # first agent expand its node, then we save the best pos as post_move in moves for that agent
        # the next agent will use the updated map to get available moves, and save as well their best move
        # and so on
        # so, there is a branch factor of 8, instead of 8 ^ n, because WAIT is processed here
        for pf_agent_data in self._pathfinding_agents:
            # print(pf_agent_data.linked_agent.agent_name, "->", pf_agent_data.linked_agent.finished)
            if not pf_agent_data.linked_agent.finished:
                # update the map of moves for the next agent and set it in its perceptions
                agent_perceptions[pf_agent_data].moves = moves
                agent_actions[pf_agent_data] = pf_agent_data.linked_agent.response(agent_perceptions[pf_agent_data])

                pre_move, post_move = moves[pf_agent_data.linked_agent.agent_name]
                if agent_actions[pf_agent_data]:
                    _, new_post_move = heappop(agent_actions[pf_agent_data])
                    moves[pf_agent_data.linked_agent.agent_name] = (pre_move, new_post_move)
                else:  # in this case, the agent WAITS
                    moves[pf_agent_data.linked_agent.agent_name] = (pre_move, pre_move)

        """
        STAGE 3: apply the agents' actions in the environment
        """
        for pf_agent_data in self._pathfinding_agents:
            if not pf_agent_data.linked_agent.finished:
                _, post_move = moves[pf_agent_data.linked_agent.agent_name]

                if post_move:
                    pf_agent_data.grid_position = post_move

        # set all the agents that reached their destinations to finished
        for pf_agent_data in self._pathfinding_agents:
            if not pf_agent_data.linked_agent.finished:
                if pf_agent_data.grid_position == self.agent_goal_map[pf_agent_data.linked_agent.agent_name][1]:
                    pf_agent_data.linked_agent.finished = True


class Tester(object):
    WIDTH = 10
    HEIGHT = 10

    DELAY = 0.1

    def __init__(self, num_agents, random_seed):
        self.env = MyEnvironment(Tester.WIDTH, Tester.HEIGHT, num_agents, random_seed)

        print("\n### INITIAL GRID ###")
        print(self.env)

        self.make_steps()

    def make_steps(self):
        steps = 0

        while not self.env.goals_completed():
            self.env.step()

            print(self.env)

            time.sleep(Tester.DELAY)

            steps += 1

        print("\nTHE AGENTS HAVE REACHED DESTINATIONS AFTER %s STEPS!!!" % str(steps))


if __name__ == "__main__":
    print('Number of arguments:', len(sys.argv), 'arguments.')
    if len(sys.argv) < 3:
        print("[Usage] -> python3 my_pathfinding_world.py $nr_agents $random_seed")
        sys.exit(1)

    print('Argument List:', str(sys.argv[1:]))
    num_agents = int(sys.argv[1])
    random_seed = int(sys.argv[2])
    tester = Tester(num_agents, random_seed)
