from gridworld import *


class PathfindingAgent(Agent):
    """
    Parent class for agents.
    """
    agent_counter = 0

    def __init__(self, agent_type):
        """
        Default constructor for PathfindingAgent
        :param agent_type: the agent type
        """
        self.agent_type = agent_type

        # Initialize the unique numeric ID of the agent
        self.id = PathfindingAgent.agent_counter

        self.agent_name = "A" + str(self.id)

        self.agent_memory = {}

        # True, because we want to run A* first
        self.agent_memory[self.agent_name] = (True, [])

        self.finished = False

        # Increase global counter
        PathfindingAgent.agent_counter += 1

    def __eq__(self, other):
        """
        Two agents are equal if their ID's are the same
        :param other: the other agent
        :return: True if the `other' agent has the same ID as this one
        """
        if isinstance(other, self.__class__):
            return self.id == other.id
        else:
            return False

    def __hash__(self):
        return self.id

    def __str__(self):
        return self.agent_name


class PathfindingAgentData(GridAgentData):
    PATHFINDER = 1

    def __init__(self, linked_agent, agent_type, grid_position):
        super(PathfindingAgentData, self).__init__(linked_agent, grid_position, GridOrientation.NORTH)

        self.agent_type = agent_type

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.agent_type == other.agent_type and self.grid_position == other.grid_position and \
                   self.linked_agent == other.linked_agent
        else:
            return False

    def __hash__(self):
        return self.linked_agent.id

    def __str__(self):
        return str(self.linked_agent)


class PathfindingEnvironment(AbstractGridEnvironment):

    def __init__(self):
        super(PathfindingEnvironment, self).__init__()

        self._pathfinding_agents = []
        self.agent_goal_map = {}
        self.str_goals_map = {}

    def add_agent(self, agent_data):
        self.__add_pathfinding_agent(agent_data)

    def __add_pathfinding_agent(self, agent_data):
        if agent_data.agent_type == PathfindingAgentData.PATHFINDER:
            self._pathfinding_agents.append(agent_data)
        else:
            raise ValueError("Illegal type of agent: %s" % str(agent_data.linked_agent))

        self._agents.append(agent_data)

    def initialize_world(self, w, h, pathfinding_agents, rand_seed=None):
        """
        Initializes the environment with the provided width, height and number of agents.
        :param w: width of the grid
        :param h: height of the grid
        :param pathfinding_agents: list of agents to place on the grid
        :param rand_seed: Seed for random number generator. May be None
        """

        num_agents = len(pathfinding_agents)

        # generate obstacles and goals
        # num_obstacles = random.randint(1, w)
        num_obstacles = 10
        self.initialize(w, h, num_agents, num_obstacles, rand_seed)

        # generate all agents
        attempts = 10 * num_agents * num_agents
        generated = 0

        if rand_seed:
            random.seed(rand_seed)

        while attempts > 0 and generated < num_agents:
            x = random.randint(1, w)
            y = random.randint(1, h)

            pos = GridPosition(x, y)
            ok = True

            # generate at minimum manhattan distance of 1 from other agents, tiles and goals
            for pf_agent_data in self._pathfinding_agents:
                if pos.get_distance_to(pf_agent_data.grid_position) < 1:
                    ok = False

            for _x_tile in self._get_x_tiles():
                if pos.get_distance_to(_x_tile) < 1:
                    ok = False

            for _goal in self._get_goal_tiles():
                if pos.get_distance_to(_goal) < 1:
                    ok = False

            if ok:
                generated += 1
                self.__add_pathfinding_agent(
                    PathfindingAgentData(pathfinding_agents.pop(), agent_type=PathfindingAgentData.PATHFINDER,
                                         grid_position=pos))

            attempts -= 1

        if generated < num_agents:
            print("Failed to generate all required agents. Wanted: %i, generated: %i" % (
                num_agents, generated))

        # assign agent to its goal
        # number of agents = number of goals
        goals = self._get_goal_tiles()
        i = 0
        for pf_agent_data in self._pathfinding_agents:
            str_agent = pf_agent_data.linked_agent.agent_name
            str_goal = "G" + str(pf_agent_data.linked_agent.id)
            self.agent_goal_map[str_agent] = (str_goal, goals[i])
            self.str_goals_map[goals[i]] = str_goal
            i += 1

        for m in self.agent_goal_map:
            print(m, " -> ", self.agent_goal_map[m][0], self.agent_goal_map[m][1])

    def get_nearby_obstacles(self, grid_position, range):
        """
        Returns the set of obstacles which are at a distance from a given position by at most `range'
        :param grid_position: the position of the agent
        :param range: the range the agent can observe
        :return: The set of GridPositions where obstacles are found
        """
        nearby_obstacles = []

        for pos in self._xtiles:
            if pos.get_distance_to(grid_position) <= range:
                nearby_obstacles.append(pos)

        # other agents are considered obstacles, too, when they are finished
        for pf_agent_data in self._pathfinding_agents:
            if pf_agent_data.linked_agent.finished:
                if grid_position.get_distance_to(pf_agent_data.grid_position) <= range:
                    nearby_obstacles.append(pf_agent_data.grid_position)

        return nearby_obstacles

    def get_nearby_agents(self, grid_position, range):
        """
        Returns the set of agents which are at a distance from a given position by at most `range'.
        :param grid_position: Position around which to determine the nearby agents
        :param range: the range the agent can observe
        :return: The set of nearby agents given as `PathfindingAgentData' instances.
        """
        nearby_agents = []
        for pf_agent_data in self._pathfinding_agents:
            if grid_position.get_distance_to(pf_agent_data.grid_position) <= range:
                nearby_agents.append(pf_agent_data)

        return nearby_agents

    def goals_completed(self):
        for pf_agent_data in self._pathfinding_agents:
            if not pf_agent_data.linked_agent.finished:
                return False

        return True

    def __str__(self):
        res = ""
        res += "  |"

        ## border top
        for i in range(self._x0, self._x1 + 1):
            step = 1
            if i >= 10:
                step = 2

            for k in range(0, self._cellW - step):
                res += " "

            res += str(i) + "|"

        res += "\n"
        res += "--+"

        for i in range(self._x0, self._x1 + 1):
            for k in range(0, self._cellW):
                res += "-"
            res += "+"

        res += "\n"

        ## for each line
        for j in range(self._y1, self._y0 - 1, -1):
            # first cell row
            if j < 10:
                res += " " + str(j) + "|"
            else:
                res += str(j) + "|"

            for i in range(self._x0, self._x1 + 1):
                pos = GridPosition(i, j)
                agent_string = ""
                for agent_data in self._agents:
                    if agent_data.grid_position == pos:
                        agent_string += str(agent_data.linked_agent)

                k = 0
                if pos in self._xtiles:
                    while k < self._cellW:
                        res += "X"
                        k += 1

                if self._cellH < 2 and pos in self._goals:
                    res += "~"
                    k += 1

                if len(agent_string) > 0:
                    if self._cellW == 1:
                        if len(agent_string) > 1:
                            res += "."
                        else:
                            res += agent_string
                        k += 1
                    else:
                        res += agent_string[:min(len(agent_string), self._cellW - k)]
                        k += min(len(agent_string), self._cellW - k)

                while k < self._cellW:
                    res += " "
                    k += 1

                res += "|"

            res += "\n"

            # second cell row
            res += "  |"
            for i in range(self._x0, self._x1 + 1):
                pos = GridPosition(i, j)
                if pos in self._goals:
                    str_goal = self.str_goals_map[pos]
                else:
                    str_goal = ""
                for k in range(0, self._cellW):
                    if pos in self._xtiles:
                        res += "X"
                    elif pos in self._goals:
                        res += str_goal[k]
                    else:
                        res += " "
                res += "|"

            res += "\n"

            # other cell rows
            for ky in range(0, self._cellH - 2):
                res += "|"
                for i in range(self._x0, self._x1 + 1):
                    for k in range(0, self._cellW):
                        if GridPosition(i, j) in self._xtiles:
                            res += "X"
                        else:
                            res += " "
                    res += "|"
                res += "\n"

            res += "--+"

            for i in range(self._x0, self._x1 + 1):
                for k in range(0, self._cellW):
                    res += "-"
                res += "+"
            res += "\n"

        return res
