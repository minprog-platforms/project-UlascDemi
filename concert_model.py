from __future__ import annotations

import numpy as np
import math
import random

from mesa import Agent, Model
from mesa.space import ContinuousSpace
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector


# TODO
# pandas dataframes fixen
# moet uiteindelijk visueel duidelijk zijn en moet plotje komen

# TODO
# Kleine bias naar rechts maken

# TODO
# ipv kijken zijn er accidents, laat weten wanneer er een accident is

# TODO
# TODO
# TODO
# FIX DAT AGENT OOK EEN ACCIDENT_NUM KRIJGT ZODAT WE DAT KUNNEN OPLOSSEN


def get_x_loc(agent: "Person") -> float:
    """Returns the x location of an agent"""
    return agent.pos[0]


def get_y_loc(agent: "Person") -> float:
    """Returns the y location of an agent"""
    return agent.pos[1]


def get_accident_x_loc(agent: "Person") -> float:
    """Returns the x location of an agent moving to an accident"""
    if agent.get_status() != "MovingToAccident":
        return np.nan

    return agent.pos[0]


def get_accident_y_loc(agent: "Person") -> float:
    """Returns the y location of an agent moving to an accident"""
    if agent.get_status() != "MovingToAccident":
        return np.nan

    return agent.pos[1]


class Person(Agent):
    """
    
    """
    def __init__(self, unique_id: int, model) -> None:
        super().__init__(unique_id, model)
        self._uid = unique_id
        self._size = 0.5
        # self._running = False
        self._walking_speed = 0.5
        self._status = ""
        self.type = ""

    def get_angle(self) -> float:
        """
        Returns a random angle with a 12% bias towards the right for the first 60
        steps, afterwards returns a uniform random angle.
        
        Returns:
            angle (float) = An angle in radians
        """
        if self.model.get_step_number() < 60:
            angle = random.uniform(-0.06, 1.06) * 2 * np.pi
        else:
            angle = random.random() * 2 * np.pi
        return angle

    def collision_check(self, new_pos: tuple[float]) -> bool:
        """
        Check if there is another agent in the new position.
        
        Arguments:
            new_pos (tuple) = the new position, in the format of (x, y)
        
        Returns:
            True = The movement is not possible
            False = The movement is possible
        """
        agents_new_pos = self.model.space.get_neighbors(new_pos, self._size)
        if len(agents_new_pos) > 1:
            return True
        return False

    def accident_check(self, new_pos: tuple[float]) -> bool:
        """
        Check if there is another agent that has an accident
        in a radius of 5 meters of the new location.
        
        Arguments:
            new_pos (tuple) = the new position, in the format of (x, y)
        
        Returns:
            True = The movement is not possible
            False = The movement is possible
        """
        agents_new_pos = self.model.space.get_neighbors(new_pos, 5)
        for agent in agents_new_pos:
            if agent.get_status() == "Accident":
                return True
        return False

    def bound_checks(self, new_pos: tuple[float]) -> bool:
        """
        Checks if the new location complies within the bounds
        of the visitor area. These bounds are the width and
        length of the area minus the area for the workers.
        
        Returns:
            True = The movement is not possible
            False = The movement is possible
        """

        new_x_pos = new_pos[0]
        new_y_pos = new_pos[1]

        bot_bound = self.model.worker_space
        right_bound = self.model.space.width - self.model.worker_space
        top_bound = self.model.space.height - self.model.worker_space
        left_bound = self.model.worker_space

        # Bound checks for the concert hall
        if self.type == "Visitor":
            if (
                new_x_pos > right_bound
                or new_x_pos < left_bound
                or new_y_pos > top_bound
                or new_y_pos < bot_bound
            ):
                return True
        return False

    def get_status(self) -> str:
        """
        Returns the status of the agent
        
        Returns:
            _status (str) = the status of the agent
        """
        return self._status


class Visitor(Person):
    """Visitor"""
    def __init__(self, unique_id, model) -> None:
        super().__init__(unique_id, model)
        self.type = "Visitor"
        self.accident_number = -1

    def move(self) -> None:
        """
        Uses the get_angle() method to receive an angle and move the
        agent in that direction. The distance travelled is dependent
        on self._walking_speed.
        
        Returns:
            None
        """
        # Get a random angle and calculate the x and y unit vectors
        angle = self.get_angle()
        x_unit = np.cos(angle)
        y_unit = np.sin(angle)

        # Calculate the displacement vectors
        x_displacement = x_unit * self._walking_speed
        y_displacement = y_unit * self._walking_speed

        # Calculate the new position
        new_x_pos = self.pos[0] + x_displacement
        new_y_pos = self.pos[1] + y_displacement
        new_pos = (new_x_pos, new_y_pos)

        # Check if the new position complies with the bound checks
        if (
            self.collision_check(new_pos)
            or self.bound_checks(new_pos)
            or self.accident_check(new_pos)
        ):
            return

        # Move the agent
        self.model.space.move_agent(self, new_pos)

    def set_accident(self):
        """
        Sets the self._status to "Accident".
        
        Returns:
            None
        """
        self._status = "Accident"

    def step(self) -> None:
        """
        Step function for the model. This function checks if
        the agent has an accident status, if not then it moves
        the agent.
        
        Returns:
            None
        """
        if self._status != "Accident":
            self.move()


class Worker(Person):
    """worker"""
    def __init__(self, unique_id, model) -> None:
        super().__init__(unique_id, model)
        self.type = "Worker"
        self._walking_speed = 1
        self.original_pos = ()
        self.accident_pos = ()
        self.accident_number = -1

    def set_status(self, status: str) -> None:
        """
        Sets the status to a given status.
        
        Arguments:
            status (string) = The new status
            
        Returns:
            None
        """
        self._status = status

    def move(self, destination: Tuple[float]) -> None:
        """
        Moves the agent in the direction of a position.
        This method first calculates the angle towards a
        position and then moves the agent towards that position
        with a distance according to the walking speed.
        
        Aguments:
            destination (tuple) = a tuple containing two float positions in the
                                    format of (x, y)
        
        Returns:
            None
        """
        # Unpacks the destination position tuple
        dest_x_pos = destination[0]
        dest_y_pos = destination[1]

        # Unpacks the agents position tuple
        self_x_pos = self.pos[0]
        self_y_pos = self.pos[1]

        # Calculates the x and y vectors
        x_vec = dest_x_pos - self_x_pos
        y_vec = dest_y_pos - self_y_pos

        # Calculates the angle and then the x and y unit vectors
        angle = math.atan2(y_vec, x_vec)
        x_unit = np.cos(angle)
        y_unit = np.sin(angle)

        # Calculate the displacement vectors
        x_displacement = x_unit * self._walking_speed
        y_displacement = y_unit * self._walking_speed

        # Calculate the new position
        new_x_pos = self.pos[0] + x_displacement
        new_y_pos = self.pos[1] + y_displacement
        new_pos = (new_x_pos, new_y_pos)

        # Move the agent
        self.model.space.move_agent(self, new_pos)

    def start_moving_to_accident(self, position: tuple[float], accident_number: int) -> None:
        """
        Starts the process of moving to an accident. It first checks if the
        agent is not already moving towards an accident, then saves the
        original position and accident number internally. This is so that
        the agent can find its original position back.
        
        Arguments:
            position (tuple) = a tuple containing two float positions in the
                                    format of (x, y)
                                    
        Returns:
            None
        """
        if self._status != "MovingToAccident":
            return
        self.original_pos = self.pos
        self.accident_pos = position
        self.accident_number = accident_number
        self.move_to_accident()

    def move_to_accident(self) -> None:
        """
        Checks if the agent should be moving to the accident
        and if the agent is already at the accident. If both
        checks are passed, the agent is moved towards the
        accident position with the move() method.
        
        returns:
            None
        """
        if self._status != "MovingToAccident":
            return

        if self.model.distance(self.accident_pos, self.pos) < 1:
            self.model.space.move_agent(self, self.accident_pos)
            self.set_status("MovingBack")
            return

        self.move(self.accident_pos)

    def move_to_orig(self):
        if self._status != "MovingBack":
            return

        if self.model.distance(self.pos, self.original_pos) < 1:
            self.model.space.move_agent(self, self.original_pos)
            self.set_status("")
            return

        self.move(self.original_pos)

    def resolve_accident(self):
        pass

    def step(self) -> None:
        if self._status == "MovingToAccident":
            self.move_to_accident()
        elif self._status == "MovingBack":
            self.move_to_orig()


class ConcertHall(Model):
    def __init__(self, n_visit: int, n_work: int, width: int, height: int) -> None:
        assert width > 2
        assert height > 2

        self._step_number = 0
        self._n_visitor = n_visit
        self._n_worker = n_work
        self._width = width
        self._height = height
        self._middle = (width / 2, height / 2)
        self.worker_space = 2
        self.space = ContinuousSpace(width, height, False)
        self.schedule = RandomActivation(self)
        self.previous_accidents = []
        self.accident_number = 0

        # Create the visitors
        for i in range(self._n_visitor):
            visitor = Visitor(i, self)
            self.schedule.add(visitor)
            x = self.random.randrange(
                self.worker_space, self.space.width - self.worker_space
            )
            y = self.random.randrange(
                self.worker_space, self.space.height - self.worker_space
            )
            self.space.place_agent(visitor, (x, y))

        i += 1
        
        # Create the workers
        for j in range(i, i + self._n_worker):
            worker = Worker(j, self)
            self.schedule.add(worker)
            rng_number = random.random()
            # Bottom side
            if rng_number < 0.25:
                x = self.random.randrange(0, self.space.width)
                y = self.random.randrange(0, self.worker_space)
            # Right side
            elif rng_number >= 0.25 and rng_number < 0.5:
                x = self.random.randrange(
                    self.space.width - self.worker_space, self.space.width
                )
                y = self.random.randrange(0, self.space.height)
            # Top side
            elif rng_number >= 0.5 and rng_number < 0.75:
                x = self.random.randrange(0, self.space.width)
                y = self.random.randrange(
                    self.space.height - self.worker_space, self.space.height
                )
            # Left side
            elif rng_number >= 0.75 and rng_number <= 1.00:
                x = self.random.randrange(0, self.worker_space)
                y = self.random.randrange(0, self.space.height)

            self.space.place_agent(worker, (x, y))

        self.datacollector = DataCollector(
            agent_reporters={"x_pos": get_x_loc, "y_pos": get_y_loc}
        )

    def get_space(self) -> "ContinuousSpace":
        return self.space

    def distance(self, pos1: tuple, pos2: tuple):
        delta_x = pos1[0] - pos2[0]
        delta_y = pos1[1] - pos2[1]

        distance = np.sqrt(delta_x**2 + delta_y**2)
        return distance

    def get_agents(self) -> list:
        agents = self.space.get_neighbors(
            self._middle,
            np.sqrt((self.space.width / 2) ** 2 + (self.space.height / 2) ** 2),
        )
        return np.array(agents)

    def check_accidents(self) -> tuple:
        agents = self.get_agents()

        accident_positions = [
            agent.pos for agent in agents if agent.get_status() == "Accident"
        ]

        return accident_positions

    def check_new_accident(self) -> bool:
        accidents = self.check_accidents()

        for accident in accidents:
            if accident not in self.previous_accidents:
                self.previous_accidents.append(accident)
                return True

        return False

    def get_accident_loc(self) -> tuple:
        accident_number = self.accident_number
        self.accident_number += 1
        return (self.previous_accidents[accident_number], accident_number)

    def move_nearest_worker(self) -> None:
        accident_loc, accident_num = self.get_accident_loc()

        all_agents = self.get_agents()
        free_agents = [
            agent
            for agent in all_agents
            if agent.get_status() != "MovingToAccident" and agent.type == "Worker"
        ]

        lowest_distance = 100000
        closest_worker = 0
        for agent in free_agents:
            distance = self.distance(agent.pos, accident_loc)

            if distance < lowest_distance:
                lowest_distance = distance
                closest_worker = agent

        closest_worker.set_status("MovingToAccident")
        closest_worker.start_moving_to_accident(accident_loc, accident_num)

    def get_step_number(self) -> int:
        return self._step_number

    def increase_step(self) -> None:
        self._step_number += 1

    def step(self) -> None:
        if self.check_new_accident():
            self.move_nearest_worker()

        if random.random() < 0.03:
            agents = self.get_agents()
            random_agent = random.choice(agents)
            while random_agent.type == "Worker":
                random_agent = random.choice(agents)

            random_agent.set_accident()

        self.datacollector.collect(self)
        self.schedule.step()
        self.increase_step()
