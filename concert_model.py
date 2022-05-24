from __future__ import annotations
from mesa import Agent, Model
from mesa.space import ContinuousSpace
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from copy import deepcopy
import numpy as np
import random

# TODO
# pandas dataframes fixen
# moet uiteindelijk visueel duidelijk zijn en moet plotje komen

# TODO
# Kleine bias naar rechts maken


def get_x_loc(agent):
    return agent.pos[0]


def get_y_loc(agent):
    return agent.pos[1]


class Person(Agent):
    def __init__(self, unique_id: int, model) -> None:
        super().__init__(unique_id, model)
        self._uid = unique_id
        self._size = 0.5
        # self._running = False
        self._walking_speed = 0.5
        self._type = ""
        self._status = ""

    def move(self) -> None:
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

    def get_angle(self) -> float:
        """Returns a random angle with a 4% bias towards the right"""
        angle = random.uniform(-0.06, 1.06) * 2 * np.pi
        return angle

    def collision_check(self, new_pos: tuple) -> bool:
        """
        Check if there is another agent in the new position
        Returns False if the movement is possible
        True when not
        """
        agents_new_pos = self.model.space.get_neighbors(new_pos, self._size)
        if len(agents_new_pos) > 1:
            return True
        return False

    def accident_check(self, new_pos: tuple) -> bool:
        agents_new_pos = self.model.space.get_neighbors(new_pos, 5)
        for agent in agents_new_pos:
            if agent.get_status == "Accident":
                return True
        return False

    def bound_checks(self, new_pos: tuple[float]) -> bool:
        """Returns true or false"""

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
        return self._status


class Visitor(Person):
    def __init__(self, unique_id, model) -> None:
        super().__init__(unique_id, model)
        self.type = "Visitor"

    def set_accident(self):
        self._status = "Accident"

    def step(self) -> None:
        if self._status != "Accident":
            self.move()

        # if random.random() < 0.01:
        #     self.set_accident()


class Worker(Person):
    def __init__(self, unique_id, model) -> None:
        super().__init__(unique_id, model)
        self.type = "Worker"

    def move_to_accident(self):
        # TODO
        pass

    # def step(self) -> None:
    #     self.move()


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

    def check_accidents(self) -> tuple:
        agents = self.space.get_neighbors(
            self._middle,
            np.sqrt((self.space.width / 2) ** 2 + (self.space.height / 2) ** 2),
        )
        accident_positions = [
            agent.pos for agent in agents if agent.get_status() == "Accident"
        ]

        return accident_positions

    def check_new_accident(self) -> bool:
        accidents = self.check_accidents()

        if accidents != self.previous_accidents:
            self.previous_accidents = accidents
            return True

        return False

    def get_step_number(self) -> int:
        return self._step_number

    def increase_step(self) -> None:
        self._step_number += 1

    def step(self) -> None:

        if self.check_new_accident():
            print(self.get_step_number())

        self.datacollector.collect(self)
        self.schedule.step()
        self.increase_step()
