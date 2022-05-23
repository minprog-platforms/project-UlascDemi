from mesa import Agent, Model
from mesa.space import ContinuousSpace
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import numpy as np
import random

# TODO
# pandas dataframes fixen
# moet uiteindelijk visueel duidelijk zijn en moet plotje komen


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
        self._walking_speed = 1.4
        self._type = ""

    def move(self):
        angle = self.get_angle()
        x_unit = np.cos(angle)
        y_unit = np.sin(angle)

        x_displacement = x_unit * self._walking_speed
        y_displacement = y_unit * self._walking_speed

        new_x_pos = self.pos[0] + x_displacement
        new_y_pos = self.pos[1] + y_displacement
        new_pos = (new_x_pos, new_y_pos)

        if self.bound_checks(new_x_pos, new_y_pos) == False:
            return

        agents_new_pos = self.model.space.get_neighbors(new_pos, self._size)

        if len(agents_new_pos) > 1:
            return

        self.model.space.move_agent(self, new_pos)

    def get_angle(self):
        angle = random.random() * 2 * np.pi
        return angle

    def bound_checks(self, new_x_pos, new_y_pos):
        """Returns true or false"""
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
                return False
        elif self.type == "Worker":
            # Bottom side
            if self.pos[1] < bot_bound:
                if (
                    new_x_pos < 0
                    or new_x_pos > self.model.space.width
                    or new_y_pos < 0
                    or new_y_pos > self.model.space.height
                ):
                    return False
            # Right side
            elif self.pos[0] > right_bound:
                if (
                    new_x_pos < right_bound
                    or new_x_pos > self.model.space.width
                    or new_y_pos < 0
                    or new_y_pos > self.model.space.height
                ):
                    return False
            # Top side
            elif self.pos[1] > top_bound:
                if (
                    new_x_pos < 0
                    or new_x_pos > self.model.space.width
                    or new_y_pos < top_bound
                    or new_y_pos > self.model.space.height
                ):
                    return False
            # Left side
            elif self.pos[0] < left_bound:
                if (
                    new_x_pos < 0
                    or new_x_pos > self.model.worker_space
                    or new_y_pos < 0
                    or new_y_pos > self.model.space.height
                ):
                    return False
        return True


class Visitor(Person):
    def __init__(self, unique_id, model) -> None:
        super().__init__(unique_id, model)
        self.type = "Visitor"

    def step(self):
        self.move()


class Worker(Person):
    def __init__(self, unique_id, model) -> None:
        super().__init__(unique_id, model)
        self.type = "Worker"

    def step(self):
        self.move()


class ConcertHall(Model):
    def __init__(self, n_visitors, n_workers, width, height) -> None:
        assert width > 2
        assert height > 2

        self._n_visitor = n_visitors
        self._n_worker = n_workers
        self._width = width
        self._height = height
        self._middle = (width / 2, height / 2)
        self.worker_space = 2
        self.space = ContinuousSpace(width, height, False)
        self.schedule = RandomActivation(self)
        self._locations = []

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

    def save_locations(self):
        agents = self.space.get_neighbors(
            self._middle,
            np.sqrt((self.space.width / 2) ** 2 + (self.space.height / 2) ** 2),
        )

        positions = np.array([agent.pos for agent in agents])
        self._locations.append(positions)

    def get_locations(self):
        return self._locations

    def get_space(self):
        return self.space

    def get_width(self):
        return self._width

    def get_heigth(self):
        return self._height

    def step(self) -> None:
        self.save_locations()
        self.datacollector.collect(self)
        self.schedule.step()
