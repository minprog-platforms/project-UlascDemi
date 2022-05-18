from mesa import Agent, Model
from mesa.space import ContinuousSpace
from mesa.time import RandomActivation
import numpy as np
import random

MALE, FEMALE = 0, 1


class Person(Agent):
    def __init__(self, unique_id: int, model) -> None:
        super().__init__(unique_id, model)
        self._uid = unique_id
        self._gender = random.randint(MALE, FEMALE)
        self._size = 0
        self._running = False
        self._walking_speed = 1.4

        if self._gender == MALE:
            self._running_speed = 2.4
        elif self._gender == FEMALE:
            self._running_speed = 2.3

    def move(self):
        angle = self.get_angle()
        x_unit = np.cos(angle)
        y_unit = np.sin(angle)

        if self._running == False:
            x_displacement = x_unit * self._walking_speed
            y_displacement = y_unit * self._walking_speed
        else:
            x_displacement = x_unit * self._running_speed
            y_displacement = y_unit * self._running_speed

        self.model.space.move_agent(self, (x_displacement, y_displacement))

    def get_angle(self):
        angle = random.randint(0, 360)
        return angle

    def get_gender(self):
        return self._gender

    def step(self):
        self.move()


class Visitor(Person):
    def __init__(self, unique_id, model) -> None:
        super().__init__(unique_id, model)


class Worker(Person):
    def __init__(self, unique_id, model) -> None:
        super().__init__(unique_id, model)


class ConcertHall(Model):
    def __init__(self, n_visitors, width, height) -> None:
        self._n_visitor = n_visitors
        self.middle = (width / 2, height / 2)
        self.space = ContinuousSpace(width, height, False)
        self._schedule = RandomActivation(self)
        self._locations = []

        for i in range(n_visitors):
            visitor = Visitor(i, ConcertHall)
            self._schedule.add(visitor)
            x = self.random.randrange(self.space.width)
            y = self.random.randrange(self.space.height)
            self._grid.place_agent(visitor, (x, y))

    def save_locations(self):
        agents = self.get_neighbours(
            self.middle,
            np.sqrt((self.space.width / 2) ** 2 + (self.space.height / 2) ** 2),
        )
        print(agents)
        # agents = np.array(agents)
        # positions = [agent.pos for agent in agents]
        # self._locations.append(positions)

    def step(self) -> None:
        self._schedule.step()
