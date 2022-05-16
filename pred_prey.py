from mesa import Model, Agent
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import random

FEMALE, MALE = 0, 1
HUNGER_TICK = 0.3


class Grass(Agent):
    def __init__(self, unique_id: int, model: Model) -> None:
        super().__init__(unique_id, model)


class Animal(Agent):
    def __init__(self, unique_id, model) -> None:
        super().__init__(unique_id, model)
        self.uid = unique_id
        self.type = ""  # kan prey of predator zijn
        self.gender = random.randint(FEMALE, MALE)
        self.pregn = False
        self.age = 0
        self.color = 0x00  # later iets met kleuren en achtergrond doen
        self.hunger = 0

    def move(self):
        pass

    def eat(self):
        pass

    def die(self):
        pass

    def aging(self):
        if self.model.step_number % 10 == 0:
            self.age += 1

    def step(self) -> None:
        if self.hunger >= 10:
            self.die()

        self.hunger += HUNGER_TICK

        self.move()


class Prey(Animal):
    def __init__(self, unique_id, model) -> None:
        super().__init__(unique_id, model)
        self.type = "prey"


class PredPreyModel(Model):
    def __init__(self, width, height, prey_count, pred_count) -> None:
        self.grid = MultiGrid(width, height, False)
        self.schedule = RandomActivation(self)
        self.step_number = 0
        self.prey_count = prey_count
        self.pred_count = pred_count
        self.last_uid = 0

        # Create prey
        for _ in self.prey_count:
            prey = Prey(self.new_uid(), PredPreyModel)
            self.schedule.add(prey)

            # Add prey to random location on the map
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(prey, (x, y))

    def new_uid(self):
        self.last_uid += 1

        return self.last_uid

    def stepy(self):
        self.step_number += 1
        self.schedule.step()
