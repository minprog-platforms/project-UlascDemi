from concert_model import ConcertHall
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap

model = ConcertHall(100, 10, 100, 100)

steps = 10
for _ in range(steps):
    print(f"step {_}")
    model.step()

data = model.datacollector.get_agent_vars_dataframe()
print(data)

# print(data.index.get_level_values(0))


# locations = model.get_locations()

# print(locations)

# fig, ax = plt.subplots()
# locations = model.get_locations()
# plt.grid()
# images = [[plt.scatter(image[:, 0], image[:, 1], c="k")] for image in locations]


# ani = animation.ArtistAnimation(fig, images, interval=100, blit=True, repeat_delay=100)
# ani.save("animations/concert.gif")
