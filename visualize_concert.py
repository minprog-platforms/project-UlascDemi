from concert_model import ConcertHall
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap

model = ConcertHall(2, 10, 100, 100)

for _ in range(100):
    model.step()

locations = model.get_locations()

fig, ax = plt.subplots()
locations = model.get_locations()
plt.grid()
images = [[plt.scatter(image[:, 0], image[:, 1], c="k")] for image in locations]


ani = animation.ArtistAnimation(fig, images, interval=100, blit=True, repeat_delay=100)
ani.save("animations/animation.gif")
