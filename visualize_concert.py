from concert_model import ConcertHall
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap

if __name__ == "__main__":
    visitors = 1000
    workers = 10
    steps = 10

    model = ConcertHall(visitors, workers, 100, 100)

    print("Computing the model")
    for i in range(steps):
        model.step()

    positions = model.datacollector.get_agent_vars_dataframe()
    positions = positions.groupby(level=0)

    print("Creating the animation")
    fig, ax = plt.subplots()
    plt.grid()

    colourmap = ["black"] * visitors + ["red"] * workers

    images = [
        [plt.scatter(image.to_numpy()[:, 0], image.to_numpy()[:, 1], c=colourmap)]
        for step, image in positions
    ]
    ani = animation.ArtistAnimation(
        fig, images, interval=100, blit=True, repeat_delay=100
    )
    ani.save("animations/concert.gif")
