from concert_model import ConcertHall
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def main():
    visitors = 3000
    workers = 15
    steps = 400

    model = ConcertHall(visitors, workers, 100, 100)

    print("Computing the model")
    for i in range(steps):
        if i % 50 == 0 and i != 0:
            print(f"Step {i}/{steps}")
        model.step()

    workers = model.get_workers()

    worker_x_loc = [worker.pos[0] for worker in workers]
    worker_y_loc = [worker.pos[1] for worker in workers]

    time_spent_accidents = [worker.time_spent_accidents for worker in workers]

    # positions = model.datacollector.get_agent_vars_dataframe()
    # positions = positions.groupby(level=0)

    # print("Creating the animation")
    # fig, ax = plt.subplots()
    # plt.grid()

    # colourmap = ["black"] * visitors + ["red"] * workers

    # images = [
    #     [
    #         plt.scatter(
    #             image.to_numpy()[:, 0],
    #             image.to_numpy()[:, 1],
    #             c=colourmap,
    #             linewidths=0.01,
    #         )
    #     ]
    #     for step, image in positions
    # ]

    # ani = animation.ArtistAnimation(
    #     fig, images, interval=100, blit=True, repeat_delay=100
    # )
    # ani.save("animations/concert.gif")


if __name__ == "__main__":
    main()
