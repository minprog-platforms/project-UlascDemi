from concert_model import ConcertHall
import numpy as np

import plotly.graph_objs as go
from plotly.subplots import make_subplots

import matplotlib.pyplot as plt
import matplotlib.animation as animation


def main():
    n_visitors = 2500
    n_workers = 15
    steps = 600

    width = 100
    height = 100

    n_simulations = 5

    avg_time_hmap = np.zeros((101, 101))
    avg_acc_hmap = np.zeros((101, 101))

    for i in range(n_simulations):
        print("------------------------------------")
        print(f"Simulation {i+1} of {n_simulations}")
        model = ConcertHall(n_visitors, n_workers, width, height)

        # print("Computing the model")
        for i in range(steps):
            if i % 50 == 0 and i != 0:
                print(f"Step {i}/{steps}")
            model.step()

        # print("Computing heatmap")
        workers = model.get_workers()

        x_loc = [round(worker.pos[0]) for worker in workers]
        y_loc = [round(worker.pos[1]) for worker in workers]

        time = [worker.get_avg_accident_time() for worker in workers]
        resolved_accidents = [worker.accident_amount for worker in workers]

        for i, value in enumerate(time):
            avg_time_hmap[y_loc[i], x_loc[i]] += value

        for i, value in enumerate(resolved_accidents):
            avg_acc_hmap[y_loc[i], x_loc[i]] += value

    conc_width = width - model.worker_space
    conc_height = height - model.worker_space

    avg_time_hmap[model.worker_space : conc_width, model.worker_space : conc_height] = 0
    avg_acc_hmap[model.worker_space : conc_width, model.worker_space : conc_height] = 0

    np.true_divide(avg_time_hmap, n_simulations)
    np.true_divide(avg_acc_hmap, n_simulations)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Average time spent on accidents",
            "Average amount of accidents resolved",
        ),
    )

    fig.add_trace(go.Heatmap(z=avg_time_hmap, connectgaps=True), 1, 1)
    fig.add_trace(go.Heatmap(z=avg_acc_hmap, connectgaps=True), 1, 2)
    fig.write_html("output_graphs.html")

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
