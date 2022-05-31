from concert_model import ConcertHall
import numpy as np

import plotly.graph_objs as go
from plotly.subplots import make_subplots

import matplotlib.pyplot as plt
import matplotlib.animation as animation


def main():
    # Number of visitors, workers and seconds of runtime
    n_visitors = 2500
    n_workers = 15
    frames = 600

    width = 100
    height = 100

    # Number of batch simulation
    n_simulations = 5

    # Creating an empty array to later append to
    avg_time_hmap = np.zeros((101, 101))
    avg_acc_hmap = np.zeros((101, 101))

    # Running the batch siulations
    for i in range(n_simulations):
        print("------------------------------------")
        print(f"Simulation {i+1} of {n_simulations}")

        # Creating the model
        model = ConcertHall(n_visitors, n_workers, width, height)

        # Computing each frame
        for i in range(frames):
            if i % 50 == 0 and i != 0:
                print(f"Step {i}/{frames}")
            model.step()

        workers = model.get_workers()

        # Create a lists of all x and y locations of the workers
        x_loc = [round(worker.pos[0]) for worker in workers]
        y_loc = [round(worker.pos[1]) for worker in workers]

        # Create a list of the average time and resolved accidents of each worker
        time = [worker.get_avg_accident_time() for worker in workers]
        resolved_accidents = [worker.accident_amount for worker in workers]

        # Insert the average time into the previously made empty array
        for i, value in enumerate(time):
            avg_time_hmap[y_loc[i], x_loc[i]] += value

        # Insert the average resolved accidents into the previously made array
        for i, value in enumerate(resolved_accidents):
            avg_acc_hmap[y_loc[i], x_loc[i]] += value

    # Visitor area width and height
    conc_width = width - model.worker_space
    conc_height = height - model.worker_space

    # Clear any bugs where a worker position appear in the visitor area
    avg_time_hmap[model.worker_space : conc_width, model.worker_space : conc_height] = 0
    avg_acc_hmap[model.worker_space : conc_width, model.worker_space : conc_height] = 0

    np.true_divide(avg_time_hmap, n_simulations)
    np.true_divide(avg_acc_hmap, n_simulations)

    # Create the subplots
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Average time spent on accidents",
            "Average amount of accidents resolved",
        ),
    )

    # Add the figures to the subplot
    fig.add_trace(go.Heatmap(z=avg_time_hmap, connectgaps=True), 1, 1)
    fig.add_trace(go.Heatmap(z=avg_acc_hmap, connectgaps=True), 1, 2)
    fig.write_html("output_graphs.html")


if __name__ == "__main__":
    main()
