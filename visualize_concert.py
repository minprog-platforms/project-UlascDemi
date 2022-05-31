#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# Created By  : Ulas
# Created Date: 31-05-2022
# ---------------------------------------------------------------------------
"""
This is the visualisation code of the concert_model. This file is meant to be
run to perform the simulation. 
"""
# ---------------------------------------------------------------------------

from concert_model import ConcertHall
import numpy as np

import plotly.graph_objs as go
from plotly.subplots import make_subplots

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import argparse


def main(output, n_visitors, n_workers, n_frames, n_simulations):
    # Number of visitors, workers and seconds of runtime
    # n_visitors = 2500
    # n_workers = 15
    # n_frames = 600

    width = 100
    height = 100

    # Number of batch simulation
    # n_simulations = 5

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
        for i in range(n_frames):
            if i % 50 == 0 and i != 0:
                print(f"Step {i}/{n_frames}")
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

        # Temporary output in case program crashes
        if i % 10 == 0:
            output1 = np.true_divide(avg_time_hmap, i)
            output2 = np.true_divide(avg_acc_hmap, i)

            output1.to_csv("temporary_output/avg_time.csv")
            output2.to_csv("temporary_output/avg_acc_resolved.csv")

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
    fig.write_html(output)

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


if __name__ == "__main__":
    # Set-up parsing command line arguments
    parser = argparse.ArgumentParser(description="Batch run a concert simulation")

    # Adding arguments
    parser.add_argument("output", help="output file (html)")
    parser.add_argument(
        "-v",
        "--n_visitors",
        type=int,
        default=3000,
        help="Amount of visitors (default: 3000)",
    )

    parser.add_argument(
        "-w",
        "--n_workers",
        type=int,
        default=15,
        help="Amount of workers (default: 15)",
    )
    parser.add_argument(
        "-t",
        "--time_steps",
        type=int,
        default=400,
        help="Amount of frames to be calculated, each frame is a second (default: 400)",
    )
    parser.add_argument(
        "-s",
        "--n_simulations",
        type=int,
        default=1,
        help="Amount of simulations to be done, this can be more for a batch simulation (default: 1)",
    )

    # Read arguments from command line
    args = parser.parse_args()

    # Run main with provide arguments
    main(
        args.output,
        args.n_visitors,
        args.n_workers,
        args.time_steps,
        args.n_simulations,
    )
