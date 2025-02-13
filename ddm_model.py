import os
import sys
import math

import numpy as np # type: ignore
import pandas as pd # type: ignore
import plotly.express as px # type: ignore


class DDM():
    def __init__(self, dt, input_a, input_b, noise_mag, threshold, initial_condition = 0):
        self.dt = dt
        self.input_a = input_a
        self.input_b = input_b
        self.noise_mag = noise_mag
        self.threshold = threshold
        self.initial_condition = initial_condition

        self.simulated_trajectories = None
        self.simulated_results = None
    
    def simulate(self,trials, timesteps):
        results = []
        trajectories = []
        for trial in range(trials):
            x = self.initial_condition
            traj = [x]
            for y in range(timesteps):
                noise_term = np.random.randn()
                dxdt = (self.input_a - self.input_b) + (self.noise_mag * noise_term)
                x += dxdt
                traj.append(x)

                if x >= self.threshold:
                    results.append("A")
                    break
                elif x <= -self.threshold:
                    results.append("B")
                    break
            else:
                results.append("None")
            trajectories.append(traj)
        
        self.simulated_trajectories = trajectories
        self.simulated_results = results
    
    def plot_trajectories(self):
        s = pd.DataFrame(self.simulated_trajectories).T
        fig = px.line(s, title="Drift-Diffusion Model Simulations", labels={"index": "Time Step", "value": "Decision Variable x(t)"})
        fig.add_hline(y=self.threshold, line_dash="dash", line_color="red", annotation_text="Decision A")
        fig.add_hline(y=-self.threshold, line_dash="dash", line_color="blue", annotation_text="Decision B")
        fig.show()