import os
import sys
import math

import numpy as np # type: ignore
import pandas as pd # type: ignore
import plotly.express as px # type: ignore
import plotly.graph_objects as go #type: ignore


class DDM():
    def __init__(self, dt, input_a, input_b, noise_mag, threshold, initial_condition = 0):
        self.dt = dt #in ms
        self.input_a = input_a
        self.input_b = input_b
        self.drift_rate = self.input_a - self.input_b
        self.noise_mag = noise_mag #noise values should go from 0 to 1.
        self.threshold = Threshold(threshold)
        self.initial_condition = initial_condition

        self.simulated_trajectories = None
        self.simulated_results = None
    
    def simulate(self,trials, timesteps):
        results = []
        trajectories = []
        noise_vector = np.random.randn(trials,timesteps)
        for trial in range(trials):
            x = self.initial_condition
            traj = np.zeros(timesteps)
            for y in range(timesteps):
                noise_term = noise_vector[trial,y] * np.sqrt(self.dt)
                dx = ((self.drift_rate)*self.dt) + (self.noise_mag * noise_term)
                x += dx
                traj[y] = x

                if x >= self.threshold.th[y]:
                    results.append("A")
                    break
                elif x <= -self.threshold.th[y]:
                    results.append("B")
                    break
            else:
                results.append("None")
            trajectories.append(traj[:y+1])

            
        self.simulated_trajectories = trajectories
        self.simulated_results = results
    
    def plot_trajectories(self):
        s = pd.DataFrame(self.simulated_trajectories).T
        s.index = s.index * self.dt
        fig = px.line(s, title="Drift-Diffusion Model Simulations", labels={"index": "Time (ms)", "value": "Decision Variable x(t)"})
        fig.add_trace(go.Scatter(x=np.linspace(0,1000,len(self.threshold.th)), y=self.threshold.th, mode='lines', name='Decision A'))
        fig.add_trace(go.Scatter(x=np.linspace(0,1000,len(self.threshold.th)), y=-self.threshold.th, mode='lines', name='Decision B'))
        fig.show()

class Threshold():
    def __init__(self, th, type = "standard"):
        self.type = type
        self.th = np.full(1000,th)
    
    def collapse(self, timesteps, a, b, c):
        self.type = "collapsing"
        x = np.linspace(0, timesteps, timesteps)
        bound = a * np.exp(-b * x) + c#(a - b * np.log(x + c))
        self.th = bound