import os
import sys
import math

import numpy as np # type: ignore
import pandas as pd # type: ignore
import plotly.express as px # type: ignore
import plotly.graph_objects as go #type: ignore


class DDM():
    def __init__(self, dt, drift_rate, noise_mag, threshold, initial_condition = 0):
        self.dt = dt #in ms
        self.drift_rate = drift_rate #positive values go towards A (top)
        self.noise_mag = noise_mag #noise values should go from 0 to 1.
        self.threshold = Threshold(threshold)
        self.initial_condition = initial_condition

        self.simulated_trajectories = None
        self.simulated_results = None
    
    def simulate(self,trials, timesteps):
        results = []
        trajectories = []

        noise_vector = np.random.randn(trials,timesteps)
        trial_objects = [Trial(prob_A=0.5) for _ in range(trials)]

        for i, trial in enumerate(trial_objects):
            trial_drift = self.drift_rate if trial.truth_value == "A" else -self.drift_rate

            x = self.initial_condition
            traj = np.zeros(timesteps)

            for y in range(timesteps):
                noise_term = noise_vector[i,y] * np.sqrt(self.dt)
                dx = ((trial_drift)*self.dt) + (self.noise_mag * noise_term)
                x += dx
                traj[y] = x

                if x >= self.threshold.th[y]:
                    results.append(("A", str(trial.truth_value)))
                    break
                elif x <= -self.threshold.th[y]:
                    results.append(("B", str(trial.truth_value)))
                    break
            else:
                results.append(("None", str(trial.truth_value)))
            trajectories.append(traj[:y+1])

            
        self.simulated_trajectories = trajectories
        self.simulated_results = results

        self.error_rate = sum((x[0]!=x[1] for x in self.simulated_results))/trials
    
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

class Trial():
    def __init__(self, prob_A=0.5):
        self.prob_A = prob_A
        self.truth_value = np.random.choice(["A", "B"], p=[prob_A, 1 - prob_A])