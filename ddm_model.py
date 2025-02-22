import os
import sys
import math

import numpy as np # type: ignore
import pandas as pd # type: ignore
import plotly.express as px # type: ignore
import plotly.graph_objects as go #type: ignore


class DDM():
    def __init__(self, drift_rate, noise_mag, threshold, initial_condition = 0):
        self.drift_rate = drift_rate #positive values go towards A (top)
        self.noise_mag = noise_mag #noise values should go from 0 to 1.
        self.threshold = Threshold(threshold)
        self.initial_condition = initial_condition

        self.simulated_trajectories = None
        self.simulated_results = None


class Threshold():
    def __init__(self, th, type = "standard", collapse_arg = (10,0.01,10)):
        self.type = type
        self.th = th
        self.a,self.b,self.c = collapse_arg
    
    def generate_threshold(self, timesteps, dt):
        self.timesteps = timesteps
        self.dt = dt

        if self.type == "standard":
            self.simulated_th = np.full(self.timesteps,self.th)
        elif self.type == "collapsing":
            x = np.linspace(0, self.timesteps * self.dt, self.timesteps)
            bound = self.a * np.exp(-self.b * x) + self.c
            self.simulated_th = bound

class Trial():
    def __init__(self, prob_A=0.5):
        self.prob_A = prob_A
        self.truth_value = np.random.choice(["A", "B"], p=[prob_A, 1 - prob_A])

class Simulation():
    def __init__(self, model, dt, timesteps, trials):
        #Type verification to make sure model is an instance of DDM
        if not isinstance(model, DDM):
            raise TypeError("Expected instance of DDM Model.")
        
        #initializing variables
        self.model = model
        self.dt = dt
        self.timesteps = timesteps
        self.trials = trials
        self.simulated_trajectories = None
        self.simulated_results = None

        #generate thresholds for the simulation
        self.model.threshold.generate_threshold(self.timesteps, self.dt)


        #creation of variables for simulation
        noise_vector = np.random.randn(trials,timesteps)
        trial_objects = [Trial(prob_A=0.5) for _ in range(trials)]
        reaction_times = np.zeros((trials,1))
        results = []
        trajectories = []

        for i, trial in enumerate(trial_objects):
            trial_drift = self.model.drift_rate if trial.truth_value == "A" else -self.model.drift_rate

            x = self.model.initial_condition
            traj = np.zeros(self.timesteps)

            for y in range(timesteps):
                noise_term = noise_vector[i,y] * np.sqrt(self.dt)
                dx = ((trial_drift)*self.dt) + (self.model.noise_mag * noise_term)
                x += dx
                traj[y] = x

                if x >= self.model.threshold.simulated_th[y]:
                    results.append(("A", str(trial.truth_value)))
                    break
                elif x <= -self.model.threshold.simulated_th[y]:
                    results.append(("B", str(trial.truth_value)))
                    break
            else:
                results.append(("None", str(trial.truth_value)))
            trajectories.append(traj[:y+1])
            reaction_times[i] = y * self.dt

            
        self.simulated_trajectories = trajectories
        self.simulated_results = results
        self.reaction_times = reaction_times #includes "no decision" as a max decision time. maybe warps the reaction times (this would be no reaction technically).
        self.error_rate = sum((x[0]!=x[1] for x in self.simulated_results))/trials #includes no decision as an error. technically correct but maybe not ideal.




    def plot_trajectories(self):
        s = pd.DataFrame(self.simulated_trajectories).T
        s.index = s.index * self.dt

        fig = px.line(s, title="Drift-Diffusion Model Simulations", labels={"index": "Time (ms)", "value": "Decision Variable x(t)"})
        fig.add_trace(go.Scatter(x=np.linspace(0,self.dt*self.timesteps,len(self.model.threshold.simulated_th)), y=self.model.threshold.simulated_th, mode='lines', name='Decision A'))
        fig.add_trace(go.Scatter(x=np.linspace(0,self.dt*self.timesteps,len(self.model.threshold.simulated_th)), y=-self.model.threshold.simulated_th, mode='lines', name='Decision B'))
        fig.show()