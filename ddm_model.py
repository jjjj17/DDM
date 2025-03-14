import os
import sys
import math
from typing import List, Tuple, Union

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import plotly.express as px  # type: ignore
import plotly.graph_objects as go  # type: ignore
import optuna  # type: ignore


class DDM:
    def __init__(self, drift_rate: float, noise_mag: float, threshold: 'Threshold', initial_condition: float = 0):
        """
        Initialize the DDM model.

        :param drift_rate: The drift rate of the model.
        :param noise_mag: The magnitude of the noise.
        :param threshold: The decision threshold.
        :param initial_condition: The initial condition of the decision variable.
        """
        self.drift_rate = drift_rate
        self.noise_mag = noise_mag
        self.threshold = threshold
        self.initial_condition = initial_condition

        self.simulated_trajectories = None
        self.simulated_results = None

    def fit(self, trials_reactiontimes: np.ndarray, true_labels: np.ndarray, fitting_trials: int = 100, method: str = "mse", accuracy_weight: float = 0.5) -> optuna.Study:
        """
        Fit the DDM model to the given trials data using Optuna for hyperparameter optimization.

        :param trials_reactiontimes: The reaction times from the trials.
        :param true_labels: The true labels for the trials.
        :param fitting_trials: The number of trials for the optimization process.
        :param method: The fitting method to use ("mse" or "nll").
        :param accuracy_weight: The weight given to accuracy in the objective function (only used for "mse" method).
        :return: The Optuna study object.
        """
        timesteps = 1000
        dt = 1

        def objective_mse(trial: optuna.Trial) -> float:
            drift_rate = trial.suggest_float("drift_rate", 0.001, 0.2)
            noise_mag = trial.suggest_float("noise_mag", 0.01, 1.0)
            threshold = trial.suggest_float("threshold", 0.1, 20.0)

            self.drift_rate = drift_rate
            self.noise_mag = noise_mag
            self.threshold = Threshold(threshold)

            simulation = Simulation(self, dt, timesteps, len(trials_reactiontimes))

            mse = np.mean((simulation.reaction_times - trials_reactiontimes) ** 2)  # MSE
            accuracy = np.mean([1 if result[0] == true_label else 0 for result, true_label in zip(simulation.simulated_results, true_labels)])

            # Combine MSE and accuracy into a single objective function
            objective_value = mse * (1 - accuracy_weight) + (1 - accuracy) * accuracy_weight
            return objective_value

        def negative_log_likelihood(simulated_reaction_times, observed_reaction_times):
            mean = np.mean(simulated_reaction_times)
            std = np.std(simulated_reaction_times)
            nll = -np.sum(np.log(1 / (std * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((observed_reaction_times - mean) / std) ** 2)))
            return nll

        def objective_nll(trial: optuna.Trial) -> float:
            drift_rate = trial.suggest_float("drift_rate", 0.001, 0.2)
            noise_mag = trial.suggest_float("noise_mag", 0.01, 1.0)
            threshold = trial.suggest_float("threshold", 0.1, 20.0)

            self.drift_rate = drift_rate
            self.noise_mag = noise_mag
            self.threshold = Threshold(threshold)

            simulation = Simulation(self, dt, timesteps, len(trials_reactiontimes))

            nll = negative_log_likelihood(simulation.reaction_times, trials_reactiontimes)
            return nll

        if method == "mse":
            objective = objective_mse
        elif method == "nll":
            objective = objective_nll
        else:
            raise ValueError("Invalid method. Choose 'mse' or 'nll'.")

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=fitting_trials)

        best_params = study.best_params
        print(f"Best parameters: {best_params}")

        self.drift_rate = best_params['drift_rate']
        self.noise_mag = best_params['noise_mag']
        self.threshold = Threshold(best_params['threshold'])

        return study


class Threshold:
    def __init__(self, th: float, type: str = "standard", collapse_arg: Tuple[float, float, float] = (10, 0.01, 10)):
        """
        Initialize the Threshold object.

        :param th: The threshold value.
        :param type: The type of threshold ("standard" or "collapsing").
        :param collapse_arg: The arguments for the collapsing threshold.
        """
        self.type = type
        self.th = th
        self.a, self.b, self.c = collapse_arg

    def generate_threshold(self, timesteps: int, dt: float) -> None:
        """
        Generate the threshold values for the given timesteps and time step size.

        :param timesteps: The number of timesteps.
        :param dt: The time step size.
        """
        self.timesteps = timesteps
        self.dt = dt

        if self.type == "standard":
            self.simulated_th = np.full(self.timesteps, self.th)
        elif self.type == "collapsing":
            x = np.linspace(0, self.timesteps * self.dt, self.timesteps)
            bound = self.a * np.exp(-self.b * x) + self.c
            self.simulated_th = bound
        else:
            raise ValueError("Incorrect threshold type.")


class Trial:
    def __init__(self, prob_A: float = 0.5):
        """
        Initialize the Trial object.

        :param prob_A: The probability of A being the correct choice.
        """
        self.prob_A = prob_A
        self.truth_value = np.random.choice(["A", "B"], p=[prob_A, 1 - prob_A])


class Simulation:
    def __init__(self, model: DDM, dt: float, timesteps: int, trials: int):
        """
        Initialize the Simulation object.

        :param model: The DDM model.
        :param dt: The time step size.
        :param timesteps: The number of timesteps.
        :param trials: The number of trials.
        """
        if not isinstance(model, DDM):
            raise TypeError("Expected instance of DDM Model.")

        self.model = model
        self.dt = dt
        self.timesteps = timesteps
        self.trials = trials
        self.simulated_trajectories = None
        self.simulated_results = None

        self.model.threshold.generate_threshold(self.timesteps, self.dt)

        self.simulate_trials()

    def simulate_trials(self) -> None:
        """
        Simulate the trials for the DDM model.
        """
        noise_vector = np.random.randn(self.trials, self.timesteps)
        trial_objects = [Trial(prob_A=0.5) for _ in range(self.trials)]
        reaction_times = np.zeros((self.trials, 1))
        results = []
        trajectories = []

        for i, trial in enumerate(trial_objects):
            trial_drift = self.model.drift_rate if trial.truth_value == "A" else -self.model.drift_rate

            x = self.model.initial_condition
            traj = np.zeros(self.timesteps)

            for y in range(self.timesteps):
                noise_term = noise_vector[i, y] * np.sqrt(self.dt)
                dx = (trial_drift * self.dt) + (self.model.noise_mag * noise_term)
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
            trajectories.append(traj[:y + 1])
            reaction_times[i] = y * self.dt

        self.simulated_trajectories = trajectories
        self.simulated_results = results
        self.reaction_times = reaction_times
        self.error_rate = sum((x[0] != x[1] for x in self.simulated_results)) / self.trials

    def plot_trajectories(self) -> None:
        """
        Plot the trajectories of the simulation.
        """
        s = pd.DataFrame(self.simulated_trajectories).T
        s.index = s.index * self.dt

        fig = px.line(s, title="Drift-Diffusion Model Simulations", labels={"index": "Time (ms)", "value": "Decision Variable x(t)"})
        fig.add_trace(go.Scatter(x=np.linspace(0, self.dt * self.timesteps, len(self.model.threshold.simulated_th)), y=self.model.threshold.simulated_th, mode='lines', name='Decision A'))
        fig.add_trace(go.Scatter(x=np.linspace(0, self.dt * self.timesteps, len(self.model.threshold.simulated_th)), y=-self.model.threshold.simulated_th, mode='lines', name='Decision B'))
        fig.show()