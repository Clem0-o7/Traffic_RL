from __future__ import absolute_import, print_function

import os
import sys
import time
import optparse
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt

# Ensure SUMO_HOME is set
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary  
import traci 

def get_vehicle_numbers(lanes):
    """
    Get the number of vehicles in each lane.

    Args:
        lanes (list): List of lane IDs.

    Returns:
        dict: Dictionary with lane IDs as keys and vehicle counts as values.
    """
    vehicle_per_lane = dict()
    for l in lanes:
        vehicle_per_lane[l] = 0
        for k in traci.lane.getLastStepVehicleIDs(l):
            if traci.vehicle.getLanePosition(k) > 10:
                vehicle_per_lane[l] += 1
    return vehicle_per_lane

def get_waiting_time(lanes):
    """
    Get the total waiting time for all vehicles in the given lanes.

    Args:
        lanes (list): List of lane IDs.

    Returns:
        int: Total waiting time.
    """
    waiting_time = 0
    for lane in lanes:
        waiting_time += traci.lane.getWaitingTime(lane)
    return waiting_time

def phaseDuration(junction, phase_time, phase_state):
    """
    Set the phase duration and state for a traffic light.

    Args:
        junction (str): Traffic light ID.
        phase_time (int): Duration of the phase.
        phase_state (str): State of the traffic light.
    """
    traci.trafficlight.setRedYellowGreenState(junction, phase_state)
    traci.trafficlight.setPhaseDuration(junction, phase_time)

class Model(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        """
        Initialize the neural network model.

        Args:
            lr (float): Learning rate.
            input_dims (int): Number of input dimensions.
            fc1_dims (int): Number of neurons in the first fully connected layer.
            fc2_dims (int): Number of neurons in the second fully connected layer.
            n_actions (int): Number of possible actions.
        """
        super(Model, self).__init__()
        self.lr = lr
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.linear1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.linear2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.linear3 = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        """
        Forward pass through the network.

        Args:
            state (torch.Tensor): Input state.

        Returns:
            torch.Tensor: Output actions.
        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        actions = self.linear3(x)
        return actions

class Agent:
    def __init__(self, gamma, epsilon, lr, input_dims, fc1_dims, fc2_dims, batch_size, n_actions, junctions, max_memory_size=100000, epsilon_dec=5e-4, epsilon_end=0.05):
        """
        Initialize the agent.

        Args:
            gamma (float): Discount factor.
            epsilon (float): Exploration rate.
            lr (float): Learning rate.
            input_dims (int): Number of input dimensions.
            fc1_dims (int): Number of neurons in the first fully connected layer.
            fc2_dims (int): Number of neurons in the second fully connected layer.
            batch_size (int): Batch size for training.
            n_actions (int): Number of possible actions.
            junctions (list): List of junction IDs.
            max_memory_size (int, optional): Maximum size of the replay memory. Defaults to 100000.
            epsilon_dec (float, optional): Epsilon decay rate. Defaults to 5e-4.
            epsilon_end (float, optional): Minimum value of epsilon. Defaults to 0.05.
        """
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.batch_size = batch_size
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.action_space = [i for i in range(n_actions)]
        self.junctions = junctions
        self.max_mem = max_memory_size
        self.epsilon_dec = epsilon_dec
        self.epsilon_end = epsilon_end
        self.mem_cntr = 0
        self.iter_cntr = 0
        self.replace_target = 100

        self.Q_eval = Model(self.lr, self.input_dims, self.fc1_dims, self.fc2_dims, self.n_actions)
        self.memory = {junction: {
            "state_memory": np.zeros((self.max_mem, self.input_dims), dtype=np.float32),
            "new_state_memory": np.zeros((self.max_mem, self.input_dims), dtype=np.float32),
            "reward_memory": np.zeros(self.max_mem, dtype=np.float32),
            "action_memory": np.zeros(self.max_mem, dtype=np.int32),
            "terminal_memory": np.zeros(self.max_mem, dtype=np.bool_),
            "mem_cntr": 0,
            "iter_cntr": 0,
        } for junction in junctions}

    def store_transition(self, state, state_, action, reward, done, junction):
        """
        Store a transition in the replay memory.

        Args:
            state (list): Current state.
            state_ (list): Next state.
            action (int): Action taken.
            reward (float): Reward received.
            done (bool): Whether the episode is done.
            junction (int): Junction ID.
        """
        index = self.memory[junction]["mem_cntr"] % self.max_mem
        self.memory[junction]["state_memory"][index] = state
        self.memory[junction]["new_state_memory"][index] = state_
        self.memory[junction]['reward_memory'][index] = reward
        self.memory[junction]['terminal_memory'][index] = done
        self.memory[junction]["action_memory"][index] = action
        self.memory[junction]["mem_cntr"] += 1

    def choose_action(self, observation):
        """
        Choose an action based on the current observation.

        Args:
            observation (list): Current observation.

        Returns:
            int: Chosen action.
        """
        state = torch.tensor([observation], dtype=torch.float).to(self.Q_eval.device)
        if np.random.random() > self.epsilon:
            actions = self.Q_eval.forward(state)
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action
    
    def reset(self, junction_numbers):
        """
        Reset the memory counters for the given junctions.

        Args:
            junction_numbers (list): List of junction IDs.
        """
        for junction_number in junction_numbers:
            self.memory[junction_number]['mem_cntr'] = 0

    def save(self, model_name):
        """
        Save the model to a file.

        Args:
            model_name (str): Name of the model file.
        """
        torch.save(self.Q_eval.state_dict(), f'models/{model_name}.bin')

    def learn(self, junction):
        """
        Perform a learning step.

        Args:
            junction (int): Junction ID.
        """
        if self.memory[junction]['mem_cntr'] < self.batch_size:
            return

        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.memory[junction]['mem_cntr'], self.max_mem)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        state_batch = torch.tensor(self.memory[junction]["state_memory"][batch]).to(self.Q_eval.device)
        new_state_batch = torch.tensor(self.memory[junction]["new_state_memory"][batch]).to(self.Q_eval.device)
        reward_batch = torch.tensor(self.memory[junction]['reward_memory'][batch]).to(self.Q_eval.device)
        terminal_batch = torch.tensor(self.memory[junction]['terminal_memory'][batch]).to(self.Q_eval.device)
        action_batch = self.memory[junction]["action_memory"][batch]

        q_eval = self.Q_eval.forward(state_batch)[np.arange(self.batch_size), action_batch]
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0
        q_target = reward_batch + self.gamma * torch.max(q_next, dim=1)[0]
        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)

        loss.backward()
        self.Q_eval.optimizer.step()

        self.iter_cntr += 1
        self.epsilon = max(self.epsilon - self.epsilon_dec, self.epsilon_end)

def run(train=True, model_name="model", epochs=50, steps=500):
    """
    Execute the TraCI control loop.

    Args:
        train (bool, optional): Whether to train the model. Defaults to True.
        model_name (str, optional): Name of the model file. Defaults to "model".
        epochs (int, optional): Number of epochs. Defaults to 50.
        steps (int, optional): Number of steps per epoch. Defaults to 500.
    """
    best_time = np.inf
    total_time_list = list()
    traci.start([checkBinary("sumo"), "-c", "configuration.sumocfg", "--tripinfo-output", "maps/tripinfo.xml"])
    all_junctions = traci.trafficlight.getIDList()
    junction_numbers = list(range(len(all_junctions)))

    brain = Agent(
        gamma=0.99,
        epsilon=1.0,
        lr=0.001,
        input_dims=4,
        fc1_dims=256,
        fc2_dims=256,
        batch_size=64,
        n_actions=4,
        junctions=junction_numbers,
    )

    if not train:
        brain.Q_eval.load_state_dict(torch.load(f'models/{model_name}.bin', map_location=brain.Q_eval.device))

    print(brain.Q_eval.device)
    traci.close()
    for e in range(epochs):
        if train:
            traci.start([checkBinary("sumo"), "-c", "configuration.sumocfg", "--tripinfo-output", "tripinfo.xml"])
        else:
            traci.start([checkBinary("sumo-gui"), "-c", "configuration.sumocfg", "--tripinfo-output", "tripinfo.xml"])

        print(f"Epoch: {e}")
        select_lane = [
            ["yyyrrrrrrrrr", "GGGrrrrrrrrr"],
            ["rrryyyrrrrrr", "rrrGGGrrrrrr"],
            ["rrrrrryyyrrr", "rrrrrrGGGrrr"],
            ["rrrrrrrrryyy", "rrrrrrrrrGGG"],
        ]

        step = 0
        total_time = 0
        min_duration = 5
        
        traffic_lights_time = dict()
        prev_wait_time = dict()
        prev_vehicles_per_lane = dict()
        prev_action = dict()
        all_lanes = list()
        
        for junction_number, junction in enumerate(all_junctions):
            prev_wait_time[junction] = 0
            prev_action[junction_number] = 0
            traffic_lights_time[junction] = 0
            prev_vehicles_per_lane[junction_number] = [0] * 4
            all_lanes.extend(list(traci.trafficlight.getControlledLanes(junction)))

        while step <= steps:
            traci.simulationStep()
            for junction_number, junction in enumerate(all_junctions):
                controled_lanes = traci.trafficlight.getControlledLanes(junction)
                waiting_time = get_waiting_time(controled_lanes)
                total_time += waiting_time
                if traffic_lights_time[junction] == 0:
                    vehicles_per_lane = get_vehicle_numbers(controled_lanes)

                    reward = -1 * waiting_time
                    state_ = list(vehicles_per_lane.values()) 
                    state = prev_vehicles_per_lane[junction_number]
                    prev_vehicles_per_lane[junction_number] = state_
                    brain.store_transition(state, state_, prev_action[junction_number], reward, (step == steps), junction_number)

                    lane = brain.choose_action(state_)
                    prev_action[junction_number] = lane
                    phaseDuration(junction, 6, select_lane[lane][0])
                    phaseDuration(junction, min_duration + 10, select_lane[lane][1])

                    traffic_lights_time[junction] = min_duration + 10
                    if train:
                        brain.learn(junction_number)
                else:
                    traffic_lights_time[junction] -= 1
            step += 1
        print("Total time:", total_time)
        total_time_list.append(total_time)

        if total_time < best_time:
            best_time = total_time
            if train:
                brain.save(model_name)

        traci.close()
        sys.stdout.flush()
        if not train:
            break
    if train:
        plt.plot(list(range(len(total_time_list))), total_time_list)
        plt.xlabel("Epochs")
        plt.ylabel("Total Time")
        plt.savefig(f'plots/time_vs_epoch_{model_name}.png')
        plt.show()

def get_options():
    """
    Parse command line options.

    Returns:
        optparse.Values: Parsed options.
    """
    optParser = optparse.OptionParser()
    optParser.add_option(
        "-m",
        dest='model_name',
        type='string',
        default="model",
        help="Name of model",
    )
    optParser.add_option(
        "--train",
        action='store_true',
        default=False,
        help="Training or testing",
    )
    optParser.add_option(
        "-e",
        dest='epochs',
        type='int',
        default=50,
        help="Number of epochs",
    )
    optParser.add_option(
        "-s",
        dest='steps',
        type='int',
        default=500,
        help="Number of steps",
    )
    
    options, args = optParser.parse_args()
    return options

if __name__ == "__main__":
    options = get_options()
    model_name = options.model_name
    train = options.train
    epochs = options.epochs
    steps = options.steps
    run(train=train, model_name=model_name, epochs=epochs, steps=steps)