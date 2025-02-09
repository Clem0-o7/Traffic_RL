import os
import sys
import optparse
import traci
from sumolib import checkBinary

# Make sure to import your Agent and TrafficEnvironment classes here
from train import Agent, TrafficEnvironment

def run_simulation(model_name="model", steps=500):
    traci.start([checkBinary("sumo-gui"), "-c", "configuration.sumocfg",
                 "--tripinfo-output", "tripinfo.xml"])

    env = TrafficEnvironment()
    env.initialize()  # Initialize after starting TraCI

    agent = Agent(input_dims=4, n_actions=4)
    agent.load(f'models/{model_name}.bin')

    total_time = 0
    step = 0

    while step <= steps:
        traci.simulationStep()
        
        for junction in env.all_junctions:
            state = env.get_state(junction)
            action = agent.choose_action(state)
            
            # Apply action
            phase_duration = 6 if action % 2 == 0 else 16
            phase_state = ["yyyrrrrrrrrr", "GGGrrrrrrrrr", 
                           "rrryyyrrrrrr", "rrrGGGrrrrrr", 
                           "rrrrrryyyrrr", "rrrrrrGGGrrr", 
                           "rrrrrrrrryyy", "rrrrrrrrrGGG"][action]
            env.set_traffic_light(junction, phase_duration, phase_state)
            
            # Get reward (negative waiting time)
            controlled_lanes = traci.trafficlight.getControlledLanes(junction)
            reward = -env.get_waiting_time(controlled_lanes)
            total_time -= reward

        step += 1

    print(f"Total waiting time: {total_time}")
    traci.close()

if __name__ == "__main__":
    parser = optparse.OptionParser()
    parser.add_option("-m", dest='model_name', default="model", help="Name of model")
    parser.add_option("-s", dest='steps', type='int', default=500, help="Number of steps")
    options, _ = parser.parse_args()

    run_simulation(model_name=options.model_name, steps=options.steps)