import optparse
import traci
from sumolib import checkBinary

# Import Agent and TrafficEnvironment classes here
from train import Agent, TrafficEnvironment

def run_simulation(model_name="model", steps=500):
    """
    Runs the traffic simulation using the specified model and number of steps.

    Parameters:
    model_name (str): The name of the model to load.
    steps (int): The number of simulation steps to run.
    """
    # Start the SUMO simulation with the specified configuration file
    traci.start([checkBinary("sumo-gui"), "-c", "configuration.sumocfg",
                 "--tripinfo-output", "tripinfo.xml"])

    # Initialize the traffic environment
    env = TrafficEnvironment()
    env.initialize()  # Initialize after starting TraCI

    # Load the trained agent model
    agent = Agent(input_dims=4, n_actions=4)
    agent.load(f'models/{model_name}.bin')

    total_time = 0
    step = 0

    # Run the simulation for the specified number of steps
    while step <= steps:
        traci.simulationStep()
        
        for junction in env.all_junctions:
            # Get the current state of the junction
            state = env.get_state(junction)
            # Choose an action based on the current state
            action = agent.choose_action(state)
            
            # Apply the chosen action
            phase_duration = 6 if action % 2 == 0 else 16
            phase_state = ["yyyrrrrrrrrr", "GGGrrrrrrrrr", 
                           "rrryyyrrrrrr", "rrrGGGrrrrrr", 
                           "rrrrrryyyrrr", "rrrrrrGGGrrr", 
                           "rrrrrrrrryyy", "rrrrrrrrrGGG"][action]
            env.set_traffic_light(junction, phase_duration, phase_state)
            
            # Get the reward (negative waiting time)
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