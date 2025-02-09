import os
import sys
import optparse
import traci
import numpy as np
import matplotlib.pyplot as plt

# SUMO_HOME setup
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary

def get_waiting_time(lanes):
    waiting_time = 0
    for lane in lanes:
        waiting_time += traci.lane.getWaitingTime(lane)
    return waiting_time

def phaseDuration(junction, phase_time, phase_state):
    traci.trafficlight.setRedYellowGreenState(junction, phase_state)
    traci.trafficlight.setPhaseDuration(junction, phase_time)

def run(gui=False, epochs=50, steps=500):
    total_time_list = []

    for e in range(epochs):
        if gui:
            traci.start([checkBinary("sumo-gui"), "-c", "configuration.sumocfg", "--tripinfo-output", "tripinfo.xml"])
        else:
            traci.start([checkBinary("sumo"), "-c", "configuration.sumocfg", "--tripinfo-output", "tripinfo.xml"])
        
        print(f"epoch: {e}")
        select_lane = [
            ["yyyrrrrrrrrr", "GGGrrrrrrrrr"],
            ["rrryyyrrrrrr", "rrrGGGrrrrrr"],
            ["rrrrrryyyrrr", "rrrrrrGGGrrr"],
            ["rrrrrrrrryyy", "rrrrrrrrrGGG"],
        ]

        step = 0
        total_time = 0
        phase_duration = 30  # Fixed time for each phase
        
        all_junctions = traci.trafficlight.getIDList()
        traffic_lights_time = {junction: 0 for junction in all_junctions}
        all_lanes = []
        
        for junction in all_junctions:
            all_lanes.extend(list(traci.trafficlight.getControlledLanes(junction)))

        while step <= steps:
            traci.simulationStep()
            for junction in all_junctions:
                controlled_lanes = traci.trafficlight.getControlledLanes(junction)
                waiting_time = get_waiting_time(controlled_lanes)
                total_time += waiting_time
                
                if traffic_lights_time[junction] == 0:
                    # Cycle through phases in a fixed order
                    current_phase = (step // phase_duration) % 4
                    phaseDuration(junction, 6, select_lane[current_phase][0])
                    phaseDuration(junction, phase_duration, select_lane[current_phase][1])
                    traffic_lights_time[junction] = phase_duration
                else:
                    traffic_lights_time[junction] -= 1
            
            step += 1

        print("total_time", total_time)
        total_time_list.append(total_time)
        traci.close()
        sys.stdout.flush()

        if gui:
            break  # Only run one epoch in GUI mode

    if not gui:
        plt.plot(list(range(len(total_time_list))), total_time_list)
        plt.xlabel("epochs")
        plt.ylabel("total time")
        plt.savefig('plots/time_vs_epoch_fixed_time.png')
        plt.show()

def get_options():
    optParser = optparse.OptionParser()
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
    optParser.add_option(
        "--gui",
        action="store_true",
        default=False,
        help="Run with GUI",
    )
    
    options, args = optParser.parse_args()
    return options

if __name__ == "__main__":
    options = get_options()
    epochs = options.epochs
    steps = options.steps
    gui = options.gui
    run(gui=gui, epochs=epochs, steps=steps)