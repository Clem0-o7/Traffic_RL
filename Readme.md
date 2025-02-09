# Traffic Light Control using Reinforcement Learning

This project aims to optimize traffic light control using reinforcement learning (RL) techniques. The RL agent is trained to minimize the waiting time of vehicles at traffic junctions.

## Requirements

- Python 3.x
- SUMO (Simulation of Urban MObility)
- PyTorch
- NumPy
- Matplotlib
Refer requirements.txt for versions

## Installation

1. Install SUMO from [here](https://www.eclipse.org/sumo/).
2. Install the required Python packages:
    ```bash
    pip install torch numpy matplotlib
    ```

## Usage

### Training the Model

To train the model, run the following command:
```bash
python train_RL.py --train -m <model_name> -e <epochs> -s <steps>
```
- `--train`: Flag to indicate training mode.
- `-m <model_name>`: Name of the model file to save.
- `-e <epochs>`: Number of epochs for training.
- `-s <steps>`: Number of steps per epoch.

### Testing the Model

To test the model, run the following command:
```bash
python train_RL.py -m <model_name> -e <epochs> -s <steps>
```
- `-m <model_name>`: Name of the model file to load.
- `-e <epochs>`: Number of epochs for testing.
- `-s <steps>`: Number of steps per epoch.

## File Structure

- `train_RL.py`: Main script for training and testing the RL agent.
- `models/`: Directory to save the trained models.
- `plots/`: Directory to save the training plots.
- `configuration.sumocfg`: SUMO configuration file.
- `maps/tripinfo.xml`: Output file for trip information.

## Functions

### `get_vehicle_numbers(lanes)`

Returns the number of vehicles in each lane.

### `get_waiting_time(lanes)`

Returns the total waiting time for all vehicles in the given lanes.

### `phaseDuration(junction, phase_time, phase_state)`

Sets the phase duration and state for a traffic light.

### `Model`

Neural network model for the RL agent.

### `Agent`

RL agent that interacts with the environment and learns to optimize traffic light control.

### `run(train=True, model_name="model", epochs=50, steps=500)`

Executes the TraCI control loop for training or testing the RL agent.

### `get_options()`

Parses command line options.

## License

This project is licensed under the MIT License.