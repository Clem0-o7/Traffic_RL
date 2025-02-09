# Traffic Signal Optimization using Reinforcement Learning

This project aims to optimize traffic signal control using reinforcement learning (RL) techniques. The goal is to reduce waiting times and improve traffic flow at intersections.

## Requirements
- Python 3.x
- SUMO (Simulation of Urban MObility)
- PyTorch
- Matplotlib

- Refer requirements.txt for version - 

## Setup
1. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```
2. Ensure that SUMO is installed and the `SUMO_HOME` environment variable is set.

## Usage

### Run Fixed Traffic Signal Control in GUI
To run the fixed traffic signal control in the SUMO GUI, use the following command:
```bash
python fixed.py --gui -s 500
```

### Train Reinforcement Learning Model
To train the RL model, use the following command:
```bash
python train_RL.py -m test -e 30 -s 500
```
- `-m`: Name of the model.
- `-e`: Number of epochs.
- `-s`: Number of steps per epoch.

### Run Dynamic Traffic Signal Control in GUI
To run the dynamic traffic signal control using the trained RL model in the SUMO GUI, use the following command:
```bash
python run_RL.py -m test -s 500
```
- `-m`: Name of the model.
- `-s`: Number of steps.

## Project Structure
- `train_RL.py`: Script to train the RL model.
- `run_RL.py`: Script to run the dynamic traffic signal control using the trained RL model.
- `fixed.py`: Script to run the fixed traffic signal control.
- `Readme.md`: Project documentation.

## Additional Information
- The `train_RL.py` script initializes an RL agent, trains it over multiple epochs, and saves the best-performing model.
- The `run_RL.py` script loads the trained model and uses it to control traffic signals dynamically.
- The project uses SUMO for traffic simulation and PyTorch for implementing the RL model.

For more details, refer to the code and comments in the respective scripts.