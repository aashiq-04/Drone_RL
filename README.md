# Drone RL Project

This repository contains code to train and test a reinforcement learning (RL) model using the **PPO (Proximal Policy Optimization)** algorithm for drone control.

## Installation

To get started with this project, follow these steps:

### 1. Clone the Repository
Clone the repository to your local machine:
```bash
git clone https://github.com/aashiq-04/Drone_RL.git
```
### 2. Navigate to the Project Directory
```bash
cd Drone_RL
```
 ### 4. Install Dependencies
Make sure to install all required dependencies listed in requirements.txt:
```bash
pip install -r requirements.txt
```
## Running the Program
### 1. Training the Model
To train the RL model using PPO, navigate to the src directory and run the train_ppo.py script:
```bash
cd new_src
python train_ppo.py
```
This will start training the model. The progress will be shown in the terminal, and you can monitor the training performance.

### 2. Testing the Model
Once the model is trained, you can test it using the test_model.py script. Navigate back to the src directory and run the following command:

```bash
python test_model.py
```
This will load the trained model and run tests on the drone control environment.
