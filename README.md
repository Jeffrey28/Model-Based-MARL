# Algorithms:
1. DMPO (Our method)
2. DPPO (Decentralized PPO)
4. CPPO (Centralized PPO)
5. IC3Net

* radius and radius_q: 
    * The observable radius of p should be 1-5.
    * The V of each agent predicts the local reward.
    
# Environments:
CACC Catchup, Slowdown.
Ring Attenuation.
Figure Eight.
ATSC Grid


# Interface:
The state, reward, done should all be numpy arrays.
For MARL, done should be given for each agent, although we assume they are the same.

# Environment setup
1. SUMO installation. Version 1.11.0

The commit number of SUMO, available at https://github.com/eclipse/sumo used to run the results is 2147d155b1.
To install SUMO, you are recommended to refer to https://sumo.dlr.de/docs/Installing/Linux_Build.html to install the specific version via repository checkout. Note that the latest version of SUMO is not compatible with Flow environments.
In brief, after you checkout to that version, run the following command to build the SUMO binaries.
```
sudo apt-get install cmake python g++ libxerces-c-dev libfox-1.6-dev libgdal-dev libproj-dev libgl2ps-dev swig
cd <sumo_dir> # please insert the correct directory name here
export SUMO_HOME="$PWD"
mkdir build/cmake-build && cd build/cmake-build
cmake ../..
make -j$(nproc)
```
After building, you need to manually ad the bin folder into your path:
```
export PATH=$PATH:$SUMO_HOME/bin
```

2. Setting up the environment.

It's recommended to set up the environment via Anaconda. The environment specification is in environment.yml.
After installing the required packages, run
```
export PYTHONPATH="$SUMO_HOME/tools:$PYTHONPATH"
```
in terminal to include the SUMO python packages.

3. Setting up WandB.

Our code uses WandB as logger. Before running our code, you should log in to WandB locally. Please refer to https://docs.wandb.ai/quickstart for more detail.

# Usage
```python
python launcher.py --env ENV --algo ALGO --name NAME --para PARA
```
`ENV` specifies which environment to run in, including `eight`, `ring`, `catchup`， `slowdown`, `Grid` .

`ALGO` specifies the algorithm to use, including `IC3Net`, `CPPO`, `DPPO`, `DMPO`.

`NAME` is the additional name for the logger, which is set to `''` as default.

`PARA` is the hyperparameter json string. The default parameters are loaded from config folder, and this would override specific parameters.

# Results in video form
## Description of the following videos in ATSC-Grid
![ATSC-Grid Net](https://user-images.githubusercontent.com/100258046/163699030-7e9ad4f4-8f62-43ad-8825-4d7ffeb7df1f.png)

This is the network structure of 5*5 intersections in ATSC-Grid. The highlighted areas of the red frame are shown in the following videos and they are also the key area for displaying execution result in DPPO and DMPO. Through the key indicator "Insertion-backlogged vehicles" in the numerical panel on the left, we can observe that "Insertion-backlogged vehicles" will gradually increase with the load of the traffic flow, and then gradually decrease with the effective decision of the traffic light. The maximum value of this indicator is 1486 vehs in DPPO and 1033 vehs in DMPO, which indicates that DMPO can reduce the backlog of vehicles at intersections. Therefore, compared with DPPO, DMPO is more effective in solving traffic jams. Through the three intersections we focused on in the video, we can also observe that DMPO can make traffic jams last for less time.


### Execution result of DPPO(Decentralized PPO) in ATSC-Grid
https://user-images.githubusercontent.com/100258046/163685533-f85cd0ba-a3f6-457c-a35f-e13bfbe47481.mp4

### Execution result of DMPO(Our method) in ATSC-Grid
https://user-images.githubusercontent.com/100258046/163685524-4b1a1ef7-3a01-44da-897e-b8f6f1d48d61.mp4



## Description of the following videos in ATSC-Monaco
![Real Net_point](https://user-images.githubusercontent.com/100258046/165713991-04522202-c3a2-4b52-b17c-70b9b06d305c.png)

This is a more challenging scenario with a heterogeneous network structure with diverse action and observation spaces: ATSC-Monaco traffic network with 28 intersections of real traffic network in Monaco city. The highlighted areas of the red frame are shown in the following videos and they are also the key area for displaying execution result in DPPO and DMPO. Through the key indicator "arrived  vehicles" in the numerical panel on the left, we can observe that " arrived  vehicles " will gradually increase with the load of the traffic flow. The maximum value of this indicator is 563 vehs in DPPO and 752 vehs in DMPO, which indicates that DMPO can reduce the backlog of vehicles at intersections and allow more vehicles to arrive at their destinations. Therefore, compared with DPPO, DMPO is more effective in solving traffic jams. Through the three intersections we focused on in the video, we can also observe that DMPO can make traffic jams last for less time.



### Execution result of DPPO(Decentralized PPO) in ATSC-Monaco
https://user-images.githubusercontent.com/100258046/165706056-cac77b41-e3af-4e2b-bda1-3e42f0b5ac17.mp4

### Execution result of DMPO(Our method) in ATSC-Monaco
https://user-images.githubusercontent.com/100258046/165706098-d8cf2009-c7e6-4a7a-9040-c417d38cbeef.mp4

