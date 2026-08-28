# Trust-Tiago
Trust: the twisty story of TIAGO, an LLM and their users 


## Setup
### Step 1: Download ROS2 WS
If using a Windows machine it is easiest to either setup dual boot with Linux or use [docker](https://docs.docker.com/guides/ros2/) 
Must use the Humble version to be compatible with Tiago.

### Step 2: Install Xlaunch
If you are using docker [Xlaunch](http://www.straightrunning.com/XmingNotes/) is needed to visualise the processes running in the container

### Step 3: Download Gazebo Sim



### Step 4: Download Tiago

### Step 5: Download Moveit2

### Step 6: Download LMStudio
[LMStudio](https://lmstudio.ai/download) is used but [Ollama](https://ollama.com/download/windows) is also an alternative

### Step 7: 

## Simulation

## How to run the entire system

### Launch Simulation
Startup Xlaunch
ros2 launch tiago_gazebo tiago_gazebo.launch.py is_public_sim:=True world_name:='pick_place_demo'


ros2 run vla_node my_node
