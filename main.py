import datetime
import random
import numpy as np
import os
import sys
import traci
import display
import matplotlib.pyplot as plt

from traci.constants import INVALID_DOUBLE_VALUE


# Rounds the given value with the given step
def roundByStep(x, step):
    roundStep = int(abs(x) / step) * step
    diff = abs(x) - roundStep
    if diff > step / 2:
        roundStep = roundStep + step
    if x < 0:
        return -roundStep
    else:
        return roundStep


# Returns the index of the state to be used in the Q table at the beginning
# Returns an arbitrary state
def initializeState():
    #    return getIndexOfState((30, 2, 0))
    return int(len(STATES) / 2)


# Returns the index of the state defined by (space_headway, relative_speed, speed)
def getIndexOfState(st):
    return hash_states_index[st]


# Resets the environment to learn again and returns the initial state index
def reset(simulation_counter):
    if simulation_counter > 0:
        traci.close(False)

    print("Starting the TRACI server...")
    traci.start(sumoCmd)

    # we take the full control of the learning car
    traci.vehicle.setSpeedMode(LEARNING_VEHICLE, 0)

    # do one step to initialize system
    traci.simulationStep()

    return initializeState()


# Returns the number of states
def getNbStates():
    return NB_STATES


def getNbActions():
    return NB_ACTIONS


# Returns the distance between the leader and the controlled car
def getSpaceHeadway():
    dist_leader = traci.vehicle.getDistance(LEADER_VEHICLE)
    dist_learner = traci.vehicle.getDistance(LEARNING_VEHICLE)

    return dist_leader - dist_learner


# Updates the current state according to SUMO infos, returns true if there is a collision between leader and learner
def updateCurrentState():
    # update the current state to the state computing by using the current 3 parameters (space headway,
    # relative speed, speed)

    coll_list = traci.simulation.getCollidingVehiclesIDList()
    if len(coll_list) > 0:
        print("Collision at time ", traci.simulation.getTime(), " between ", coll_list)
        output_f.write("Collision at time " + str(traci.simulation.getTime()) + " between " + str(coll_list) + "\n")
        return 0, True

    space_headway = getSpaceHeadway()

    if space_headway == INVALID_DOUBLE_VALUE:
        # it means that one of the vehicle is not on the ring, stop the simulation
        print(" leader_position = ", traci.vehicle.getPosition(LEADER_VEHICLE))
        print(" learner_position = ", traci.vehicle.getPosition(LEARNING_VEHICLE))

    speed = traci.vehicle.getSpeed(LEARNING_VEHICLE)

    relative_speed = traci.vehicle.getSpeed(LEADER_VEHICLE) - speed

    # Convert measurements to be in ranges using projection
    space_headway = min(MAX_SPACE_HEADWAY, max(0, space_headway))
    relative_speed = min(MAX_RELATIVE_SPEED, max(- MAX_RELATIVE_SPEED, relative_speed))
    speed = min(MAX_SPEED, max(0, speed))

    # Discretize values to get a state in Q
    space_headway = roundByStep(space_headway, STEP_SPACE_HEADWAY)
    relative_speed = roundByStep(relative_speed, STEP_RELATIVE_SPEED)
    speed = roundByStep(speed, STEP_SPEED)

    new_state_id = getIndexOfState((space_headway, relative_speed, speed))

    # No collision
    return new_state_id, False


# Updates the speed of the controlled car by applying the action specified as parameter (action index)
def updateSpeed(action):
    current_learning_vehicle_speed = traci.vehicle.getSpeed(LEARNING_VEHICLE)

    if current_learning_vehicle_speed < 1 and action == 2:  # action = 2 <=> accelerate
        new_speed = 2
    else:
        new_speed = current_learning_vehicle_speed * ACTIONS[action]
    return new_speed


# Compute the reward depending on if the controlled car is in a safety distance range.
def getRewardSafetyDistance():
    safeDist = traci.vehicle.getSpeed(LEARNING_VEHICLE) * 3
    safeDistRangeLow = safeDist - traci.vehicle.getSpeed(LEARNING_VEHICLE) / 3
    safeDistRangeHigh = safeDist + traci.vehicle.getSpeed(LEARNING_VEHICLE) / 3

    space_headway = getSpaceHeadway()

    if safeDistRangeLow <= space_headway <= safeDistRangeHigh:
        return MAX_SPEED / 2
    else:
        if space_headway < safeDistRangeLow:
            return -MAX_SPEED / 2
        else:
            return 0


# apply the action and returns the new state, the reward, if the simulation is finished (case of a collision)
def takeAction(action):
    new_speed = updateSpeed(action)

    traci.vehicle.setSpeed(LEARNING_VEHICLE, new_speed)

    # next step with this action
    traci.simulationStep()

    # compute the new state
    new_state_id, coll = updateCurrentState()

    if coll:
        reward = -20
    else:
        # compute reward (equal to the speed)
        # reward = getRewardSafetyDistance()
        reward = traci.vehicle.getSpeed(LEARNING_VEHICLE)

    return new_state_id, reward, coll


# Returns a random action index in [0;NB_ACTIONS-1]
def randomAction():
    # Returns a random action id
    random_action_id = random.randint(0, NB_ACTIONS - 1)

    return random_action_id


# Returns the parameter used for the greedy policy
# Add more randomness at the beginning of the learning
def getEpsilon(nb_iter):
    if nb_iter < 200:
        epsilon = 0.2
    else:
        epsilon = 0.1

    return epsilon


# MAIN PROGRAM
# initialization of the Sumo command lines
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

sumoGuiBinary = "C:/Program Files (x86)/Eclipse/Sumo/bin/sumo-gui"

sumoBinary = "C:/Program Files (x86)/Eclipse/Sumo/bin/sumo"

sumoConfig = ["-c", "ring.sumocfg", "-S", "--collision.mingap-factor", "0", "--collision.action", "warn",
              "--no-warnings", "true"]

sumoCmd = [sumoBinary]
sumoGuiCmd = [sumoGuiBinary]
for i in range(len(sumoConfig)):
    sumoCmd.append(sumoConfig[i])
    sumoGuiCmd.append(sumoConfig[i])

# Possible actions : decelerate, stabilize, accelerate
ACTIONS = [0.9, 1, 1.1]  # [0.8, 1, 1.2]
NB_ACTIONS = len(ACTIONS)

# Different parameters of the states

MAX_SPACE_HEADWAY = 50  # 100  # 1500 m
MAX_SPEED = 5  # 28  # 28 m/s <=> 100 km/h
MAX_RELATIVE_SPEED = 4  # 8  # 28 m/s <=> 100 km/h

STEP_SPACE_HEADWAY = 1
STEP_SPEED = 1
STEP_RELATIVE_SPEED = 1

# Possible states are added to the list STATES
STATES = []

# This hash table is indexed by the states (sh, rs, s) and contains the corresponding index of the state
hash_states_index = {}

counter = 0

# Creation of the results output file
startDateStr = str(datetime.datetime.today()).replace(" ", "_").replace(":", "-").replace(".", "_")
output_f = open("resultsQLearning" + startDateStr + ".txt", 'w')

# States are defined by 3 parameters :
# * the space headway from 0 to MAX_SPACE_HEADWAY
for i in np.arange(0, MAX_SPACE_HEADWAY + 1, step=STEP_SPACE_HEADWAY):
    # * the relative speed from -MAX_RELATIVE_SPEED to MAX_RELATIVE_SPEED (m/s)
    for j in np.arange(-MAX_RELATIVE_SPEED, MAX_RELATIVE_SPEED + 1, step=STEP_RELATIVE_SPEED):
        # * the speed from 0 to MAX_SPEED (m/s)
        for k in np.arange(0, MAX_SPEED + 1, step=STEP_SPEED):
            hs = roundByStep(i, STEP_SPACE_HEADWAY)
            rs = roundByStep(j, STEP_RELATIVE_SPEED)
            sp = roundByStep(k, STEP_SPEED)

            s = (hs, rs, sp)
            STATES.append(s)
            hash_states_index[s] = counter

            counter += 1

NB_STATES = len(STATES)

print("Nb of states : ", NB_STATES)
output_f.write("Nb of states : " + str(NB_STATES) + "\n")

# Names of the vehicles
LEARNING_VEHICLE = "learning_vehicle"
LEADER_VEHICLE = "leader"

# Implementation of Q-learning algorithm

# Initialize parameters
alpha = 0.9  # 0.6
gamma = 1.0

# Initialize Q with 0
Q = np.zeros([getNbStates(), getNbActions()])

# We will run nb_scenarios learning sessions
nb_scenarios = 3000
count_steps = 0

count_nb_iter_successive_without_coll = 0

collision = False

# Arrays with the total driven distance of the learning car at each end of iteration
dataDistanceLearningCar = []
nbIter = []

for i in range(nb_scenarios):

    # when 100 successive iterations without collision, stop learning
    #    if not collision:
    #        count_nb_iter_successive_without_coll += 1
    #        if count_nb_iter_successive_without_coll > 100:
    #            break
    #    else:
    #        count_nb_iter_successive_without_coll = 0

    print("************************** ", i, "th scenario **************************")
    output_f.write("************************** " + str(i) + "th scenario **************************" + "\n")

    S = reset(i)

    collision = False
    # Initialize simulation (does as many steps as needed to have learner and leader on the ring
    while traci.vehicle.getSpeed(LEARNING_VEHICLE) == INVALID_DOUBLE_VALUE:
        traci.simulationStep()

    traci.vehicle.setSpeed(LEARNING_VEHICLE, 3)  # set the speed to 3 at the beginning, otherwise the speed stays to 0

    #    for j in range(30):
    #        traci.simulationStep()

    count_steps = 0

    while not collision and count_steps < 1000:  # maximum of steps in one learning, because it can loop otherwise
        count_steps += 1

        # choose the action by applying a greedy policy
        p = random.random()

        # case of "random" action
        if p < getEpsilon(count_steps):
            action_id = randomAction()

        # case of "best" action at this state
        else:
            action_id = np.argmax(Q[S])
            if Q[S, 0] == 0 and Q[S, 1] == 0 and Q[S, 2] == 0:
                # no action has been explored for this state, choose accelerate
                action_id = 2

        # take the action :
        # observe R at the next state S_prime
        # if collision, this simulation is done, we have to reset the environment
        S_prime, R, collision = takeAction(action_id)

        # Update Q
        current_Q = Q[S, action_id]
        next_best_Q = np.max(Q[S_prime, :])
        if collision:  # if the action triggers a collision, S_prime does not exist, thus does not appear in the formula
            Q[S][action_id] += alpha * (R + Q[S][action_id])
            print("*** COLLISION : Q de ", S, " et ", action_id, " = ", Q[S, action_id])
            output_f.write(
                "*** COLLISION : Q de " + str(S) + " et " + str(action_id) + " = " + str(Q[S, action_id]) + "\n")
            print("Distance de learning : ", traci.vehicle.getDistance(LEARNING_VEHICLE))
        else:
            best_next_action_id = np.argmax(Q[S_prime])
            Q[S][action_id] += alpha * (R + gamma * Q[S_prime][best_next_action_id] - Q[S][action_id])

        S = S_prime

    print("Speed of learning car : ", traci.vehicle.getSpeed(LEARNING_VEHICLE))
    # End of the iteration : either collision or 1000 steps
    dataDistanceLearningCar.append(traci.vehicle.getDistance(LEARNING_VEHICLE))
    nbIter.append(i)

print("End of Q-learning")

# Sumo GUI launching to see the result of the learning
traci.close()

# Displays the driven distance at each iteration (should increase)
plt.plot(nbIter, dataDistanceLearningCar, c='green')
plt.title('Driven distance at each iteration')
plt.xlabel('Iteration')
plt.ylabel('Driven distance')
plt.show()

traci.start(sumoGuiCmd)

# we take the full control of the learning car
traci.vehicle.setSpeedMode(LEARNING_VEHICLE, 0)

# do one step to initialize system
traci.simulationStep()

collision = False
count_steps = 0

traci.vehicle.setSpeed(LEARNING_VEHICLE, 3)  # set the speed to 3 at the beginning, otherwise the speed stays to 0

for j in range(30):
    traci.simulationStep()

S = initializeState()

# run the simulation to see the results
while not collision and count_steps < 1000:
    count_steps += 1

    # choose the best action
    action_id = np.argmax(Q[S])

    S_prime, R, collision = takeAction(action_id)

# Save the Q table value in the result file
output_f.write("\n\n")
output_f.write("  $$$$$$$$$$$$     Q TABLE    $$$$$$$$$$$$$  \n")
for i in range(getNbStates()):
    for j in range(getNbActions()):
        output_f.write(str(Q[i, j]) + "  ")
    output_f.write("\n")

output_f.close()

# Displays the 3D graph of Q_normalized
display.displayQin3D(STATES, Q)

# Displays the actions (best, medium, worst) for each state
display.displayQForEachAction(Q, startDateStr)
