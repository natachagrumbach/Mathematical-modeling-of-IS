import numpy as np
import matplotlib.pyplot as plt


def displayQForEachAction(Q, startSimulationDate):
    valuesAcceleration = []
    valuesStabilize = []
    valuesDeceleration = []
    indexStates = []

    # Displays 2 lines of points, one for values of Q for acceleration
    # One for values of Q for deceleration
    # for each state
    for ind in range(len(Q)):
        valuesDeceleration.append(Q[ind, 0])
        valuesStabilize.append(Q[ind, 1])
        valuesAcceleration.append(Q[ind, 2])
        indexStates.append(ind)

    plt.scatter(indexStates, valuesStabilize, c='blue')
    plt.scatter(indexStates, valuesDeceleration, c='red')
    plt.scatter(indexStates, valuesAcceleration, c='green')

    plt.title('Q values for acceleration (green), stabilization (blue) and deceleration (red)')
    plt.xlabel('Index of state')
    plt.ylabel('Q')
    plt.savefig("Index_of_state_Q" + startSimulationDate + ".png")
    plt.show()


def displayQin3D(STATES, Q):
    x_decelerate = []
    y_decelerate = []
    z_decelerate = []
    x_stabilize = []
    y_stabilize = []
    z_stabilize = []
    x_accelerate = []
    y_accelerate = []
    z_accelerate = []

    # Displays in specific color points depending on best action
    # for each state
    for ind in range(len(STATES)):
        best_action_index = np.argmax(Q[ind])
        if best_action_index == 0:  # decelerate
            if Q[ind, 0] != 0:
                x_decelerate.append(STATES[ind][0])  # space headway
                z_decelerate.append(STATES[ind][1])  # relative speed
                y_decelerate.append(STATES[ind][2])  # speed

        if best_action_index == 1:  # stabilize
            if Q[ind, 1] != 0:
                x_stabilize.append(STATES[ind][0])  # space headway
                z_stabilize.append(STATES[ind][1])  # relative speed
                y_stabilize.append(STATES[ind][2])  # speed

        if best_action_index == 2:  # accelerate
            if Q[ind, 2] != 0:
                x_accelerate.append(STATES[ind][0])  # space headway
                z_accelerate.append(STATES[ind][1])  # relative speed
                y_accelerate.append(STATES[ind][2])  # speed

    # Creating figure
    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection="3d")

    # Creating plot
    ax.scatter3D(x_decelerate, y_decelerate, z_decelerate, color="red")
    ax.scatter3D(x_stabilize, y_stabilize, z_stabilize, color="blue")
    ax.scatter3D(x_accelerate, y_accelerate, z_accelerate, color="green")

    plt.title("3D Q plot")
    plt.title('Chosen action depending on space headway, speed and relative speed')
    ax.set_xlabel('Space headway', fontweight='bold')
    ax.set_ylabel('Speed', fontweight='bold')
    ax.set_zlabel('Relative speed', fontweight='bold')

    # show plot
    plt.show()
