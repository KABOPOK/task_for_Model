# main.py
import pybullet as p
import pybullet_data
import time
import numpy as np
from config import *
from utils import log_data, plot_trajectory

# Инициализация PyBullet
client = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.setTimeStep(SIMULATION_STEP)

# Загрузка модели
p.loadURDF("plane.urdf")
robotId = p.loadURDF("robot.urdf", [0, 0, 0])

# Инициализация логов
trajectory_log = []

for step in range(SIMULATION_STEPS):
    p.stepSimulation()

    # Получение состояний сочленений
    joint_states = p.getJointStates(robotId, [0, 1])
    joint_positions = [state[0] for state in joint_states]

    # Положение конечного эффектора
    link_state = p.getLinkState(robotId, END_EFFECTOR_INDEX)
    current_position = np.array(link_state[4])

    # Ошибка позиции
    position_error = TARGET_POSITION - current_position

    # Якобиан
    jacobian = p.calculateJacobian(
        robotId,
        END_EFFECTOR_INDEX,
        [0, 0, 0],
        list(joint_positions),
        [0.0, 0.0],
        [0.0, 0.0]
    )

    linear_jacobian = np.array(jacobian[0])[:, :2]

    try:
        dq = np.linalg.pinv(linear_jacobian).dot(position_error * KP)
    except np.linalg.LinAlgError:
        dq = np.zeros(2)

    target_positions = [joint_positions[i] + dq[i] for i in range(2)]

    for i in range(2):
        p.setJointMotorControl2(
            bodyUniqueId=robotId,
            jointIndex=i,
            controlMode=p.POSITION_CONTROL,
            targetPosition=target_positions[i],
            force=MAX_FORCE
        )

    log_data(trajectory_log, step * SIMULATION_STEP, current_position, target_positions)
    time.sleep(SIMULATION_STEP)

# Завершение
plot_trajectory(trajectory_log, TARGET_POSITION)
p.disconnect()
