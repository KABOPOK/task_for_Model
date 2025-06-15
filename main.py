import pybullet as p
import pybullet_data
import time
import numpy as np
from scipy.linalg import solve_continuous_are
import math
from control import lqr
import matplotlib.pyplot as plt

# --- System Parameters ---
m = 5.0       # Mass of the payload (kg)
l = 2.0       # Length of the link (m)
g = 3.71      # Mars gravity (m/s²) i belive that i am on the Mars

# ẋ = Ax + Bu
# ẋ = Ax + B(-Kx) = (A - BK)x
# Цель: чтобы матрица (A - BK) имела устойчивые (отрицательные) собственные значения

# Moment of inertia about the rotation axis
I = m * l**2
A = np.array([
    [0, 1],
    [m*g*l/I, 0]
])
B = np.array([[0], [1/I]])
Q = np.diag([10, 1])
R = np.array([[0.1]])

K_lqr, _, _ = lqr(A, B, Q, R)
K = -K_lqr  #A-BK

print("LQR Gain Matrix:", K)

# --- PyBullet Simulation ---
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -g)
planeId = p.loadURDF("plane.urdf")
robot = p.loadURDF("two-link.urdf", basePosition=[0, 0, 0])

# Find joint
joint_index = next((i for i in range(p.getNumJoints(robot))
                    if p.getJointInfo(robot, i)[1].decode() == "joint_1"), None)
if joint_index is None:
    raise ValueError("Joint 'joint_1' not found")

p.setJointMotorControl2(robot, joint_index, p.VELOCITY_CONTROL, force=0)

# Simulation setup
dt = 1.0 / 240.0
sim_time = 5.0
initial_angle = np.pi + 0.5
p.resetJointState(robot, joint_index, targetValue=initial_angle)

# Data logging
log_time = []
log_angle = []
log_control = []

# Stability detection
stabilization_time = None
STABILITY_THRESHOLD = 0.02  # rad
STABILITY_DURATION = 1.0  # sec

# Main loop
start_time = time.time()
current_time = 0.0
stable_start = None

while current_time < sim_time:
    # Get state
    angle, angle_vel = p.getJointState(robot, joint_index)[:2]
    angle_error = angle - math.pi

    # Check stability
    if abs(angle_error) < STABILITY_THRESHOLD:
        if stable_start is None:
            stable_start = current_time
        elif current_time - stable_start >= STABILITY_DURATION and stabilization_time is None:
            stabilization_time = current_time
    else:
        stable_start = None

    # Control
    state = np.array([[angle_error], [angle_vel]])
    torque = (K @ state)[0, 0]
    p.setJointMotorControl2(robot, joint_index, p.TORQUE_CONTROL, force=torque)

    # Simulation step
    p.stepSimulation()

    # Logging
    log_time.append(current_time)
    log_angle.append(angle)
    log_control.append(torque)

    time.sleep(dt)
    current_time = time.time() - start_time

p.disconnect()

plt.figure(figsize=(14, 8))

# Angle plot with improved stabilization marker
plt.subplot(2, 1, 1)
plt.plot(log_time, log_angle, label='Actual Angle', linewidth=2)
plt.axhline(y=np.pi, color='r', linestyle='--', label='Target (π)')

# Only draw stabilization line if truly stabilized
if stabilization_time and stabilization_time < log_time[-1] - 1.0:
    plt.axvline(x=stabilization_time, color='g', linestyle='-',
               linewidth=2, alpha=0.7, label=f'Stabilization ({stabilization_time:.2f}s)')
    plt.axvspan(stabilization_time, log_time[-1], color='g', alpha=0.05)

plt.title('Pendulum Angle Stabilization on Mars', fontsize=14)
plt.ylabel('Angle [rad]', fontsize=12)
plt.legend(fontsize=10, loc='upper right')
plt.grid(True, linestyle=':', alpha=0.7)
plt.ylim(min(log_angle)-0.1, max(log_angle)+0.1)

# Control torque plot
plt.subplot(2, 1, 2)
plt.plot(log_time, log_control, label='Control Torque', color='purple', linewidth=2)
if stabilization_time and stabilization_time < log_time[-1] - 1.0:
    plt.axvline(x=stabilization_time, color='g', linestyle='-', linewidth=2, alpha=0.7)
    plt.axvspan(stabilization_time, log_time[-1], color='g', alpha=0.05)

plt.title('Control Torque', fontsize=14)
plt.xlabel('Time [s]', fontsize=12)
plt.ylabel('Torque [Nm]', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle=':', alpha=0.7)

plt.tight_layout()
plt.show()