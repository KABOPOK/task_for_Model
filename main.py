import pybullet as p
import pybullet_data
import time
import numpy as np
from scipy.linalg import solve_continuous_are
import math
import matplotlib.pyplot as plt

# --- System Parameters ---
m = 5.0       # Mass of the payload (kg)
l = 2.0       # Length of the link (m)
g = 3.71      # Mars gravity (m/s²) i belive that i am on the Mars

# Moment of inertia about the rotation axis
I = m * l**2

# --- LQR Calculation ---
# System matrices (linearized around upright position)
# State: [angle, angular velocity]
A = np.array([
    [0, 1],
    [m*g*l/I, 0]  # Restoring force coefficient
])

# Control matrix (torque)
B = np.array([[0], [1/I]])

# Weight matrices
Q = np.diag([10, 1])  # State weights: angle and angular velocity
R = np.array([[0.1]])  # Control weight

# Solve Riccati equation
P = solve_continuous_are(A, B, Q, R)

# Calculate LQR gain matrix
K = np.linalg.inv(R) @ B.T @ P

print("LQR Gain Matrix:", K)

# --- PyBullet Initialization ---
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -g)
planeId = p.loadURDF("plane.urdf")

# Load the two-link pendulum from external URDF
robot = p.loadURDF("two-link.urdf", basePosition=[0, 0, 0])

# Find joint index
joint_index = None
for i in range(p.getNumJoints(robot)):
    joint_info = p.getJointInfo(robot, i)
    if joint_info[1].decode("utf-8") == "joint_1":
        joint_index = i
        break

if joint_index is None:
    raise ValueError("Joint 'joint_1' not found in the URDF file")

# Disable default motor controller
p.setJointMotorControl2(
    robot, 
    joint_index,
    p.VELOCITY_CONTROL,
    force=0
)

# Simulation parameters
dt = 1.0 / 240.0  # Simulation timestep
sim_time = 10.0    # Total simulation time in seconds

# Initial state: slight offset from vertical
initial_angle = 0.1  # radians (~5.7 degrees)
p.resetJointState(robot, joint_index, targetValue=initial_angle)

# Data logging
log_time = []
log_angle = []
log_control = []

# Main simulation loop
start_time = time.time()
current_time = 0.0

while current_time < sim_time:
    # Get current state
    joint_state = p.getJointState(robot, joint_index)
    angle = joint_state[0]  # Current angle
    angle_vel = joint_state[1]  # Angular velocity
    
    # State vector (deviation from vertical)
    angle_error = angle - math.pi
    
    state = np.array([[angle_error], [angle_vel]])
    
    # Calculate control input
    control = -K @ state
    torque = control[0, 0]
    
    # Apply torque to joint
    p.setJointMotorControl2(
        robot,
        joint_index,
        p.TORQUE_CONTROL,
        force=torque
    )
    
    # Simulation step
    p.stepSimulation()
    
    # Log data
    log_time.append(current_time)
    log_angle.append(angle_error)
    log_control.append(torque)
    
    # Real-time synchronization
    time.sleep(dt)
    current_time = time.time() - start_time

p.disconnect()

# Visualization
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(log_time, log_angle)
plt.title('Angle Deviation from Vertical (Mars Gravity)')
plt.ylabel('Angle (rad)')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(log_time, log_control)
plt.title('Control Torque')
plt.xlabel('Time (s)')
plt.ylabel('Torque (N·m)')
plt.grid(True)

plt.tight_layout()
plt.show()