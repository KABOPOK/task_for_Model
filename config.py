# config.py
import numpy as np

SIMULATION_STEP = 1.0 / 240.0
SIMULATION_DURATION = 5.0
SIMULATION_STEPS = int(SIMULATION_DURATION / SIMULATION_STEP)

TARGET_POSITION = np.array([0.2, 0.2, 0.1])  # Целевая позиция для ЭЭ
END_EFFECTOR_INDEX = 2  # Номер последнего звена в URDF

KP = 1.0  # Коэффициент пропорционального усиления
MAX_FORCE = 10.0
