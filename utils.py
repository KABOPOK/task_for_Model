import matplotlib.pyplot as plt

def log_data(log, time, position, joint_targets):
    log.append({
        'time': time,
        'position': position,
        'joint_targets': joint_targets
    })

def plot_trajectory(log, target):
    times = [entry['time'] for entry in log]
    xs = [entry['position'][0] for entry in log]
    ys = [entry['position'][1] for entry in log]
    zs = [entry['position'][2] for entry in log]

    plt.figure(figsize=(10, 6))
    plt.plot(times, xs, label='X')
    plt.plot(times, ys, label='Y')
    plt.plot(times, zs, label='Z')
    plt.axhline(target[0], color='r', linestyle='--', label='Target X')
    plt.axhline(target[1], color='g', linestyle='--', label='Target Y')
    plt.axhline(target[2], color='b', linestyle='--', label='Target Z')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.legend()
    plt.title('End Effector Position Over Time')
    plt.grid()
    plt.tight_layout()
    plt.show()
