import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.constants as const


Q = [-const.e, -const.e, -const.e, -const.e]

m = const.m_e
k = 9 * 10**9
N = 4
tmax = 7
dt = 0.1

def accel():

    x = [0, 2, 0, -2]
    y = [2, 0, -2, 0]
    z = [1, 1, 1, 1]

    ax, ay, az = [0] * N, [0] * N, [0] * N

    for i in range(N):
        for j in range(N):
            if i != j:
                r = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                ax[i] += (k * Q[i] * Q[j]) * (x[i] - x[j]) / (r**3 * m)
                ay[i] += (k * Q[i] * Q[j]) * (y[i] - y[j]) / (r**3 * m)
                az[i] = 0
    return ax, ay, az


def pos_and_vel():
    t = 0

    vx = [0, 0, 0, 0]
    vy = [0, 0, 0, 0]
    vz = [50, 50, 50, 50]
    x = [0, 2, 0, -2]
    y = [2, 0, -2, 0]
    z = [1, 1, 1, 1]

    x_data = open(r'data/x_data.txt', 'w')
    y_data = open(r'data/y_data.txt', 'w')
    z_data = open(r'data/z_data.txt', 'w')

    vx_data = open(r'data/vx.txt', 'w')
    vy_data = open(r'data/vy.txt', 'w')
    vz_data = open(r'data/vz.txt', 'w')

    np.savetxt(x_data, np.array([x]))
    np.savetxt(y_data, np.array([y]))
    np.savetxt(z_data, np.array([z]))
    np.savetxt(vx_data, np.array([vx]))
    np.savetxt(vy_data, np.array([vy]))
    np.savetxt(vz_data, np.array([vz]))

    while(t <= tmax):
        for i in range(N):
            a0x, a0y, a0z = accel()

            vx[i] += a0x[i] * dt
            vy[i] += a0y[i] * dt
            vz[i] += a0z[i] * dt

            x[i] += vx[i] * dt
            y[i] += vy[i] * dt
            z[i] += vz[i] * dt

        np.savetxt(x_data, np.array([x]))
        np.savetxt(y_data, np.array([y]))
        np.savetxt(z_data, np.array([z]))
        np.savetxt(vx_data, np.array([vx]))
        np.savetxt(vy_data, np.array([vy]))
        np.savetxt(vz_data, np.array([vz]))

        t += dt

    x_data.close()
    y_data.close()
    z_data.close()
    vx_data.close()
    vy_data.close()
    vz_data.close()


def plot():
    x = pd.read_csv(r'data/x_data.txt', header = None, sep = ' ')
    y = pd.read_csv(r'data/y_data.txt', header = None, sep = ' ')
    z = pd.read_csv(r'data/z_data.txt', header = None, sep = ' ')

    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')  # elev = 90, azim = -90
    ax.scatter(x, y, z)
    ax.set_xlabel("x", fontsize = 16)
    ax.set_ylabel("y", fontsize = 16)
    ax.set_zlabel("z", fontsize = 16)
    ax.set_xlim(-35, 35)
    ax.set_ylim(-35, 35)
    ax.set_title("Space charge effect", fontweight = 'bold', fontsize = 16)
    plt.show()

def animating_data(skip = 1, dt = 0.05):
    x = pd.read_csv(r'data/x_data.txt', header=None, sep=' ')
    y = pd.read_csv(r'data/y_data.txt', header=None, sep=' ')
    z = pd.read_csv(r'data/z_data.txt', header=None, sep=' ')

    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    line, _ = np.shape(x)
    time = np.linspace(0, dt * (line - 1), line)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')  ## azim = 90, elev = 90

    for frames, t in enumerate(time[::skip]):
        fs = frames * skip
        ax.clear()
        ax.scatter(x[fs], y[fs], z[fs], marker = 'o')
        ax.set_xlabel("x", fontsize = 16)
        ax.set_ylabel("y", fontsize = 16)
        ax.set_zlabel("z", fontsize = 16)
        ax.set_xlim(-1600, 1600)
        ax.set_ylim(-1600, 1600)
        ax.set_zlim(0, 400)
        ax.set_title("Space charge effect", fontweight = 'bold', fontsize = 16)
        plt.pause(0.000001)
    plt.show()


pos_and_vel()
# plot()
animating_data()
