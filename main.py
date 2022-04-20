import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import atan, degrees


class GPS:
    data: pd.DataFrame
    linestart: int
    lineend: int
    x: np.ndarray
    y: np.ndarray

    def __init__(self, filename: str):
        self.data: pd.DataFrame = self.read_data(filename)
        self.linestart = 460
        self.lineend = 619
        self.x = self.data['x'][self.linestart:self.lineend].to_numpy(dtype=float)
        self.y = self.data['y'][self.linestart:self.lineend].to_numpy(dtype=float)

    def read_data(self, name: str) -> pd.DataFrame:
        data = pd.read_csv(name, sep=",", header=None)
        data.columns = ["time", "x", "y", "z"]
        return data

    def LSM(self, x: np.ndarray, y: np.ndarray) -> (float, float):
        N = len(x)
        mx = x.sum() / N
        my = y.sum() / N
        a2 = np.dot(x.T, x) / N
        a11 = np.dot(x.T, y) / N
        kk = (a11 - mx * my) / (a2 - mx ** 2)
        bb = my - kk * mx
        return kk, bb

    def GLSM(self, x: np.ndarray, y: np.ndarray) -> (float, float):
        A = np.vstack([x, np.ones(len(x)), y]).T
        U, S, V = np.linalg.svd(A)
        V = V.T.conj()
        paramsGLSM = -V[:, 2] / V[2, 2]
        return paramsGLSM[0], paramsGLSM[1]

    def show_full_trajectory(self):
        plt.plot(self.data['x'], self.data['y'], 'o', label='Full trajectory', markersize=2)
        plt.legend()
        plt.show()

    def calculate_angles(self, k1: float, k2: float) -> float:
        angle_in_radians = atan(k1) - atan(k2)
        angle_in_degrees = degrees(angle_in_radians)
        return angle_in_degrees

    def task1(self):
        # y=kx+b
        kk, bb = self.LSM(self.x, self.y)
        kk2, bb2 = self.GLSM(self.x, self.y)

        plt.plot(self.x, self.y, 'o', label='Original data', markersize=3)
        plt.plot(self.x, kk * self.x + bb, 'r', label='LSM line', linewidth=3)
        plt.plot(self.x, kk2 * self.x + bb2, 'g', label='GLSM line')

        plt.xlabel('X-coord (m)')
        plt.ylabel('Y-coord (m)')
        plt.title('Angles between line is {0} degrees'.format(self.calculate_angles(kk, kk2)))

        plt.legend()
        plt.show()

    def task2(self):
        delta_x = np.diff(self.x)
        np.insert(delta_x, 0, 0)
        delta_y = np.diff(self.y)
        np.insert(delta_y, 0, 0)
        c = np.array([delta_x,
                      delta_y])
        velocity_arr = np.linalg.norm(c, axis=0)
        time_step_arr = np.arange(len(delta_x))

        kk, bb = self.LSM(time_step_arr, velocity_arr)
        kk2, bb2 = self.GLSM(time_step_arr, velocity_arr)
        plt.plot(time_step_arr, velocity_arr, 'y', label='Velocity original', linewidth=2)
        plt.plot(time_step_arr, kk * time_step_arr + bb, 'r', label='LSM line', linewidth=3)
        plt.plot(time_step_arr, kk2 * time_step_arr + bb2, 'g', label='GLSM line')
        plt.xlabel('Time (sec)')
        plt.ylabel('Velocity abs (m/s)')
        plt.title('Angles between line is {0} degrees'.format(self.calculate_angles(kk, kk2)))
        plt.legend()
        plt.show()


if __name__ == '__main__':
    gps = GPS('Rover1.cleanest')
    gps.task1()
    gps.task2()
