#!/usr/bin/env python

import numpy as np
import os, sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch import tensor as ts
from mpl_toolkits.mplot3d import Axes3D


class KalmanFilter():
    '''
    Description:
        Build up a Kalman filter for prediction.
    Arguments:
        state_space <list of matrices> - Contain the initial state and all matrices for a linear state space.
                                       - [X0, A, B, C D]
        P0          <np matrix>        - The initial state covairance matrix.
        Q           <np matrix>        - The state noise covairance matrix.
        R           <np matrix>        - The measurement noise covairance matrix.
    Attributes:
        tk <int>    - Discrete time step.
        Xs <ndarry> - States of every time step.
    Functions
        one_step     <run> - Evolve one step given control input U and measurement Y.
        append_state <set> - Append the given state to the state array Xs.
        predict_more <run> - Predict more steps to the future.
        predict      <run> - The predict step in KF.
        update       <run> - The update step in KF.
    '''

    def __init__(self, state_space, P0, Q, R, pred_offset=10):
        super().__init__() 
        self.ss = state_space # [X0, A, B, C, D]
        self.X = state_space[0]
        self.P = P0
        self.Q = Q
        self.R = R
        self.tK = 0 # discrete time step
        self.offset = pred_offset

        self.Xs = state_space[0]

    def one_step(self, U, Y):
        self.tK += 1
        self.predict(U)
        self.update(U, Y)   
        self.Xs = np.concatenate((self.Xs, self.X), axis=1)
        return self.X

    def append_state(self, X):
        self.Xs = np.concatenate((self.Xs, X), axis=1)
        self.tK += 1

    def predict_more(self, T, evolve_P=True):
        for _ in range(T):
            self.predict(np.zeros((np.shape(self.ss[2])[1],1)), evolve_P)
            self.Xs = np.concatenate((self.Xs, self.X), axis=1)

    def predict(self, U, evolve_P=True):
        A = self.ss[1]
        B = self.ss[2]
        self.X = np.dot(A, self.X) + np.dot(B, U)
        if evolve_P:
            self.P = np.dot(A, np.dot(self.P, A.T)) + self.Q
        return self.X

    def update(self, U, Y):
        C = self.ss[3]
        D = self.ss[4]
        Yh = np.dot(C, self.X) + np.dot(D, U)
        S = self.R + np.dot(C, np.dot(self.P, C.T)) # innovation: covariance of Yh
        K = np.dot(self.P, np.dot(C.T, np.linalg.inv(S))) # Kalman gain
        self.X = self.X + np.dot(K, (Y-Yh))
        self.P = self.P - np.dot(K, np.dot(S, K.T))
        return (self.X,K,S,Yh)

    def inference(self, traj):
        Y = [np.array(traj[1,:]), np.array(traj[2,:]), 
            np.array(traj[3,:]), np.array(traj[4,:])]
        for kf_i in range(len(Y) + self.offset):
            if kf_i<len(Y):
                self.one_step(np.array([[0]]), np.array(Y[kf_i]).reshape(2,1))
            else:
                self.predict(np.array([[0]]), evolve_P=False)
                self.append_state(self.X)
        return self.X, self.P
    
def model_CV(X0, Ts=1):
    A = np.array([[1,0,0,Ts,0,0], [0,1,0,0,Ts,0], [0,0,1,0,0,Ts], [0,0,0,1,0,0], [0,0,0,0,1,0], [0,0,0,0,0,1]])
    B = np.zeros((6,1))
    C = np.array([[1,0,0,0,0,0], [0,1,0,0,0,0], [0,0,1,0,0,0]])
    D = np.zeros((3,1))
    return [X0, A, B, C, D]

def model_CA(X0, Ts=1):
    A = np.array([[1,0,0,Ts,0,0], [0,1,0,0,Ts,0], [0,0,1,0,0,Ts], [0,0,0,1,0,0], [0,0,0,0,1,0], [0,0,0,0,0,1]])
    B = np.array([[0,0,0], [0,0,0], [0,0,0], [Ts,0,0], [0,Ts,0], [0,0,Ts]])
    C = np.array([[1,0,0,0,0,0], [0,1,0,0,0,0], [0,0,1,0,0,0]])
    D = np.zeros((3,3))
    return [X0, A, B, C, D]


def fill_diag(diag):
    M = np.zeros((len(diag),len(diag)))
    for i in range(len(diag)):
        M[i,i] = diag[i]
    return M

# Code Running Functions

def generate_demo_data():
    motion_data = [
        (26, -16, 6.9),
        (24.5, -13.5, 6.8),
        (23, -11, 6.7),
        (21.5, -8.5, 6.6),
        (20, -6, 6.5),
        (18.5, -3.5, 6.4),
        (17, -1, 6.3),
        (15.5, 1.5, 6.2),
        (14, 4, 6.1),
        (12.5, 6.5, 6.0),
        (11, 9, 5.9),
        (9.5, 11, 5.8),
        (8, 12, 5.7),
        (6.5, 12.5, 5.6),
        (5, 12, 5.5),
        (4, 10.5, 5.4),
        (3.5, 8.5, 5.3),
        (3.5, 6.5, 5.2),
        (3, 4.5, 5.1),
        (2, 3, 5),
        (1.5, 2, 5.1),
        (1.25, 1.5, 5.2),
        (1.5, 1.25, 5.3),
        (2, 1.5, 5.4),
        (2.5, 2, 5.5),
        (3.25, 2.75, 5.6),
        (4, 3.5, 5.7),
        (4.75, 4.25, 5.8),
        (5.5, 5, 5.9),
        (6.25, 5.75, 6.0)
    ]

    # Print the first data point from motion data
    # print("First Data Point:", motion_data[0])

    # Print x y and z coordinates of the first data point
    x_init = motion_data[0][0]
    y_init = motion_data[0][1]
    z_init = motion_data[0][2]

    X0 = np.array([[x_init, y_init, z_init, 0, 0, 0]]).transpose() # Assuming initial velocity is 0

    return X0, motion_data


def initialize_kalman_filters(X0):
    # Initialize the starting point (X0) with the first x and y coordinates from motion_data
    model1 = model_CV(X0)
    model2 = model_CA(X0)

    P0 = fill_diag((1,1,1,1,1,1))
    Q1 = np.eye(6)  # Adjust these matrices as needed
    Q2 = np.eye(6)
    R  = np.eye(3)  # Measurement noise covariance matrix

    U1 = np.array([[0]])  # Assuming no control input for model1
    U2 = np.array([[0], [0]])  # Assuming no control input for model2

    KF1 = KalmanFilter(model1, P0, Q1, R)
    KF2 = KalmanFilter(model2, P0, Q2, R)
    return KF1, KF2, U1, U2

import matplotlib.pyplot as plt

def predict_motion(KF1, KF2, U1, U2, motion_data, num_predictions):
    for i in range(len(motion_data) + num_predictions):
        if i < len(motion_data):
            # Process actual data points
            U2 = (np.random.rand(3,1)-0.5) / 10
            KF1.one_step(U1, np.array(motion_data[i]).reshape(3,1))
            KF2.one_step(U2, np.array(motion_data[i]).reshape(3,1))
        else:
            # Predict future data points
            KF1.predict(U1, evolve_P=True)
            KF2.predict(U2, evolve_P=True)
            KF1.append_state(KF1.X)
            KF2.append_state(KF2.X)

    return KF1, KF2

def plot_predictions(KF1, KF2, Y, ax):
    ax.scatter(KF1.Xs[0,:len(Y)], KF1.Xs[1,:len(Y)], KF1.Xs[2,:len(Y)], c='b', marker='o', label='KF Model 1 Actual')
    ax.scatter(KF2.Xs[0,:len(Y)], KF2.Xs[1,:len(Y)], KF2.Xs[2,:len(Y)], c='g', marker='o', label='KF Model 2 Actual')
    ax.scatter(np.array(Y)[:,0], np.array(Y)[:,1], np.array(Y)[:,2], c='r', marker='x', label='Original Data')

    # Plot predictions
    ax.plot(KF1.Xs[0,len(Y):], KF1.Xs[1,len(Y):], KF1.Xs[2,len(Y):], 'bo--', label='KF Model 1 Predictions')
    ax.plot(KF2.Xs[0,len(Y):], KF2.Xs[1,len(Y):], KF2.Xs[2,len(Y):], 'go--', label='KF Model 2 Predictions')

    # Add axis labels and title
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    plt.title('Kalman Filter Predictions vs. Original Data')

    plt.draw()
    plt.pause(5)

def plot_original(Y, ax):
    ax.scatter(np.array(Y)[:,0], np.array(Y)[:,1], np.array(Y)[:,2], c='r', marker='x', label='Original Data')

    # Add axis labels and title
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    plt.title('Original Data')

    plt.draw()
    plt.pause(1)

def generate_sine_wave_coordinates(coordinates, num_new_points, z_value=0):
    if coordinates:
        start_x = coordinates[-1][0] + (2 * np.pi / len(coordinates))
    else:
        start_x = 0

    x = np.linspace(start_x, start_x + 2 * np.pi, num_new_points)
    y = np.sin(x)
    z = np.full_like(x, z_value)  # Create an array filled with the specified z_value
    new_coordinates = list(zip(x, y, z))

    # Add the new coordinates to the existing ones
    coordinates.extend(new_coordinates)
    
    x_init = coordinates[0][0]
    y_init = coordinates[0][1]
    z_init = coordinates[0][2]

    X0 = np.array([[x_init, y_init, z_init, 0, 0, 0]]).transpose() # Assuming initial velocity is 0

    return X0, coordinates
    
