from __future__ import print_function
import time
import numpy as np
import argparse

import pybullet as p
import time
import pybullet_data as pd

p.connect(p.GUI)
p.setAdditionalSearchPath(pd.getDataPath())
dt = 1. / 240.

p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
p.loadURDF("plane.urdf")
robot = p.loadURDF("quadruped/spirit40.urdf", [0, 0, .5], useFixedBase=False)
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
p.setGravity(0, 0, -9.8)


toes = [3, 7, 11, 15]
revolute_joints = [1, 2, 5, 6, 9, 10, 13, 14, 0, 4, 8, 12]

joint_positions = [0.11321902, 0.1833849, 0.11935401, 0.1723926, 0.13162422, 0.2013433,
                   0.09736967, 0.16855788, -0.08001328, -0.140342, 0.0006392, 0.11989117]

joint_positions = [0.74833727, 1.4664032, 0.76712584, 1.4589894, 0.74642015, 1.4782903,
                   0.7488482, 1.4464638, -0.01239824, -0.02952504, 0.02939725, 0.05764484]

for j in range(p.getNumJoints(robot)):
    print("j=", p.getJointInfo(robot, j))

for j in range(12):
    joint_index = revolute_joints[j]
    print("revolute_joint index=", joint_index)
    p.resetJointState(robot, joint_index, joint_positions[j])
    p.setJointMotorControl(robot, joint_index, p.POSITION_CONTROL, joint_positions[j])

count = 0
while p.isConnected():
    count += 1
    p.stepSimulation()
    time.sleep(dt)
print("sitting!")