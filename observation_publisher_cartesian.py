#!/usr/bin/env python3

import numpy as np
import rospy
from custom_arm_reaching.msg import ObsMessage
from kortex_driver.srv import *
from kortex_driver.msg import *
from sensor_msgs.msg import JointState

current_observation = np.zeros(3)

def eef_pose(data):
    x_pose = data.base.tool_pose_x 
    y_pose = data.base.tool_pose_y 
    z_pose = data.base.tool_pose_z
    x_twist = np.deg2rad(data.base.tool_pose_theta_x)
    y_twist = np.deg2rad(data.base.tool_pose_theta_y)
    z_twist = np.deg2rad(data.base.tool_pose_theta_z)
    current_observation[0] = x_pose
    current_observation[1] = y_pose
    current_observation[2] = z_pose
    # current_observation[3] = x_twist
    # current_observation[4] = y_twist
    # current_observation[5] = z_twist


def joint_pose(data):
    joint_pose = data.position
    current_observation[6] = joint_pose[1]
    current_observation[7] = joint_pose[2]
    current_observation[8] = joint_pose[3]
    current_observation[9] = joint_pose[4]
    current_observation[10] = joint_pose[5]
    current_observation[11] = joint_pose[6]
    current_observation[12] = joint_pose[7]


def observation_publisher():
    pub = rospy.Publisher("rl_observation", ObsMessage, queue_size=1)
    rospy.Subscriber("/my_gen3/base_feedback", BaseCyclic_Feedback, callback=eef_pose)
    #rospy.Subscriber("/my_gen3/joint_states", JointState, callback=joint_pose)
    rospy.init_node("observation_pub", anonymous=True)
    rate = rospy.Rate(1000)

    while not rospy.is_shutdown():
        pub.publish(ObsMessage(current_observation.tolist()))
        rate.sleep()


if __name__ == '__main__':
    try:
        print("publishing observations")
        observation_publisher()
    except rospy.ROSInterruptException:
        pass