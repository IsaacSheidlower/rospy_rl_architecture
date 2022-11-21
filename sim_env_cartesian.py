#!/usr/bin/env python3

from turtle import speed
import numpy as np
import rospy 
import time
from gen3_movement_utils import Arm
from custom_arm_reaching.msg import ObsMessage
from kortex_driver.srv import *
from kortex_driver.msg import *
from moveit_msgs.srv import GetStateValidity, GetStateValidityRequest, GetStateValidityResponse
from moveit_msgs.msg import RobotState
from sensor_msgs.msg import JointState
import os

z_min = .03
z_max = .18
x_min = .6
x_max = .8
y_min = -.3
y_max = .3
wrist_rotate_min = np.deg2rad(-25)
wrist_rotate_max = np.deg2rad(25)

start_pose = [-0.1487268942365656, 1.3957608630977656, 3.058740913129101, -2.1208424225465548, -0.1734628670734395, 1.9523837683194285,1.5977496475553705]

class SimDrawing():
    def __init__(self, max_action=.1, min_action=-.1, n_actions=1, action_duration=.2, reset_pose=None, episode_time=60, 
        stack_size=4, sparse_rewards=False, success_threshold=.16, home_arm=True, with_pixels=False, max_vel=.12, 
        duration_timeout=1, speed=.1, sim=True, use_goals_file=None):
        
        self.max_action = max_action
        self.min_action = min_action
        self.action_duration = action_duration
        self.n_actions = n_actions
        self.reset_pose = reset_pose
        self.episode_time = episode_time
        self.stack_size = stack_size
        self.sparse_rewards = sparse_rewards
        self.success_threshhold = success_threshold
        self.home_arm = home_arm
        self.with_pixels = with_pixels
        self.max_vel = max_vel
        self.duration_timeout = duration_timeout
        self.sim = sim
        self.arm = Arm()
        self.goal = [0.0,0.0]
        self.speed = speed
        self.arm.goto_cartesian_relative_sim([0, 0,.04,0,0,0], duration=self.duration_timeout, speed=self.speed)
        if use_goals_file is not None:
            self.goals = np.genfromtxt(use_goals_file, delimiter=',')
            self.use_goals = True
        else:
            self.use_goals = False
        self.goal_index = 0
        
        
    def get_obs(self):
        temp = np.array(rospy.wait_for_message("/rl_observation", ObsMessage).obs)
        temp = np.concatenate((temp, self.goal))
        return temp

    def reset(self):
        global goal
        time.sleep(.5)
        if self.use_goals:
            self.goal_index = 0
            self.goal = self.goals[self.goal_index]
            self.goal_index += 1
            print("GOAL" , self.goal)
        else:
            self.goal[0] = np.random.uniform(x_min, x_max)
            self.goal[1] = np.random.uniform(y_min, y_max)
            print("GOAL" , self.goal)
        self.arm.home_arm()
        self.arm.goto_joint_pose_sim(start_pose)
        rospy.sleep(2.5)
        os.system(f"gz marker -m 'action: ADD_MODIFY, type: SPHERE, id: 2, scale: {{x:0.1, y:0.1, z:.1}}, pose: {{position: {{x:{self.goal[0]} y:{self.goal[1]}, z:0.06701185554265976}}, orientation: {{x:0.0, y:0.0, z:0.0, w:1.0}}}}'")
        #os.system("gz marker -x")
        return self.get_obs()

    def step(self, action):
        global x_min, x_max, y_min, y_max, z_min, z_max
        #print(action)
        action = np.clip(np.array(action)*self.max_action, self.min_action, self.max_action)
        #action-=.5
        #action = action*2
        #action = action*self.max_vel
        #action = np.clip(np.array(action), self.min_action, self.max_action)
        #print(action)
        #assert np.all(np.array(action)<=self.max_action and np.array(action)>=self.min_action)
        #action = action*self.max_action*.3
        if self.sim:
            self.arm.goto_cartesian_relative_sim([action[0],action[1],0.0,0,0,0], duration=self.duration_timeout, speed=self.speed)
            rospy.sleep(self.duration_timeout+.05)
        else:
            #self.arm.goto_cartesian_pose([action[0],action[1],action[2],0.0,0.0,0.0], relative=True, radians=True, wait_for_end=False)
            self.arm.cartesian_velocity_command(.5*np.array([action[0],action[1],0.0,0.0,0.0,0.0]), duration=.1)
            #rospy.sleep(.2)
        #rospy.sleep(self.duration_timeout+.05)
        obs = self.get_obs()
        reward = 0
        done = False
        reward = np.abs((obs[0]-self.goal[0]) + (obs[1]-self.goal[1])) 
        #print(obs[0:2], self.goal)
        if reward <= self.success_threshhold:
            reward = -reward
            print("SUCCESS")
            if self.sparse_rewards:
                reward = 0
            reward = 20
            if self.use_goals:
                if self.goal_index >= len(self.goals):
                    done = True
                else:
                    self.goal = self.goals[self.goal_index]
                    self.goal_index += 1
            else:
                self.goal[0] = np.random.uniform(x_min, x_max)
                self.goal[1] = np.random.uniform(y_min, y_max)
            #os.system(f"gz marker -m 'action: ADD_MODIFY, type: SPHERE, id: 2, scale: {{x:0.1, y:0.1, z:.1}}, pose: {{position: {{x:{self.goal[0]} y:{self.goal[1]}, z:0.06701185554265976}}, orientation: {{x:0.0, y:0.0, z:0.0, w:1.0}}}}'")
        else:
            reward = -reward
            if self.sparse_rewards:
                reward = -1
            done = False
        
        if obs[0] < x_min-.1 or obs[0] > x_max+.1 or obs[1] < y_min-.1 or obs[1] > y_max+.1:
           print("eef out of bounds")
            # print(obs[0] < x_min, "xmin")
            # print(obs[0] > x_max, "xmax")
            # print(obs[1] < y_min, "ymin")
            # print(obs[1] > y_max, "ymax")
           reward = -100
           done = True
        
        if obs[2] < z_min or obs[2] > z_max:
           reward = -50
           print('bad z')
            #done = True
                
        #if obs[4] < wrist_rotate_min or obs[4] > wrist_rotate_max:
        #    print("wrist out of bounds")
        #    reward = -10
            #done = True
        
        # if obs[6] > self.max_vel or obs[7] > self.max_vel or obs[8] > self.max_vel:
        #     print("vel out of bounds")
        #     reward = -10
        #     #done = True
            
        rospy.wait_for_service("/my_gen3/check_state_validity")
        state_check = rospy.ServiceProxy("/my_gen3/check_state_validity", GetStateValidity)
        robot_state = RobotState()
        joint_states = rospy.wait_for_message("/my_gen3/joint_states", JointState)
        robot_state.joint_state = joint_states

        state_msg = GetStateValidityRequest()
        state_msg.robot_state = robot_state
        state_result = state_check(state_msg)
        #print(state_result)
        if not state_result.valid:
            print("Collision")
            reward = -20
            #done = True

        return obs, reward, done

if __name__ == '__main__':
    try:
        rospy.init_node("custom_reacher")
        SimDrawing()
    except rospy.ROSInterruptException:
        pass

        