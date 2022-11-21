#!/usr/bin/env python3

from glob import glob
import numpy as np
import rospy 
import pickle
from sac_torch import Agent
from custom_arm_reaching.msg import *
from custom_arm_reaching.srv import GetActionFromObs, GetActionFromObsResponse
from buffer import ReplayBuffer

# TODO: make agent parameters passable
global agent
agent = pickle.load(open("agents/sim_drawing_REALvel_standardRL_960.p", "rb" ))
# agent.memory = ReplayBuffer(100000, (8,), 2)
# agent = Agent(alpha=0.00314854, beta=0.00314854, input_dims=(6,), batch_size=256,
#             tau=.02, max_size=100000, layer1_size=400, layer2_size=300, n_actions=2, \
#             max_action=1, reward_scale=10, auto_entropy=True)
#def learn():
#    agent.learn

def store_transition(data):
    old_observation = np.array(data.old_observation)
    action = np.array(data.action)
    reward = np.array(data.reward)
    observation = np.array(data.observation)
    done = np.array(data.done)
    global agent
    agent.remember(old_observation, action, reward, observation, done)

def learn(data):
    update_params = data.update_params
    update_disc = data.update_disc
    global agent

    if update_params and update_disc:
        try:
            act_loss, disc1_loss, disc1_log_probs, \
                        disc1_crit = agent.learn(update_params=True, update_disc=True)
            # act_loss, disc1_loss = agent.learn(update_params=True)
            if not act_loss == 0 and not disc1_loss == 0:
                # act_loss = np.mean(act_loss.item())
                #disc1_loss =  np.mean(disc1_loss.item())
                loss_pub = rospy.Publisher("loss_info", LossInfo, queue_size=1, latch=True)
                #print("ENTROPY:", agent.entropy)

                losses = LossInfo()
                losses.act_loss = act_loss
                losses.disc_loss = disc1_loss
                losses.disc_log_probs = disc1_log_probs
                losses.disc_crit = disc1_crit
                loss_pub.publish(losses)
        except Exception as e:
            #print(e)
            pass
    elif update_params and not update_disc:
        #agent.learn(update_params=True, update_disc=False)
        agent.learn(update_params=True)
        #agent.learn(update_params=True, update_disc=False)
        #agent.learn(update_params=True, update_disc=False)
    elif update_disc and not update_params:
        #agent.learn(update_params=False, update_disc=True)
        agent.learn(update_params=True)
    else:
        #agent.learn(update_params=False, update_disc=False)
        try:
            # act_loss, disc1_loss, disc1_log_probs, \
            #             disc1_crit = agent.learn(update_params=True, update_disc=True)
            act_loss, disc1_loss = agent.learn(update_params=True)
            # act_loss = np.mean(act_loss.item())
            disc1_loss =  np.mean(disc1_loss.item())
            loss_pub = rospy.Publisher("loss_info", LossInfo, queue_size=1, latch=True)
            #print("ENTROPY:", agent.entropy)
            losses = LossInfo()
            losses.act_loss = act_loss
            losses.disc_loss = disc1_loss
            losses.disc_log_probs = 0
            losses.disc_crit = 0
            loss_pub.publish(losses)
        except Exception as e:
            #print(e)
            pass
        #agent.learn(update_params=False, update_disc=False)
        #agent.learn(update_params=False, update_disc=False)
        #agent.learn(update_params=False, update_disc=False)

def action_callback(observation):
    observation = observation.observation
    action = agent.choose_action(observation)
    return GetActionFromObsResponse(action)

def save_callback(data):
    save = data.save
    global agent
    if save:
        pickle.dump(agent, open(data.file_name, "wb" ) )
    else:
        return

def learning_listener():
    rospy.init_node('learner', anonymous=True)

    action_service = rospy.Service('get_action', GetActionFromObs, action_callback)
    #rospy.Service('save_agent', GetActionFromObs, action_callback)
    rospy.Subscriber("/rl_transition", RLTransitionPlusReward, store_transition)
    rospy.Subscriber("/save_agent", SaveAgentRequest, save_callback)
    rospy.Subscriber("/rl_learn", LearnRequest, learn)
    print("Ready to recieve actions")
    # learn()

    rospy.spin()

if __name__ == '__main__':
    learning_listener()
