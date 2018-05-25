import os

current_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
parent_dir = os.path.dirname(current_dir)
os.sys.path.append(parent_dir)

import gym
import numpy as np
import cust_envs
import time


def main():
    env = gym.make("Walker2DCustomEnv-v0")
    env.render(mode="human")

    env.reset()

    while 1:
        frame = 0
        score = 0
        restart_delay = 0
        obs = env.reset()

        while 1:
            time.sleep(1. / 60.)
            obs, r, done, _ = env.step(np.zeros(6))
            score += r
            frame += 1

            still_open = env.render("human")
            if still_open == False:
                return
            # if not done: continue
            # if restart_delay==0:
            #     print("score=%0.2f in %i frames" % (score, frame))
            #     restart_delay = 60*2  # 2 sec at 60 fps
            # else:
            #     restart_delay -= 1
            #     if restart_delay==0: break


if __name__ == "__main__":
    main()
