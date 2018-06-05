from .scene_stadium import SinglePlayerStadiumScene
from .env_bases import MJCFBaseBulletEnv
import numpy as np
import pybullet
import gym
import time
from .robots import Walker2D, Crab2D


class WalkerBaseBulletEnv(MJCFBaseBulletEnv):

    def __init__(self, robot, render=False):
        print("WalkerBase::__init__ start")
        MJCFBaseBulletEnv.__init__(self, robot, render)

        self.camera_x = 0
        self.walk_target_x = 1e3  # kilometer away
        self.walk_target_y = 0
        self.stateId = -1

    def create_single_player_scene(self, bullet_client):
        self.stadium_scene = SinglePlayerStadiumScene(
            bullet_client, gravity=9.8, timestep=0.0165 / 4, frame_skip=4
        )
        return self.stadium_scene

    def _reset(self):
        if self.stateId >= 0:
            # print("restoreState self.stateId:",self.stateId)
            self._p.restoreState(self.stateId)

        r = MJCFBaseBulletEnv._reset(self)
        self._p.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0)

        self.parts, self.jdict, self.ordered_joints, self.robot_body = self.robot.addToScene(
            self._p, self.stadium_scene.ground_plane_mjcf
        )
        self.ground_ids = set(
            [
                (
                    self.parts[f].bodies[self.parts[f].bodyIndex],
                    self.parts[f].bodyPartIndex,
                )
                for f in self.foot_ground_object_names
            ]
        )
        self._p.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)
        if self.stateId < 0:
            self.stateId = self._p.saveState()
        # print("saving state self.stateId:",self.stateId)

        return r

    def move_robot(self, init_x, init_y, init_z):
        "Used by multiplayer stadium to move sideways, to another running lane."
        self.cpp_robot.query_position()
        pose = self.cpp_robot.root_part.pose()
        pose.move_xyz(
            init_x, init_y, init_z
        )  # Works because robot loads around (0,0,0), and some robots have z != 0 that is left intact
        self.cpp_robot.set_pose(pose)

    electricity_cost = (
        -2.0
    )  # cost for using motors -- this parameter should be carefully tuned against reward for making progress, other values less improtant
    stall_torque_cost = (
        -0.1
    )  # cost for running electric current through a motor even at zero rotational speed, small
    foot_collision_cost = (
        -1.0
    )  # touches another leg, or other objects, that cost makes robot avoid smashing feet into itself
    foot_ground_object_names = set(["floor"])  # to distinguish ground and other objects
    joints_at_limit_cost = -0.1  # discourage stuck joints

    def _step(self, a):
        if (
            not self.scene.multiplayer
        ):  # if multiplayer, action first applied to all robots, then global step() called, then _step() for all robots with the same actions
            self.robot.apply_action(a)
            self.scene.global_step()

        state = self.robot.calc_state()  # also calculates self.joints_at_limit

        alive = float(
            self.robot.alive_bonus(
                state[0] + self.robot.initial_z, self.robot.body_rpy[1]
            )
        )  # state[0] is body height above ground, body_rpy[1] is pitch
        done = alive < 0
        if not np.isfinite(state).all():
            print("~INF~", state)
            done = True

        potential_old = self.potential
        self.potential = self.robot.calc_potential()
        progress = float(self.potential - potential_old)

        feet_collision_cost = 0.0
        for i, f in enumerate(
            self.robot.feet
        ):  # TODO: Maybe calculating feet contacts could be done within the robot code
            contact_ids = set((x[2], x[4]) for x in f.contact_list())
            # print("CONTACT OF '%d' WITH %d" % (contact_ids, ",".join(contact_names)) )
            if self.ground_ids & contact_ids:
                # see Issue 63: https://github.com/openai/roboschool/issues/63
                # feet_collision_cost += self.foot_collision_cost
                self.robot.feet_contact[i] = 1.0
            else:
                self.robot.feet_contact[i] = 0.0

        electricity_cost = self.electricity_cost * float(
            np.abs(a * self.robot.joint_speeds).mean()
        )  # let's assume we have DC motor with controller, and reverse current braking
        electricity_cost += self.stall_torque_cost * float(np.square(a).mean())

        joints_at_limit_cost = float(
            self.joints_at_limit_cost * self.robot.joints_at_limit
        )
        debugmode = 0
        if debugmode:
            print("alive=")
            print(alive)
            print("progress")
            print(progress)
            print("electricity_cost")
            print(electricity_cost)
            print("joints_at_limit_cost")
            print(joints_at_limit_cost)
            print("feet_collision_cost")
            print(feet_collision_cost)

        self.rewards = [
            alive,
            progress,
            electricity_cost,
            joints_at_limit_cost,
            feet_collision_cost,
        ]
        if debugmode:
            print("rewards=")
            print(self.rewards)
            print("sum rewards")
            print(sum(self.rewards))
        self.HUD(state, a, done)
        self.reward += sum(self.rewards)

        return state, sum(self.rewards), bool(done), {}

    def camera_adjust(self, distance=10, yaw=10):
        x, y, z = self.robot.body_xyz
        self.camera_x = 0.98 * self.camera_x + (1 - 0.98) * x
        # self.camera.move_and_look_at(self.camera_x, y - 2.0, 1.4, x, y, 1.0)
        lookat = [x, y, z]
        self._p.resetDebugVisualizerCamera(distance, yaw, -20, lookat)


class Walker2DCustomEnv(WalkerBaseBulletEnv):

    def __init__(self):
        self.robot = Walker2D()
        WalkerBaseBulletEnv.__init__(self, self.robot)


class Crab2DCustomEnv(WalkerBaseBulletEnv):

    def __init__(self):
        self.robot = Crab2D()
        WalkerBaseBulletEnv.__init__(self, self.robot)

    def set_gravity(self, g):
        self._p.setGravity(0, 0, g)


class PDCrab2DCustomEnv(Crab2DCustomEnv):
    """
        Just `Crab2DCustomEnv` but driven with a PDcontroller
        The control step is actually `nb_pd_steps` times faster than the original
    """

    def __init__(self, nb_pd_steps=5):  # TODO: how to get nb_pd_steps from the code?
        super(PDCrab2DCustomEnv, self).__init__()
        self.nb_pd_steps = nb_pd_steps
        self.pd_controller = PDController(self)
        self.render_mode = None

    def _render(self, mode, *args, **kwargs):
        super(PDCrab2DCustomEnv, self)._render(mode=mode, *args, **kwargs)
        self.render_mode = mode

    @property
    def get_num_joints(self):
        return len(self.robot.ordered_joints)

    def get_joints_state(self):
        thetas_and_omegas = self.robot.calc_state()[8 : 8 + 2 * self.get_num_joints]
        return thetas_and_omegas[0::2], thetas_and_omegas[1::2]

    def _step(self, action):
        thetas, omegas = self.get_joints_state()

        done = False
        reward = 0
        velocities = np.zeros(3)
        sum_omegas = np.zeros(
            omegas.shape[0]
        )  # required to calculate the mean of the omegas

        for _ in range(self.nb_pd_steps):
            if not done:
                # drive the motors with PD angles
                torques = self.pd_controller.drive_torques(action, thetas, omegas)
                obs, r, done, _ = super()._step(torques)
            else:
                # stop driving motors after the episode ends
                # print('episode has ended')
                obs, r, _, _ = super()._step(
                    np.zeros(len(action))
                )

            # accumulate velocities to output the correct state
            thetas, omegas = self.get_joints_state()
            sum_omegas += omegas
            velocities += obs[3:6]
            reward += r

            # the motion looks too fast for human eye. just a hack to make it seem like a realtime simulation
            if self.render_mode == "human":
                time.sleep(1. / 60.)

        # take the mean of the velocities as the observed value
        obs[3:6] = velocities / self.nb_pd_steps
        obs[9 : 9 + 2 * self.get_num_joints : 2] = sum_omegas / self.nb_pd_steps

        return obs, reward / self.nb_pd_steps, done, {}


class NabiRosCustomEnv(PDCrab2DCustomEnv):
    """
        Just `PDCrab2DCustomEnv` with the feet fixed
        the feet use a PDController, so it's more like a spring/damper than a rigid body

        It's supposed to be something (kinda) like the NABI-Ros robot: https://www.youtube.com/watch?v=Y5UoQsHJskw

        TODO: increase the PD coefficients for the feet
    """

    def __init__(self, *args, **kwargs):
        super(NabiRosCustomEnv, self).__init__(*args, **kwargs)
        self.action_space = gym.spaces.Box(
            self.action_space.low[:-2], self.action_space.high[:-2]
        )

    def _step(self, action):
        return super(NabiRosCustomEnv, self)._step(
            np.concatenate([action[0:2], [0], action[2:4], [0]])
        )


class PDController:

    def __init__(self, env):
        self.action_dim = env.action_space.shape[0]
        self.high = env.action_space.high
        self.low = env.action_space.low
        frequency = 2
        self.k_p = (2 * np.pi * frequency) ** 2
        damping_ratio = 2
        self.k_d = 2 * damping_ratio * 2 * np.pi * frequency

    def drive_torques(self, target_thetas, thetas, omegas):
        # States and targets should be [theta, omega] * action_dim
        diff = target_thetas - thetas
        torques = self.k_p * diff - self.k_d * omegas
        return np.clip(
            torques / 40,
            self.low,
            self.high
        )
