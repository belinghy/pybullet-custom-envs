from .robot_bases import XmlBasedRobot, MJCFBasedRobot, URDFBasedRobot, BodyPart
import numpy as np
import pybullet as p
import os
import pybullet_data


class WalkerBase(MJCFBasedRobot):

    def __init__(self, fn, robot_name, action_dim, obs_dim, power):
        MJCFBasedRobot.__init__(self, fn, robot_name, action_dim, obs_dim)
        self.power = power
        self.camera_x = 0
        self.start_pos_x, self.start_pos_y, self.start_pos_z = 0, 0, 0
        self.walk_target_x = 1e3  # kilometer away
        self.walk_target_y = 0
        self.body_xyz = [0, 0, 0]

    def robot_specific_reset(self, bullet_client):
        self._p = bullet_client
        for j in self.ordered_joints:
            j.reset_current_position(self.np_random.uniform(low=-0.1, high=0.1), 0)

        self.feet = [self.parts[f] for f in self.foot_list]
        self.feet_contact = np.array([0.0 for f in self.foot_list], dtype=np.float32)
        self.scene.actor_introduce(self)
        self.initial_z = None

    def apply_action(self, a):
        assert np.isfinite(a).all()
        for n, j in enumerate(self.ordered_joints):
            j.set_motor_torque(self.power * j.power_coef * float(np.clip(a[n], -1, +1)))

    def calc_state(self):
        j = np.array(
            [j.current_relative_position() for j in self.ordered_joints],
            dtype=np.float32,
        ).flatten()
        # even elements [0::2] position, scaled to -1..+1 between limits
        # odd elements  [1::2] angular speed, scaled to show -1..+1
        self.joint_speeds = j[1::2]
        self.joints_at_limit = np.count_nonzero(np.abs(j[0::2]) > 0.99)

        body_pose = self.robot_body.pose()
        parts_xyz = np.array([p.pose().xyz() for p in self.parts.values()]).flatten()
        self.body_xyz = (
            parts_xyz[0::3].mean(),
            parts_xyz[1::3].mean(),
            body_pose.xyz()[2],
        )  # torso z is more informative than mean z
        self.body_rpy = body_pose.rpy()
        z = self.body_xyz[2]
        if self.initial_z == None:
            self.initial_z = z
        r, p, yaw = self.body_rpy
        self.walk_target_theta = np.arctan2(
            self.walk_target_y - self.body_xyz[1], self.walk_target_x - self.body_xyz[0]
        )
        self.walk_target_dist = np.linalg.norm(
            [
                self.walk_target_y - self.body_xyz[1],
                self.walk_target_x - self.body_xyz[0],
            ]
        )
        angle_to_target = self.walk_target_theta - yaw

        rot_speed = np.array(
            [
                [np.cos(-yaw), -np.sin(-yaw), 0],
                [np.sin(-yaw), np.cos(-yaw), 0],
                [0, 0, 1],
            ]
        )
        vx, vy, vz = np.dot(
            rot_speed, self.robot_body.speed()
        )  # rotate speed back to body point of view

        more = np.array(
            [
                z - self.initial_z,
                np.sin(angle_to_target),
                np.cos(angle_to_target),
                0.3 * vx,
                0.3 * vy,
                0.3 * vz,
                # 0.3 is just scaling typical speed into -1..+1, no physical sense here
                r,
                p,
            ],
            dtype=np.float32,
        )
        return np.clip(np.concatenate([more] + [j] + [self.feet_contact]), -5, +5)

    def calc_potential(self):
        # progress in potential field is speed*dt, typical speed is about 2-3 meter per second, this potential will change 2-3 per frame (not per second),
        # all rewards have rew/frame units and close to 1.0
        debugmode = 0
        if debugmode:
            print("calc_potential: self.walk_target_dist")
            print(self.walk_target_dist)
            print("self.scene.dt")
            print(self.scene.dt)
            print("self.scene.frame_skip")
            print(self.scene.frame_skip)
            print("self.scene.timestep")
            print(self.scene.timestep)
        return -self.walk_target_dist / self.scene.dt


class Walker2D(WalkerBase):
    foot_list = ["foot", "foot_left"]

    def __init__(self):
        WalkerBase.__init__(
            self, "walker2d.xml", "pelvis", action_dim=6, obs_dim=22, power=0.40
        )

    def alive_bonus(self, z, pitch):
        return +1 if z > 0.8 and abs(pitch) < 1.0 else -1

    def robot_specific_reset(self, bullet_client):
        WalkerBase.robot_specific_reset(self, bullet_client)
        for n in ["foot_joint", "foot_left_joint"]:
            self.jdict[n].power_coef = 30.0


class Crab2D(WalkerBase):
    foot_list = ["foot", "foot_left"]

    def __init__(self, obs_dim=22):
        WalkerBase.__init__(
            self, "crab2d.xml", "pelvis", action_dim=6, obs_dim=obs_dim, power=0.40
        )

    def alive_bonus(self, z, pitch):
        return +1 if z > 0.2 and abs(pitch) < 1.0 else -1

    def robot_specific_reset(self, bullet_client):
        WalkerBase.robot_specific_reset(self, bullet_client)
        for n in ["foot_joint", "foot_left_joint"]:
            self.jdict[n].power_coef = 30.0

    def is_balanced(self):
        """ Check whether robot is balanced using centre of mass and feet position.
            Second return value is the foot that should be moved to maintain balance. """

        all_contact = (self.feet_contact == np.ones(self.feet_contact.shape)).all()
        none_contact = (self.feet_contact == np.zeros(self.feet_contact.shape)).all()
        if all_contact or none_contact:
            # If _both_ or _neither_ foot are in contact, check if centre of mass lies in bounding box
            left_x, right_x = float("inf"), -float("inf")
            for i, in_contact in enumerate(self.feet_contact):
                foot = self.feet[i]
                if in_contact:
                    # First foot is "foot_left_geom", second is "foot_geom"
                    contacts = foot.contact_list()  # Non-empty if foot touching ground
                    contact_info = contacts[0]  # Ground
                    # Position 6 contains (x,y,z) of contact in world coordinates
                    x, _, _ = contact_info[6]
                    left_x = x if x <= left_x else left_x
                    right_x = x if x >= right_x else right_x
                else:
                    x, _, _ = foot.pose().xyz()
                    left_x = x if x <= left_x else left_x
                    right_x = x if x >= right_x else right_x

            # Centre of mass
            x, _, _ = self.body_xyz
            # mask return 1 for foot not in contact
            mask = np.array([0. if x <= right_x else 1., 0. if left_x <= x else 1.])
            return left_x <= x <= right_x, mask
        else:
            # If one of the feet is not in contact, we want body to not sway past the contacting foot
            com_vx, _, _ = self.robot_body.speed()
            com_x, _, _ = self.body_xyz

            # mask return 1 for foot not in contact
            mask = np.ones(self.feet_contact.shape) - self.feet_contact

            if self.feet_contact[0] == 1:
                # Right foot in contact
                foot = self.feet[0]
            else:
                # Left foot in contact
                foot = self.feet[1]

            # Get contact info
            contacts = foot.contact_list()
            contact_info = contacts[0]
            # Position 6 contains (x,y,z) of contact in world coordinates
            foot_x, _, _ = contact_info[6]
            foot_vx, _, _ = foot.speed()

            if self.feet_contact[0] == 1:
                # Right foot in contact
                # 1. CoM on the left side of right foot
                # 2. CoM on the right side, but is moving to towards balance
                delta_x = com_x - foot_x
                delta_nx = (com_x + com_vx) - (foot_x + foot_vx)
                return delta_x <= 0 or (delta_x > 0 and delta_nx <= delta_x), mask
            else:
                # Left foot in contact
                delta_x = com_x - foot_x
                delta_nx = (com_x + com_vx) - (foot_x + foot_vx)
                return delta_x >= 0 or (delta_x < 0 and delta_nx >= delta_x), mask
