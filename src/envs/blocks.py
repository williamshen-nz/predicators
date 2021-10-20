"""Blocks domain. This environment IS downward refinable and DOESN'T
require any backtracking (as long as all the blocks can fit comfortably
on the table, which is true here because the block size and number of blocks
are much less than the table dimensions). The simplicity of this environment
makes it a good testbed for predicate invention.
"""

from typing import List, Set, Sequence, Dict, Tuple, Optional
import numpy as np
from gym.spaces import Box
import pybullet as p
from predicators.src.envs.pybullet_utils import get_kinematic_chain, \
    inverse_kinematics, get_asset_path
from predicators.src.envs import BaseEnv
from predicators.src.structs import Type, Predicate, State, Task, \
    ParameterizedOption, Object, Action, GroundAtom, Image, Array
from predicators.src.settings import CFG
from predicators.src import utils


class BlocksEnv(BaseEnv):
    """Blocks domain.
    """
    # Parameters that aren't important enough to need to clog up settings.py
    table_height = 0.2
    x_lb = 1.3
    x_ub = 1.4
    y_lb = 0.15
    y_ub = 20.85
    held_tol = 0.5
    clear_tol = 0.5
    open_fingers = 0.8
    pick_tol = 0.08
    assert pick_tol < CFG.blocks_block_size
    lift_amt = 1.0

    def __init__(self) -> None:
        super().__init__()
        # Types
        self._block_type = Type(
            "block", ["pose_x", "pose_y", "pose_z", "held", "clear"])
        self._robot_type = Type("robot", ["fingers"])
        # Predicates
        self._On = Predicate(
            "On", [self._block_type, self._block_type], self._On_holds)
        self._OnTable = Predicate(
            "OnTable", [self._block_type], self._OnTable_holds)
        self._GripperOpen = Predicate(
            "GripperOpen", [self._robot_type], self._GripperOpen_holds)
        self._Holding = Predicate(
            "Holding", [self._block_type], self._Holding_holds)
        self._Clear = Predicate(
            "Clear", [self._block_type], self._Clear_holds)
        # Options
        self._Pick = ParameterizedOption(
            # variables: [object to pick]
            # params: [delta x, delta y, delta z]
            "Pick", types=[self._block_type],
            params_space=Box(-1, 1, (3,)),
            _policy=self._Pick_policy,
            _initiable=self._Pick_initiable,
            _terminal=self._Pick_terminal)
        self._Stack = ParameterizedOption(
            # variables: [object on which to stack currently-held-object]
            # params: [delta x, delta y, delta z]
            "Stack", types=[self._block_type],
            params_space=Box(-1, 1, (3,)),
            _policy=self._Stack_policy,
            _initiable=self._Stack_initiable,
            _terminal=self._Stack_terminal)
        self._PutOnTable = ParameterizedOption(
            # params: [x, y] (normalized coordinates on the table surface)
            "PutOnTable", types=[],
            params_space=Box(0, 1, (2,)),
            _policy=self._PutOnTable_policy,
            _initiable=self._PutOnTable_initiable,
            _terminal=self._PutOnTable_terminal)
        # Objects
        self._robot = Object("robby", self._robot_type)
        # Rendering
        if CFG.make_videos:
            self._obj_to_pybullet_obj = {}
            self._last_pb_state = {"gripper": np.array([0.9, 0.3, 0.3])}
            self._initialize_pybullet()

    def simulate(self, state: State, action: Action) -> State:
        assert self.action_space.contains(action.arr)
        _, _, z, fingers = action.arr
        # Infer which transition function to follow
        if fingers < 0.5:
            transition_fn = self._transition_pick
        elif z < self.table_height + CFG.blocks_block_size:
            transition_fn = self._transition_putontable
        else:
            transition_fn = self._transition_stack
        next_state = transition_fn(state, action)
        return next_state

    def _transition_pick(self, state: State, action: Action) -> State:
        next_state = state.copy()
        # Can only pick if fingers are open
        if state.get(self._robot, "fingers") < self.open_fingers:
            return next_state
        x, y, z, fingers = action.arr
        block = self._get_block_at_xyz(state, x, y, z)
        if block is None:  # no block at this pose
            return next_state
        # Can only pick if object is clear
        if state.get(block, "clear") < self.clear_tol:
            return next_state
        # Execute pick
        next_state.set(block, "pose_x", x)
        next_state.set(block, "pose_y", y)
        next_state.set(block, "pose_z", z+self.lift_amt)
        next_state.set(block, "held", 1.0)
        next_state.set(block, "clear", 0.0)
        next_state.set(self._robot, "fingers", fingers)
        # Update clear bit of block below, if there is one
        cur_x = state.get(block, "pose_x")
        cur_y = state.get(block, "pose_y")
        cur_z = state.get(block, "pose_z")
        poss_below_block = self._get_highest_block_below(
            state, cur_x, cur_y, cur_z)
        assert poss_below_block != block
        if poss_below_block is not None:
            next_state.set(poss_below_block, "clear", 1.0)
        return next_state

    def _transition_putontable(self, state: State, action: Action) -> State:
        next_state = state.copy()
        # Can only putontable if fingers are closed
        if state.get(self._robot, "fingers") >= self.open_fingers:
            return next_state
        block = self._get_held_block(state)
        assert block is not None
        x, y, z, fingers = action.arr
        # Check that table surface is clear at this pose
        poses = [[state.get(b, "pose_x"),
                  state.get(b, "pose_y"),
                  state.get(b, "pose_z")] for b in state
                 if b.type == self._block_type]
        existing_xys = {(float(p[0]), float(p[1])) for p in poses}
        if not self._table_xy_is_clear(x, y, existing_xys):
            return next_state
        # Execute putontable
        next_state.set(block, "pose_x", x)
        next_state.set(block, "pose_y", y)
        next_state.set(block, "pose_z", z)
        next_state.set(block, "held", 0.0)
        next_state.set(block, "clear", 1.0)
        next_state.set(self._robot, "fingers", fingers)
        return next_state

    def _transition_stack(self, state: State, action: Action) -> State:
        next_state = state.copy()
        # Can only stack if fingers are closed
        if state.get(self._robot, "fingers") >= self.open_fingers:
            return next_state
        # Check that both blocks exist
        block = self._get_held_block(state)
        assert block is not None
        x, y, z, fingers = action.arr
        other_block = self._get_highest_block_below(state, x, y, z)
        if other_block is None:  # no block to stack onto
            return next_state
        # Can't stack onto yourself!
        if block == other_block:
            return next_state
        # Need block we're stacking onto to be clear
        if state.get(other_block, "clear") < self.clear_tol:
            return next_state
        # Execute stack by snapping into place
        cur_x = state.get(other_block, "pose_x")
        cur_y = state.get(other_block, "pose_y")
        cur_z = state.get(other_block, "pose_z")
        next_state.set(block, "pose_x", cur_x)
        next_state.set(block, "pose_y", cur_y)
        next_state.set(block, "pose_z", cur_z+CFG.blocks_block_size)
        next_state.set(block, "held", 0.0)
        next_state.set(block, "clear", 1.0)
        next_state.set(other_block, "clear", 0.0)
        next_state.set(self._robot, "fingers", fingers)
        return next_state

    def get_train_tasks(self) -> List[Task]:
        return self._get_tasks(num_tasks=CFG.num_train_tasks,
                               possible_num_blocks=CFG.blocks_num_blocks_train,
                               rng=self._train_rng)

    def get_test_tasks(self) -> List[Task]:
        return self._get_tasks(num_tasks=CFG.num_test_tasks,
                               possible_num_blocks=CFG.blocks_num_blocks_test,
                               rng=self._test_rng)

    @property
    def predicates(self) -> Set[Predicate]:
        return {self._On, self._OnTable, self._GripperOpen, self._Holding,
                self._Clear}

    @property
    def types(self) -> Set[Type]:
        return {self._block_type, self._robot_type}

    @property
    def options(self) -> Set[ParameterizedOption]:
        return {self._Pick, self._Stack, self._PutOnTable}

    @property
    def action_space(self) -> Box:
        # dimensions: [x, y, z, fingers]
        lowers = np.array([self.x_lb, self.y_lb, 0.0, 0.0], dtype=np.float32)
        uppers = np.array([self.x_ub, self.y_ub, 10.0, 1.0], dtype=np.float32)
        return Box(lowers, uppers)

    def render(self, state: State, task: Task,
               action: Optional[Action] = None) -> List[Image]:
        assert CFG.make_videos
        print("RENDERING")
        blocks = [o for o in state if o.type == self._block_type]
        if not self._obj_to_pybullet_obj:
            self._rebuild_pybullet_objects(blocks)
        if action is None:
            # Nothing to render, since we're using simulate. Set up for
            # the next episode.
            self._rebuild_pybullet_objects(blocks)
            self._last_pb_state = {"gripper": np.array([0.9, 0.3, 0.3])}
            return []
        # Render in between state and next_state
        next_state = self.simulate(state, action)
        pbtraj = self._get_interpolated_pybullet_states(state, next_state)
        images = []
        for pb_state in pbtraj:
            self._update_pybullet_state(pb_state)
            images.append(self._get_pybullet_image())
        return images

    def _get_tasks(self, num_tasks: int, possible_num_blocks: List[int],
                   rng: np.random.Generator) -> List[Task]:
        tasks = []
        for _ in range(num_tasks):
            num_blocks = rng.choice(possible_num_blocks)
            piles = self._sample_initial_piles(num_blocks, rng)
            init_state = self._sample_state_from_piles(piles, rng)
            atoms = utils.abstract(init_state, self.predicates)
            while True:  # repeat until goal is not satisfied
                goal = self._sample_goal_from_piles(num_blocks, piles, rng)
                if not goal.issubset(atoms):
                    break
            tasks.append(Task(init_state, goal))
        return tasks

    def _sample_initial_piles(self, num_blocks: int, rng: np.random.Generator
                              ) -> List[List[Object]]:
        piles: List[List[Object]] = []
        for block_num in range(num_blocks):
            block = Object(f"block{block_num}", self._block_type)
            # If coin flip, start new pile
            if block_num == 0 or rng.uniform() < 0.2:
                piles.append([])
            # Add block to pile
            piles[-1].append(block)
        return piles

    def _sample_state_from_piles(self, piles: List[List[Object]],
                                 rng: np.random.Generator) -> State:
        data: Dict[Object, Array] = {}
        # Create objects
        block_to_pile_idx = {}
        for i, pile in enumerate(piles):
            for j, block in enumerate(pile):
                assert block not in block_to_pile_idx
                block_to_pile_idx[block] = (i, j)
        # Sample pile (x, y)s
        pile_to_xy: Dict[int, Tuple[float, float]] = {}
        for i in range(len(piles)):
            pile_to_xy[i] = self._sample_initial_pile_xy(
                rng, set(pile_to_xy.values()))
        # Create block states
        for block, pile_idx in block_to_pile_idx.items():
            pile_i, pile_j = pile_idx
            x, y = pile_to_xy[pile_i]
            z = self.table_height + CFG.blocks_block_size * (0.5 + pile_j)
            max_j = max(j for i, j in block_to_pile_idx.values() if i == pile_i)
            # [pose_x, pose_y, pose_z, held, clear]
            data[block] = np.array([x, y, z, 0.0, int(pile_j == max_j)*1.0])
        # [fingers]
        data[self._robot] = np.array([1.0])  # fingers start off open
        return State(data)

    def _sample_goal_from_piles(self, num_blocks: int,
                                piles: List[List[Object]],
                                rng: np.random.Generator) -> Set[GroundAtom]:
        # Sample goal pile that is different from initial
        while True:
            goal_piles = self._sample_initial_piles(num_blocks, rng)
            if goal_piles != piles:
                break
        # Create goal from piles
        goal_atoms = set()
        for pile in goal_piles:
            goal_atoms.add(GroundAtom(self._OnTable, [pile[0]]))
            if len(pile) == 1:
                continue
            for block1, block2 in zip(pile[1:], pile[:-1]):
                goal_atoms.add(GroundAtom(self._On, [block1, block2]))
        return goal_atoms

    def _sample_initial_pile_xy(self, rng: np.random.Generator,
                                existing_xys: Set[Tuple[float, float]]
                                ) -> Tuple[float, float]:
        while True:
            x = rng.uniform(self.x_lb, self.x_ub)
            y = rng.uniform(self.y_lb, self.y_ub)
            if self._table_xy_is_clear(x, y, existing_xys):
                return (x, y)

    @staticmethod
    def _table_xy_is_clear(x: float, y: float,
                           existing_xys: Set[Tuple[float, float]]) -> bool:
        if all(abs(x-other_x) > 2*CFG.blocks_block_size
               for other_x, _ in existing_xys):
            return True
        if all(abs(y-other_y) > 2*CFG.blocks_block_size
               for _, other_y in existing_xys):
            return True
        return False

    @staticmethod
    def _On_holds(state: State, objects: Sequence[Object]) -> bool:
        block1, block2 = objects
        cls = BlocksEnv
        if state.get(block1, "held") >= cls.held_tol or \
           state.get(block2, "held") >= cls.held_tol:
            return False
        x1 = state.get(block1, "pose_x")
        y1 = state.get(block1, "pose_y")
        z1 = state.get(block1, "pose_z")
        x2 = state.get(block2, "pose_x")
        y2 = state.get(block2, "pose_y")
        z2 = state.get(block2, "pose_z")
        return np.allclose([x1, y1, z1], [x2, y2, z2+CFG.blocks_block_size],
                           atol=cls.pick_tol)

    @staticmethod
    def _OnTable_holds(state: State, objects: Sequence[Object]) -> bool:
        block, = objects
        z = state.get(block, "pose_z")
        cls = BlocksEnv
        desired_z = cls.table_height + CFG.blocks_block_size * 0.5
        return (state.get(block, "held") < cls.held_tol) and \
            (desired_z-cls.pick_tol < z < desired_z+cls.pick_tol)

    @staticmethod
    def _GripperOpen_holds(state: State, objects: Sequence[Object]) -> bool:
        robot, = objects
        return state.get(robot, "fingers") >= BlocksEnv.open_fingers

    def _Holding_holds(self, state: State, objects: Sequence[Object]) -> bool:
        block, = objects
        return self._get_held_block(state) == block

    @staticmethod
    def _Clear_holds(state: State, objects: Sequence[Object]) -> bool:
        block, = objects
        return state.get(block, "clear") >= BlocksEnv.clear_tol

    def _Pick_policy(self, state: State, objects: Sequence[Object],
                     params: Array) -> Action:
        block, = objects
        block_pose = np.array([state.get(block, "pose_x"),
                               state.get(block, "pose_y"),
                               state.get(block, "pose_z")])
        arr = np.r_[block_pose+params, 0.0].astype(np.float32)
        arr = np.clip(arr, self.action_space.low, self.action_space.high)
        return Action(arr)

    @staticmethod
    def _Pick_initiable(state: State, objects: Sequence[Object],
                        params: Array) -> bool:
        del state, objects, params  # unused
        return True  # can be run from anywhere

    @staticmethod
    def _Pick_terminal(state: State, objects: Sequence[Object],
                       params: Array) -> bool:
        del state, objects, params  # unused
        return True  # always 1 timestep

    def _Stack_policy(self, state: State, objects: Sequence[Object],
                      params: Array) -> Action:
        block, = objects
        block_pose = np.array([state.get(block, "pose_x"),
                               state.get(block, "pose_y"),
                               state.get(block, "pose_z")])
        arr = np.r_[block_pose+params, 1.0].astype(np.float32)
        arr = np.clip(arr, self.action_space.low, self.action_space.high)
        return Action(arr)

    @staticmethod
    def _Stack_initiable(state: State, objects: Sequence[Object],
                         params: Array) -> bool:
        del state, objects, params  # unused
        return True  # can be run from anywhere

    @staticmethod
    def _Stack_terminal(state: State, objects: Sequence[Object],
                        params: Array) -> bool:
        del state, objects, params  # unused
        return True  # always 1 timestep

    def _PutOnTable_policy(self, state: State, objects: Sequence[Object],
                           params: Array) -> Action:
        del state, objects  # unused
        # Un-normalize parameters to actual table coordinates
        x_norm, y_norm = params
        x = self.x_lb + (self.x_ub - self.x_lb) * x_norm
        y = self.y_lb + (self.y_ub - self.y_lb) * y_norm
        z = self.table_height + 0.5*CFG.blocks_block_size
        arr = np.array([x, y, z, 1.0], dtype=np.float32)
        arr = np.clip(arr, self.action_space.low, self.action_space.high)
        return Action(arr)

    @staticmethod
    def _PutOnTable_initiable(state: State, objects: Sequence[Object],
                              params: Array) -> bool:
        del state, objects, params  # unused
        return True  # can be run from anywhere

    @staticmethod
    def _PutOnTable_terminal(state: State, objects: Sequence[Object],
                             params: Array) -> bool:
        del state, objects, params  # unused
        return True  # always 1 timestep

    def _get_held_block(self, state: State) -> Optional[Object]:
        for block in state:
            if block.type != self._block_type:
                continue
            if state.get(block, "held") >= self.held_tol:
                return block
        return None

    def _get_block_at_xyz(self, state: State, x: float, y: float,
                          z: float) -> Optional[Object]:
        close_blocks = []
        for block in state:
            if block.type != self._block_type:
                continue
            block_pose = np.array([state.get(block, "pose_x"),
                                   state.get(block, "pose_y"),
                                   state.get(block, "pose_z")])
            if np.allclose([x, y, z], block_pose, atol=self.pick_tol):
                dist = np.linalg.norm(np.array([x, y, z])-  # type: ignore
                                      block_pose)
                close_blocks.append((block, dist))
        if not close_blocks:
            return None
        return min(close_blocks, key=lambda x: x[1])[0]  # min distance

    def _get_highest_block_below(self, state: State, x: float, y: float,
                                 z: float) -> Optional[Object]:
        blocks_here = []
        for block in state:
            if block.type != self._block_type:
                continue
            block_pose = np.array([state.get(block, "pose_x"),
                                   state.get(block, "pose_y")])
            block_z = state.get(block, "pose_z")
            if np.allclose([x, y], block_pose, atol=self.pick_tol) and \
               block_z < z:
                blocks_here.append((block, block_z))
        if not blocks_here:
            return None
        return max(blocks_here, key=lambda x: x[1])[0]  # highest z

    def _initialize_pybullet(self):
        # Load things into environment.
        if not p.getConnectionInfo()["isConnected"]:
            self._physics_client_id = p.connect(p.GUI)
            self._reset_camera()
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0,
                                       physicsClientId=self._physics_client_id)
            p.resetSimulation(physicsClientId=self._physics_client_id)
            p.setAdditionalSearchPath("src/envs/assets/")
            p.loadURDF(get_asset_path("urdf/plane.urdf"), [0, 0, -1],
                       useFixedBase=True,
                       physicsClientId=self._physics_client_id)
            self._fetch_id = p.loadURDF(
                get_asset_path("urdf/robots/fetch.urdf"), useFixedBase=True,
                physicsClientId=self._physics_client_id)
            global PHYSICS_CLIENT_ID, FETCH_ID  # pylint:disable=global-variable-undefined
            PHYSICS_CLIENT_ID = self._physics_client_id
            FETCH_ID = self._fetch_id
        else:
            self._physics_client_id = PHYSICS_CLIENT_ID
            self._fetch_id = FETCH_ID
        base_position = [0.8, 0.7441, 0]
        base_orientation = [0., 0., 0., 1.]
        p.resetBasePositionAndOrientation(
            self._fetch_id, base_position, base_orientation,
            physicsClientId=self._physics_client_id)
        # Get joints info.
        joint_names = [p.getJointInfo(
            self._fetch_id, i,
            physicsClientId=self._physics_client_id)[1].decode("utf-8")
                       for i in range(p.getNumJoints(
                           self._fetch_id,
                           physicsClientId=self._physics_client_id))]
        self._ee_id = joint_names.index("gripper_axis")
        self._ee_orn_down = p.getQuaternionFromEuler((0, np.pi/2, -np.pi))
        self._ee_orn_side = p.getQuaternionFromEuler((0, 0, 0))
        self._arm_joints = get_kinematic_chain(
            self._fetch_id, self._ee_id,
            physics_client_id=self._physics_client_id)
        self._left_finger_id = joint_names.index("l_gripper_finger_joint")
        self._right_finger_id = joint_names.index("r_gripper_finger_joint")
        self._arm_joints.append(self._left_finger_id)
        self._arm_joints.append(self._right_finger_id)
        self._init_joint_values = inverse_kinematics(
            self._fetch_id, self._ee_id, [1., 0, 0.75], self._ee_orn_down,
            self._arm_joints, physics_client_id=self._physics_client_id)
        # Add table.
        table_urdf = get_asset_path("urdf/table.urdf")
        self._table_id = p.loadURDF(table_urdf, useFixedBase=True,
                                    physicsClientId=self._physics_client_id)
        p.resetBasePositionAndOrientation(
            self._table_id, (1.65, 0.5, 0.0), [0., 0., 0., 1.],
            physicsClientId=self._physics_client_id)

    def _get_pybullet_image(self):
        camera_distance, yaw, pitch, camera_target = self._get_camera_params()
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=camera_target,
            distance=camera_distance,
            yaw=yaw,
            pitch=pitch,
            roll=0,
            upAxisIndex=2,
            physicsClientId=self._physics_client_id)

        width = height = 1800

        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=float(width / height),
            nearVal=0.1, farVal=100.0,
            physicsClientId=self._physics_client_id)

        (_, _, px, _, _) = p.getCameraImage(
            width=width, height=height, viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            physicsClientId=self._physics_client_id)

        rgb_array = np.array(px).reshape((width, height, 4))
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    @staticmethod
    def _get_camera_params():
        camera_distance = 1.65
        yaw = 90
        pitch = -5
        camera_target = [1.05, 0.5, 0.42]
        return camera_distance, yaw, pitch, camera_target

    def _reset_camera(self):
        camera_distance, yaw, pitch, camera_target = self._get_camera_params()
        p.resetDebugVisualizerCamera(
            camera_distance, yaw, pitch,
            camera_target, physicsClientId=self._physics_client_id)

    def _update_pybullet_state(self, pb_state):
        grip = pb_state["gripper"]
        active_constraint = pb_state["constraint"]
        block_states = {k: v for k, v in pb_state.items()
                        if k not in ["gripper", "constraint"]}

        target_position = np.add(grip, [0.0, 0.0, 0.075])
        ee_orien_to_use = self._ee_orn_down
        hint_joint_values = [
            0.47979457172467466, -1.576409316226008,
            1.8756301813146756, 0.8320363798078769,
            1.3659745447630645, -0.22762065844250637,
            -0.32964011684942474, 0.034577873746798826,
            0.03507221623551996]
        # Trick to make IK work: reset to either
        # side grasp or top grasp general position
        for joint_idx, joint_val in zip(self._arm_joints,
                                        hint_joint_values):
            p.resetJointState(self._fetch_id, joint_idx, joint_val,
                              physicsClientId=self._physics_client_id)

        # Target gripper
        joint_values = inverse_kinematics(
            self._fetch_id, self._ee_id, target_position,
            ee_orien_to_use, self._arm_joints,
            physics_client_id=self._physics_client_id)
        for joint_idx, joint_val in zip(self._arm_joints, joint_values):
            p.resetJointState(self._fetch_id, joint_idx, joint_val,
                              physicsClientId=self._physics_client_id)

        # Close fingers if holding
        finger_val = 0.05
        for finger_idx in [self._left_finger_id, self._right_finger_id]:
            p.resetJointState(self._fetch_id, finger_idx, finger_val,
                              physicsClientId=self._physics_client_id)

        for obj, py_obj in self._obj_to_pybullet_obj.items():
            if active_constraint is not None and \
                obj == active_constraint[0]:
                pose, orn = self._apply_pick_constraint(active_constraint)
            else:
                pose = block_states[obj].copy()
                orn = [0, 0, 0, 1]
            p.resetBasePositionAndOrientation(
                py_obj, pose, orn,
                physicsClientId=self._physics_client_id)

    def _apply_pick_constraint(self, active_constraint):
        base_link = np.r_[p.getLinkState(
            self._fetch_id, self._ee_id,
            physicsClientId=self._physics_client_id)[:2]]
        _, transf = active_constraint
        obj_loc, orn = p.multiplyTransforms(
            base_link[:3], base_link[3:], transf[0], transf[1])
        return obj_loc, orn

    def _rebuild_pybullet_objects(self, blocks):
        # Remove any existing objects.
        for obj_id in self._obj_to_pybullet_obj.values():
            p.removeBody(obj_id, physicsClientId=self._physics_client_id)
        self._obj_to_pybullet_obj = {}
        # Add new blocks.
        colors = [
            (0.95, 0.05, 0.1, 1.),
            (0.05, 0.95, 0.1, 1.),
            (0.1, 0.05, 0.95, 1.),
            (0.4, 0.05, 0.6, 1.),
            (0.6, 0.4, 0.05, 1.),
            (0.05, 0.04, 0.6, 1.),
            (0.95, 0.95, 0.1, 1.),
            (0.95, 0.05, 0.95, 1.),
            (0.05, 0.95, 0.95, 1.),
        ]
        for i, block in enumerate(blocks):
            assert block.type == self._block_type
            color = colors[i%len(colors)]
            width = CFG.blocks_block_size
            length = CFG.blocks_block_size
            height = CFG.blocks_block_size
            mass, friction = 0.04, 1.2
            orn_x, orn_y, orn_z, orn_w = 0, 0, 0, 1
            half_extents = [width/2, length/2, height/2]
            collision_id = p.createCollisionShape(
                p.GEOM_BOX, halfExtents=half_extents,
                physicsClientId=self._physics_client_id)
            visual_id = p.createVisualShape(
                p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color,
                physicsClientId=self._physics_client_id)
            block_id = p.createMultiBody(
                baseMass=mass, baseCollisionShapeIndex=collision_id,
                baseVisualShapeIndex=visual_id, basePosition=[0, 0, 0],
                baseOrientation=[orn_x, orn_y, orn_z, orn_w],
                physicsClientId=self._physics_client_id)
            p.changeDynamics(block_id, -1, lateralFriction=friction,
                             physicsClientId=self._physics_client_id)
            self._obj_to_pybullet_obj[block] = block_id

    def _get_interpolated_pybullet_states(self, state, next_state):
        num_interp = 3#TODO:36

        manipulated_block = None
        free_blocks = []
        for obj in state:
            if obj.type != self._block_type:
                continue
            if state.get(obj, "held") > self.held_tol or \
               next_state.get(obj, "held") > self.held_tol:
                assert manipulated_block is None, "Multiple blocks held?!"
                manipulated_block = obj
            else:
                free_blocks.append(obj)

        # Interpolate block states
        block_to_traj = {}
        for block in free_blocks + [manipulated_block]:
            block_traj = np.array([[state.get(block, "pose_x"),
                                    state.get(block, "pose_y"),
                                    state.get(block, "pose_z")]
                                   for _ in range(num_interp)])
            block_traj[:, 1] /= 20  # NOTE: compensate for scaled-up y_ub
            z_pose = state.get(block, "pose_z")
            if z_pose > self.lift_amt:
                z_pose -= self.lift_amt
            block_traj[:, 2] = z_pose
            block_to_traj[block] = block_traj

        assert manipulated_block is not None, "No block manipulated"

        # Detect whether pick or place
        held_before = state.get(manipulated_block, "held") > self.held_tol
        held_after = next_state.get(manipulated_block, "held") > self.held_tol
        assert not (held_before and held_after)
        # Pick
        if held_after:
            constraint = None
        # Place:
        else:
            assert held_before
            tf = ((0.12, 0.0, 0.0), (0.7, 0.0, -0.7, 0.0))
            constraint = (manipulated_block, tf)
            block_to_traj[manipulated_block][-1] = np.array([
                next_state.get(manipulated_block, "pose_x"),
                next_state.get(manipulated_block, "pose_y"),
                next_state.get(manipulated_block, "pose_z")])
            block_to_traj[manipulated_block][-1][1] /= 20

        # Get gripper trajectory
        start_gripper_pose = self._last_pb_state['gripper']
        end_gripper_pose = np.array([
            next_state.get(manipulated_block, "pose_x"),
            next_state.get(manipulated_block, "pose_y"),
            next_state.get(manipulated_block, "pose_z")])
        end_gripper_pose[1] /= 20
        if end_gripper_pose[2] > self.lift_amt:
            end_gripper_pose[2] -= self.lift_amt
        if held_after:
            end_gripper_pose[2] -= 0.05

        # Use fixed up/down movements between waypoints
        wp_height = -np.inf
        for block, traj in block_to_traj.items():
            wp_height = max(wp_height, max(traj[:, 2])+0.05)
        waypoint1 = start_gripper_pose.copy()
        waypoint1[2] = wp_height
        waypoint2 = end_gripper_pose.copy()
        waypoint2[2] = wp_height
        waypoints = [
            start_gripper_pose,
            waypoint1,
            waypoint2,
            end_gripper_pose,
        ]
        assert num_interp % (len(waypoints)-1) == 0
        num_interp_per_wp = num_interp // (len(waypoints)-1)
        gripper_traj = []
        for wp1, wp2 in zip(waypoints[:-1], waypoints[1:]):
            gripper_traj.extend(np.linspace(wp1, wp2, num_interp_per_wp))
        assert len(gripper_traj) == num_interp

        # Uncomment to use linear interpolation instead
        # gripper_traj = np.linspace(
        #     self._last_pb_state['gripper'],
        #     end_gripper_pose,
        #     num_interp)

        # Create pybullet state trajectory
        final_traj = []
        for t in range(num_interp):
            s_t = {"gripper": gripper_traj[t],
                   "constraint": constraint if t < num_interp - 1 else None}
            for block, traj in block_to_traj.items():
                s_t[block] = traj[t]
            final_traj.append(s_t)

        self._last_pb_state = {"gripper": gripper_traj[-1]}

        return final_traj
