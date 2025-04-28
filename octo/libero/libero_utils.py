from typing import Any, Optional, Tuple, Dict

import gym
from gym import spaces

import numpy as np

import math
import cv2


def resize_image(img, resize_size):
    """
    Takes numpy array corresponding to a single image and returns resized image as numpy array.
    Uses OpenCV for image processing instead of TensorFlow.

    NOTE (Moo Jin): To make input images in distribution with respect to the inputs seen at training time, we follow
                    the same resizing scheme used in the Octo dataloader, which OpenVLA uses for training.
    """
    assert isinstance(resize_size, tuple)

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
    _, encoded = cv2.imencode(".jpg", img, encode_param)
    img = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    img_resized = cv2.resize(img, resize_size, interpolation=cv2.INTER_LANCZOS4)
    img_resized = np.clip(np.round(img_resized), 0, 255).astype(np.uint8)
    return img_resized


def get_libero_image(obs, resize_size):
    """Extracts image from observations and preprocesses it."""
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    img = obs["agentview_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    img = resize_image(img, resize_size)
    return img


def invert_gripper_action(action):
    """
    Flips the sign of the gripper action (last dimension of action vector).
    This is necessary for some environments where -1 = open, +1 = close, since
    the RLDS dataloader aligns gripper actions such that 0 = close, 1 = open.
    """
    action[..., -1] = action[..., -1] * -1.0
    return action


def normalize_gripper_action(action, binarize=True):
    """
    Changes gripper action (last dimension of action vector) from [0,1] to [-1,+1].
    Necessary for some environments (not Bridge) because the dataset wrapper standardizes gripper actions to [0,1].
    Note that unlike the other action dimensions, the gripper action is not normalized to [-1,+1] by default by
    the dataset wrapper.

    Normalization formula: y = 2 * (x - orig_low) / (orig_high - orig_low) - 1
    """
    # Just normalize the last action to [-1,+1].
    orig_low, orig_high = 0.0, 1.0
    action[..., -1] = 2 * (action[..., -1] - orig_low) / (orig_high - orig_low) - 1

    if binarize:
        # Binarize to -1 or +1.
        action[..., -1] = np.sign(action[..., -1])

    return action


def quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


class LiberoGymWrapper(gym.Env):
    """
    Wrapper for LIBERO environments that implements the standard gym interface
    and returns observations in the specified format:

    obs = {
        "image_primary": ...
    }

    Also implements env.get_task() function that returns:
    task = {
        "language_instruction": "some string"
        "goal": {
            "image_primary": ...
        }
    }
    """

    def __init__(
        self,
        env,
        camera_height: int = 128,
        camera_width: int = 128,
        max_episode_len: int = 300,
        seed: Optional[int] = None,
    ):
        """
        Initialize the LiberoGymWrapper.

        Args:
            bddl_file_name: Path to the BDDL file for the task
            camera_heights: Height of camera images
            camera_widths: Width of camera images
            max_episode_steps: Maximum number of steps per episode
            seed: Random seed
        """
        # Initialize the LIBERO environment
        self._env = env
        self._camera_height = camera_height
        self._camera_width = camera_width
        self._max_episode_len = max_episode_len

        if seed is not None:
            self._env.seed(seed)

        # Store parameters
        self.current_step = 0

        # Get observation and action spaces from the underlying env
        self._setup_spaces()

        # Store task information
        self._language_instruction = self._env.language_instruction

        # Initialize goal image (will be set during reset)
        self._goal_image = None

    def _setup_spaces(self):
        """Set up the observation and action spaces."""
        # Reset the environment to get example observation
        init_obs = self._env.reset()

        # Get primary camera image shape
        primary_image = get_libero_image(init_obs, (self._camera_height, self._camera_width))

        # proprio_data = np.concatenate([init_obs["robot0_eef_pos"], quat2axisangle(init_obs["robot0_eef_quat"]), init_obs["robot0_gripper_qpos"]], axis=0)

        # Define observation space
        self.observation_space = spaces.Dict(
            {
                "image_primary": spaces.Box(low=0, high=255, shape=primary_image.shape, dtype=np.uint8),
                # "proprio": spaces.Box(low=-np.inf, high=np.inf, shape=proprio_data.shape, dtype=np.float32),
            }
        )

        # Define action space based on the LIBERO environment's action space
        action_spec = self._env.env.action_spec
        self.action_space = spaces.Box(low=action_spec[0], high=action_spec[1], dtype=np.float32)

    def reset(self, **kwargs) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Reset the environment and return the initial observation and info.

        Returns:
            Tuple of (initial observation dictionary, info dictionary)
        """
        self.current_step = 0
        obs = self._env.reset(**kwargs)

        # Extract proprioceptive data
        # proprio_data = np.concatenate([obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]], axis=0)

        # Convert to the required format
        image = get_libero_image(obs, (self._camera_height, self._camera_width))
        gym_obs = {"image_primary": image}  # , "proprio": proprio_data}

        return gym_obs, {}

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        """
        Take a step in the environment.

        Args:
            action: Action to take

        Returns:
            Tuple of (observation, reward, done, info)
        """
        self.current_step += 1

        # Take step in the underlying environment
        obs, reward, done, info = self._env.step(action)

        terminated = done
        truncated = self.current_step >= self._max_episode_len

        proprio_data = np.concatenate(
            [obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]], axis=0
        )

        # Convert to the required format
        image = get_libero_image(obs, (self._camera_height, self._camera_width))
        gym_obs = {"image_primary": image}  # , "proprio": proprio_data}

        return gym_obs, reward, terminated, truncated, info

    def render(self, mode: str = "rgb_array", width: int = 256, height: int = 256) -> np.ndarray:
        """
        Render the environment.

        Args:
            mode: Rendering mode
            width: Width of the rendered image
            height: Height of the rendered image

        Returns:
            Rendered image
        """
        if mode == "rgb_array":
            return self._env.render(mode=mode, width=width, height=height)
        else:
            raise NotImplementedError(f"Rendering mode {mode} not supported")

    def get_task(self) -> Dict[str, Any]:
        """
        Get task information including language instruction.

        Returns:
            Task dictionary
        """
        task = {"language_instruction": [self._language_instruction]}
        return task

    def close(self):
        """Close the environment."""
        self._env.close()

    def seed(self, seed: Optional[int] = None):
        """Set random seed."""
        return self._env.seed(seed)

    def get_sim_state(self):
        """Get simulation state from the underlying environment."""
        return self._env.get_sim_state()

    def __getattr__(self, name):
        """Forward any other attributes to the underlying environment."""
        return getattr(self._env, name)
