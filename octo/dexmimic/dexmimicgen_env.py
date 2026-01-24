"""
DexMimicGen environment wrapper for Octo evaluation.

This module provides a Gym environment wrapper that adapts dexmimicgen environments
to match Octo's expected interface, following the same pattern as the ALOHA sim environment.
"""

import os
import sys
import gym
import gym.spaces
import numpy as np
from typing import List, Optional, Dict, Any

# Add dexmimicgen to path - user should set this appropriately
# Add dexmimicgen to path - adjust this path as needed
if "DEXMIMICGEN_PATH" in os.environ:
    dexmimicgen_path = os.environ["DEXMIMICGEN_PATH"]
else:
    print("Warning: Could not find dexmimicgen path. Please set DEXMIMICGEN_PATH environment variable. Using default")
    dexmimicgen_path = "/home/mila/a/artur.kuramshin/dexmimicgen" # Example path; change as needed

if dexmimicgen_path:
    sys.path.append(dexmimicgen_path) 
else:
    print("Warning: Could not find dexmimicgen path. Please set DEXMIMICGEN_PATH environment variable.")

try:
    import dexmimicgen
    from dexmimicgen.utils.pca_action_wrapper import PCAActionWrapper, create_pca_from_dataset
    import robosuite
    from robosuite import load_composite_controller_config
    DEXMIMICGEN_AVAILABLE = True
    PCA_AVAILABLE = True

except ImportError as e:
    print(f"DexMimicGen not available: {e}. Please install and set the correct path.")
    DEXMIMICGEN_AVAILABLE = False
    PCA_AVAILABLE = False


class DexMimicGenGymEnv(gym.Env):
    """
    Gym wrapper for dexmimicgen environments to match Octo's expected interface.

    This follows the same pattern as AlohaGymEnv but adapts dexmimicgen environments.
    """

    def __init__(
        self,
        env_name: str,
        robots: Optional[List[str]] = None,
        camera_names: Optional[List[str]] = None,
        primary_im_size: int = 256,
        wrist_im_size: int = 128,
        seed: int = 1234,
        pca_config: Optional[Dict[str, Any]] = None,
        **env_kwargs
    ):
        if not DEXMIMICGEN_AVAILABLE:
            raise RuntimeError("DexMimicGen is not available. Please install and set the correct path.")

        # Map environment names to default robots
        ENV_ROBOTS = {
            "SingleArmDrawerCleanup": ["PandaDexRH"],
            "TwoArmDrawerCleanup": ["PandaDexRH", "PandaDexLH"],
            "TwoArmBoxCleanup": ["PandaDexRH", "PandaDexLH"],
            "TwoArmThreading": ["Panda", "Panda"],
            "TwoArmThreePieceAssembly": ["Panda", "Panda"],
            "TwoArmTransport": ["Panda", "Panda"],
            "TwoArmLiftTray": ["PandaDexRH", "PandaDexLH"],
            "TwoArmCoffee": ["GR1FixedLowerBody"],
            "TwoArmPouring": ["GR1FixedLowerBody"],
            "TwoArmCanSortRandom": ["GR1ArmsOnly"],
        }

        if env_name not in ENV_ROBOTS:
            raise ValueError(f"Environment {env_name} not supported. Available: {list(ENV_ROBOTS.keys())}")

        # Set defaults
        robots = robots or ENV_ROBOTS[env_name]
        camera_names = camera_names or ["agentview", "robot0_eye_in_hand"]
        camera_heights = [primary_im_size, wrist_im_size]
        camera_widths = [primary_im_size, wrist_im_size]

        # Set up environment configuration
        self.env_name = env_name
        self.primary_im_size = primary_im_size
        self.wrist_im_size = wrist_im_size

        default_env_kwargs = {
            "env_name": env_name,
            "robots": robots,
            "controller_configs": load_composite_controller_config(robot=robots[0]),
            "has_renderer": False,
            "has_offscreen_renderer": True,
            "ignore_done": False,
            "use_camera_obs": True,
            "control_freq": 20,
            "camera_names": camera_names,
            "camera_heights": camera_heights,
            "camera_widths": camera_widths,
            "seed": seed,
        }

        # Override defaults with provided kwargs
        default_env_kwargs.update(env_kwargs)

        # Create the robosuite environment
        self._env = robosuite.make(**default_env_kwargs)

        # Set up action and observation spaces
        action_spec = self._env.action_spec
        self.action_space = gym.spaces.Box(low=action_spec[0], high=action_spec[1], dtype=np.float32)

        # Set up observation space following Octo's expected format
        self.observation_space = gym.spaces.Dict({
            "image_primary": gym.spaces.Box(
                low=0, high=255, shape=(self.primary_im_size, self.primary_im_size, 3), dtype=np.uint8
            ),
            "image_wrist": gym.spaces.Box(
                low=0, high=255, shape=(self.wrist_im_size, self.wrist_im_size, 3), dtype=np.uint8
            ),
            "proprio": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32),
        })

        self.camera_names = camera_names
        self._rng = np.random.default_rng(seed)

        # Store PCA configuration (will be applied by RolloutVisualizer)
        self.pca_config = pca_config

    def reset(self, **kwargs):
        obs = self._env.reset(**kwargs)
        octo_obs = self._convert_obs_to_octo_format(obs)
        self._episode_is_success = 0
        return octo_obs, {}

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        octo_obs = self._convert_obs_to_octo_format(obs)

        # Add episode success tracking
        if not hasattr(self, '_episode_is_success'):
            self._episode_is_success = 0

        # Check success if environment supports it
        if hasattr(self._env, '_check_success'):
            if self._env._check_success():
                self._episode_is_success = 1

        return octo_obs, reward, done, False, info

    def _convert_obs_to_octo_format(self, obs):
        """Convert robosuite observations to Octo's expected format."""
        octo_obs = {}

        # Primary image observation (agentview)
        if "agentview_image" in obs:
            image = obs["agentview_image"][::-1]
            if len(image.shape) == 3 and image.shape[-1] == 3:
                octo_obs["image_primary"] = image
            elif len(image.shape) == 3 and image.shape[0] == 3:
                octo_obs["image_primary"] = np.transpose(image, (1, 2, 0))

        # Wrist image observation (robot0_eye_in_hand)
        if "robot0_eye_in_hand_image" in obs:
            image = obs["robot0_eye_in_hand_image"][::-1]
            if len(image.shape) == 3 and image.shape[-1] == 3:
                octo_obs["image_wrist"] = image
            elif len(image.shape) == 3 and image.shape[0] == 3:
                octo_obs["image_wrist"] = np.transpose(image, (1, 2, 0))

        # Resize images to expected size if needed
        for key in ["image_primary", "image_wrist"]:
            if key in octo_obs:
                image = octo_obs[key]
                if key == "image_primary":
                    expected_size = (self.primary_im_size, self.primary_im_size)
                else:
                    expected_size = (self.wrist_im_size, self.wrist_im_size)
                if image.shape[0] != expected_size[0] or image.shape[1] != expected_size[1]:
                    # Simple resize - in practice you might want more sophisticated resizing
                    import cv2
                    octo_obs[key] = cv2.resize(image, expected_size)

        # Proprioceptive observations
        # Using the same format as in the CLI script
        proprio = np.asarray(np.concatenate((
            obs["robot0_joint_pos"],
            obs["robot0_gripper_qpos"][[0,2,4,6,8,11]]
        ), axis=-1), dtype=np.float32)
        octo_obs["proprio"] = proprio

        return octo_obs

    def get_instruction(self):
        """Return task specification with language instruction."""
        return self._env.get_task()["language_instruction"]

    def get_episode_metrics(self):
        """Return episode metrics for evaluation."""
        return {
            "success_rate": getattr(self, '_episode_is_success', 0),
        }

    def render(self):
        return self._env.render()

    def close(self):
        return self._env.close()

    def apply_pca_wrapper(self, pca_config: Optional[Dict[str, Any]] = None):
        """
        Apply PCA action wrapper to reduce action space dimensionality.

        Args:
            pca_config: Dictionary containing PCA configuration with keys:
                - dataset_path: Path to HDF5 dataset for PCA fitting
                - hand_indices: Tuple (start, end) for hand action indices (default: (6, 12))
                - n_components: Number of PCA components (default: 2)

        Returns:
            Wrapped environment with PCA action processing
        """
        if not PCA_AVAILABLE:
            print("Warning: PCA not available, skipping PCA wrapper application")
            return self

        config = pca_config or self.pca_config
        if config is None:
            return self

        dataset_path = config.get("dataset_path")
        hand_indices = config.get("hand_indices", (6, 12))
        n_components = config.get("n_components", 2)

        if not dataset_path:
            print("Warning: No dataset_path provided for PCA, skipping PCA wrapper")
            return self

        if not os.path.exists(dataset_path):
            print(f"Warning: PCA dataset path {dataset_path} not found, skipping PCA wrapper")
            return self

        try:
            # Create PCA model from dataset
            if PCA_AVAILABLE:
                pca_model, mean_action, std_action = create_pca_from_dataset(
                    dataset_path, hand_indices=hand_indices, n_components=n_components
                )

                # Apply PCA wrapper
                pca_wrapped_env = PCAActionWrapper(self, pca_model, mean_action, std_action, hand_indices)
                print(f"Applied PCA wrapper with {n_components} components for hand actions")
                return pca_wrapped_env
            else:
                print("Warning: PCA not available, skipping PCA wrapper")
                return self

        except Exception as e:
            print(f"Warning: Failed to apply PCA wrapper: {e}")
            return self


# Register gym environments following the same pattern as ALOHA
def _register_dexmimicgen_envs():
    """Register common dexmimicgen environments with gym."""

    # Single arm environments
    gym.register(
        "dexmimicgen-single-arm-drawer-cleanup-v0",
        entry_point=lambda: DexMimicGenGymEnv(
            "SingleArmDrawerCleanup",
            robots=["PandaDexRH"],
            camera_names=["agentview", "robot0_eye_in_hand"],
            camera_heights=[256, 128],
            camera_widths=[256, 128],
        ),
    )

    # Two arm environments
    gym.register(
        "dexmimicgen-two-arm-drawer-cleanup-v0",
        entry_point=lambda: DexMimicGenGymEnv(
            "TwoArmDrawerCleanup",
            robots=["PandaDexRH", "PandaDexLH"],
            camera_names=["agentview", "robot0_eye_in_hand"],
            camera_heights=[256, 128],
            camera_widths=[256, 128],
        ),
    )

    # Add more environments as needed...


if DEXMIMICGEN_AVAILABLE:
    _register_dexmimicgen_envs()
