"""
This script demonstrates how to load and rollout a finetuned Octo model on dexmimicgen environments.

Usage:
    cd examples
    python3 04_eval_finetuned_on_dexmimicgen.py --finetuned_path=<path_to_finetuned_checkpoint> --env_name=SingleArmDrawerCleanup
"""
from functools import partial
import sys
import os

from absl import app, flags, logging
import gym
import jax
import numpy as np
import wandb

# Import the dexmimicgen environment wrapper
from envs.dexmimicgen_env import DexMimicGenGymEnv  # noqa

# Add dexmimicgen to path - adjust this path as needed
if "DEXMIMICGEN_PATH" in os.environ:
    dexmimicgen_path = os.environ["DEXMIMICGEN_PATH"]
else:
    print("Warning: Could not find dexmimicgen path. Please set DEXMIMICGEN_PATH environment variable. Using default")
    dexmimicgen_path = "/home/mila/a/artur.kuramshin/dexmimicgen" # Example path; change as needed

if dexmimicgen_path:
    sys.path.append(dexmimicgen_path) 

from octo.model.octo_model import OctoModel
from octo.utils.gym_wrappers import HistoryWrapper, NormalizeProprio, RHCWrapper
from octo.utils.train_callbacks import supply_rng

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "finetuned_path", None, "Path to finetuned Octo checkpoint directory."
)
flags.DEFINE_string(
    "env_name", "TwoArmDrawerCleanup", "Name of the dexmimicgen environment to evaluate on."
)
flags.DEFINE_integer(
    "num_episodes", 3, "Number of episodes to evaluate."
)
flags.DEFINE_integer(
    "max_steps", 400, "Maximum steps per episode."
)
flags.DEFINE_boolean(
    "render", False, "Whether to render the environment during evaluation."
)
flags.DEFINE_boolean(
    "save_video", False, "Whether to save videos of the rollouts."
)


def main(_):
    # setup wandb for logging
    wandb.init(name=f"eval_dexmimicgen_{FLAGS.env_name}", project="octo")

    # load finetuned model
    logging.info("Loading finetuned model...")
    model = OctoModel.load_pretrained(FLAGS.finetuned_path)

    # make gym environment
    ##################################################################################################################
    # environment needs to implement standard gym interface + return observations of the following form:
    #   obs = {
    #     "image_primary": ...
    #     "image_wrist": ...  (optional)
    #     "proprio": ...
    #   }
    # it should also implement an env.get_task() function that returns a task dict with goal and/or language instruct.
    #   task = {
    #     "language_instruction": "some string"
    #   }
    ##################################################################################################################
    logging.info(f"Creating {FLAGS.env_name} environment...")

    # Use the registered gym environment
    env_name_map = {
        "SingleArmDrawerCleanup": "dexmimicgen-single-arm-drawer-cleanup-v0",
        "TwoArmDrawerCleanup": "dexmimicgen-two-arm-drawer-cleanup-v0",
    }

    if FLAGS.env_name in env_name_map:
        env = gym.make(env_name_map[FLAGS.env_name])
    else:
        # Fallback to direct instantiation for other environments
        env = DexMimicGenGymEnv(FLAGS.env_name)

    # wrap env to normalize proprio
    env = NormalizeProprio(env, model.dataset_statistics)

    # add wrappers for history and "receding horizon control", i.e. action chunking
    env = HistoryWrapper(env, horizon=1)
    env = RHCWrapper(env, exec_horizon=50)

    # the supply_rng wrapper supplies a new random key to sample_actions every time it's called
    policy_fn = supply_rng(
        partial(
            model.sample_actions,
            unnormalization_statistics=model.dataset_statistics["action"],
        ),
    )

    # running rollouts
    episode_returns = []
    successes = []
    opened_drawers = []

    for episode_idx in range(FLAGS.num_episodes):
        logging.info(f"Running episode {episode_idx + 1}/{FLAGS.num_episodes}")
        obs, info = env.reset()

        # create task specification --> use model utility to create task dict with correct entries
        task_spec = env.get_task()
        language_instruction = task_spec.get("language_instruction", f"Complete the {FLAGS.env_name} task")
        task = model.create_tasks(texts=[language_instruction])

        # run rollout for max_steps
        images = [obs["image_primary"][0]]
        episode_return = 0.0
        success = False

        step_count = 0
        while step_count < FLAGS.max_steps:
            # model returns actions of shape [batch, pred_horizon, action_dim] -- remove batch
            actions = policy_fn(jax.tree_map(lambda x: x[None], obs), task)
            actions = actions[0]

            # step env -- info contains full "chunk" of observations for logging
            # obs only contains observation for final step of chunk
            obs, reward, done, trunc, info = env.step(actions)
            images.extend([o["image_primary"][0] for o in info["observations"]])
            episode_return += reward
            step_count += len(info["observations"])

            # Check success
            if hasattr(env.env, '_check_success'):
                success = env.env._check_success()

            if done or trunc:
                break

        episode_returns.append(episode_return)
        successes.append(success)

        # Check if drawer was opened (for drawer environments)
        opened = False
        if hasattr(env.env, 'drawer_opened'):
            opened = env.env.drawer_opened
        opened_drawers.append(opened)

        print(f"Episode {episode_idx + 1}: Return={episode_return:.3f}, Success={success}, Opened={opened}")

        # log rollout video to wandb -- subsample temporally 2x for faster logging
        if FLAGS.save_video:
            wandb.log({
                f"rollout_video_episode_{episode_idx + 1}": wandb.Video(
                    np.array(images).transpose(0, 3, 1, 2)[::2]
                )
            })

    # Log summary metrics
    success_rate = np.mean(successes)
    mean_return = np.mean(episode_returns)
    std_return = np.std(episode_returns)

    wandb.log({
        "eval/success_rate": success_rate,
        "eval/mean_return": mean_return,
        "eval/std_return": std_return,
        "eval/num_episodes": FLAGS.num_episodes,
        "eval/env_name": FLAGS.env_name,
    })

    print(f"\nEvaluation Results:")
    print(f"Environment: {FLAGS.env_name}")
    print(f"Success Rate: {success_rate:.1%}")
    print(f"Mean Return: {mean_return:.3f} ± {std_return:.3f}")
    print(f"Individual Episodes: {episode_returns}")


if __name__ == "__main__":
    app.run(main)
