from functools import partial
import os

from absl import app, flags, logging
import jax
# import wandb
import json
import numpy as np
import cv2

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from octo.libero.libero_utils import LiberoGymWrapper, normalize_gripper_action, invert_gripper_action

from octo.model.octo_model import OctoModel
from octo.utils.gym_wrappers import HistoryWrapper, NormalizeProprio, RHCWrapper
from octo.utils.train_callbacks import supply_rng

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "finetuned_path", None, "Path to finetuned Octo checkpoint directory."
)


def main(_):
    # setup wandb for logging
    # wandb.init(name="eval_aloha", project="octo")

    # load finetuned model
    # MAKE SURE TO INCLUDE VANILLA OCTO AS BASELINE
    logging.info("Loading finetuned model...")
    model = OctoModel.load_pretrained(FLAGS.finetuned_path, 60000)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite_name = "libero_90" # can also choose libero_spatial, libero_object, etc.
    task_suite = benchmark_dict[task_suite_name]()
    # task_id = 0


    task_id = [task.name for task in task_suite.tasks].index("KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet")
    task = task_suite.get_task(task_id)
    task_name = task.name
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    print(f"[info] retrieving task {task_id} from suite {task_suite_name}, the " + \
        f"language instruction is {task_description}, and the bddl file is {task_bddl_file}")

    # step over the environment
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": 256,
        "camera_widths": 256
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    env.reset()
    # init_states = task_suite.get_task_init_states(task_id) # for benchmarking purpose, we fix the a set of initial states
    init_state_id = 0

    env = LiberoGymWrapper(env, camera_height=256, camera_width=256, max_episode_len=520)

    # add wrappers for history and "receding horizon control", i.e. action chunking
    env = HistoryWrapper(env, horizon=1)
    env = RHCWrapper(env, exec_horizon=4)

    # the supply_rng wrapper supplies a new random key to sample_actions every time it's called
    policy_fn = supply_rng(
        partial(
            model.sample_actions,
            unnormalization_statistics=model.dataset_statistics['libero_90']["action"],#dataset_stats["action"],
        ),
    )

    # running rollouts
    for _ in range(10):
        obs, info = env.reset()

        # create task specification --> use model utility to create task dict with correct entries
        # language_instruction = env.get_task()["language_instruction"]
        language_instruction = ["put the wine bottle in the drawer and close it"] # put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket
        task = model.create_tasks(texts=language_instruction)

        # run rollout for 400 steps
        images = [obs["image_primary"][0]]
        episode_return = 0.0
        t = 0
        # libero_90 longest training demo has 373 steps
        # libero_10 longest training demo has 505 steps
        while t < 520 + 15:
             # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
            # and we need to wait for them to fall
            if t < 15:
                obs, reward, done, trunc, info = env.step([[0, 0, 0, 0, 0, 0, -1]]*4)
                t += 1
                continue

            # model returns actions of shape [batch, pred_horizon, action_dim] -- remove batch
            actions = policy_fn(jax.tree_map(lambda x: x[None], obs), task)
            action = np.array(actions[0])

            # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
            action = normalize_gripper_action(action, binarize=True)
            # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
            action = invert_gripper_action(action)

            # step env -- info contains full "chunk" of observations for logging
            # obs only contains observation for final step of chunk
            obs, reward, done, trunc, info = env.step(action)
            images.extend([o["image_primary"][0] for o in info["observations"]])
            episode_return += reward
            if done or trunc:
                break
            t += 1
        print(f"Episode return: {episode_return}")

        # log rollout video to wandb -- subsample temporally 2x for faster logging
        # wandb.log(
        #     {"rollout_video": wandb.Video(np.array(images).transpose(0, 3, 1, 2)[::2])}
        # )
        # log rollout video to folder -- subsample temporally 2x for faster logging
        video_path = f"/home/artur/Downloads/run_{_}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(video_path, fourcc, 10.0, (256, 256))
        for img in images:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            out.write(img_bgr)
        out.release()
        


if __name__ == "__main__":
    app.run(main)
