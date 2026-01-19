from ml_collections import ConfigDict
from ml_collections.config_dict import FieldReference, placeholder

from octo.utils.spec import ModuleSpec


def get_config(config_string="full,language_conditioned"):
    mode, task = config_string.split(",")
    assert task in ["image_conditioned", "language_conditioned", "multimodal"]
    assert mode in ["full", "head_only", "head_mlp_only"]

    if mode == "full":
        frozen_keys = None
    elif mode == "head_only":
        frozen_keys = ("octo_transformer.*",)
    elif mode == "head_mlp_only":
        frozen_keys = (
            "octo_transformer.*",
            "heads_*.map_head.probe",
            "heads_*.map_head.MultiHeadDotProductAttention_0.*",
        )
    else:
        raise ValueError("Invalid mode")

    max_steps = FieldReference(30000)
    window_size = FieldReference(default=1)

    workspace_augment_kwargs = dict(
        random_resized_crop=dict(scale=[0.8, 1.0], ratio=[0.9, 1.1]),
        random_brightness=[0.1],
        random_contrast=[0.9, 1.1],
        random_saturation=[0.9, 1.1],
        random_hue=[0.05],
        augment_order=[
            "random_resized_crop",
            "random_brightness",
            "random_contrast",
            "random_saturation",
            "random_hue",
        ],
    )
    wrist_augment_kwargs = dict(
        random_brightness=[0.1],
        random_contrast=[0.9, 1.1],
        random_saturation=[0.9, 1.1],
        random_hue=[0.05],
        augment_order=[
            "random_brightness",
            "random_contrast",
            "random_saturation",
            "random_hue",
        ],
    )

    if task == "image_conditioned":
        goal_relabeling_strategy = "uniform"
        keep_image_prob = 1.0
    elif task == "language_conditioned":
        goal_relabeling_strategy = None
        keep_image_prob = 0.0
    elif task == "multimodal":
        goal_relabeling_strategy = "uniform"
        keep_image_prob = 0.5
    else:
        raise ValueError("Invalid modality")

    config = dict(
        pretrained_path=placeholder(str),
        pretrained_step=placeholder(int),
        shuffle_buffer_size=75000,
        batch_size=256,
        num_steps=max_steps,
        log_interval=100,
        eval_interval=5000,
        eval_datasets=["bridge_original", "bridge_subtasks"],
        save_interval=5000,
        save_dir=placeholder(str),
        seed=42,
        prefetch_num_batches=0,
        wandb=dict(project="octo_finetune", group=placeholder(str), entity=placeholder(str)),
        
dataset_kwargs=dict(
                name="dex_mimic",
                data_dir="/project/datasets/", # "/network/projects/real-g-grp/tensorflow_datasets/"
                image_obs_keys={"primary": "image", "wrist": "wrist_image"},
                proprio_obs_key="joint_state",
                language_key="language_instruction",
                standardize_fn=ModuleSpec.create(
                    "octo.data.oxe.oxe_standardization_transforms:dex_mimic",
                ),
        ),
        traj_transform_kwargs=dict(
            window_size=window_size,
            action_horizon=4,
            goal_relabeling_strategy=goal_relabeling_strategy,
            task_augment_strategy="delete_task_conditioning",
            task_augment_kwargs=dict(
                keep_image_prob=keep_image_prob,
            ),
            # If the default data loading speed is too slow, try these:
            # num_parallel_calls=16,  # for less CPU-intensive ops
        ),
        modality=task,
        finetuning_mode=mode,
        window_size=window_size,
        optimizer=dict(
            learning_rate=dict(
                name="cosine",
                init_value=0.0,
                peak_value=3e-4,
                warmup_steps=2000,
                decay_steps=max_steps,
                end_value=0.0,
            ),
            weight_decay=0.01,
            clip_gradient=1.0,
            frozen_keys=frozen_keys,
            grad_accumulation_steps=None,  # if you are using grad accumulation, you need to adjust max_steps accordingly
        ),
        val_kwargs=dict(
            val_shuffle_buffer_size=1000,
            num_val_batches=16,
        ),
        viz_kwargs=dict(
            eval_batch_size=128,
            trajs_for_metrics=100,
            trajs_for_viz=8,
            samples_per_state=8,
        ),
        model=dict(
            repeat_task_tokens=True,
        )
    )

    frame_transform_kwargs = dict(
        resize_size={
            "primary": (256, 256),  # workspace (3rd person) camera is at 256x256
            "wrist": (128, 128),
        },
        image_augment_kwargs=dict(
            primary=workspace_augment_kwargs,
            wrist=wrist_augment_kwargs,
        ),
    )
    # If the default data loading speed is too slow, try these:
    # config["frame_transform_threads"] = 16  # for the most CPU-intensive ops (decoding, resizing, augmenting)

    config["frame_transform_kwargs"] = frame_transform_kwargs

    # Add rollout evaluation configuration for dexmimicgen environments
    config["rollout_kwargs"] = dict(
        dataset_name="action",  # Use action statistics for normalization
        trajs_for_rollouts=3,  # Number of rollouts per evaluation
        modes_to_evaluate=("text_conditioned",),  # Evaluate language-conditioned mode
        visualizer_kwargs_list=[
            dict(
                name="dexmimicgen_two_arm_drawer_cleanup",
                env_name="dexmimicgen-two-arm-drawer-cleanup-v0",
                history_length=1,
                exec_horizon=50,
                max_episode_length=400,
                env_kwargs=dict(
                    seed=42,  # Fixed seed for reproducible evaluation
                    pca_config=dict(  # Optional: PCA reduces 6D hand actions to 2D latent space
                        dataset_path="/home/artur/dexmimicgen/datasets/generated/two_arm_drawer_cleanup.hdf5",
                        hand_indices=(6, 12),  # Indices of hand actions in action vector
                        n_components=2,  # Number of PCA components
                    ),
                ),
                use_temp_ensembling=False,  # Use RHC instead of temporal ensembling
                vis_fps=10,
                video_subsample_rate=2,
            ),
        ],
    )

    return ConfigDict(config)
