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

    max_steps = FieldReference(200000) #FieldReference(50000)
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

    # FINETUNING_KWARGS = { 
    #     "name":  "libero_original_no_noops",
    #     "data_dir": "/home/mila/m/michael.przystupa/scratch/modified_libero_rlds/libero_10_no_noops/1.0.0",
    #     "image_obs_keys": {"primary": "image_0", "wrist": None},
    #     "proprio_obs_key": "proprio",
    #     "language_key": "language_instruction",
    #     "action_proprio_normalization_type": "normal",
    #     # We want to avoid normalizing the gripper
    #     "action_normalization_mask": [True, True, True, True, True, True, False],
    #     # standardize_fn is dynamically loaded from a file
    #     # for example: "experiments/kevin/custom_standardization_transforms.py:aloha_dataset_transform"
    #     "standardize_fn": ModuleSpec.create(
    #         "octo.data.oxe.oxe_standardization_transforms:libero_dataset_transform",
    #         #"octo.data.oxe.oxe_standardization_transforms:bridge_dataset_transform",
    #     ),
    #     # If the default data loading speed is too slow, try these:
    #     # "num_parallel_reads": 8,  # for reading from disk / GCS
    #     # "num_parallel_calls": 16,  # for initial dataset construction
    # }

    config = dict(
        pretrained_path="/home/mila/n/nearyc/scratch/octo/octo-base-1.5",
        pretrained_step=300000,
        shuffle_buffer_size=10000,
        num_steps=max_steps,
        log_interval=100,
        eval_interval=5000,
        eval_datasets=["liber_o10", "libero_spatial", "libero_goal", "libero_object"],
        save_interval=5000,
        save_dir="/home/mila/n/nearyc/scratch/octo/2025-06-21_octo_base_1p5_libero_finetune",
        seed=42,
        prefetch_num_batches=0,
        wandb=dict(project="octo_finetune", group=placeholder(str), entity=placeholder(str)),
        dataset_kwargs=dict(
            oxe_kwargs=dict(
                data_mix="libero_spatial_goal_object_10",
                data_dir="/home/mila/n/nearyc/scratch/datasets/modified_libero_rlds",
                load_camera_views=("primary",),
                load_depth=False,
                force_recompute_dataset_statistics=False,
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
            batch_size=256,
            shuffle_buffer_size=10000,
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
        },
        image_augment_kwargs=dict(
            primary=workspace_augment_kwargs,
            # wrist=wrist_augment_kwargs,
        ),
    )
    # If the default data loading speed is too slow, try these:
    # config["frame_transform_threads"] = 16  # for the most CPU-intensive ops (decoding, resizing, augmenting)

    config["dataset_kwargs"]["frame_transform_kwargs"] = frame_transform_kwargs
    return ConfigDict(config)
