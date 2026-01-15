"""Defines dataset mixtures and weights for the Open X-Embodiment Datasets."""


BRIDGE_MIX = [
    ("bridge_dataset", 1.0),
]

RT_X_MIX = [
    ("fractal20220817_data", 0.54087122203),
    ("kuka", 0.8341046294),
    ("bridge_dataset", 1.0),
    ("taco_play", 2.0),
    ("jaco_play", 2.0),
    ("berkeley_cable_routing", 3.0),
    ("roboturk", 1.0),
    ("nyu_door_opening_surprising_effectiveness", 5.0),
    ("viola", 2.0),
    ("berkeley_autolab_ur5", 1.0),
    ("toto", 1.0),
]


OXE_FRANKA_MIX = [
    ("taco_play", 1.0),
    ("berkeley_cable_routing", 1.0),
    ("viola", 1.0),
    ("toto", 1.0),
    ("stanford_hydra_dataset_converted_externally_to_rlds", 1.0),
    ("austin_buds_dataset_converted_externally_to_rlds", 3.0),
    ("nyu_franka_play_dataset_converted_externally_to_rlds", 3.0),
    ("maniskill_dataset_converted_externally_to_rlds", 0.1),
    ("furniture_bench_dataset_converted_externally_to_rlds", 0.1),
    ("cmu_franka_exploration_dataset_converted_externally_to_rlds", 5.0),
    ("austin_sailor_dataset_converted_externally_to_rlds", 1.0),
    ("austin_sirius_dataset_converted_externally_to_rlds", 1.0),
    ("berkeley_rpt_converted_externally_to_rlds", 1.0),
    ("kaist_nonprehensile_converted_externally_to_rlds", 3.0),
    ("stanford_robocook_converted_externally_to_rlds", 1.0),
    ("iamlab_cmu_pickup_insert_converted_externally_to_rlds", 1.0),
    ("utaustin_mutex", 1.0),
    # ("cmu_playing_with_food", 1.0),
    ("cmu_play_fusion", 1.0),
]


OXE_MAGIC_SOUP = [
    ("fractal20220817_data", 0.54087122203),
    ("kuka", 0.8341046294),
    ("bridge_dataset", 1.0),
    ("taco_play", 2.0),
    ("jaco_play", 1.0),
    ("berkeley_cable_routing", 1.0),
    ("roboturk", 2.0),
    ("nyu_door_opening_surprising_effectiveness", 1.0),
    ("viola", 2.0),
    ("berkeley_autolab_ur5", 2.0),
    ("toto", 1.0),
    ("language_table", 0.1),
    ("stanford_hydra_dataset_converted_externally_to_rlds", 2.0),
    ("austin_buds_dataset_converted_externally_to_rlds", 1.0),
    ("nyu_franka_play_dataset_converted_externally_to_rlds", 3.0),
    ("furniture_bench_dataset_converted_externally_to_rlds", 0.1),
    ("ucsd_kitchen_dataset_converted_externally_to_rlds", 2.0),
    ("austin_sailor_dataset_converted_externally_to_rlds", 1.0),
    ("austin_sirius_dataset_converted_externally_to_rlds", 1.0),
    ("bc_z", 0.2),
    ("dlr_edan_shared_control_converted_externally_to_rlds", 1.0),
    ("iamlab_cmu_pickup_insert_converted_externally_to_rlds", 1.0),
    # ("uiuc_d3field", 1.0),  --> somehow raw data is broken
    ("utaustin_mutex", 1.0),
    ("berkeley_fanuc_manipulation", 2.0),
    ("cmu_stretch", 1.0),
]


OXE_FLEX_ACT_SOUP = [
    ("fractal20220817_data", 0.54087122203),
    ("kuka", 0.8341046294),
    ("bridge_dataset", 1.0),
    ("taco_play", 2.0),
    ("jaco_play", 1.0),
    ("berkeley_cable_routing", 1.0),
    ("roboturk", 2.0),
    ("nyu_door_opening_surprising_effectiveness", 1.0),
    ("viola", 2.0),
    ("berkeley_autolab_ur5", 2.0),
    ("toto", 1.0),
    ("language_table", 0.1),
    ("stanford_hydra_dataset_converted_externally_to_rlds", 2.0),
    ("austin_buds_dataset_converted_externally_to_rlds", 1.0),
    ("nyu_franka_play_dataset_converted_externally_to_rlds", 3.0),
    ("furniture_bench_dataset_converted_externally_to_rlds", 0.1),
    ("ucsd_kitchen_dataset_converted_externally_to_rlds", 2.0),
    ("austin_sailor_dataset_converted_externally_to_rlds", 1.0),
    ("austin_sirius_dataset_converted_externally_to_rlds", 1.0),
    ("bc_z", 0.2),
    ("berkeley_mvp_converted_externally_to_rlds", 1.0),
    # ("berkeley_rpt_converted_externally_to_rlds", 1.0),
    ("dlr_edan_shared_control_converted_externally_to_rlds", 1.0),
    ("iamlab_cmu_pickup_insert_converted_externally_to_rlds", 1.0),
    # ("uiuc_d3field", 1.0),  --> somehow raw data is broken
    ("utaustin_mutex", 1.0),
    ("berkeley_fanuc_manipulation", 2.0),
    ("cmu_stretch", 1.0),
    ("gnm_dataset", 1.0),
    ("aloha_static_dataset", 3.0),
    # ("aloha_dagger_dataset", 1.0),
    ("aloha_mobile_dataset", 2.0),
    # ("fmb_dataset", 1.0),
    ("dobbe", 1.0),
    ("roboset", 0.5),
    ("rh20t", 0.5),
]


OXE_FULL_MIX = [
    ("fractal20220817_data", 1.0),
    ("kuka", 1.0),
    ("bridge_dataset", 1),
    ("taco_play", 1.0),
    ("jaco_play", 1.0),
    ("berkeley_cable_routing", 1.0),
    ("roboturk", 1.0),
    ("nyu_door_opening_surprising_effectiveness", 1.0),
    ("viola", 1.0),
    ("berkeley_autolab_ur5", 1.0),
    ("toto", 1.0),
    ("language_table", 1.0),
    ("columbia_cairlab_pusht_real", 1.0),
    ("stanford_kuka_multimodal_dataset_converted_externally_to_rlds", 1.0),
    ("nyu_rot_dataset_converted_externally_to_rlds", 1.0),
    ("stanford_hydra_dataset_converted_externally_to_rlds", 1.0),
    ("austin_buds_dataset_converted_externally_to_rlds", 1.0),
    ("nyu_franka_play_dataset_converted_externally_to_rlds", 1.0),
    ("maniskill_dataset_converted_externally_to_rlds", 1.0),
    ("furniture_bench_dataset_converted_externally_to_rlds", 1.0),
    ("cmu_franka_exploration_dataset_converted_externally_to_rlds", 1.0),
    ("ucsd_kitchen_dataset_converted_externally_to_rlds", 1.0),
    ("ucsd_pick_and_place_dataset_converted_externally_to_rlds", 1.0),
    ("austin_sailor_dataset_converted_externally_to_rlds", 1.0),
    ("austin_sirius_dataset_converted_externally_to_rlds", 1.0),
    ("bc_z", 1.0),
    ("utokyo_pr2_opening_fridge_converted_externally_to_rlds", 1.0),
    ("utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds", 1.0),
    ("utokyo_xarm_pick_and_place_converted_externally_to_rlds", 1.0),
    ("utokyo_xarm_bimanual_converted_externally_to_rlds", 1.0),
    ("robo_net", 1.0),
    ("berkeley_mvp_converted_externally_to_rlds", 1.0),
    ("berkeley_rpt_converted_externally_to_rlds", 1.0),
    ("kaist_nonprehensile_converted_externally_to_rlds", 1.0),
    ("stanford_mask_vit_converted_externally_to_rlds", 1.0),
    ("tokyo_u_lsmo_converted_externally_to_rlds", 1.0),
    ("dlr_sara_pour_converted_externally_to_rlds", 1.0),
    ("dlr_sara_grid_clamp_converted_externally_to_rlds", 1.0),
    ("dlr_edan_shared_control_converted_externally_to_rlds", 1.0),
    ("asu_table_top_converted_externally_to_rlds", 1.0),
    ("stanford_robocook_converted_externally_to_rlds", 1.0),
    ("imperialcollege_sawyer_wrist_cam", 1.0),
    ("iamlab_cmu_pickup_insert_converted_externally_to_rlds", 1.0),
    ("uiuc_d3field", 1.0),
    ("utaustin_mutex", 1.0),
    ("berkeley_fanuc_manipulation", 1.0),
    ("cmu_playing_with_food", 1.0),
    ("cmu_play_fusion", 1.0),
    ("cmu_stretch", 1.0),
    ("gnm_dataset", 1.0),
]

LIBERO_90_AUGMENTED = [
    ("libero_90_subtasks", 1.0),
    ("libero_90_original", 1.0),
]

LIBERO_90_ORIGINAL = [
    ("libero_90_original", 1.0),
]

LIBERO_90_SINGLE_TASK = [
    ("libero_90_single_task", 1.0),
]

LIBERO_90_SINGLE_TASK_SEGMENTED = [
    ("libero_90_single_task_segmented", 1.0),
]

LIBERO_90_SINGLE_TASK_MIX = [
    ("libero_90_single_task", 1.0),
    ("libero_90_single_task_segmented", 1.0),
]

LIBERO_90_TWO_TASK = [
    ("libero_90_two_task", 1.0),
]

LIBERO_90_TWO_TASK_SEGMENTED = [
    ("libero_90_two_task_segmented", 1.0),
]

LIBERO_90_TWO_TASK_MIX = [
    ("libero_90_two_task", 1.0),
    ("libero_90_two_task_segmented", 1.0),
]

LIBERO_90_SUBTASKS = [
    ("libero_90_subtasks", 1.0)
]

LIBERO_90 = [
    ("libero_90", 1.0)
]

LIBERO_10_AUGMENTED = [
    ("libero_10_subtasks", 1.0),
    ("libero_10_original_no_noops", 1.0),
]

LIBERO_10_ORIGINAL = [
    ("libero_10_original_no_noops", 1.0),
]

LIBERO_10_SUBTASKS = [
    ("libero_10_subtasks", 1.0)
]

LIBERO_100_AUGMENTED = [
    ("libero_90_subtasks", 1.0),
    ("libero_90_original", 2.0),
    ("libero_10_subtasks", 1.0),
    ("libero_10_original_no_noops", 1.0),
]

LIBERO_100_AUGMENTED_REMIX = [
    ("libero_90_subtasks", 0.21817),
    ("libero_90_original", 0.26868),
    ("libero_10_subtasks", 0.22871),
    ("libero_10_original_no_noops", 0.28444),
]

LIBERO_100_AUGMENTED_DIVERSITY = [
    ("libero_90_subtasks_augmented", 0.15861),
    ("libero_90_original", 0.27521),
    ("libero_10_subtasks_augmented", 0.10757),
    ("libero_10_original_no_noops", 1.0),
]

LIBERO_100_AUGMENTED_DIVERSITY_REMIX = [
    ("libero_90_subtasks_augmented", 0.23429),
    ("libero_90_original", 0.26724),
    ("libero_10_subtasks_augmented", 0.24023),
    ("libero_10_original_no_noops", 0.25997),
]

LIBERO_FULL = [
    ("libero_90_subtasks_augmented", 0.15861),
    ("libero_90_original", 0.27521),
    ("libero_10_subtasks_augmented", 0.10757),
    ("libero_10_original_no_noops", 0.19181),
    ("libero_90_subtasks", 0.13721),
    ("libero_10_subtasks", 0.12958),
]

LIBERO_100_AUGMENTED_CUMUL_DIVERSITY = [
    ("libero_90_subtasks_cumul_aug", 0.75),
    ("libero_90_original", 2.0),
    ("libero_10_subtasks_cumul_aug", 0.75),
    ("libero_10_original_no_noops", 1.0),
]

LIBERO_100_AUGMENTED_GROUPED_DIVERSITY = [
    ("libero_90_subtasks_grouped_aug", 0.45),
    ("libero_90_original", 2.0),
    ("libero_10_subtasks_grouped_aug", 0.3),
    ("libero_10_original_no_noops", 1.0),
]

LIBERO_100_ORIGINAL = [
    ("libero_90_original", 2.0),
    ("libero_10_original_no_noops", 1.0),
]

LIBERO_100_SUBTASKS = [
    ("libero_90_subtasks", 1.0),
    ("libero_10_subtasks", 1.0),
]

LIBERO_100_ORIGINAL_REMIX = [
    ("libero_90_original", 0.48303),
    ("libero_10_original_no_noops", 0.51697),
]


LIBERO_100_SUBTASKS_AUGMENTED = [
    ("libero_90_subtasks_augmented", 1.0),
    ("libero_10_subtasks_augmented", 1.0),
]

BRIDGE_SUBTASKS_MIX = [
    ("bridge_original", 0.55438),
    ("bridge_subtasks", 0.44562),
]

BRIDGE_ORIGINAL= [
    ("bridge_original", 1.0),
]

OXE_NAMED_MIXES = {
    "bridge": BRIDGE_MIX,
    "bridge_mix": BRIDGE_SUBTASKS_MIX,
    "bridge_original": BRIDGE_SUBTASKS_MIX,
    "rtx": RT_X_MIX,
    "rtx_franka": RT_X_MIX + OXE_FRANKA_MIX,
    "oxe_magic_soup": OXE_MAGIC_SOUP,
    "oxe_flex_act_soup": OXE_FLEX_ACT_SOUP,
    "libero_90_augmented": LIBERO_90_AUGMENTED,
    "libero_90_original": LIBERO_90_ORIGINAL,
    "libero_90_single_task": LIBERO_90_SINGLE_TASK,
    "libero_90_single_task_segmented": LIBERO_90_SINGLE_TASK_SEGMENTED,
    "libero_90_single_task_mix": LIBERO_90_SINGLE_TASK_MIX,
    "libero_90_two_task": LIBERO_90_TWO_TASK,
    "libero_90_two_task_segmented": LIBERO_90_TWO_TASK_SEGMENTED,
    "libero_90_two_task_mix": LIBERO_90_TWO_TASK_MIX,
    "libero_90": LIBERO_90,
    "libero_90_subtasks": LIBERO_90_SUBTASKS,
    "libero_10_augmented": LIBERO_10_AUGMENTED,
    "libero_10_original": LIBERO_10_ORIGINAL,
    "libero_10_subtasks": LIBERO_10_SUBTASKS,
    "libero_100_augmented": LIBERO_100_AUGMENTED,
    "libero_100_augmented_remix": LIBERO_100_AUGMENTED_REMIX,
    "libero_100_original": LIBERO_100_ORIGINAL,
    "libero_100_original_remix": LIBERO_100_ORIGINAL_REMIX,
    "libero_100_subtasks": LIBERO_100_SUBTASKS,
    "libero_100_subtasks_augmented": LIBERO_100_SUBTASKS_AUGMENTED,
    "libero_100_augmented_diversity": LIBERO_100_AUGMENTED_DIVERSITY,
    "libero_100_augmented_diversity_remix": LIBERO_100_AUGMENTED_DIVERSITY_REMIX,
    "libero_100_augmented_grouped": LIBERO_100_AUGMENTED_CUMUL_DIVERSITY,
    "libero_100_augmented_grouped_diversity": LIBERO_100_AUGMENTED_GROUPED_DIVERSITY,
    "libero_full": LIBERO_FULL,
}
