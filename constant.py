import os
import tensorflow as tf
# Import the Classifier.
from sklearn.svm import SVC

CATEGORY_LABELS = {'cup': 8, 'timber': 17, 'bottle': 4, 'tin': 18, 'ball': 1, 'weight': 20, 'eggcoloringcup': 9, 'basket': 2, 'cone': 7, 'cannedfood': 6, 'noodle': 13, 'egg': 10, 'medicine': 11, 'pvc': 15, 'can': 5, 'pasta': 14, 'tupperware': 19, 'bigstuffedanimal': 3, 'smallstuffedanimal': 16, 'metal': 12}
OBJECT_LABELS = {'cup_yellow': 40, 'basket_handle': 9, 'pvc_1': 71, 'smallstuffedanimal_moose': 79, 'smallstuffedanimal_headband_bear': 78, 'noodle_2': 62, 'timber_square': 84, 'tupperware_ground_coffee': 92, 'medicine_calcium': 54, 'basket_cylinder': 6, 'egg_cardboard': 46, 'cannedfood_tomato_paste': 30, 'egg_smooth_styrofoam': 49, 'noodle_1': 61, 'basket_funnel': 7, 'can_starbucks': 25, 'weight_3': 98, 'cone_2': 32, 'weight_2': 97, 'bottle_red': 19, 'medicine_aspirin': 52, 'eggcoloringcup_orange': 43, 'bottle_green': 18, 'egg_wood': 50, 'egg_plastic_wrap': 47, 'noodle_3': 63, 'timber_squiggle': 85, 'pasta_pipette': 69, 'noodle_5': 65, 'cannedfood_tomatoes': 29, 'pasta_cremette': 66, 'ball_transparent': 4, 'ball_basket': 2, 'tupperware_coffee_beans': 91, 'metal_thermos': 60, 'bottle_google': 17, 'smallstuffedanimal_otter': 80, 'tin_tea': 90, 'eggcoloringcup_blue': 41, 'tupperware_pasta': 94, 'cup_blue': 36, 'egg_rough_styrofoam': 48, 'bigstuffedanimal_tan_dog': 15, 'timber_semicircle': 83, 'eggcoloringcup_pink': 44, 'cone_5': 35, 'timber_rectangle': 82, 'cannedfood_cowboy_cookout': 27, 'noodle_4': 64, 'tupperware_marbles': 93, 'cone_3': 33, 'pasta_penne': 68, 'pasta_rotini': 70, 'bigstuffedanimal_pink_dog': 14, 'cannedfood_soup': 28, 'tin_snowman': 89, 'metal_flower_cylinder': 56, 'eggcoloringcup_yellow': 45, 'weight_4': 99, 'cup_metal': 38, 'weight_5': 100, 'bigstuffedanimal_frog': 13, 'medicine_ampicillin': 51, 'smallstuffedanimal_bunny': 76, 'cone_4': 34, 'tin_poker': 87, 'can_red_bull_small': 24, 'cannedfood_chili': 26, 'ball_blue': 3, 'smallstuffedanimal_chick': 77, 'ball_base': 1, 'pvc_4': 74, 'medicine_bilberry_extract': 53, 'pvc_2': 72, 'timber_pentagon': 81, 'medicine_flaxseed_oil': 55, 'cup_isu': 37, 'metal_tea_jar': 59, 'ball_yellow_purple': 5, 'cone_1': 31, 'metal_food_can': 57, 'metal_mix_covered_cup': 58, 'tin_pokemon': 86, 'can_arizona': 21, 'bigstuffedanimal_bear': 11, 'can_red_bull_large': 23, 'tupperware_rice': 95, 'bigstuffedanimal_bunny': 12, 'can_coke': 22, 'eggcoloringcup_green': 42, 'pasta_macaroni': 67, 'basket_green': 8, 'pvc_5': 75, 'basket_semicircle': 10, 'tin_snack_depot': 88, 'bottle_sobe': 20, 'weight_1': 96, 'pvc_3': 73, 'cup_paper_green': 39, 'bottle_fuse': 16}
BEHAVIORS = ["Press", "Grasp", "Hold", "Lift", "Drop", "Poke", "Push", "Shake", "Tap"]
MODALITIES = ['Haptic', 'Audio', 'Surf', 'Vibro']

NUM_OF_CATEGORY = len(CATEGORY_LABELS)
OBJECTS_PER_CATEGORY = 5
TRIALS_PER_OBJECT = 5
NUM_OF_BEHAVIORS = len(BEHAVIORS)
NUM_OF_MODALITIES = len(MODALITIES)

NUM_OF_CATEGORY_FOR_TRAINING = 15  # should be less than 20
NUM_OF_CATEGORY_FOR_TESTING = NUM_OF_CATEGORY - NUM_OF_CATEGORY_FOR_TRAINING
TEST_TRAIN_RATIO = NUM_OF_CATEGORY_FOR_TESTING / NUM_OF_CATEGORY

# Hyper-Parameters
TRAINING_EPOCHS = 10 #1000  # 5000
LEARNING_RATE = 0.0001 #0.0001
CODE_VECTOR = 125
HIDDEN_LAYER_UNITS = [1000, 500, 250]
ACTIVATION_FUNCTION = tf.nn.elu
RUNS = 2 #10
BETA = 0.0001

CLF = SVC(gamma='auto', kernel='rbf')
CLF_NAME = "SVM-RBF"

AUDIO_DATASETS = ["crush_audio.bin", "grasp_audio.bin", "hold_audio.bin", "lift_slow_audio.bin", "low_drop_audio.bin", "poke_audio.bin",
    "push_audio.bin", "shake_audio.bin", "tap_audio.bin"]
HAPTIC_DATASETS = ["crush_proprioception_10bin_features.bin", "grasp_proprioception_10bin_features.bin", "hold_proprioception_10bin_features.bin",
    "lift_slow_proprioception_10bin_features.bin", "low_drop_proprioception_10bin_features.bin", "poke_proprioception_10bin_features.bin",
    "push_proprioception_10bin_features.bin", "shake_proprioception_10bin_features.bin", "tap_proprioception_10bin_features.bin"]
SURF_DATASETS = ["crush_surf.bin", "grasp_surf.bin", "hold_surf.bin", "lift_slow_surf.bin", "low_drop_surf.bin", "poke_surf.bin",
                       "push_surf.bin", "shake_surf.bin", "tap_surf.bin"]
VIBRO_DATASETS =["crush_vibro.bin", "grasp_vibro.bin", "hold_vibro.bin", "lift_slow_vibro.bin", "low_drop_vibro.bin", "poke_vibro.bin",
                       "push_vibro.bin", "shake_vibro.bin", "tap_vibro.bin"]

DATASETS_FOLDERS = {'A': ['audio_10x10_datasets', AUDIO_DATASETS], 'H': ['proprioception_10x10_datasets', HAPTIC_DATASETS],
                    'S': ['surf_128_datasets', SURF_DATASETS], 'V': ['vibro_5x20_datasets', VIBRO_DATASETS]}

# Random Objects for 10 Runs
RANDOM_OBJECTS_DICT = []

train_cat = [int(e) for e in "0 1 5 6 7 8 10 11 12 13 14 15 17 18 19".split(" ")]
test_cat = [int(e) for e in "2 3 4 9 16".split(" ")]
RANDOM_OBJECTS_DICT.append({"train":train_cat, "test":test_cat})

train_cat = [int(e) for e in "0 1 2 5 6 7 8 10 11 12 14 15 17 18 19".split(" ")]
test_cat = [int(e) for e in "3 4 9 13 16".split(" ")]
RANDOM_OBJECTS_DICT.append({"train":train_cat, "test":test_cat})

train_cat = [int(e) for e in "0 1 2 5 6 7 8 10 11 12 13 14 15 17 18".split(" ")]
test_cat = [int(e) for e in "3 4 9 16 19".split(" ")]
RANDOM_OBJECTS_DICT.append({"train":train_cat, "test":test_cat})

train_cat = [int(e) for e in "8 6 15 5 17 4 7 0 10 2 19 11 18 9 14".split(" ")]
test_cat = [int(e) for e in "3 12 13 16 1".split(" ")]
RANDOM_OBJECTS_DICT.append({"train":train_cat, "test":test_cat})

train_cat = [int(e) for e in "0 2 8 7 13 16 4 5 15 3 14 6 17 19 9".split(" ")]
test_cat = [int(e) for e in "10 1 11 12 18".split(" ")]
RANDOM_OBJECTS_DICT.append({"train":train_cat, "test":test_cat})

train_cat = [int(e) for e in "7 16 12 10 6 11 5 8 13 9 1 14 0 4 3".split(" ")]
test_cat = [int(e) for e in "18 19 17 2 15".split(" ")]
RANDOM_OBJECTS_DICT.append({"train":train_cat, "test":test_cat})

train_cat = [int(e) for e in "11 17 5 19 6 18 16 9 1 15 12 2 8 14 4".split(" ")]
test_cat = [int(e) for e in "13 0 3 10 7".split(" ")]
RANDOM_OBJECTS_DICT.append({"train":train_cat, "test":test_cat})

train_cat = [int(e) for e in "3 17 16 18 14 11 5 19 4 6 10 8 7 9 0".split(" ")]
test_cat = [int(e) for e in "12 15 13 1 2".split(" ")]
RANDOM_OBJECTS_DICT.append({"train":train_cat, "test":test_cat})

train_cat = [int(e) for e in "17 5 3 14 18 16 11 6 8 1 13 4 9 15 10".split(" ")]
test_cat = [int(e) for e in "7 12 0 2 19".split(" ")]
RANDOM_OBJECTS_DICT.append({"train":train_cat, "test":test_cat})

train_cat = [int(e) for e in "13 12 19 11 17 7 2 5 8 15 9 18 16 0 3".split(" ")]
test_cat = [int(e) for e in "6 14 1 10 4".split(" ")]
RANDOM_OBJECTS_DICT.append({"train":train_cat, "test":test_cat})
