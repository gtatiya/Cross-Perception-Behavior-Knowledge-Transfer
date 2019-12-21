import pickle

import numpy as np
import matplotlib.pyplot as plt

from model import classifier
from constant import *


def time_taken(start, end):
    """Human readable time between `start` and `end`

    :param start: time.time()
    :param end: time.time()

    :returns: day:hour:minute:second.millisecond
    """

    my_time = end-start
    day = my_time // (24 * 3600)
    my_time = my_time % (24 * 3600)
    hour = my_time // 3600
    my_time %= 3600
    minutes = my_time // 60
    my_time %= 60
    seconds = my_time
    milliseconds = ((end - start)-int(end - start))
    day_hour_min_sec = str('%02d' % int(day))+":"+str('%02d' % int(hour))+":"+str('%02d' % int(minutes))+":"+str('%02d' % int(seconds)+"."+str('%.3f' % milliseconds)[2:])

    return day_hour_min_sec


def find_modality_bin_behavior(a_path, db_file_name):
    """
    Finds modality, bins, behavior by using `path` and `dataset` file name

    :param a_path: Dataset path
    :param db_file_name: Dataset file name

    :return: modality, bins, behavior
    """

    modality = a_path.split(os.sep)[1].split("_")[0].capitalize()
    bins = a_path.split(os.sep)[1].split("_")[1]

    if modality == "Proprioception":
        modality = "Haptic"

    if (db_file_name.split(".")[0].split("_")[0]) == 'low':
        behavior = "Drop"
    else:
        behavior = db_file_name.split(".")[0].split("_")[0].capitalize()

    if behavior == "Crush":
        behavior = 'Press'

    return modality, bins, behavior


def reshape_full_data(data):
    """
    Reshape data into (Categories, Objects, Trials)

    :param data: Dataset list

    :return: reshaped Dataset list
    """
    return data.reshape(NUM_OF_CATEGORY, OBJECTS_PER_CATEGORY, TRIALS_PER_OBJECT, -1)


def read_dataset(a_path, db_file_name):
    """
    Read dataset

    :param a_path: Dataset path
    :param db_file_name: Dataset file name

    :return: interaction_data, category_labels, object_labels
    """

    bin_file = open(a_path + os.sep + db_file_name, "rb")
    interaction_data = pickle.load(bin_file)
    category_labels = pickle.load(bin_file)
    object_labels = pickle.load(bin_file)
    bin_file.close()

    return reshape_full_data(interaction_data), reshape_full_data(category_labels), reshape_full_data(object_labels)


def repeat_trials(interaction_data_1_train, interaction_data_2_train):
    """
    Repeat trials for both robots

    :param interaction_data_1_train: Source robot dataset
    :param interaction_data_2_train: Target robot dataset

    :return: Repeated source robot dataset, Repeated target robot dataset
    """

    # Source
    # One example of the source robot can be mapped to all the example of the target robot
    # So, repeating each example of the source robot for each example of target robot
    interaction_data_1_train_repeat = np.repeat(interaction_data_1_train, TRIALS_PER_OBJECT, axis=2)

    # Target
    # Concatenating same examples of target robot to make it same size as source robot
    interaction_data_2_train_repeat = interaction_data_2_train
    for _ in range(TRIALS_PER_OBJECT - 1):
        interaction_data_2_train_repeat = np.concatenate((interaction_data_2_train_repeat, interaction_data_2_train),
                                                         axis=2)

    return interaction_data_1_train_repeat, interaction_data_2_train_repeat


def object_recognition_classifier(clf, data_train, data_test, label_train, label_test, num_of_features):
    """
    Train a classifier and test it based on provided data

    :param clf:
    :param data_train:
    :param data_test:
    :param label_train:
    :param label_test:
    :param num_of_features:

    :return: accuracy, prediction
    """

    train_cats_data = data_train.reshape(-1, num_of_features)
    train_cats_label = label_train.reshape(-1, 1).flatten()

    test_cats_data = data_test.reshape(-1, num_of_features)
    test_cats_label = label_test.reshape(-1, 1).flatten()

    y_acc, y_pred = classifier(clf, train_cats_data, test_cats_data, train_cats_label, test_cats_label)

    return y_acc, y_pred


def print_discretized_data(data, x_values, y_values, modality, behavior, file_path=None):
    """
    prints the data point and save it

    :param data: one data point
    :param x_values: temporal bins
    :param y_values:
    :param modality:
    :param behavior:
    :param file_path:

    :return:
    """
    data = data.reshape(x_values, y_values)

    plt.imshow(data.T)

    title_name = " ".join([behavior, modality, "Features"])
    plt.title(title_name, fontsize=16)
    plt.xlabel("Temporal Bins", fontsize=16)

    if modality == 'Haptic':
        y_label = "Joints"
    elif modality == 'Audio':
        y_label = "Frequency Bins"
    else:
        y_label = ""
    plt.ylabel(y_label, fontsize=16)

    ax = plt.gca()
    ax.set_xticks(np.arange(0, x_values, 1))
    ax.set_yticks(np.arange(0, y_values, 1))
    ax.set_xticklabels(np.arange(1, x_values + 1, 1))
    ax.set_yticklabels(np.arange(1, y_values + 1, 1))

    plt.colorbar()

    if file_path != None:
        plt.savefig(file_path, bbox_inches='tight', dpi=100)

    #plt.show()
    plt.close()


""" Setting 1 """
# Target Robot never interacts with a few categories


def reshape_data_setting1(num_of_category, data):
    """
    Reshape data into (Categories, Objects, Trials)

    :param num_of_category:
    :param data: Dataset list

    :return: reshaped Dataset list
    """
    return data.reshape(num_of_category, OBJECTS_PER_CATEGORY, TRIALS_PER_OBJECT, -1)


def get_data_label_for_given_labels(given_labels, interaction_data, category_labels):
    """
    Get all the examples of the given labels

    :param given_labels: labels to find
    :param interaction_data: examples
    :param category_labels: labels

    :return: Dataset, labels
    """

    data = []
    label = []

    for a_label in given_labels:
        data.append(interaction_data[a_label])
        label.append(category_labels[a_label])

    return np.array(data), np.array(label)


def train_test_splits(num_of_objects):
    """
    Split the data into object based 5 fold cross validation

    :param num_of_objects:

    :return: dictionary containing train test index of 5 folds
    """

    n_folds = 5
    tt_splits = {}

    for a_fold in range(n_folds):
        train_index = []

        test_index = np.arange(a_fold, (a_fold + 1))

        if a_fold > 0:
            train_index.extend(np.arange(0, a_fold))

        if (a_fold + 1) - 1 < num_of_objects - 1:
            train_index.extend(np.arange((a_fold + 1), num_of_objects))

        tt_splits.setdefault("fold_" + str(a_fold), {}).setdefault("train", []).extend(train_index)
        tt_splits.setdefault("fold_" + str(a_fold), {}).setdefault("test", []).extend(test_index)

    return tt_splits


def object_based_5_fold_cross_validation(clf, data_train, data_test, labels, num_of_features):
    """
    Perform object based 5 fold cross validation and return mean accuracy

    :param clf: classifier
    :param data_train: Training dataset
    :param data_test: Testing dataset
    :param labels: True labels
    :param num_of_features: Number of features of the robot

    :return: mean accuracy of 5 fold validation
    """

    tts = train_test_splits(OBJECTS_PER_CATEGORY)

    my_acc = []

    for a_fold in sorted(tts):
        train_cats_index = tts[a_fold]["train"]
        test_cats_index = tts[a_fold]["test"]

        train_cats_data = data_train[:, train_cats_index]
        train_cats_label = labels[:, train_cats_index]
        train_cats_data = train_cats_data.reshape(-1, num_of_features)
        train_cats_label = train_cats_label.reshape(-1, 1).flatten()

        test_cats_data = data_test[:, test_cats_index]
        test_cats_label = labels[:, test_cats_index]
        test_cats_data = test_cats_data.reshape(-1, num_of_features)
        test_cats_label = test_cats_label.reshape(-1, 1).flatten()

        y_acc, y_pred = classifier(clf, train_cats_data, test_cats_data, train_cats_label, test_cats_label)
        my_acc.append(y_acc)

    return np.mean(my_acc)


""" Setting 2 """
# Target Robot never interacts with a few objects
"""
Worst Case:
If target robot interacts with 1 object, generate features of rest 4 obj.
Train KNN using 1 real + 4 gen, test on 4 real.

Best Case:
If target robot interacts with 4 object, generate features of rest 1 obj.
Train KNN using 4 real + 1 gen, test on 1 real.
"""

""" Setting 2 version 2"""
# Target Robot never interacts with a few objects
"""
Both robots does not interacts with this one object that is used for testing KNN using 5 fold.
This process is repeated 5 times until each obj is tested once.

Worst Case:
If target robot interacts with 1 object, generate features of rest 3 obj.
Train KNN using 1 real + 3 gen, test on 1 real.

Best Case:
If target robot interacts with 3 object, generate features of rest 1 obj.
Train KNN using 3 real + 1 gen, test on 1 real.
"""


def train_test_split_setting2(num_of_obj_for_training):
    """
    Train test split for Setting 2

    :return: train, test objects
    """

    train_objects = []
    test_objects = []
    for _ in range(NUM_OF_CATEGORY):
        # randomly choose the object target robot interacts with for each category
        train_obj = np.random.choice(np.arange(OBJECTS_PER_CATEGORY), size=num_of_obj_for_training, replace=False)
        # print(train_obj)
        train_objects.append(train_obj)

        # put the rest objects for testing
        test_obj = np.arange(OBJECTS_PER_CATEGORY)
        for a_obj in train_obj:
            test_obj = np.delete(test_obj, np.where(test_obj == a_obj), axis=0)
        # print(test_obj)
        test_objects.append(test_obj)

    return np.array(train_objects), np.array(test_objects)


def train_test_split_setting2_v2(train_objs_index, num_of_trials_for_training):
    """
    Train test split for Setting 2_v2
    Given the training objects, put objects for training and testing EDN based on number of obj target robot interacts with
    It chooses objects to train randomly

    :param: num_of_trials_for_training

    :return: train, test objects
    """

    # randomly choose the object target robot interacts with
    train_obj_edn = np.random.choice(train_objs_index, size=num_of_trials_for_training, replace=False)

    # put the rest objects for testing
    test_objs_edn = train_objs_index
    for a_obj in train_obj_edn:
        test_objs_edn = np.delete(test_objs_edn, np.where(test_objs_edn == a_obj), axis=0)

    return train_obj_edn, test_objs_edn


def train_test_split_setting2_v3(train_objs_index, num_of_trials_for_training):
    """
    Train test split for Setting 2_v2
    Given the training objects, put objects for training and testing EDN based on number of obj target robot interacts with
    It chooses objects to train in sequential order

    :param: num_of_trials_for_training

    :return: train, test objects
    """

    return np.array(train_objs_index[:num_of_trials_for_training]), np.array(train_objs_index[num_of_trials_for_training:])


def get_data_label_for_given_objects(given_objects, interaction_data, category_labels, object_labels):
    """
    Create dataset for given objects

    :param given_objects: 2D list of objects of each category
    :param interaction_data:
    :param category_labels:
    :param object_labels:

    :return: interaction_data, category_labels, object_labels
    """

    data = []
    cat_labels = []
    obj_labels = []
    for i in range(NUM_OF_CATEGORY):
        data.append(interaction_data[i][given_objects[i]])
        cat_labels.append(category_labels[i][given_objects[i]])
        obj_labels.append(object_labels[i][given_objects[i]])

    return np.array(data), np.array(cat_labels), np.array(obj_labels)


def get_data_label_for_given_objects_v2(given_objects, interaction_data, category_labels, object_labels):
    """
    Create dataset for given objects

    :param given_objects: 2D list of objects of each category
    :param interaction_data:
    :param category_labels:
    :param object_labels:

    :return: interaction_data, category_labels, object_labels
    """

    data = []
    cat_labels = []
    obj_labels = []
    for i in range(NUM_OF_CATEGORY):
        data.append(interaction_data[i][given_objects])
        cat_labels.append(category_labels[i][given_objects])
        obj_labels.append(object_labels[i][given_objects])

    return np.array(data), np.array(cat_labels), np.array(obj_labels)


def reshape_data_setting2(num_of_objects, data):
    """
    Reshape data into (Categories, Objects, Trials)

    :param num_of_objects:
    :param data: Dataset list

    :return: reshaped Dataset list
    """
    return data.reshape(NUM_OF_CATEGORY, num_of_objects, TRIALS_PER_OBJECT, -1)


""" Setting 3 """
# Target Robot only interacts with a few trials
"""
Worst Case:
If target robot interacts with 1 trial, generate features of rest 4 trials.
Train KNN using 1 real + 4 gen, test on 4 real.

Best Case:
If target robot interacts with 4 trial, generate features of rest 1 trials.
Train KNN using 4 real + 1 gen, test on 1 real.
"""

""" Setting 3 version 2"""
# Target Robot only interacts with a few trials
"""
Both robots does not interacts with this one object that is used for testing KNN using 5 fold.
This process is repeated 5 times until each obj is tested once.

Worst Case:
If target robot interacts with 1 trial, generate features of rest 4 trials.
Train KNN using 1 real + 4 gen, test on 1 real.

Best Case:
If target robot interacts with 4 trial, generate features of rest 1 trials.
Train KNN using 4 real + 1 gen, test on 1 real.
"""


def train_test_split_setting3(num_of_trials_for_training):
    """
    Train test split for Setting 3

    :param: num_of_trials_for_training

    :return: train, test objects
    """

    train_trials = []
    test_trials = []
    for _ in range(NUM_OF_CATEGORY):
        train_trials_cat = []
        test_trials_cat = []
        for _ in range(OBJECTS_PER_CATEGORY):
            # randomly choose the trial target robot interacts with for each object of each categoty
            train_tri = np.random.choice(np.arange(TRIALS_PER_OBJECT), size=num_of_trials_for_training, replace=False)
            # print(train_tri)
            train_trials_cat.append(train_tri)

            # put the rest trials for testing
            test_tri = np.arange(TRIALS_PER_OBJECT)
            for a_tri in train_tri:
                test_tri = np.delete(test_tri, np.where(test_tri == a_tri), axis=0)
            # print(test_tri)
            test_trials_cat.append(test_tri)

        train_trials.append(train_trials_cat)
        test_trials.append(test_trials_cat)

    return np.array(train_trials), np.array(test_trials)


def get_data_label_for_given_trials(given_trials, interaction_data, category_labels, object_labels):
    """
    Create dataset for given objects

    :param given_trials: 2D list of objects of each category
    :param interaction_data:
    :param category_labels:
    :param object_labels:

    :return: interaction_data, category_labels, object_labels
    """

    data = []
    cat_labels = []
    obj_labels = []
    for i in range(NUM_OF_CATEGORY):
        data_cat = []
        cat_labels_cat = []
        obj_labels_cat = []
        for j in range(OBJECTS_PER_CATEGORY):
            data_cat.append(interaction_data[i][j][given_trials[i][j]])
            cat_labels_cat.append(category_labels[i][j][given_trials[i][j]])
            obj_labels_cat.append(object_labels[i][j][given_trials[i][j]])
        data.append(data_cat)
        cat_labels.append(cat_labels_cat)
        obj_labels.append(obj_labels_cat)

    return np.array(data), np.array(cat_labels), np.array(obj_labels)


def repeat_trials_setting3(num_of_trials_for_training, interaction_data_1_train, interaction_data_2_train):
    """
    Repeat trials for both robots

    :param num_of_trials_for_training: num_of_trials_for_training
    :param interaction_data_1_train: Source robot dataset
    :param interaction_data_2_train: Target robot dataset

    :return: Repeated source robot dataset, Repeated target robot dataset
    """

    # Source
    # One example of the source robot can be mapped to all the example of the target robot
    # So, repeating each example of the source robot for each example of target robot
    interaction_data_1_train_repeat = np.repeat(interaction_data_1_train, num_of_trials_for_training, axis=2)

    # Target
    # Concatenating same examples of target robot to make it same size as source robot
    interaction_data_2_train_repeat = interaction_data_2_train
    for _ in range(num_of_trials_for_training - 1):
        interaction_data_2_train_repeat = np.concatenate((interaction_data_2_train_repeat, interaction_data_2_train),
                                                         axis=2)

    return interaction_data_1_train_repeat, interaction_data_2_train_repeat


def reshape_data_setting3(num_of_trials, data):
    """
    Reshape data into (Categories, Objects, Trials)

    :param num_of_trials:
    :param data: Dataset list

    :return: reshaped Dataset list
    """
    return data.reshape(NUM_OF_CATEGORY, OBJECTS_PER_CATEGORY, num_of_trials, -1)


def train_test_split_setting3_v2(num_of_trials_for_training):
    """
    Train test split for Setting 3

    :param: num_of_trials_for_training

    :return: train, test objects
    """

    train_trials = []
    test_trials = []
    for _ in range(NUM_OF_CATEGORY):
        train_trials_cat = []
        test_trials_cat = []
        for _ in range(OBJECTS_PER_CATEGORY-1):
            # randomly choose the trial target robot interacts with for each object of each categoty
            train_tri = np.random.choice(np.arange(TRIALS_PER_OBJECT), size=num_of_trials_for_training, replace=False)
            # print(train_tri)
            train_trials_cat.append(train_tri)

            # put the rest trials for testing
            test_tri = np.arange(TRIALS_PER_OBJECT)
            for a_tri in train_tri:
                test_tri = np.delete(test_tri, np.where(test_tri == a_tri), axis=0)
            # print(test_tri)
            test_trials_cat.append(test_tri)

        train_trials.append(train_trials_cat)
        test_trials.append(test_trials_cat)

    return np.array(train_trials), np.array(test_trials)


def train_test_split_setting3_v3(num_of_trials_for_training):
    """
    Train test split for Setting 3

    :param: num_of_trials_for_training

    :return: train, test objects
    """

    train_trials = []
    test_trials = []
    for _ in range(NUM_OF_CATEGORY):
        train_trials_cat = []
        test_trials_cat = []
        for _ in range(OBJECTS_PER_CATEGORY-1):
            # randomly choose the trial target robot interacts with for each object of each categoty
            train_tri = np.arange(TRIALS_PER_OBJECT)[:num_of_trials_for_training]
            train_trials_cat.append(train_tri)

            # put the rest trials for testing
            test_tri = np.arange(TRIALS_PER_OBJECT)[num_of_trials_for_training:]
            test_trials_cat.append(test_tri)

        train_trials.append(train_trials_cat)
        test_trials.append(test_trials_cat)

    return np.array(train_trials), np.array(test_trials)


def get_data_label_for_given_trials_v2(train_objs_index, given_trials, interaction_data, category_labels, object_labels):
    """
    Create dataset for given objects

    :param given_trials: 2D list of objects of each category
    :param interaction_data:
    :param category_labels:
    :param object_labels:

    :return: interaction_data, category_labels, object_labels
    """

    data = []
    cat_labels = []
    obj_labels = []
    for i in range(NUM_OF_CATEGORY):
        data_cat = []
        cat_labels_cat = []
        obj_labels_cat = []
        for j in range(len(train_objs_index)):
            data_cat.append(interaction_data[i][train_objs_index[j]][given_trials[i][j]])
            cat_labels_cat.append(category_labels[i][train_objs_index[j]][given_trials[i][j]])
            obj_labels_cat.append(object_labels[i][train_objs_index[j]][given_trials[i][j]])
        data.append(data_cat)
        cat_labels.append(cat_labels_cat)
        obj_labels.append(obj_labels_cat)

    return np.array(data), np.array(cat_labels), np.array(obj_labels)


def reshape_data_setting3_v2(obj_per_cat, num_of_trials, data):
    """
    Reshape data into (Categories, Objects, Trials)

    :param num_of_trials:
    :param data: Dataset list

    :return: reshaped Dataset list
    """
    return data.reshape(NUM_OF_CATEGORY, obj_per_cat, num_of_trials, -1)


def get_indices(categories, category_labels):
    """
    Return indices of given categories in the category_labels 
    """
    all_indices = []
    for a_label in sorted(categories):
        indices = np.where((category_labels.flatten()-1) == a_label)
        all_indices.extend(indices[0])    

    return np.array(all_indices)


def get_indices_of_selected_trials(k, cluster_labels, test_indices, num_of_train_examples):

    selected_train_indices = []
    sample_probability = {}
    selected_train_indices_clusters = {}
    max_possible_examples = 0

    for a_label in range(k):
        if a_label in set(cluster_labels[test_indices]):
            indices = np.where(cluster_labels == a_label)

            num_of_test_examples = len(np.intersect1d(indices, test_indices))
            num_of_examples = len(indices[0])

            proba = num_of_test_examples/num_of_examples
            if proba < 1:
                sample_probability[a_label] = num_of_test_examples/num_of_examples
            else:
                sample_probability[a_label] = 0

            indices = np.setdiff1d(indices, test_indices)
            selected_train_indices_clusters[a_label] = indices
            max_possible_examples += len(indices)
        else:
            sample_probability[a_label] = 0

    sample_probability_norm = {}
    for a_cluster in sample_probability:
        sample_probability_norm[a_cluster] = sample_probability[a_cluster] / sum(sample_probability.values())

    while True:
        objects = np.random.choice(k, size=1, p=list(sample_probability_norm.values()))
        index = np.random.choice(selected_train_indices_clusters[objects[0]], size=1)
        
        if index not in selected_train_indices:
            selected_train_indices.append(int(index))
        else:
            print(num_of_train_examples, max_possible_examples, len(selected_train_indices))

        if (len(selected_train_indices) == num_of_train_examples):
            return np.array(selected_train_indices)
        elif (len(selected_train_indices) == max_possible_examples):
            print("HIT MAX POSSIBLE TRIALS")
            return []

def get_indices_of_selected_objects(k, cluster_labels, test_indices, labels, num_of_train_objects):

    sample_probability = {}
    selected_train_objects_clusters = {}
    selected_train_objects_set = []
    for a_label in range(k):
        if a_label in set(cluster_labels[test_indices]):
            indices = np.where(cluster_labels == a_label)

            # common objects
            objects = np.intersect1d(list(set(labels.flatten()[indices[0]])), 
                                     list(set(labels.flatten()[test_indices])))
                    
            num_of_test_objects = len(objects)
            num_of_objects = len(set(labels.flatten()[indices[0]]))

            proba = num_of_test_objects/num_of_objects
            if proba < 1:
                sample_probability[a_label] = proba
            else:
                sample_probability[a_label] = 0

            # uncommon objects
            objects = np.setdiff1d(list(set(labels.flatten()[indices[0]])),
                                   list(set(labels.flatten()[test_indices])))

            selected_train_objects_set.extend(objects)
            selected_train_objects_clusters[a_label] = objects
                
        else:
            sample_probability[a_label] = 0

    sample_probability_norm = {}
    for a_cluster in sample_probability:
        sample_probability_norm[a_cluster] = sample_probability[a_cluster] / sum(sample_probability.values())

    selected_train_objects = []
    max_possible_objects = len(set(selected_train_objects_set))
    while True:
        cluster = np.random.choice(k, size=1, p=list(sample_probability_norm.values()))
        
        an_object = np.random.choice(selected_train_objects_clusters[cluster[0]], size=1)
                
        if an_object not in selected_train_objects:
            selected_train_objects.append(int(an_object))
        else:
            print(num_of_train_objects, max_possible_objects, len(selected_train_objects))
        
        if len(selected_train_objects) == num_of_train_objects:
            break
        elif len(selected_train_objects) == max_possible_objects:
            print("HIT MAX POSSIBLE OBJECTS")
            selected_train_objects = []
            break
    selected_train_objects = np.array(selected_train_objects)

    selected_train_indices = []
    for an_object in selected_train_objects:
        indices = np.where(labels.flatten() == an_object)        
        selected_train_indices.extend(indices[0])

    return selected_train_indices, selected_train_objects
