import sys
import csv
import glob
import time
import numpy as np
from sklearn.model_selection import train_test_split
from numpy import genfromtxt
import matplotlib.pyplot as plt

from utils import find_modality_bin_behavior, read_dataset, get_data_label_for_given_labels, reshape_data_setting1, \
    object_based_5_fold_cross_validation, repeat_trials, time_taken
from model import EncoderDecoderNetwork, EncoderDecoderNetwork_b_VEDN, EncoderDecoderNetwork_b_VAE

from constant import *

tf.set_random_seed(1)

"""
1 Source Robot and 1 Target Robot

One to one projections:
A, H, S, V
A2A, A2H, A2S, A2V, H2A, H2H, H2S, H2V, S2A, S2H, S2S, S2V, V2A, V2H, V2S, V2V

python 1_source_b-ved_b-vae.py H2H
"""

if len(sys.argv) != 2:
    print("Pass one of 1st arguments: A2A, A2H, A2S, A2V, H2A, H2H, H2S, H2V, S2A, S2H, S2S, S2V, V2A, V2H, V2S, V2V")
    print("For example: python 1_source_b-ved_b-vae.py H2H")
    exit()

LOGS_PATH = r".." + os.sep + "Cross-Perception-Behavior-Knowledge-Transfer_" + sys.argv[1] + os.sep
os.makedirs(LOGS_PATH, exist_ok=True)

# Source Robot data
A_PATH1 = r"Datasets" + os.sep + DATASETS_FOLDERS[sys.argv[1][0]][0]
SOURCE_DATASETS = DATASETS_FOLDERS[sys.argv[1][0]][1]
# Target Robot data
A_PATH2 = r"Datasets" + os.sep + DATASETS_FOLDERS[sys.argv[1][2]][0]
TARGET_DATASETS = DATASETS_FOLDERS[sys.argv[1][2]][1]


def plot_loss_curve(cost, save_path, title_name_end, xlabel, ylabel):
    """
    Plot loss over iterations and save a plot

    :param cost:
    :param save_path:
    :param title_name_end:
    :param xlabel:
    :param ylabel:

    :return:
    """

    plt.plot(range(1, len(cost)+1), cost)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    title_name = " ".join([behavior1, modality1, "TO", behavior2, modality2])
    plt.title(title_name)
    title_name_ = "_".join([behavior1, modality1, "TO", behavior2, modality2])+title_name_end
    plt.savefig(save_path+os.sep+title_name_, bbox_inches='tight', dpi=100)
    plt.close()


def save_cost_csv(cost, save_path, csv_name_end):
    """
    Save loss over iterations in a csv file

    :param cost:
    :param save_path:
    :param csv_name_end:

    :return:
    """

    csv_name = "_".join([behavior1, modality1, "TO", behavior2, modality2])+csv_name_end

    with open(save_path+os.sep+csv_name, 'w') as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["epoch", "Loss"])
        for i in range(1, len(cost)+1):
            writer.writerow([i, cost[i-1]])


def get_feed_dict(placeholder, domains_data, num_of_domains):
    feed_dict = {}
    for a_domain in range(num_of_domains):
        #print(a_domain)
        feed_dict[placeholder["input"][a_domain]] = domains_data['domain_'+str(a_domain)]
        feed_dict[placeholder["output"][a_domain]] = domains_data['domain_'+str(a_domain)]
    return feed_dict


# Writing log file for execution time
with open(LOGS_PATH + 'time_log.txt', 'w') as time_log_file:
    time_log_file.write('Time Log\n')
    main_start_time = time.time()

"""
For all the datasets in SOURCE_DATASETS, project to all the datasets in TARGET_DATASETS
Then train classifier for generated and real data and save results
"""
for a_source_dataset in SOURCE_DATASETS:

    modality1, bins1, behavior1 = find_modality_bin_behavior(A_PATH1, a_source_dataset)
    interaction_data_1, category_labels_1, object_labels_1 = read_dataset(A_PATH1, a_source_dataset)
    num_of_features_1 = interaction_data_1.shape[-1]

    print("Source Robot: ", modality1, bins1, behavior1)
    print("Source Robot: ", interaction_data_1.shape, category_labels_1.shape)

    # Writing log file for execution time
    file = open(LOGS_PATH + 'time_log.txt', 'a')  # append to the file created
    file.write("\n\nSource Robot: " + behavior1 + " " + modality1)
    file.close()

    for a_target_dataset in TARGET_DATASETS:
        
        modality2, bins2, behavior2 = find_modality_bin_behavior(A_PATH2, a_target_dataset)
        interaction_data_2, category_labels_2, object_labels_2 = read_dataset(A_PATH2, a_target_dataset)
        num_of_features_2 = interaction_data_2.shape[-1]

        # Both behaviors cannot be same and if both modalities are same
        if (behavior1 == behavior2) and (sys.argv[1][0] == sys.argv[1][2]):
            continue

        print("Target Robot: ", modality2, bins2, behavior2)
        print("Target Robot: ", interaction_data_2.shape, category_labels_2.shape)
        start_time = time.time()

        a_map_log_path = LOGS_PATH + "_".join([behavior1, modality1, "TO", behavior2, modality2]) + \
                         "_Category_" + CLF_NAME + os.sep

        os.makedirs(a_map_log_path, exist_ok=True)

        with open(a_map_log_path+os.sep+"results.csv", 'w') as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(["S. No", "EDN accuracy (generated features)",
                             "bVEDN accuracy (generated features)",
                             "bVAE accuracy (generated features)",
                             "Target robot accuracy for real features corresponding to generated features",
                             "Train categories", "Test categories"])

        for a_run in range(1, RUNS+1):
            train_cat, test_cat = RANDOM_OBJECTS_DICT[a_run]["train"], RANDOM_OBJECTS_DICT[a_run]["test"]

            print("Object Categories used for Training: ", train_cat)
            print("Object Categories used for Testing: ", test_cat)

            interaction_data_1_train, category_labels_1_train = get_data_label_for_given_labels(train_cat, interaction_data_1, category_labels_1)
            interaction_data_2_train, category_labels_2_train = get_data_label_for_given_labels(train_cat, interaction_data_2, category_labels_2)

            interaction_data_1_test, category_labels_1_test = get_data_label_for_given_labels(test_cat, interaction_data_1, category_labels_1)
            interaction_data_2_test, category_labels_2_test = get_data_label_for_given_labels(test_cat, interaction_data_2, category_labels_2)

            a_map_run_log_path = a_map_log_path+os.sep+str(a_run)
            os.makedirs(a_map_run_log_path, exist_ok=True)

            # Repeat trials for both robots to map each trial of the source to all trials of the target
            interaction_data_1_train_repeat, interaction_data_2_train_repeat = repeat_trials(interaction_data_1_train, interaction_data_2_train)

            # Implement the network
            tf.reset_default_graph()
            edn = EncoderDecoderNetwork(input_channels=num_of_features_1,
                                        output_channels=num_of_features_2,
                                        hidden_layer_sizes=HIDDEN_LAYER_UNITS,
                                        n_dims_code=CODE_VECTOR,
                                        learning_rate=LEARNING_RATE,
                                        activation_fn=ACTIVATION_FUNCTION)
            
            # Train the network
            cost_log = edn.train_session(interaction_data_1_train_repeat, interaction_data_2_train_repeat, None)  # Repeat trials
            plot_loss_curve(cost_log, a_map_run_log_path, title_name_end="_Loss_EDN.png", xlabel='Training Iterations', ylabel='Loss')
            save_cost_csv(cost_log, a_map_run_log_path, csv_name_end="_Loss_EDN.csv")

            # Generate features using trained network
            generated_dataset = edn.generate(interaction_data_1_test)
            generated_dataset = np.array(generated_dataset)
            generated_dataset = reshape_data_setting1(NUM_OF_CATEGORY_FOR_TESTING, generated_dataset)

            # Test data loss
            test_loss = edn.rmse_loss(generated_dataset, interaction_data_2_test)
            with open(a_map_run_log_path + os.sep + "test_loss_EDN.csv", 'w') as f:
                writer = csv.writer(f, lineterminator="\n")
                writer.writerow(["Test Loss", test_loss])

            # Training on generated data and testing on real data
            generated_acc_EDN = object_based_5_fold_cross_validation(clf=CLF, data_train=generated_dataset,
                                                                 data_test=interaction_data_2_test,
                                                                 labels=category_labels_2_test,
                                                                 num_of_features=num_of_features_2)
            # If the target robot actually interacts
            # Training and testing on real data
            actual_acc = object_based_5_fold_cross_validation(clf=CLF, data_train=interaction_data_2_test,
                                                              data_test=interaction_data_2_test,
                                                              labels=category_labels_2_test,
                                                              num_of_features=num_of_features_2)

            tf.reset_default_graph()
            edn = EncoderDecoderNetwork_b_VEDN(input_channels=num_of_features_1,
                            output_channels=num_of_features_2,
                            beta=BETA,
                            hidden_layer_sizes=HIDDEN_LAYER_UNITS,
                            n_dims_code=CODE_VECTOR,
                            learning_rate=LEARNING_RATE,
                            activation_fn=ACTIVATION_FUNCTION,
                            training_epochs=TRAINING_EPOCHS)

            # Train the network
            cost_log = edn.train_session(interaction_data_1_train_repeat, interaction_data_2_train_repeat, None)  # Repeat trials
            plot_loss_curve(cost_log, a_map_run_log_path, title_name_end="_Loss_bVEDN.png", xlabel='Training Iterations', ylabel='Loss')
            save_cost_csv(cost_log, a_map_run_log_path, csv_name_end="_Loss_bVEDN.csv")

            # Generate features using trained network
            generated_dataset = edn.generate(interaction_data_1_test)
            generated_dataset = np.array(generated_dataset)
            generated_dataset = reshape_data_setting1(NUM_OF_CATEGORY_FOR_TESTING, generated_dataset)

            # Test data loss
            test_loss = edn.rmse_loss(generated_dataset, interaction_data_2_test)
            with open(a_map_run_log_path + os.sep + "test_loss_bVEDN.csv", 'w') as f:
                writer = csv.writer(f, lineterminator="\n")
                writer.writerow(["Test Loss", test_loss])

            # Training on generated data and testing on real data
            generated_acc_bVEDN = object_based_5_fold_cross_validation(clf=CLF, data_train=generated_dataset,
                                                                 data_test=interaction_data_2_test,
                                                                 labels=category_labels_2_test,
                                                                 num_of_features=num_of_features_2)

            num_of_domains = 2
            num_of_features = [num_of_features_1, num_of_features_2]
            domain_names = [behavior1+"_"+modality1, behavior2+"_"+modality2]
            tf.reset_default_graph()
            edn = EncoderDecoderNetwork_b_VAE(num_of_domains=num_of_domains,
                                  num_of_features=num_of_features,
                                  domain_names=domain_names,
                                  activation_fn=ACTIVATION_FUNCTION,
                                  beta=BETA,
                                  hidden_layer_sizes=HIDDEN_LAYER_UNITS,
                                  learning_rate=LEARNING_RATE,
                                  training_epochs=TRAINING_EPOCHS
                                 )

            domains_data_train = {}
            domains_label_train = {}
            domains_data_train['domain_'+str(0)] = interaction_data_1_train.reshape(-1, num_of_features[0])
            domains_label_train['domain_'+str(0)] = category_labels_1_train.reshape(-1, 1)
            domains_data_train['domain_'+str(1)] = interaction_data_2_train.reshape(-1, num_of_features[1])
            domains_label_train['domain_'+str(1)] = category_labels_2_train.reshape(-1, 1)

            # Train the network
            cost_log = edn.train_session(domains_data_train, None)
            plot_loss_curve(cost_log, a_map_run_log_path, title_name_end="_Loss_bVAE.png", xlabel='Training Iterations', ylabel='Loss')
            save_cost_csv(cost_log, a_map_run_log_path, csv_name_end="_Loss_bVAE.csv")

            # Generate features using trained network
            domains_data_test = {}
            domains_label_test = {}
            domains_data_test['domain_'+str(0)] = interaction_data_1_test.reshape(-1, num_of_features[0])
            domains_label_test['domain_'+str(0)] = category_labels_1_test.reshape(-1, 1)
            domains_data_test['domain_'+str(1)] = np.zeros(interaction_data_2_test.reshape(-1, num_of_features[1]).shape)
            domains_label_test['domain_'+str(1)] = category_labels_2_test.reshape(-1, 1)

            feed_dict_test = get_feed_dict(edn.placeholder, domains_data_test, num_of_domains)

            domain_num = num_of_domains-1 # The last domain is the target domain
            generated_dataset = edn.sess.run(edn.placeholder["prediction"][domain_num], feed_dict=feed_dict_test)
            generated_dataset = np.array(generated_dataset)
            if np.any(np.isnan(generated_dataset)) == True:
                print("NaN: ", np.any(np.isnan(generated_dataset)))
                generated_dataset = np.nan_to_num(generated_dataset)
            generated_dataset = reshape_data_setting1(NUM_OF_CATEGORY_FOR_TESTING, generated_dataset)

            # Test data loss
            test_loss = edn.rmse_loss(generated_dataset, interaction_data_2_test)
            with open(a_map_run_log_path + os.sep + "test_loss_bVAE.csv", 'w') as f:
                writer = csv.writer(f, lineterminator="\n")
                writer.writerow(["Test Loss", test_loss])

            # Training on generated data and testing on real data
            generated_acc_bVAE = object_based_5_fold_cross_validation(clf=CLF, data_train=generated_dataset,
                                                                 data_test=interaction_data_2_test,
                                                                 labels=category_labels_2_test,
                                                                 num_of_features=num_of_features_2)

            # Writing results of the run
            with open(a_map_log_path+os.sep+"results.csv", 'a') as f:  # append to the file created
                writer = csv.writer(f, lineterminator="\n")
                writer.writerow([a_run, generated_acc_EDN, generated_acc_bVEDN, generated_acc_bVAE,
                                actual_acc, ' '.join(str(e) for e in train_cat),
                                 ' '.join(str(e) for e in test_cat)])

        print(str(RUNS)+" runs completed :)")

        # Writing log file for execution time
        file = open(LOGS_PATH + 'time_log.txt', 'a')  # append to the file created
        end_time = time.time()
        file.write("\nTarget Robot: " + behavior2+" "+modality2)
        file.write("\nTime: " + time_taken(start_time, end_time))
        file.write("\nTotal Time: " + time_taken(main_start_time, end_time))
        file.close()

        # Writing overall results
        my_data = genfromtxt(a_map_log_path+os.sep+"results.csv", delimiter=',')
        my_data = my_data[1:]
        a_list = []
        b_list = []
        a_list.append("Mean Accuracy")
        b_list.append("Standard Deviation")
        A = my_data[:, 1]
        B = my_data[:, 2]
        C = my_data[:, 3]
        D = my_data[:, 4]
        a_list.extend([np.mean(A), np.mean(B), np.mean(C), np.mean(D)])
        b_list.extend([np.std(A), np.std(B), np.std(C), np.std(D)])
        with open(a_map_log_path+os.sep+"results.csv", 'a') as f:  # append to the file created
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(a_list)
            writer.writerow(b_list)

        # Create lists for the plot
        materials = ['Truth Features', 'EDN Features', 'bVEDN Features', 'bVAE Features']
        x_pos = np.arange(len(materials))
        means = [np.mean(D), np.mean(A), np.mean(B), np.mean(C)]
        stds = [np.std(D), np.std(A), np.std(B), np.std(C)]
        title = behavior1+" "+modality1+" to "+behavior2+" "+modality2+" Category Recognition ("+CLF_NAME+")"

        # Build the plot
        fig, ax = plt.subplots()
        ax.bar(x_pos, means, yerr=stds, align='center', alpha=0.5, ecolor='black', capsize=10)
        ax.set_ylim(0, 1)
        ax.set_ylabel('% Recognition Accuracy')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(materials)
        ax.set_title(title)
        ax.yaxis.grid(True)

        # Save the figure and show
        plt.tight_layout()
        plt.savefig(a_map_log_path+os.sep+"bar_graph.png", bbox_inches='tight', dpi=100)
        plt.close()
