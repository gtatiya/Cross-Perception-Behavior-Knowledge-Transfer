import sys
import csv
import glob
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from numpy import genfromtxt
import matplotlib.pyplot as plt

from utils import find_modality_bin_behavior, read_dataset, get_data_label_for_given_labels, reshape_data_setting1, \
    object_based_5_fold_cross_validation, repeat_trials, time_taken
from model import EncoderDecoderNetwork_b_VAE

from constant import *

tf.set_random_seed(1)

"""
2 Source Robots and 1 Target Robot

Two to one projections:
A, H, S, V
AS2H, AS2V, HH2H, HV2S

python 2_sources_b-vae.py HH2H
"""

if len(sys.argv) != 2:
    print("Pass one of 1st arguments: HH2H")
    print("For example: python 2_sources_b-vae.py HH2H")
    exit()

LOGS_PATH = r".." + os.sep + "Cross-Perception-Behavior-Knowledge-Transfer_" + sys.argv[1] + os.sep
os.makedirs(LOGS_PATH, exist_ok=True)

# Source Robot data
A_PATH1a = r"Datasets" + os.sep + DATASETS_FOLDERS[sys.argv[1][0]][0]
SOURCE_DATASETS1a = DATASETS_FOLDERS[sys.argv[1][0]][1]
A_PATH1b = r"Datasets" + os.sep + DATASETS_FOLDERS[sys.argv[1][1]][0]
SOURCE_DATASETS1b = DATASETS_FOLDERS[sys.argv[1][1]][1]
# Target Robot data
A_PATH2 = r"Datasets" + os.sep + DATASETS_FOLDERS[sys.argv[1][3]][0]
TARGET_DATASETS = DATASETS_FOLDERS[sys.argv[1][3]][1]


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
    title_name = " ".join([behavior1a+"."+behavior1b, modality1a+"."+modality1b, "TO", behavior2, modality2])
    plt.title(title_name)
    title_name_ = "_".join([behavior1a+"."+behavior1b, modality1a+"."+modality1b, "TO", behavior2, modality2])+title_name_end
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

    csv_name = "_".join([behavior1a+"."+behavior1b, modality1a+"."+modality1b, "TO", behavior2, modality2])+csv_name_end

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
df = pd.DataFrame(columns = ['robot1a', 'robot1b', 'robot2'])

for a_source_dataset in SOURCE_DATASETS1a:

    modality1a, bins1a, behavior1a = find_modality_bin_behavior(A_PATH1a, a_source_dataset)
    interaction_data_1a, category_labels_1a, object_labels_1a = read_dataset(A_PATH1a, a_source_dataset)
    num_of_features_1a = interaction_data_1a.shape[-1]

    print("Source Robot 1: ", modality1a, bins1a, behavior1a)
    print("Source Robot 1: ", interaction_data_1a.shape, category_labels_1a.shape)

    # Writing log file for execution time
    file = open(LOGS_PATH + 'time_log.txt', 'a')  # append to the file created
    file.write("\n\nSource Robot 1: " + behavior1a + " " + modality1a)
    file.close()

    for a_source_dataset in SOURCE_DATASETS1b:

        modality1b, bins1b, behavior1b = find_modality_bin_behavior(A_PATH1b, a_source_dataset)
        interaction_data_1b, category_labels_1b, object_labels_1b = read_dataset(A_PATH1b, a_source_dataset)
        num_of_features_1b = interaction_data_1b.shape[-1]

        # Both behaviors and modalities cannot be same
        if (behavior1a == behavior1b) and (modality1a == modality1b):
            continue

        for a_target_dataset in TARGET_DATASETS:
            
            modality2, bins2, behavior2 = find_modality_bin_behavior(A_PATH2, a_target_dataset)
            interaction_data_2, category_labels_2, object_labels_2 = read_dataset(A_PATH2, a_target_dataset)
            num_of_features_2 = interaction_data_2.shape[-1]

            # Target robot behavior and modality1 cannot be same as source robot 1
            if (behavior2 == behavior1a) and (modality2 == modality1a):
                continue
            # Target robot behavior and modality1 cannot be same as source robot 2
            if (behavior2 == behavior1b) and (modality2 == modality1b):
                continue

            # Skipping projection if already exists
            df_temp = df.loc[df['robot2'] == behavior2+"_"+modality2]
            projections_exists = False
            for index, row in df_temp.iterrows():                
                if row.robot1a == behavior1b+"_"+modality1b and row.robot1b == behavior1a+"_"+modality1a:
                    projections_exists = True            
            if projections_exists:
                continue
            df = df.append({'robot1a':behavior1a+"_"+modality1a, 'robot1b':behavior1b+"_"+modality1b, 'robot2':behavior2+"_"+modality2}, ignore_index=True)

            print("Source Robot 2: ", modality1b, bins1b, behavior1b)
            print("Source Robot 2: ", interaction_data_1b.shape, category_labels_1b.shape)

            # Writing log file for execution time
            file = open(LOGS_PATH + 'time_log.txt', 'a')  # append to the file created
            file.write("\n\nSource Robot 2: " + behavior1b + " " + modality1b)
            file.close()

            print("Target Robot: ", modality2, bins2, behavior2)
            print("Target Robot: ", interaction_data_2.shape, category_labels_2.shape)
            start_time = time.time()

            a_map_log_path = LOGS_PATH + "_".join([behavior1a+"."+behavior1b, modality1a+"."+modality1b, "TO", behavior2, modality2]) + \
                             "_Category_" + CLF_NAME + os.sep
            os.makedirs(a_map_log_path, exist_ok=True)

            with open(a_map_log_path+os.sep+"results.csv", 'w') as f:
                writer = csv.writer(f, lineterminator="\n")
                writer.writerow(["S. No", "Target robot accuracy for only generated features",
                                 "Target robot accuracy for real features corresponding to generated features",
                                 "Train categories", "Test categories"])

            for a_run in range(1, RUNS+1):
                train_cat, test_cat = RANDOM_OBJECTS_DICT[a_run-1]["train"], RANDOM_OBJECTS_DICT[a_run-1]["test"]

                print("Object Categories used for Training: ", train_cat)
                print("Object Categories used for Testing: ", test_cat)

                interaction_data_1a_train, category_labels_1a_train = get_data_label_for_given_labels(train_cat, interaction_data_1a, category_labels_1a)
                interaction_data_1b_train, category_labels_1b_train = get_data_label_for_given_labels(train_cat, interaction_data_1b, category_labels_1b)
                interaction_data_2_train, category_labels_2_train = get_data_label_for_given_labels(train_cat, interaction_data_2, category_labels_2)

                interaction_data_1a_test, category_labels_1a_test = get_data_label_for_given_labels(test_cat, interaction_data_1a, category_labels_1a)
                interaction_data_1b_test, category_labels_1b_test = get_data_label_for_given_labels(test_cat, interaction_data_1b, category_labels_1b)
                interaction_data_2_test, category_labels_2_test = get_data_label_for_given_labels(test_cat, interaction_data_2, category_labels_2)

                a_map_run_log_path = a_map_log_path+os.sep+str(a_run)
                os.makedirs(a_map_run_log_path, exist_ok=True)

                tf.reset_default_graph()
                # Implement the network
                num_of_domains = 3
                num_of_features = [num_of_features_1a, num_of_features_1b, num_of_features_2]
                domain_names = [behavior1a+"_"+modality1a, behavior1b+"_"+modality1b, behavior2+"_"+modality2]
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
                domains_data_train['domain_'+str(0)] = interaction_data_1a_train.reshape(-1, num_of_features[0])
                domains_label_train['domain_'+str(0)] = category_labels_1a_train.reshape(-1, 1)
                domains_data_train['domain_'+str(1)] = interaction_data_1b_train.reshape(-1, num_of_features[1])
                domains_label_train['domain_'+str(1)] = category_labels_1b_train.reshape(-1, 1)
                domains_data_train['domain_'+str(2)] = interaction_data_2_train.reshape(-1, num_of_features[2])
                domains_label_train['domain_'+str(2)] = category_labels_2_train.reshape(-1, 1)

                # Train the network
                cost_log = edn.train_session(domains_data_train, None)
                plot_loss_curve(cost_log, a_map_run_log_path, title_name_end="_Loss.png", xlabel='Training Iterations', ylabel='Loss')
                save_cost_csv(cost_log, a_map_run_log_path, csv_name_end="_Loss.csv")

                # Generate features using trained network
                domains_data_test = {}
                domains_label_test = {}
                domains_data_test['domain_'+str(0)] = interaction_data_1a_test.reshape(-1, num_of_features[0])
                domains_label_test['domain_'+str(0)] = category_labels_1a_test.reshape(-1, 1)
                domains_data_test['domain_'+str(1)] = interaction_data_1b_test.reshape(-1, num_of_features[1])
                domains_label_test['domain_'+str(1)] = category_labels_1b_test.reshape(-1, 1)
                domains_data_test['domain_'+str(2)] = np.zeros(interaction_data_2_test.reshape(-1, num_of_features[2]).shape)
                domains_label_test['domain_'+str(2)] = category_labels_2_test.reshape(-1, 1)

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
                with open(a_map_run_log_path + os.sep + "test_loss.csv", 'w') as f:
                    writer = csv.writer(f, lineterminator="\n")
                    writer.writerow(["Test Loss", test_loss])

                # Training on generated data and testing on real data
                generated_acc = object_based_5_fold_cross_validation(clf=CLF, data_train=generated_dataset,
                                                                     data_test=interaction_data_2_test,
                                                                     labels=category_labels_2_test,
                                                                     num_of_features=num_of_features_2)
                # If the target robot actually interacts
                # Training and testing on real data
                actual_acc = object_based_5_fold_cross_validation(clf=CLF, data_train=interaction_data_2_test,
                                                                  data_test=interaction_data_2_test,
                                                                  labels=category_labels_2_test,
                                                                  num_of_features=num_of_features_2)

                # Writing results of the run
                with open(a_map_log_path+os.sep+"results.csv", 'a') as f:  # append to the file created
                    writer = csv.writer(f, lineterminator="\n")
                    writer.writerow([a_run, generated_acc, actual_acc, ' '.join(str(e) for e in train_cat),
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
            a_list.extend([np.mean(A), np.mean(B)])
            b_list.extend([np.std(A), np.std(B)])
            with open(a_map_log_path+os.sep+"results.csv", 'a') as f:  # append to the file created
                writer = csv.writer(f, lineterminator="\n")
                writer.writerow(a_list)
                writer.writerow(b_list)

            # Plotting average loss on training data
            all_loss = []
            for a_mapping_folder in glob.iglob(a_map_log_path + '/*/', recursive=True):                
                csv_name = "_".join([behavior1a+"."+behavior1b, modality1a+"."+modality1b, "TO", behavior2, modality2]) + "_Loss.csv"
                my_data = genfromtxt(a_mapping_folder + os.sep + csv_name, delimiter=',', usecols=(1))
                my_data = my_data[1:]
                all_loss.append(my_data)
            avg_loss = np.mean(all_loss, axis=0)
            plot_loss_curve(avg_loss, a_map_log_path, title_name_end="_Avg_Loss.png", xlabel='Training Iterations',
                            ylabel='Loss')
            save_cost_csv(avg_loss, a_map_log_path, csv_name_end="_Avg_Loss.csv")

            # Computing average loss on test data
            all_loss = []
            for a_mapping_folder in glob.iglob(a_map_log_path + '/*/', recursive=True):
                my_data = genfromtxt(a_mapping_folder + os.sep + 'test_loss.csv', delimiter=',', usecols=(1))
                all_loss.append(my_data)
            avg_loss = np.mean(all_loss, axis=0)
            with open(a_map_log_path + os.sep + "test_loss.csv", 'w') as f:
                writer = csv.writer(f, lineterminator="\n")
                writer.writerow(["Test Loss", avg_loss])

            # Create lists for the plot
            materials = ['Projected Features', 'Ground Truth Features']
            x_pos = np.arange(len(materials))
            means = [np.mean(A), np.mean(B)]
            stds = [np.std(A), np.std(B)]
            title = behavior1a+"."+behavior1b+" "+modality1a+"."+modality1b+" to "+behavior2+" "+modality2+" Category Recognition ("+CLF_NAME+")"

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
