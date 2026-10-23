import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os
from sklearn.metrics import r2_score
import seaborn as sns

from gixd_utility import *


def plot_confusion_matrix(confusion_matrix):
    # Plotting the confusion matrix as a heatmap
    plt.figure(figsize=(20, 18))  # Adjust the figure size as needed

    # Create a heatmap with annotations
    sns.heatmap(confusion_matrix, annot=True, fmt=".2f",
                cmap="YlGnBu", linewidths=.5, annot_kws={"size": 8})

    # Customize the plot
    plt.title("Confusion Matrix Heatmap", fontsize=20)
    plt.xlabel("Predicted label", fontsize=15)
    plt.ylabel("True label", fontsize=15)

    # Set x and y ticks with rotation for better readability
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)

    # Show the plot
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()
        
def plot_benchmark_classification(hkl_symmetry, num_top_entries=1):
    prediction_df = pd.read_pickle("./prediction_df_classificationuniform")
    prediction_df["true_cell_tuple"] = prediction_df["true_cell"].apply(tuple)
    col_names = prediction_df.columns.tolist()
    #print(col_names)
    used_top_layers = list(prediction_df["target_tl"].unique())
    #print(used_top_layers)
    num_top_layers = len(used_top_layers)
    PRINT_DEBUG = False
    
    #for index, row in prediction_df.iterrows():
    #    print(f'true HKL {row["target_tl"]} try {row["trial_tl"]} first {row["first_loss"]}, second {row["second_loss"]}')
    
    grouped_df = prediction_df.groupby("true_cell_tuple")
    
    #print(len(grouped_df)*len(used_top_layers))
    #print(len(prediction_df))
    
    #print(f"Found {len(grouped_df)} unique crystal cells")
    
    confusion_matrix = np.zeros((num_top_layers, num_top_layers), np.float64)
    normalization_matrix = np.zeros_like(confusion_matrix)
    
    classification_count = 0
    success_count = np.zeros(num_top_entries, int)
    
    for true_cell_, grouped_sample in grouped_df:
        for target_tl in used_top_layers:
            losses = list()
            trial_tls = list()
            for iter, row in grouped_sample.iterrows():
                if target_tl != row["target_tl"]:
                    continue
                lattice_system = row["lattice_system"]
                #print(f"lattice = {lattice_system}")
                unique_target_tl, target_tl_num = hkl_symmetry[LatticeSystem(lattice_system).name][target_tl]
                losses.append(row["second_loss"])
                trial_tls.append(hkl_symmetry[LatticeSystem(lattice_system).name][row["trial_tl"]][0])
            combined = list(zip(losses, trial_tls))
            combined_sorted = sorted(combined, key=lambda x:x[0])
            sorted_losses, sorted_trial_tls = zip(*combined_sorted)
            if PRINT_DEBUG:
                print(f"unique_target_tl = {unique_target_tl}")
                #print(losses)
                #print(trial_tls)
                print(sorted_losses)
                print(sorted_trial_tls)
            # QUICK FIX TO DEAL WITH NON NIGGLI CELLS
            best_loss = sorted_losses[0]
            for k in range(num_top_entries):
                if sorted_trial_tls[k] == unique_target_tl:
                    if sorted_losses[k] < 0.1 and np.abs(sorted_losses[k] - best_loss) < 0.01:
                        success_count[:] += 1
                    else:
                        success_count[k:num_top_entries] += 1
                    break
            confusion_matrix[used_top_layers.index(unique_target_tl),
                             used_top_layers.index(sorted_trial_tls[0])] += 1.0
            normalization_matrix[used_top_layers.index(unique_target_tl),:] += 1.0
            classification_count += 1

    df_cm = pd.DataFrame(confusion_matrix/normalization_matrix,
                         index=used_top_layers, columns=used_top_layers)
    plot_confusion_matrix(df_cm)
    return 100*success_count/classification_count
     
      
def plot_prediction_df_dropout():
    config = GixdConfig()
    
    a_median_list = list()
    b_median_list = list()
    c_median_list = list()
    alpha_median_list = list()
    beta_median_list = list()
    gamma_median_list = list()
    
    a_95_list = list()
    b_95_list = list()
    c_95_list = list()
    alpha_95_list = list()
    beta_95_list = list()
    gamma_95_list = list()
    
    #dropout_rate_list = [0.000, 0.103, 0.113, 0.123, 0.133, 0.144,
    #                     0.154, 0.164, 0.174, 0.185, 0.195, 0.205,
    #                     0.215, 0.226, 0.236, 0.246, 0.256, 0.267,
    #                     0.277, 0.287, 0.297]
    
    dropout_rate_list = [0.0, 0.05, 0.10,
                      0.125, 0.150, 0.175, 0.200,
                      0.215, 0.230, 0.245, 0.260, 0.275, 0.290, 0.305,
                      0.320, 0.335, 0.350, 0.365, 0.380, 0.395]
    
    dropout_rate_list = [0.000, 0.100, 0.150, 0.200,
                      0.250, 0.300, 0.325, 0.350,
                      0.375, 0.400, 0.425, 0.450,
                      0.475, 0.500, 0.525, 0.550]
    
    dropout_rate_plot = list()

    all_r2_scores = list()
    
    for drop_out_rate in dropout_rate_list:
        df_name = "./prediction_df_dropout_test_20_"+str(np.int16(np.round(1000*drop_out_rate)))
        if not os.path.exists(df_name):
            continue
        dropout_rate_plot.append(drop_out_rate)
        prediction_df = pd.read_pickle(df_name)
        prediction_df["true_cell_tuple"] = prediction_df["true_cell"].apply(tuple)
        used_top_layers = list(prediction_df["target_tl"].unique())
        num_top_layers = len(used_top_layers)
        
        grouped_df = prediction_df.groupby("true_cell_tuple")
        
        all_true_cell_normalised = list()
        all_predict_cell_noramlised = list()
        true_cell = list()
        true_top_layer = list()
        first_pred = list()
        first_loss = list()
        first_top_layer = list()
        second_pred = list()
        second_loss = list()
        second_top_layer = list()
        
        first_confusion_matrix = np.zeros((num_top_layers, num_top_layers), np.float64)
        second_confusion_matrix = np.zeros((num_top_layers, num_top_layers), np.float64)
        
        for true_cell_, grouped_sample in grouped_df:
            true_cell.append(true_cell_)
            true_top_layer.append(grouped_sample["target_tl"].values[0])
            
            best_first = grouped_sample.loc[grouped_sample["first_loss"].idxmin()]
            first_pred.append(best_first["first_pred"])
            first_loss.append(best_first["first_loss"])
            first_top_layer.append(best_first["trial_tl"])
            first_confusion_matrix[used_top_layers.index(grouped_sample["target_tl"].values[0]),
                                used_top_layers.index(best_first["trial_tl"])] += 1.0
            
            best_second = grouped_sample.loc[grouped_sample["second_loss"].idxmin()]
            all_predict_cell_noramlised.append(config.normalize_cell(best_second["second_pred"]))
            all_true_cell_normalised.append(config.normalize_cell(true_cell_))
            second_pred.append(best_second["second_pred"])
            second_loss.append(best_second["second_loss"])
            second_top_layer.append(best_second["trial_tl"])
            second_confusion_matrix[used_top_layers.index(grouped_sample["target_tl"].values[0]),
                                used_top_layers.index(best_second["trial_tl"])] += 1.0
            
        
        first_pred = np.array(first_pred)
        true_cell = np.array(true_cell)
        second_pred = np.array(second_pred)

        all_predict_cell_noramlised = np.array(all_predict_cell_noramlised)
        all_true_cell_normalised = np.array(all_true_cell_normalised)

        r2_score_value = r2_score(all_true_cell_normalised, all_predict_cell_noramlised)
        all_r2_scores.append(r2_score_value)
        
        second_mae = np.abs(second_pred-true_cell)


        a_q1, a_median, a_q3, a_95 = np.percentile(second_mae[:, 0], q=[25, 50, 75, 95])
        b_q1, b_median, b_q3, b_95 = np.percentile(second_mae[:, 1], q=[25, 50, 75, 95])
        c_q1, c_median, c_q3, c_95 = np.percentile(second_mae[:, 2], q=[25, 50, 75, 95])

        alpha_q1, alpha_median, alpha_q3, alpha_95 = np.percentile(second_mae[:, 3], q=[25, 50, 75, 95])
        beta_q1, beta_median, beta_q3, beta_95 = np.percentile(second_mae[:, 4], q=[25, 50, 75, 95])
        gamma_q1, gamma_median, gamma_q3, gamma_95 = np.percentile(second_mae[:, 5], q=[25, 50, 75, 95])
        
        a_median_list.append(a_median)
        b_median_list.append(b_median)
        c_median_list.append(c_median)
        alpha_median_list.append(alpha_median)
        beta_median_list.append(beta_median)
        gamma_median_list.append(gamma_median)
        
        a_95_list.append(a_95)
        b_95_list.append(b_95)
        c_95_list.append(c_95)
        alpha_95_list.append(alpha_95)
        beta_95_list.append(beta_95)
        gamma_95_list.append(gamma_95)
        
    fig, axs = plt.subplots(2, 3, figsize=(20,10))

    fig.suptitle("Prediction Error for Different Dropout Rates\nTrained with no dropout")
    
    
    axs[0,0].plot(dropout_rate_plot, a_median_list, ".-r", label="median")
    axs[0,0].plot(dropout_rate_plot, a_95_list, ".-g", label="95 bound")
    axs[0,0].set_ylabel("a MAE / A")
    axs[0,0].set_title("a")
    
    axs[0,1].plot(dropout_rate_plot, b_median_list, ".-r", label="median")
    axs[0,1].plot(dropout_rate_plot, b_95_list, ".-g", label="95 bound")
    axs[0,1].set_ylabel("b MAE / A")
    axs[0,1].set_title("b")
    
    axs[0,2].plot(dropout_rate_plot, c_median_list, ".-r", label="median")
    axs[0,2].plot(dropout_rate_plot, c_95_list, ".-g", label="95 bound")
    axs[0,2].set_ylabel("c MAE / A")
    axs[0,2].set_title("c")
    
    axs[1,0].plot(dropout_rate_plot, alpha_median_list, ".-r", label="median")
    axs[1,0].plot(dropout_rate_plot, alpha_95_list, ".-g", label="95 bound")
    axs[1,0].set_ylabel("alpha MAE / °")
    axs[1,0].set_title("alpha")
    
    axs[1,1].plot(dropout_rate_plot, beta_median_list, ".-r", label="median")
    axs[1,1].plot(dropout_rate_plot, beta_95_list, ".-g", label="95 bound")
    axs[1,1].set_ylabel("beta MAE / °")
    axs[1,1].set_title("beta")
    
    axs[1,2].plot(dropout_rate_plot, gamma_median_list, ".-r", label="median")
    axs[1,2].plot(dropout_rate_plot, gamma_95_list, ".-g", label="95 bound")
    axs[1,2].set_ylabel("gamma MAE / °")
    axs[1,2].set_title("gamma")
    
    
    
    for i in range(2):
        for j in range(3):
            axs[i,j].grid()
            axs[i,j].set_ylim([0,1.0])
            axs[i,j].set_xlabel("dropout")
            axs[i,j].legend(loc="upper left")
    
    plt.savefig("dropout_rate_error_20_trained.png")
    plt.close()

    fig, ax = plt.subplots(1,1,figsize=(10, 7))
    ax.plot(dropout_rate_plot, all_r2_scores, ".-r")
    ax.grid()
    ax.set_ylabel("R2 score")
    ax.set_xlabel("Dropout Rate")
    plt.savefig("dropout_r2_score_trained_20.png")
    plt.close()
   
        
def plot_prediction_df_bravais(data_mode):
    config = GixdConfig()
    bravais_types_plot = list()

    bravais_types = ["triclinic", "orthorhombic", "monoclinic",
                     "tetragonal", "cubic", "rhombohedral",
                     "hexagonal"]
    
    lattice_colors = ["r", "g", "b", "k", "c", "m", "y"]
    fig_vio, axs_vio = plt.subplots(2, 1, figsize=(15,10))

    for lattice_idx, bravais_type in enumerate(bravais_types):
        drop_out_rate = 0.0
        print(f"Dropout rate for testing = {drop_out_rate}")
        print("Bravais type = "+bravais_type)
        df_name = "./prediction_df_"+bravais_type
        if not os.path.exists(df_name):
            continue
        bravais_types_plot.append(bravais_type)
        prediction_df = pd.read_pickle(df_name)
        prediction_df["true_cell_tuple"] = prediction_df["true_cell"].apply(tuple)
        used_top_layers = list(prediction_df["target_tl"].unique())
        num_top_layers = len(used_top_layers)
        
        grouped_df = prediction_df.groupby("true_cell_tuple")
        
        all_true_cell_normalised = list()
        all_predict_cell_noramlised = list()
        true_cell = list()
        true_top_layer = list()
        first_pred = list()
        first_loss = list()
        first_top_layer = list()
        second_pred = list()
        second_loss = list()
        second_top_layer = list()
        
        first_confusion_matrix = np.zeros((num_top_layers, num_top_layers), np.float64)
        second_confusion_matrix = np.zeros((num_top_layers, num_top_layers), np.float64)
        
        for true_cell_, grouped_sample in grouped_df:
            true_cell.append(true_cell_)
            true_top_layer.append(grouped_sample["target_tl"].values[0])
            
            best_first = grouped_sample.loc[grouped_sample["first_loss"].idxmin()]
            first_pred.append(best_first["first_pred"])
            first_loss.append(best_first["first_loss"])
            first_top_layer.append(best_first["trial_tl"])
            first_confusion_matrix[used_top_layers.index(grouped_sample["target_tl"].values[0]),
                                used_top_layers.index(best_first["trial_tl"])] += 1.0
            
            best_second = grouped_sample.loc[grouped_sample["second_loss"].idxmin()]
            all_predict_cell_noramlised.append(config.normalize_cell(best_second["second_pred"]))
            all_true_cell_normalised.append(config.normalize_cell(true_cell_))
            second_pred.append(best_second["second_pred"])
            second_loss.append(best_second["second_loss"])
            second_top_layer.append(best_second["trial_tl"])
            second_confusion_matrix[used_top_layers.index(grouped_sample["target_tl"].values[0]),
                                used_top_layers.index(best_second["trial_tl"])] += 1.0
            
        
        first_pred = np.array(first_pred)
        true_cell = np.array(true_cell)
        second_pred = np.array(second_pred)

        all_predict_cell_noramlised = np.array(all_predict_cell_noramlised)
        all_true_cell_normalised = np.array(all_true_cell_normalised)
        
        second_mae = np.abs(second_pred-true_cell)
        distances_mae = np.log10(np.concatenate([second_mae[:,0],
                                        second_mae[:,1],
                                        second_mae[:,2]]))
        angles_mae = np.log10(np.concatenate([second_mae[:,3],
                                        second_mae[:,4],
                                        second_mae[:,5]]))
        violin_dist = axs_vio[0].violinplot(distances_mae, positions=[lattice_idx*1.0],
                                            showextrema=False, showmeans=False,
                                            showmedians=False,
                                            points = 1000)
        violin_angle = axs_vio[1].violinplot(angles_mae, positions=[lattice_idx*1.0],
                                            showextrema=False, showmeans=False,
                                            showmedians=False,
                                            points = 1000)
        
        for part in violin_dist["bodies"]:
            part.set_facecolor(lattice_colors[lattice_idx])
            part.set_edgecolor("black")
            part.set_alpha(0.7)
        for part in violin_angle["bodies"]:
            part.set_facecolor(lattice_colors[lattice_idx])
            part.set_edgecolor("black")
            part.set_alpha(0.7)
            
    axs_vio[0].set_title("log(MAE) for Distances")
    axs_vio[0].set_xticks(np.arange(len(bravais_types)), labels=bravais_types)
    axs_vio[1].set_title("log(MAE) for Angles")
    axs_vio[1].set_xticks(np.arange(len(bravais_types)), labels=bravais_types)
    axs_vio[0].set_ylim((-8, 2))
    axs_vio[1].set_ylim((-8, 2))
    
    plt.savefig("mae_violinplot_lattice_"+data_mode+".png")


def plot_prediction_direct_inversion(data_mode):
    config = GixdConfig()

    direct_inv_q_tols = [0.03, 0.06, 0.09, 0.12, 0.15, 0.2, 0.3, 0.4, 0.5]
    
    lattice_colors = ["r", "g", "b", "k", "c", "m", "y", "r", "g", "b"]
    fig_vio, axs_vio = plt.subplots(2, 1, figsize=(15,10))

    for lattice_idx, direct_inv_q_tol in enumerate(direct_inv_q_tols):
        df_name = "./benchmark_df_q_cutoff_"+str(direct_inv_q_tol)
        if not os.path.exists(df_name):
            continue
        prediction_df = pd.read_pickle(df_name)
        prediction_df["true_cell_tuple"] = prediction_df["true_cell"].apply(tuple)
        used_top_layers = list(prediction_df["target_tl"].unique())
        num_top_layers = len(used_top_layers)
        
        grouped_df = prediction_df.groupby("true_cell_tuple")
        
        all_true_cell_normalised = list()
        all_predict_cell_noramlised = list()
        true_cell = list()
        true_top_layer = list()
        first_pred = list()
        first_loss = list()
        first_top_layer = list()
        second_pred = list()
        second_loss = list()
        second_top_layer = list()
        
        first_confusion_matrix = np.zeros((num_top_layers, num_top_layers), np.float64)
        second_confusion_matrix = np.zeros((num_top_layers, num_top_layers), np.float64)
        
        for true_cell_, grouped_sample in grouped_df:
            true_cell.append(true_cell_)
            true_top_layer.append(grouped_sample["target_tl"].values[0])
            
            best_first = grouped_sample.loc[grouped_sample["first_loss"].idxmin()]
            first_pred.append(best_first["first_pred"])
            first_loss.append(best_first["first_loss"])
            first_top_layer.append(best_first["trial_tl"])
            first_confusion_matrix[used_top_layers.index(grouped_sample["target_tl"].values[0]),
                                used_top_layers.index(best_first["trial_tl"])] += 1.0
            
            best_second = grouped_sample.loc[grouped_sample["second_loss"].idxmin()]
            all_predict_cell_noramlised.append(config.normalize_cell(best_second["second_pred"]))
            all_true_cell_normalised.append(config.normalize_cell(true_cell_))
            second_pred.append(best_second["second_pred"])
            second_loss.append(best_second["second_loss"])
            second_top_layer.append(best_second["trial_tl"])
            second_confusion_matrix[used_top_layers.index(grouped_sample["target_tl"].values[0]),
                                used_top_layers.index(best_second["trial_tl"])] += 1.0
            
        
        first_pred = np.array(first_pred)
        true_cell = np.array(true_cell)
        second_pred = np.array(second_pred)

        all_predict_cell_noramlised = np.array(all_predict_cell_noramlised)
        all_true_cell_normalised = np.array(all_true_cell_normalised)
        
        second_mae = np.abs(second_pred-true_cell)
        distances_mae = np.log10(np.concatenate([second_mae[:,0],
                                        second_mae[:,1],
                                        second_mae[:,2]]))
        angles_mae = np.log10(np.concatenate([second_mae[:,3],
                                        second_mae[:,4],
                                        second_mae[:,5]]))
        violin_dist = axs_vio[0].violinplot(distances_mae, positions=[lattice_idx*1.0],
                                            showextrema=False, showmeans=False,
                                            showmedians=False,
                                            points = 1000)
        violin_angle = axs_vio[1].violinplot(angles_mae, positions=[lattice_idx*1.0],
                                            showextrema=False, showmeans=False,
                                            showmedians=False,
                                            points = 1000)
        
        print(f"q_cutoff = {direct_inv_q_tol}")
        print(f"Distances MAE = {np.mean(np.mean(second_mae[:,:3],1))}")
        print(f"Angles MAE = {np.mean(np.mean(second_mae[:,3:],1))}")
        
        for part in violin_dist["bodies"]:
            part.set_facecolor(lattice_colors[lattice_idx])
            part.set_edgecolor("black")
            part.set_alpha(0.7)
        for part in violin_angle["bodies"]:
            part.set_facecolor(lattice_colors[lattice_idx])
            part.set_edgecolor("black")
            part.set_alpha(0.7)
            
    axs_vio[0].set_title("log(MAE) for Distances")
    axs_vio[0].set_xticks(np.arange(len(direct_inv_q_tols)), labels=direct_inv_q_tols)
    axs_vio[1].set_title("log(MAE) for Angles")
    axs_vio[1].set_xticks(np.arange(len(direct_inv_q_tols)), labels=direct_inv_q_tols)
    axs_vio[0].set_ylim((-8, 2))
    axs_vio[1].set_ylim((-8, 2))
    
    plt.savefig("mae_violinplot_q_cutoff_"+data_mode+".png")