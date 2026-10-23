import numpy as np

from symmetry_dict import get_symmetry_orientations, map_hkl_to_unique_hkl
from gixd_utility import LatticeSystem

class GixdPrediction():
    def __init__(self, sample_idx:int, true_top_layer:tuple,
                 true_cell_deg:np.ndarray, true_lattice_system:LatticeSystem):
        self.sample_idx = sample_idx
        self.first_predictions = dict()
        self.second_predictions = dict()
        self.has_true_values = False
        self.predicted_top_layer = None
        if true_cell_deg is not None or true_top_layer is not None or\
          true_lattice_system is not None:
            self.add_true_values(true_top_layer, true_cell_deg, true_lattice_system)
        self.uncertainty = {}


    def add_first_prediction(self, trial_top_layer, first_predicted_cell_deg,
                       first_loss, first_num_matches=None,
                       plot_dict=None, first_num_sim_peaks=0,
                       num_measured_peaks=0):
        self.first_predictions[trial_top_layer]=(first_predicted_cell_deg,
                                       first_loss, first_num_matches,
                                       plot_dict, first_num_sim_peaks,
                                       num_measured_peaks)
        
    def add_uncertainty(self, trial_top_layer, uncertainty):
        self.uncertainty[trial_top_layer] = uncertainty
        

    def add_second_prediction(self, trial_top_layer, second_predicted_cell_deg,
                              second_loss, second_num_matches=None, plot_dict=None,
                              second_num_sim_peaks=0,num_measured_peaks=0, is_converged=True,
                              best_top_layer=None):
        if trial_top_layer in self.second_predictions.keys():
            raise RuntimeError("Can not add another second prediction for same trial_top_layer")
        self.second_predictions[trial_top_layer] = (second_predicted_cell_deg,
                                        second_loss, second_num_matches,
                                        plot_dict, second_num_sim_peaks,
                                        num_measured_peaks, is_converged,
                                        best_top_layer)

    def add_true_values(self, true_top_layer, true_cell_deg,
                        true_lattice_system:LatticeSystem):
        self.true_top_layer = true_top_layer
        self.true_cell_deg = true_cell_deg
        self.true_lattice_system = true_lattice_system
        if self.true_top_layer is not None and\
            self.true_cell_deg is not None and\
            self.true_lattice_system is not None:
                self.has_true_values = True

        if self.true_top_layer is not None:
            #print(f"lattice_system is of type {type(self.true_lattice_system)}")
            lattice_system_str = self.true_lattice_system.name
            if lattice_system_str == "monoclinic":
                if np.abs(self.true_cell_deg[3]-90.0)>0.001:
                    lattice_system_str = "monoclinic_alpha_not_90"
                elif np.abs(self.true_cell_deg[4]-90.0)>0.001:
                    lattice_system_str = "monoclinic_beta_not_90"
                else:
                    lattice_system_str = "monoclinic_gamma_not_90"
            elif lattice_system_str == "tetragonal" or lattice_system_str == "hexagonal":
                if np.abs(self.true_cell_deg[0]-self.true_cell_deg[1]) < 0.001:
                    lattice_system_str += "_a_is_b"
                else:
                    lattice_system_str += "_b_is_c"
            unique_true_top_layer = map_hkl_to_unique_hkl(lattice_system_str,
                                                          self.true_top_layer)
            self.correct_top_layers = get_symmetry_orientations(lattice_system_str,
                                                                unique_true_top_layer)

    def get_converged_accuracy_normalization(self, max_second_loss):
        best_second_loss = float("inf")
        is_converged = False
        best_trial_top_layer = None

        best_true_niggli_second_loss = float("inf")
        best_true_niggli_trial_top_layer = None

        acc_norm = 0.0
        accuracy = 0.0
        converged = 0.0
        conv_norm = 1.0

        assert self.has_true_values, "Convergence and accuracy can only be evaluated if true values are provided"

        #...[7] = niggli_fulfilled
        #...[6] = is_converged


        for trial_top_layer in self.second_predictions:
            if self.second_predictions[trial_top_layer][1] < best_second_loss and\
                self.second_predictions[trial_top_layer][6] == True:
                best_second_loss = self.second_predictions[trial_top_layer][1]
                best_trial_top_layer = trial_top_layer

        #here consider only solutions which already fulfilled the niggli criterion
        for trial_top_layer in self.second_predictions:
            if self.second_predictions[trial_top_layer][1] < best_true_niggli_second_loss and\
                self.second_predictions[trial_top_layer][6] == True and\
                self.second_predictions[trial_top_layer][7] == True:
                best_true_niggli_second_loss = self.second_predictions[trial_top_layer][1]
                best_true_niggli_trial_top_layer = trial_top_layer

        if best_true_niggli_second_loss < best_second_loss + 0.001:
            self.predicted_top_layer = best_true_niggli_trial_top_layer
            #if the niggli correct solution is very close to the one which does not fulfill
            #the niggli criterion then select the one which fulfills niggli criterion
        else:
            self.predicted_top_layer = best_trial_top_layer

        if best_second_loss < max_second_loss:
            is_converged = True
            acc_norm = 1.0

        if is_converged and self.predicted_top_layer in self.correct_top_layers:
            accuracy = 1.0

        converged = 1.0 if is_converged else 0.0

        return accuracy, acc_norm, converged, conv_norm


    def get_confusion_matrix_normalization(self, max_second_loss, used_miller_indices:list):
        best_second_loss = float("inf")
        best_trial_top_layer = None

        confusion_matrix = np.zeros((len(used_miller_indices),
                                     len(used_miller_indices)), float)

        assert self.has_true_values, "Convergence and accuracy can only be evaluated if true values are provided"


        for trial_top_layer in self.second_predictions:
            if self.second_predictions[trial_top_layer][1] < best_second_loss and\
                self.second_predictions[trial_top_layer][6] == True:
                best_second_loss = self.second_predictions[trial_top_layer][1]
                best_trial_top_layer = trial_top_layer

        if best_second_loss < max_second_loss:
            if best_trial_top_layer in self.correct_top_layers:
                lattice_system_str = self.true_lattice_system.name
                if lattice_system_str == "monoclinic":
                    if np.abs(self.true_cell_deg[3]-90.0)>0.001:
                        lattice_system_str = "monoclinic_alpha_not_90"
                    elif np.abs(self.true_cell_deg[4]-90.0)>0.001:
                        lattice_system_str = "monoclinic_beta_not_90"
                    else:
                        lattice_system_str = "monoclinic_gamma_not_90"
                elif lattice_system_str == "tetragonal" or lattice_system_str == "hexagonal":
                    if np.abs(self.true_cell_deg[0]-self.true_cell_deg[1]) < 0.001:
                        lattice_system_str += "_a_is_b"
                    else:
                        lattice_system_str += "_b_is_c"
                best_unique_layer = map_hkl_to_unique_hkl(lattice_system_str,
                                                        best_trial_top_layer)
                equivalent_best_layers = get_symmetry_orientations(lattice_system_str,
                                                                    best_unique_layer)
                for top_layer in equivalent_best_layers:
                    top_layer_idx = used_miller_indices.index(top_layer)
                    confusion_matrix[top_layer_idx, top_layer_idx] = 1/len(equivalent_best_layers)
            else:
                best_top_layer_idx = used_miller_indices.index(best_trial_top_layer)
                true_top_layer_idx = used_miller_indices.index(self.true_top_layer)
                confusion_matrix[true_top_layer_idx, best_top_layer_idx] = 1.0

        return confusion_matrix

    def __str__(self):
        repr = ["                                     \n"]
        if self.has_true_values:
            repr.append("True values:\n")
            repr.append(f"\tCell = {self.true_cell_deg}\n")
            repr.append(f"\tLattice System = {self.true_lattice_system}\n")
            repr.append(f"\tTop layer HKL = {self.true_top_layer}\n")

        if len(self.second_predictions) > 0:
            repr.append("Prediction\n")
            for second_pred_hkl in self.second_predictions:
                repr.append(f"  HKL = {second_pred_hkl}\n")
                if True and self.first_predictions.get(second_pred_hkl) is not None:
                    repr.append(f"\tFirst:")
                    repr.append(f"\tCell = {self.first_predictions[second_pred_hkl][0]}\n")
                    repr.append(f"\tLoss = {self.first_predictions[second_pred_hkl][1]}\n")
                    repr.append(f"\t#matches = {self.first_predictions[second_pred_hkl][2]}\n\n")
                repr.append(f"\tSecond:")
                repr.append(f"\tBest HKL = {self.second_predictions[second_pred_hkl][7]}\n")
                repr.append(f"\tCell = {self.second_predictions[second_pred_hkl][0]}\n")
                if hasattr(self, "uncertainty") and  self.uncertainty.get(second_pred_hkl) is not None:
                    repr.append(f"\tError = {self.uncertainty[second_pred_hkl]}\n")
                repr.append(f"\tLoss = {self.second_predictions[second_pred_hkl][1]}\n")
                repr.append(f"\t#matches = {self.second_predictions[second_pred_hkl][2]}\n")
                repr.append(f"\t#sim peaks = {self.second_predictions[second_pred_hkl][4]}\n")
                repr.append(f"\t#measured peaks = {self.second_predictions[second_pred_hkl][5]}\n\n")

        repr.append("                                     ")
        return ''.join(repr)

        

        