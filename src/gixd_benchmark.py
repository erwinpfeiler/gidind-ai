import numpy as np
import torch
from torch import Tensor
import os
from tqdm import trange, tqdm
import pandas as pd
import heapq

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from corner import corner
from itertools import combinations
from sklearn.cluster import AffinityPropagation



from gixd_data_generator import GixdDataGenerator, create_data_generators
from gixd_utility import *
from gixd_regressor import *
from config_loader import *
from polishing_fit import *

class GixdBenchmark():
    def __init__(self, data_mode:str, num_phases:int=1, path:str="none"):
        self.data_mode = data_mode
        self.num_phases = num_phases
        self.training_data_generators, self.validation_data_generators, self.test_data_generators = create_data_generators(
                "uniform", ["cell", "top_layer"], num_phases=self.num_phases)
        self.regressor = GixdRegressor(training_data_generators=self.training_data_generators,
                                validation_data_generators=self.validation_data_generators,
                                test_data_generators=self.test_data_generators,
                                num_phases=self.num_phases)
        self.path = path
        if self.path != "none":
            if not os.path.exists(self.path):
                os.mkdir(self.path)
                if not os.path.exists(self.path): raise RuntimeError(f"Directory {self.path} could not be created")
                print(f"[Benchmark]: Created directory for benchmark.")

        self.missing_test_peaks = 0
        self.extra_test_peaks = 0

        self.data = []
        self.ann_worst_cases = []
        self.ann_best_cases = []

        self.feature_config = FeatureConfig()
        self.target_config = TargetConfig()
        self.machine_learning_parameters = None
        #self.columns = ["cell_target_rad", "images", "lattice_system_target",
        #                "q_xy", "q_z", "top_layer_indices",
        #                "target_a", "target_b", "target_c", "target_alpha_deg", "target_beta_deg", "target_gamma_deg"]

    def save_data(self, file_name:str="data.pd"):
        data_frame = pd.DataFrame(self.data)
        data_frame.to_pickle(os.path.join(self.path, file_name))
        print(f"[Benchmark]: Data stored to {os.path.join(self.path, file_name)}")

    def load_data(self, file_name:str="data.pd"):
        data_frame = pd.read_pickle(os.path.join(self.path, file_name))
        self.data = data_frame.to_dict(orient="records")
        print(f"[Benchmark]: Data loaded from {os.path.join(self.path, file_name)}")



    def add_target_variables(self, label:str="target"):
        for entry in self.data:
            cell_rad = entry.get(label+'_cell_rad')
            if cell_rad is not None:
                entry[label+'_a'] = cell_rad[0]
                entry[label+'_b'] = cell_rad[1]
                entry[label+'_c'] = cell_rad[2]
                entry[label+'_alpha_deg'] = np.rad2deg(cell_rad[3])
                entry[label+'_beta_deg'] = np.rad2deg(cell_rad[4])
                entry[label+'_gamma_deg'] = np.rad2deg(cell_rad[5])


    def add_number_of_peaks(self):
        for entry in self.data:
            images = entry.get('images')
            if images is not None:
                entry['number_of_peaks'] = np.int32(np.sum(np.mean(images,2))/self.feature_config.max_intensity)

    def add_true_top_layer(self):
        for entry in self.data:
            top_layer_indices = entry.get("top_layer_indices")
            if top_layer_indices is not None:
                entry["top_layer"] = self.feature_config.miller_indices[top_layer_indices]

        
    def create_data_set(self, num_samples, batch_size=1024):
        num_batches = num_samples// batch_size
        print(f"[Benchmark]: Creating data set for benchmarking")
        for bidx in trange(num_batches):
            batch = self.test_data_generators["uniform"].generate_data_batch(
                batch_size, "uniform", self.missing_test_peaks, self.extra_test_peaks,
                None, False)
            cell_targets, images, lattice_system_targets, q_xy, q_z, top_layer_indices = zip(batch)
            for sidx in range(len(cell_targets[0])):
                self.data.append({
                    "sidx": sidx + batch_size*bidx,
                    "target_cell_rad": cell_targets[0][sidx],
                    "images": images[0][sidx].cpu().numpy(),
                    "lattice_system_targets": lattice_system_targets[0][sidx],
                    "q_xy": q_xy[0][sidx].cpu().numpy(),
                    "q_z": q_z[0][sidx].cpu().numpy(),
                    "top_layer_indices": top_layer_indices[0][sidx]
                })

        self.add_target_variables(label="target")
        self.add_number_of_peaks()
        self.add_true_top_layer()
        print(f"[Benchmark]: Data set created")


    def predict_data_with_neural_network(self):
        self.regressor.get_model()
        self.machine_learning_parameters = MachineLearningHyperparameter()
        batch_size = self.machine_learning_parameters.validation_batch_size

        num_batches = len(self.data) // batch_size

        print(f"[Benchmark]: Predicting unit cells with neural network...")

        for bidx in trange(num_batches):
            input_list = []
            for sidx in range(bidx*batch_size, min((bidx+1)*batch_size, len(self.data))):
                input_list.append(self.data[sidx]['images'])
            input_tensor = torch.Tensor(np.array(input_list)).to(torch.float32).to(self.regressor.device)
            input_tensor = torch.permute(input_tensor, dims=[0,3,1,2])
            outputs = self.regressor.gixd_model(input_tensor).cpu().detach().numpy()
            for sidx in range(bidx*batch_size, min((bidx+1)*batch_size, len(self.data))):
                self.data[sidx]["neural_network_cell_rad"] = GixdUtility.denormalize_cell(outputs[sidx-bidx*batch_size])

        self.add_target_variables(label="neural_network")

        print(f"[Benchmark]: Done with neural network prediction")

    def plot_correlogramm(self, label:str, plot_type:str="scatter"):
        fig, axs = plt.subplots(2, 3, figsize=(20,15))
        for idx, dimension in enumerate(['a', 'b', 'c', 'alpha_deg', 'beta_deg', 'gamma_deg']):
            ax = axs[idx//3, idx%3]
            data = list()
            for entry in self.data:
                target = entry.get("target_"+dimension)
                prediction = entry.get(label+"_"+dimension)
                if not target is None and not prediction is None:
                    data.append((target, prediction))
            data = np.array(data)
            ax.grid(True)
            ax.set_xlim(left=self.target_config.min_distance if len(dimension)==1 else self.target_config.min_angle,
                        right=self.target_config.max_distance if len(dimension)==1 else self.target_config.max_angle)
            ax.set_ylim(bottom=self.target_config.min_distance if len(dimension)==1 else self.target_config.min_angle,
                        top=self.target_config.max_distance if len(dimension)==1 else self.target_config.max_angle)
            ax.set_xlabel(f"True {dimension}")
            ax.set_ylabel(f"Predicted {dimension}")
            if plot_type == "scatter":
                ax.scatter(data[:,0], data[:,1], c="g")

                
        plt.savefig(os.path.join(self.path,"correlogramm.png"))
        plt.close()

    def calculate_metrics(self, label:str):
        all_losses = []
        for idx, dimension in enumerate(['a', 'b', 'c', 'alpha_deg', 'beta_deg', 'gamma_deg']):
            data = list()
            for entry in self.data:
                target = entry.get("target_"+dimension)
                prediction = entry.get(label+"_"+dimension)
                if not target is None and not prediction is None:
                    data.append((target, prediction))
            data = np.array(data)
            if np.count_nonzero(np.isnan(data)) > 0:
                print(f"Found {np.count_nonzero(np.isnan(data[:,0]))} NaN in targets and {np.count_nonzero(np.isnan(data[:,1]))} in {label} predictions for {dimension}")
            mae = np.nanmean(np.abs(data[:,0]-data[:,1]))
            mse = np.sqrt(np.nanmean(np.square(data[:,0]-data[:,1])))
            if dimension in ['a', 'b', 'c']:
                loss = np.nanmean(np.square((data[:,0]-self.target_config.min_distance)/(self.target_config.max_distance-self.target_config.min_distance)-
                                            (data[:,1]-self.target_config.min_distance)/(self.target_config.max_distance-self.target_config.min_distance)))
            else:
                loss = np.nanmean(np.square((data[:,0]-self.target_config.min_angle)/(self.target_config.max_angle-self.target_config.min_angle)-
                                            (data[:,1]-self.target_config.min_angle)/(self.target_config.max_angle-self.target_config.min_angle)))
            all_losses.append(loss)

            print(f"{dimension}: MAE = {mae:.3f}\t RMSE = {mse:.3f} loss(mse) = {loss:.5f}")

            print(f"Overall loss(mse) = {np.mean(all_losses)}")

    def run_direct_inversion_on_data_set(self, num_processes=1):
        raise NotImplementedError
        num_batches = len(self.data) // num_processes

        print(f"[Benchmark]: Predicting unit cells with neural network...")

        for bidx in trange(num_batches):
            input_list = []
            for sidx in range(bidx*num_processes, min((bidx+1)*num_processes, len(self.data))):
                input_list.append(self.data[sidx]['images'])

            input_tensor = torch.Tensor(np.array(input_list)).to(torch.float32).to(self.regressor.device)
            input_tensor = torch.permute(input_tensor, dims=[0,3,1,2])
            outputs = self.regressor.gixd_model(input_tensor).cpu().detach().numpy()


            for sidx in range(bidx*num_processes, min((bidx+1)*num_processes, len(self.data))):
                self.data[sidx]["direct_inversion_cell_rad"] = GixdUtility.denormalize_cell(outputs[sidx-bidx*num_processes])

        self.add_target_variables(label="direct_inversion")

        print(f"[Benchmark]: Done with neural network prediction")

    def collect_ann_toxic_cases(self, fraction=0.1):
        for entry in self.data:
            target_norm = GixdUtility.normalize_cell(entry["target_cell_rad"])
            nn_prediction_norm = GixdUtility.normalize_cell(entry["neural_network_cell_rad"])
            mae = np.nanmean(np.abs(target_norm-nn_prediction_norm))
            if not np.isnan(mae):
                if len(self.ann_worst_cases) < int(len(self.data)*fraction):
                    heapq.heappush(self.ann_worst_cases, (mae, entry))
                else:
                    heapq.heappushpop(self.ann_worst_cases, (mae, entry))

                if len(self.ann_best_cases) < int(len(self.data)*fraction):
                    heapq.heappush(self.ann_best_cases, (-mae, entry))
                else:
                    heapq.heappushpop(self.ann_best_cases, (-mae, entry))

    def best_vs_worst_cases_stats(self, stat='number_of_peaks'):
    
        best_cases_stats = []
        for (mae, entry) in self.ann_best_cases:
            stat_entry = entry.get(stat)
            if stat_entry is not None:
                best_cases_stats.append(stat_entry)
        worst_cases_stats = []
        for (mae, entry) in self.ann_worst_cases:
            stat_entry = entry.get(stat)
            if stat_entry is not None:
                worst_cases_stats.append(stat_entry)
        default_cases_stats = []
        for entry in self.data:
            stat_entry = entry.get(stat)
            if stat_entry is not None:
                default_cases_stats.append(stat_entry)

        plt.hist([best_cases_stats, worst_cases_stats],
                bins=50 if stat=="number_of_peaks" else 7, density=False, alpha=0.5, label=['Best', 'Worst'], histtype='stepfilled')
        #plt.hist([best_cases_stats, default_cases_stats, worst_cases_stats],
        #         bins=300, density=True, alpha=0.5, label=['Best', 'All', 'Worst'], histtype='stepfilled')
        plt.legend()
        plt.savefig(os.path.join(self.path, f"best_vs_worst_stats_{stat}.png"))
        plt.close()


    def fit_worst_cases_with_gmm(self):
        data_list = []
        for (mae, entry) in self.ann_worst_cases:
            data_list.append(GixdUtility.normalize_cell(entry["target_cell_rad"]))
        data = np.array(data_list)

        if False:
            # Determine the optimal number of Gaussians using BIC
            bic_scores = []
            num_components = np.arange(1, 10)

            for k in num_components:
                gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=42)
                gmm.fit(data)
                bic_scores.append(gmm.bic(data))

            optimal_k = num_components[np.argmin(bic_scores)]
            print(f"Optimal number of Gaussians: {optimal_k}")

            # Fit the best GMM
            best_gmm = GaussianMixture(n_components=optimal_k, covariance_type='full', random_state=42)
            best_gmm.fit(data)

            # Report Gaussian centers
            print("Gaussian Centers:")
            for mean in best_gmm.means_:
                print(f"Mean = {GixdUtility.denormalize_cell_deg(mean)}")

            # Create corner plot
            labels = [f"Dim {i+1}" for i in range(6)]
            fig = corner(data, labels=labels, color="blue", bins=30, hist_bin_factor=2)

            # Overlay Gaussian means on all relevant subplots
            axes = np.array(fig.axes).reshape((6, 6))  # Reshape into 6x6 grid

            for (i, j) in combinations(range(6), 2):  # Iterate over all 2D projections
                ax = axes[j, i]  # Get the correct subplot
                ax.scatter(best_gmm.means_[:, i], best_gmm.means_[:, j], color="red", marker="x", s=100)
        else:
            # Fit Affinity Propagation
            affinity_propagation = AffinityPropagation(random_state=42, damping=0.9)
            affinity_propagation.fit(data)

            # Get cluster centers (always elements of original data)
            cluster_centers = data[affinity_propagation.cluster_centers_indices_]

            # Get cluster labels
            labels = affinity_propagation.labels_

            print(f"Number of clusters found: {len(cluster_centers)}")
            print("Cluster Centers:")
            cluster_centers = GixdUtility.denormalize_cell_deg(cluster_centers)
            print(cluster_centers)

            # Create corner plot
            labels = [f"Dim {i+1}" for i in range(6)]
            fig = corner(GixdUtility.denormalize_cell_deg(data), labels=labels, color="blue", bins=30, hist_bin_factor=2)

            # Overlay cluster centers
            axes = np.array(fig.axes).reshape((6, 6))
            from itertools import combinations
            for (i, j) in combinations(range(6), 2):
                ax = axes[j, i]
                ax.scatter(cluster_centers[:, i],
                           cluster_centers[:, j],
                           color="red", marker="x", s=100)


        plt.savefig(os.path.join(self.path, "corner_plot_worst_cases_neural_network.png"))
        plt.close()



if __name__=="__main__":
    print(f"Running gixd_benchmark.py main()")
    benchmark = GixdBenchmark("test", 1, "./one_phase_no_noise_default")
    #benchmark.create_data_set(10000, 100)
    #benchmark.predict_data_with_neural_network()
    #benchmark.save_data("data.pd")
    
    benchmark.load_data('data.pd')
    #benchmark.plot_correlogramm("neural_network", "scatter")
    #benchmark.calculate_metrics("neural_network")
    benchmark.collect_ann_toxic_cases()
    #benchmark.fit_worst_cases_with_gmm()
    benchmark.best_vs_worst_cases_stats("number_of_peaks")
    benchmark.best_vs_worst_cases_stats("lattice_system_targets")



    #benchmark.load_data('data.pd')
