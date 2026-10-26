import numpy as np
from matplotlib import pyplot as plt
from multiprocessing import Pool
from scipy.optimize import minimize, least_squares
from scipy.spatial.transform import Rotation
from timeit import default_timer as timer

import torch
#torch.backends.cuda.preferred_linalg_library("magma")
from torch import Tensor
from types import SimpleNamespace
import torch.nn.functional as F
from torch.autograd import grad
from torch.autograd.functional import hessian

from gixd_utility import GixdUtility, LatticeSystem
from config_loader import RefinementHyperparameters
from gixd_data_generator import GixdDataGenerator
from torchmin import minimize as torch_minimize
from scipy.optimize import linear_sum_assignment

# monkey-patch correct scattering_vector to handle both 2D and 3D reciprocal_mtx
import torch as _torch  # prevent name clash
import numpy as _np

#torch.autograd.detect_anomaly(True)
from config_loader import FeatureConfig
from gidind_ai.uncertainty import diagnostic_row, guard_uncertainties

feature_config = FeatureConfig()

def _patched_get_scattering_vector(reciprocal_mtx, laue_indices):
    '''
    Compute the scattering vectors using PyTorch.
    Supports 2D ([3,3]) and 3D ([B,3,3]) reciprocal_mtx.
    Always returns a 3D tensor of shape [B, N, 3].
    '''
    two_pi = 2.0 * _np.pi
    li = laue_indices.to(device=reciprocal_mtx.device, dtype=reciprocal_mtx.dtype)
    if reciprocal_mtx.dim() == 2:
        scattering = two_pi * _torch.matmul(li, reciprocal_mtx.transpose(0, 1))
        return scattering.unsqueeze(0)
    elif reciprocal_mtx.dim() == 3:
        B = reciprocal_mtx.shape[0]
        li_expanded = li.unsqueeze(0).expand(B, -1, -1)
        scattering = two_pi * _torch.matmul(li_expanded, reciprocal_mtx.transpose(1, 2))
        return scattering
    else:
        raise ValueError(f'Unexpected reciprocal_mtx dimension: {reciprocal_mtx.dim()}')

GixdDataGenerator.get_scattering_vector = staticmethod(_patched_get_scattering_vector)


def direct_inversion_loss(unit_cell_normalized: Tensor,
                           indices_match: Tensor,
                           data_generator: GixdDataGenerator,
                           trial_top_layer,
                           loss_type: str = 'l2') -> Tensor:
    cell = GixdUtility.denormalize_cell_torch(unit_cell_normalized)
    v1, v2, v3 = data_generator.calc_real_vectors_from_unit_cell_rad(cell)
    R = data_generator.calc_rotation_matrix(
        v1, v2, v3,
        torch.tensor([trial_top_layer], device=cell.device, dtype=torch.int64)
    )
    v1r, v2r, v3r = data_generator.rotate_cell_by_matrix(v1, v2, v3, R)
    k1, k2, k3 = data_generator.calc_reciprocal_cell(v1r, v2r, v3r)
    k1, k2, k3 = k1[0], k2[0], k3[0]
    qf = 2.0 * torch.pi
    h, k_val, l_val = indices_match[:, 0], indices_match[:, 1], indices_match[:, 2]
    obs_qxy, obs_qz = indices_match[:, 4], indices_match[:, 5]
    weights = indices_match[:,6]

    theo_x = qf * (h * k1[0] + k_val * k2[0] + l_val * k3[0])
    theo_y = qf * (h * k1[1] + k_val * k2[1] + l_val * k3[1])
    theo_qxy = torch.sqrt(theo_x**2 + theo_y**2)
    theo_qz = qf * torch.abs(h * k1[2] + k_val * k2[2] + l_val * k3[2])

    err_xy = (theo_qxy - obs_qxy)**2
    err_z = (theo_qz - obs_qz)**2
    total_w = weights.sum().clamp(min=1e-6)
    loss_xy = (weights * err_xy).sum() / total_w
    loss_z = (weights * err_z).sum() / total_w
    return loss_xy + loss_z

def fixed_assignment_residuals(unit_cell_normalized, indices_match, data_generator, trial_top_layer):
    """Weighted residuals for identifiability, without changing the fitted loss."""
    cell = GixdUtility.denormalize_cell_torch(unit_cell_normalized)
    vectors = data_generator.calc_real_vectors_from_unit_cell_rad(cell)
    rotation = data_generator.calc_rotation_matrix(
        *vectors,
        torch.tensor([trial_top_layer], device=cell.device, dtype=torch.int64)
    )
    rotated = data_generator.rotate_cell_by_matrix(*vectors, rotation)
    k1, k2, k3 = [k[0] for k in data_generator.calc_reciprocal_cell(*rotated)]
    h, k, l = indices_match[:, 0], indices_match[:, 1], indices_match[:, 2]
    tx = 2 * torch.pi * (h * k1[0] + k * k2[0] + l * k3[0])
    ty = 2 * torch.pi * (h * k1[1] + k * k2[1] + l * k3[1])
    tz = 2 * torch.pi * torch.abs(h * k1[2] + k * k2[2] + l * k3[2])
    weights = indices_match[:, 6]
    scale = torch.sqrt(weights / weights.sum().clamp_min(1e-6))
    return torch.cat(((torch.sqrt(tx**2 + ty**2) - indices_match[:, 4]) * scale,
                      (tz - indices_match[:, 5]) * scale))


def match_laue_indices(used_laue_indices, k1, k2, k3,
                                        test_qxy, test_qz,
                                        data_generator: GixdDataGenerator,
                                        q_cutoff,  method='observed',
                                        return_plot_dict:bool=False):
    #print(f"method = {method}")
    assert method in ['observed', 'simulated', 'one2one', 'simulated_fractional', 'sinkhorn']
    device = data_generator.device
    test_qxy = torch.as_tensor(test_qxy, device=device)
    test_qz = torch.as_tensor(test_qz, device=device)
    if feature_config.use_specular_scan:
        mask = (test_qxy > 0.0) | (test_qz > 0.0)
    else:
        mask = test_qxy > feature_config.specular_cutoff
    #test_q = torch.stack([test_qxy, test_qz], dim=1)
    test_q = torch.stack([test_qxy[mask], test_qz[mask]], dim=1)
    #non_q = torch.stack([test_qxy[torch.logical_not(mask)], test_qz[torch.logical_not(mask)]], dim=1)
    #print(f"non_test_q = {non_q}")
    #print(f"test_qxy has {len(test_qxy)}, maks has {torch.count_nonzero(mask)} test_q has {test_q.shape[0]}")
    k1, k2, k3 = [torch.as_tensor(x, device=device, dtype=test_q.dtype) for x in (k1, k2, k3)]
    rec = torch.stack([k1, k2, k3], dim=-1)
    with torch.no_grad():
        sim_xy, sim_z = data_generator.calc_q_points(rec)
    sim_xy = sim_xy.to(device).squeeze(0)
    sim_z = sim_z.to(device).squeeze(0)
    if feature_config.use_specular_scan:
        mask2 = (sim_xy > 0.0) | (sim_z > 0.0)
    else:
        mask2 = sim_xy > feature_config.specular_cutoff
    sim_q = torch.stack([sim_xy[mask2], sim_z[mask2]], dim=1)
    #non_sim_q = torch.stack([sim_xy[torch.logical_not(mask2)], sim_z[torch.logical_not(mask2)]], dim=1)
    #print(f"non_sim_q = {non_sim_q}")
    used = torch.as_tensor(used_laue_indices, device=device, dtype=test_q.dtype)
    rel = used[mask2]

    def build_match_row(laue_values, obs_index, obs_qxy_value, obs_qz_value, weight_value):
        if not torch.is_tensor(weight_value):
            weight_value = test_q.new_tensor(float(weight_value))
        else:
            weight_value = weight_value.to(dtype=test_q.dtype)
        return torch.cat([
            laue_values.to(dtype=test_q.dtype),
            torch.stack([
                test_q.new_tensor(float(obs_index)),
                obs_qxy_value.to(dtype=test_q.dtype),
                obs_qz_value.to(dtype=test_q.dtype),
                weight_value
            ])
        ])

    # handle no simulated or no observed points
    if test_q.size(0) == 0 or sim_q.size(0) == 0:
        plot_dict = {
            "test_qxy": test_qxy.detach().cpu().numpy(),  # 1D tensor (N,)
            "test_qz": test_qz.detach().cpu().numpy(),    # 1D tensor (N,)
            "sim_q": sim_q.detach().cpu().numpy(),        # tensor of shape (N_sim_valid, 2)
            "indices_match": torch.empty((0, 7), device=device, dtype=test_q.dtype).detach().cpu().numpy(),  # tensor of shape (num_matches, 6)
            "plot_title": "no obs q" if test_q.size(0)==0 else "no simulated q"
        }
        return torch.empty((0, 7), device=device, dtype=test_q.dtype), plot_dict, sim_q.size(0)
    dists = torch.norm(test_q.unsqueeze(1) - sim_q.unsqueeze(0), dim=2)
    matches = []
    if method == 'observed':
        for i in range(test_q.size(0)):
            md, idx = torch.min(dists[i], dim=0)
            if md <= q_cutoff:
                matches.append(build_match_row(rel[idx], i, test_q[i, 0], test_q[i, 1], 1.0))
    elif method == 'simulated':
        for j in range(sim_q.size(0)):
            md, obs_idx = torch.min(dists[:,j], dim=0)
            if md <= q_cutoff:
                obs_qxy, obs_qz = test_q[obs_idx]
                matches.append(build_match_row(rel[j], obs_idx.item(), obs_qxy, obs_qz, 1.0))
    elif method == 'one2one':
        cost = dists.cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost)
        keep = cost[row_ind, col_ind] <= q_cutoff
        row_ind, col_ind = row_ind[keep], col_ind[keep]
        for obs_idx, sim_j in zip(row_ind, col_ind):
            obs_qxy, obs_qz = test_q[obs_idx]
            matches.append(build_match_row(rel[sim_j], obs_idx, obs_qxy, obs_qz, 1.0))
    elif method == 'simulated_fractional':
        min_vals, obs_idx = torch.min(dists, dim=0)
        mask_cutoff = min_vals <= q_cutoff
        counts = torch.bincount(obs_idx[mask_cutoff], minlength=test_q.size(0)).to(dtype=test_q.dtype)
        weights = torch.zeros(sim_q.size(0), device=device, dtype=test_q.dtype)
        weights[mask_cutoff] = 1.0 / counts[obs_idx[mask_cutoff]]
        for j in range(sim_q.size(0)):
            if not mask_cutoff[j]:
                continue
            i = obs_idx[j].item()
            obs_qxy, obs_qz = test_q[i]
            w = weights[j]
            matches.append(build_match_row(rel[j], i, obs_qxy, obs_qz, w))
    elif method == 'sinkhorn':
        epsilon_frac = 0.1
        
        min_vals, obs_idx = torch.min(dists, dim=0) # for each sim peak find nearest neighbor
        sim_mask = (min_vals <= q_cutoff)
        if not sim_mask.any():
            plot_dict = {
                "test_qxy": test_qxy.detach().cpu().numpy(),  # 1D tensor (N,)
                "test_qz": test_qz.detach().cpu().numpy(),    # 1D tensor (N,)
                "sim_q": sim_q.detach().cpu().numpy(),        # tensor of shape (N_sim_valid, 2)
                "indices_match": torch.empty((0, 7), device=device, dtype=test_q.dtype).detach().cpu().numpy(),  # tensor of shape (num_matches, 6)
                "plot_title": "no obs q" if test_q.size(0)==0 else "no simulated q"
            }
            return torch.empty((0, 7), device=device, dtype=test_q.dtype), plot_dict, sim_q.size(0)
        C = dists[:,sim_mask] #distances only thos smaller than cutoff in cost matrix
        epsilon = epsilon_frac * torch.median(C.flatten()).item()
        eps_tensor = torch.tensor(epsilon, device=device, dtype=C.dtype)
        #Sinkhorn Knopp
        K = torch.exp(-C/eps_tensor)
        u = torch.ones(C.size(0), device=device, dtype=C.dtype)
        v = torch.ones(C.size(1), device=device, dtype=C.dtype)
        prev_P = None
        for _ in range(100):
            u = 1.0 / (K @ v)
            v = 1.0 / (K.t() @ u)
            P = torch.diag(u) @ K @ torch.diag(v)
            if prev_P is not None and torch.norm(P-prev_P)<1e-3:
                print(f"Sinkhorn Knopp converged after {_} iter")
                break
            prev_P = P
        sim_indices = torch.nonzero(sim_mask, as_tuple=False).squeeze(1)
        for j_loc, j in enumerate(sim_indices):
            w_vec = P[:,j_loc]
            obs_is = torch.nonzero(w_vec>0, as_tuple=False).squeeze(1)
            for i in obs_is:
                w = w_vec[i]
                obs_xy, obs_z = test_q[i]
                matches.append(build_match_row(rel[j], i.item(), obs_xy, obs_z, w))
    im = torch.stack(matches, dim=0) if matches else torch.empty((0, 7), device=device, dtype=test_q.dtype)
    if return_plot_dict:
        a_val, b_val, c_val, alpha_val, beta_val, gamma_val = GixdUtility.reciprocal_lattice_to_unit_cell_deg(
            k1.cpu().numpy(), k2.cpu().numpy(), k3.cpu().numpy())
        plot_title = f"[{a_val:.3f} {b_val:.3f} {c_val:.3f} {alpha_val:.3f} {beta_val:.3f} {gamma_val:.3f}]"
        plot_dict = {
            "test_qxy": test_qxy.detach().cpu().numpy(),  # 1D tensor (N,)
            "test_qz": test_qz.detach().cpu().numpy(),    # 1D tensor (N,)
            "sim_q": sim_q.detach().cpu().numpy(),        # tensor of shape (N_sim_valid, 2)
            "indices_match": im.detach().cpu().numpy(),  # tensor of shape (num_matches, 6)
            "plot_title": plot_title
        }
    else:
        plot_dict = {"return_plot_dict": return_plot_dict}
    return im, plot_dict, sim_q.size(0)

def direct_inversion_pytorch(first_predicted_cell_deg: Tensor,
                             observed_qxy, observed_qz,
                             trial_top_layer,
                             data_generator: GixdDataGenerator,
                             method: str = 'L-BFGS-B',
                             loss_type: str = 'l2',
                             q_cutoff: float = 0.1,
                             maxiter: int = 50,
                             outer_loops: int = 3,
                             matching_method: str = 'one2one',
                             return_uncertainties: bool = False,
                             torch_lbfgs_lr: float = 1.0,
                             refinement_backend: str = "serial") -> Tensor:
    from gidind_ai.batched_refinement import Geometry, refine_batch, resolve_backend, result_rows
    if resolve_backend(refinement_backend, first_predicted_cell_deg.device) == "batched":
        if loss_type != "l2":
            raise ValueError("Batched refinement currently supports loss_type='l2' only")
        observed = torch.stack((torch.as_tensor(observed_qxy, device=first_predicted_cell_deg.device).reshape(-1),
                                torch.as_tensor(observed_qz, device=first_predicted_cell_deg.device).reshape(-1)), -1)
        result = refine_batch(first_predicted_cell_deg.reshape(1, 6), observed,
                              torch.as_tensor(trial_top_layer, device=first_predicted_cell_deg.device).reshape(1, 3),
                              Geometry.from_generator(data_generator, first_predicted_cell_deg.device),
                              matching_method=matching_method, q_cutoff=q_cutoff,
                              maxiter=maxiter, outer_loops=outer_loops,
                              return_uncertainties=return_uncertainties)
        row = result_rows(result, observed)[0]
        row['timings'] = result['timings']
        return row
    assert first_predicted_cell_deg.device.type == data_generator.device.type, \
        f"Device mismatch: cell on {first_predicted_cell_deg.device}, generator on {data_generator.device}"
    if method == "torch-lbfgs" and torch_lbfgs_lr <= 0.0:
        raise ValueError("torch_lbfgs_lr must be positive for method='torch-lbfgs'.")
    best_x = GixdUtility.normalize_cell_deg(first_predicted_cell_deg).to(data_generator.device)
    last_loss = 1e3 # maximum is 4, so this is kind of inf
    last_num_matches = 0
    refinement_iterations = 0
    refinement_loss_history = []
    for outer in range(outer_loops):
        x_t = best_x.clone().detach() #start with fresh gradients at beginning of outer loop
        #with torch.no_grad():
        if True:
            denorm = GixdUtility.denormalize_cell_torch(x_t)
            v1, v2, v3 = data_generator.calc_real_vectors_from_unit_cell_rad(denorm)
            R = data_generator.calc_rotation_matrix(
                v1, v2, v3,
                torch.tensor([trial_top_layer], device=data_generator.device, dtype=torch.int64)
            )
            v1r, v2r, v3r = data_generator.rotate_cell_by_matrix(v1, v2, v3, R)
            k1, k2, k3 = data_generator.calc_reciprocal_cell(v1r, v2r, v3r)
            k1, k2, k3 = k1[0], k2[0], k3[0]

            if outer==0:
                indices_match, init_plot_dict, init_num_sim = match_laue_indices(
                    data_generator.laue_indices_tensor,
                    k1, k2, k3,
                    observed_qxy, observed_qz,
                    data_generator,
                    q_cutoff = q_cutoff,
                    method="one2one" if matching_method=='hybrid' else matching_method,
                    return_plot_dict=True
                )
                #print(f"indices matched = {len(indices_match)}")
            else:
                if matching_method == 'hybrid':
                    if outer < outer_loops/2:
                        curr_matching_method = 'one2one'
                    else:
                        curr_matching_method = 'simulated_fractional'
                else:
                    curr_matching_method = matching_method
                indices_match, _, _ = match_laue_indices(
                    data_generator.laue_indices_tensor,
                    k1, k2, k3,
                    observed_qxy, observed_qz,
                    data_generator,
                    q_cutoff,
                    method=curr_matching_method,
                    return_plot_dict=False
                )
                #print(f"indices matched = {len(indices_match)}")
            if len(indices_match) == 0: #  matching failed
                empty_uncertainty, empty_diagnostics = guard_uncertainties(
                    torch.zeros_like(x_t), x_t.new_zeros((0, 6))
                )
                return {"refined_predicted_cell_deg": GixdUtility.denormalize_cell_deg(x_t),
                        "refined_uncertainty_cell_deg": empty_uncertainty if return_uncertainties else torch.zeros_like(x_t),
                        "convergence_status": False,
                        "init_plot_dict": init_plot_dict,
                        "refined_plot_dict": init_plot_dict,
                        "refined_q_loss": last_loss,
                        "refined_num_matches": 0,
                        "refined_num_sim_peaks": init_num_sim,
                        "identifiability": diagnostic_row(empty_diagnostics),
                        "refinement_backend": "serial", "optimizer": method,
                        "iterations": refinement_iterations,
                        "loss_history": refinement_loss_history}

        def loss_fun(x):
            return direct_inversion_loss(x, indices_match, data_generator, trial_top_layer, loss_type)

        if method == "torch-lbfgs":
            outer_fraction = 0.0 if outer_loops <= 1 else outer / (outer_loops - 1)
            outer_start_lr = float(torch_lbfgs_lr * np.exp(np.log(0.2) * outer_fraction))
            outer_end_lr = 0.1 * outer_start_lr
            x_param = torch.nn.Parameter(x_t.clone().detach())
            num_lr_stages = min(10, max(4, int(np.ceil(maxiter / 50))))
            stage_bounds = np.linspace(0, maxiter, num_lr_stages + 1, dtype=int)
            optimizer = torch.optim.LBFGS(
                [x_param],
                lr=outer_start_lr,
                max_iter=max(1, int(np.ceil(maxiter / num_lr_stages))),
                max_eval=max(2, int(np.ceil(1.25 * maxiter / num_lr_stages))),
                tolerance_grad=1e-10,
                tolerance_change=1e-10,
                history_size=min(100, max(10, maxiter)),
                line_search_fn="strong_wolfe"
            )

            last_closure_loss = None
            best_step_loss = float("inf")
            stagnation_steps = 0
            stagnation_patience = min(4, max(2, num_lr_stages // 2))

            def closure():
                nonlocal last_closure_loss
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fun(x_param)
                if not torch.isfinite(loss):
                    raise RuntimeError("torch-lbfgs encountered a non-finite loss.")
                loss.backward()
                last_closure_loss = loss.detach()
                return loss

            for stage_idx in range(num_lr_stages):
                stage_iters = int(stage_bounds[stage_idx + 1] - stage_bounds[stage_idx])
                if stage_iters <= 0:
                    continue
                if num_lr_stages <= 1:
                    current_lr = outer_start_lr
                else:
                    stage_fraction = stage_idx / (num_lr_stages - 1)
                    cosine_weight = 0.5 * (1.0 + np.cos(np.pi * stage_fraction))
                    current_lr = outer_end_lr + (outer_start_lr - outer_end_lr) * cosine_weight
                optimizer.param_groups[0]["lr"] = float(current_lr)
                optimizer.param_groups[0]["max_iter"] = stage_iters
                optimizer.param_groups[0]["max_eval"] = max(stage_iters + 1, int(np.ceil(1.5 * stage_iters)))
                step_loss = optimizer.step(closure)
                if torch.is_tensor(step_loss):
                    step_loss = float(step_loss.detach().item())
                elif step_loss is None:
                    if last_closure_loss is None:
                        continue
                    step_loss = float(last_closure_loss.item())
                else:
                    step_loss = float(step_loss)

                improvement_tol = max(1e-12, 1e-6 * max(1.0, abs(best_step_loss)))
                if step_loss < best_step_loss - improvement_tol:
                    best_step_loss = step_loss
                    stagnation_steps = 0
                else:
                    stagnation_steps += 1
                    if stagnation_steps >= stagnation_patience:
                        break

            with torch.no_grad():
                final_loss = loss_fun(x_param).detach()

            result = SimpleNamespace(
                x=x_param.detach().clone(),
                fun=float(final_loss.item()),
                status="torch-lbfgs",
                hess=None,
                hess_inv=None
            )
        else:
            if method in ['dogleg', 'trust-ncg', 'cg', 'trust-exact']:
                options = {'gtol': 1e-10}
            elif method=='newton-exact':
                options = {'xtol': 1e-10,
                           'handle_npd': 'lu',
                           'line_search': 'strong-wolfe'}
            elif method=='bfgs':
                options = {'gtol': 1e-10,
                           'xtol': 1e-10}
            elif method=='newton-cg':
                options = {'xtol': 1e-10}

            result = torch_minimize(
                loss_fun,
                x_t,
                method=method,
                options=options,
                max_iter=maxiter,
                disp=0,
                tol=1e-10
            )

        refinement_iterations += int(getattr(result, "nit", 0))
        refinement_loss_history.append(float(result.fun))
        best_x = result.x.clone().detach()
        last_loss = float(result.fun) if not torch.is_tensor(result.fun) else result.fun.detach().item()
        last_num_matches = len(indices_match)

        #if last_num_matches >= len(indices_match) and result.fun >= last_loss:
        #    break
        #last_num_matches = len(indices_match)
        #last_loss = result.fun

    #print(f"result = {result}")
    denorm = GixdUtility.denormalize_cell_torch(best_x)
    v1, v2, v3 = data_generator.calc_real_vectors_from_unit_cell_rad(denorm)
    R = data_generator.calc_rotation_matrix(
        v1, v2, v3,
        torch.tensor([trial_top_layer], device=data_generator.device, dtype=torch.int64)
    )
    v1r, v2r, v3r = data_generator.rotate_cell_by_matrix(v1, v2, v3, R)
    k1, k2, k3 = data_generator.calc_reciprocal_cell(v1r, v2r, v3r)
    k1, k2, k3 = k1[0], k2[0], k3[0]

    refined_indices_match, refined_plot_dict, refined_num_sim_peaks = match_laue_indices(
        data_generator.laue_indices_tensor,
        k1, k2, k3,
        observed_qxy, observed_qz,
        data_generator,
        q_cutoff,
        method='one2one' if matching_method=='hybrid' else matching_method,
        return_plot_dict=True
    )
    
    refined = GixdUtility.denormalize_cell_deg(best_x)
    
    identifiability = None
    if return_uncertainties:
        loss_at_refined_match = lambda x: direct_inversion_loss(
            x,
            refined_indices_match,
            data_generator,
            trial_top_layer,
            loss_type
        )

        hess_inv = getattr(result, "hess_inv", None)
        H = getattr(result, "hess", None)

        if H is not None:
            H = torch.as_tensor(H, device=best_x.device, dtype=best_x.dtype)
            H = 0.5 * (H + H.T)
            H = H + torch.eye(6, device=H.device, dtype=H.dtype) * 1e-6
            invH = torch.linalg.pinv(H)
        elif hess_inv is not None:
            invH = torch.as_tensor(hess_inv, device=best_x.device, dtype=best_x.dtype)
            invH = 0.5 * (invH + invH.T)
        else:
            # Some solvers such as BFGS do not expose `result.hess`.
            # Fall back to an explicit Hessian of the final matched loss.
            H = hessian(loss_at_refined_match, best_x)
            H = 0.5 * (H + H.T)
            H = H + torch.eye(6, device=H.device, dtype=H.dtype) * 1e-6
            invH = torch.linalg.pinv(H)

        # 3) compute residual vector at xopt
        r_np = ls_res(
            best_x.detach().cpu().numpy(),
            data_generator, trial_top_layer,
            observed_qxy, observed_qz,
            q_cutoff,
            matching_method=matching_method,
        )
        SSE = np.sum(r_np**2)
        N_obs = r_np.size
        N_par = best_x.detach().cpu().numpy().size
        sigma2 = SSE / max(1, (N_obs - N_par))

        # 4) covariance in normalized units = σ² · H⁻¹
        cov_norm = invH * sigma2

        # 5) extract 1σ in normalized units then rescale to Å /°
        sigma_norm = torch.sqrt(torch.diag(cov_norm))
        scale_device = sigma_norm.device
        scale_dtype = sigma_norm.dtype
        tc = data_generator.target_config
        min_val = torch.tensor(
            [tc.min_distance] * 3 + [tc.min_angle] * 3,
            device=scale_device,
            dtype=scale_dtype
        )
        max_val = torch.tensor(
            [tc.max_distance] * 3 + [tc.max_angle] * 3,
            device=scale_device,
            dtype=scale_dtype
        )
        scale = max_val - min_val
        uncertainty_cell_deg = sigma_norm * scale
        jacobian = torch.autograd.functional.jacobian(
            lambda x: fixed_assignment_residuals(x, refined_indices_match, data_generator, trial_top_layer),
            best_x,
            vectorize=True,
        )
        uncertainty_cell_deg, diagnostics = guard_uncertainties(uncertainty_cell_deg, jacobian)
        identifiability = diagnostic_row(diagnostics)
    else:
        uncertainty_cell_deg = torch.zeros_like(refined)

    return_dict = {
        "refined_predicted_cell_deg": refined,
        "refined_uncertainty_cell_deg": uncertainty_cell_deg,
        "convergence_status": result.status,
        "init_plot_dict": init_plot_dict,
        "refined_plot_dict": refined_plot_dict,
        "refined_q_loss": result.fun,
        "refined_num_matches": len(refined_indices_match),
        "refined_num_sim_peaks": refined_num_sim_peaks,
        "refinement_backend": "serial",
        "optimizer": method,
        "iterations": refinement_iterations,
        "loss_history": refinement_loss_history
    }
    return_dict["identifiability"] = identifiability
    return return_dict

def ls_res(x_np, data_generator, trial_top_layer, obs_xy, obs_z, q_cutoff, matching_method):
    device = data_generator.device
    x = torch.tensor(x_np, requires_grad=False, device=device)
    cell = GixdUtility.denormalize_cell_torch(x)
    v1, v2, v3 = data_generator.calc_real_vectors_from_unit_cell_rad(cell)
    R = data_generator.calc_rotation_matrix(
        v1, v2, v3,
        torch.tensor([trial_top_layer], device=device, dtype=torch.int64)
    )
    v1r, v2r, v3r = data_generator.rotate_cell_by_matrix(v1, v2, v3, R)
    k1, k2, k3 = data_generator.calc_reciprocal_cell(v1r, v2r, v3r)
    k1, k2, k3 = k1[0], k2[0], k3[0]
    im, _, _ = match_laue_indices(
        data_generator.laue_indices_tensor,
        k1, k2, k3,
        obs_xy, obs_z,
        data_generator,
        q_cutoff,
        method='one2one' if matching_method=='hybrid' else matching_method,
    )
    #print(f"[DEBUG-ls] x={x_np}, matches={im.size(0)}")
    if im.size(0) < 1:
        return np.ones_like(x_np) * 1e3
    h, k_val, l_val = im[:, 0], im[:, 1], im[:, 2]
    obsx, obsz = im[:, 4], im[:, 5]
    qf = 2.0 * torch.pi
    tx = qf * (h * k1[0] + k_val * k2[0] + l_val * k3[0])
    ty = qf * (h * k1[1] + k_val * k2[1] + l_val * k3[1])
    tz = qf * torch.abs(h * k1[2] + k_val * k2[2] + l_val * k3[2])
    txy = torch.sqrt(tx**2 + ty**2)
    res = torch.cat([txy - obsx, tz - obsz], dim=0)
    res_np = res.detach().cpu().numpy()
    #print(f"[DEBUG-ls] residual_norm={np.linalg.norm(res_np)}")
    return res_np
