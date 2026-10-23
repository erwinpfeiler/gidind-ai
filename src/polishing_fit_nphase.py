import numpy as np
from matplotlib import pyplot as plt
from multiprocessing import Pool
from scipy.optimize import minimize, least_squares
from scipy.spatial.transform import Rotation
from timeit import default_timer as timer

import torch
#torch.backends.cuda.preferred_linalg_library("magma")
from torch import Tensor
import torch.nn.functional as F
from torch.autograd import grad
from torch.autograd.functional import hessian

from gixd_utility import GixdUtility, LatticeSystem
from config_loader import RefinementHyperparameters
from gixd_data_generator import GixdDataGenerator
from torchmin import minimize as torch_minimize

# monkey-patch correct scattering_vector to handle both 2D and 3D reciprocal_mtx
import torch as _torch  # prevent name clash
import numpy as _np

from scipy.optimize import linear_sum_assignment

#torch.autograd.detect_anomaly(True)

def _patched_get_scattering_vector(reciprocal_mtx, laue_indices):
    '''
    Compute the scattering vectors using PyTorch.
    Supports 2D ([3,3]) and 3D ([B,3,3]) reciprocal_mtx.
    Always returns a 3D tensor of shape [B, N, 3].
    '''
    two_pi = 2.0 * _np.pi
    li = laue_indices.float()
    if reciprocal_mtx.dim() == 2:
        scattering = two_pi * _torch.matmul(li.to(reciprocal_mtx.device), reciprocal_mtx.transpose(0, 1).to(torch.float32))
        return scattering.unsqueeze(0)
    elif reciprocal_mtx.dim() == 3:
        B = reciprocal_mtx.shape[0]
        li_expanded = li.unsqueeze(0).expand(B, -1, -1).to(reciprocal_mtx.device)
        scattering = two_pi * _torch.matmul(li_expanded, reciprocal_mtx.transpose(1, 2))
        return scattering
    else:
        raise ValueError(f'Unexpected reciprocal_mtx dimension: {reciprocal_mtx.dim()}')

GixdDataGenerator.get_scattering_vector = staticmethod(_patched_get_scattering_vector)

from config_loader import FeatureConfig

feature_config = FeatureConfig()

def direct_inversion_loss_nphase(unit_cell_normalized: Tensor,
                           indices_match_list: Tensor,
                           data_generator: GixdDataGenerator,
                           trial_top_layer,
                           loss_type: str = 'l2') -> Tensor:
    loss = 0.0

    for p, indices_match in enumerate(indices_match_list): 
        cell = GixdUtility.denormalize_cell_torch(unit_cell_normalized[p*6:(p+1)*6])
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
        theo_x = qf * (h * k1[0] + k_val * k2[0] + l_val * k3[0])
        theo_y = qf * (h * k1[1] + k_val * k2[1] + l_val * k3[1])
        theo_qxy = torch.sqrt(theo_x**2 + theo_y**2)
        theo_qz = qf * torch.abs(h * k1[2] + k_val * k2[2] + l_val * k3[2])

        if loss_type == 'l2':
            loss += (F.mse_loss(theo_qxy, obs_qxy, reduction='mean') +
                F.mse_loss(theo_qz, obs_qz, reduction='mean'))
        loss += (F.l1_loss(theo_qxy, obs_qxy, reduction='mean') +
            F.l1_loss(theo_qz, obs_qz, reduction='mean'))
    return loss


def match_laue_indices_nphase(
    used_laue_indices,
    k1_list, k2_list, k3_list,
    test_qxy, test_qz,
    data_generator,
    q_cutoff,
    method: str = 'observed',
    return_plot_dict: bool = False
):
    """
    Multi‐phase version of match_laue_indices.  For each phase i it computes:
      rec_i → sim_q_i, rel_i
    then runs the chosen method to match sim_q_i ↔ test_q,
    returning lists of im_i, plus sim_counts_i.
    """

    assert method in [
        'observed', 'simulated', 'one2one',
        'simulated_fractional', 'sinkhorn'
    ], f"Unknown method {method}"

    device = data_generator.device

    # 1) stack observed q's
    txy = torch.as_tensor(test_qxy, device=device)
    tz  = torch.as_tensor(test_qz, device=device)
    if feature_config.use_specular_scan:
        obs_mask = (txy > 0.0) | (tz > 0.0)
    else:
        obs_mask = txy > feature_config.specular_cutoff

    test_q = torch.stack([txy[obs_mask], tz[obs_mask]], dim=1)
    N_obs = test_q.size(0)

    # 2) build sim_q_list, rel_list
    num_phases = len(k1_list)
    sim_q_list, rel_list = [], []
    sim_counts = []

    for k1, k2, k3 in zip(k1_list, k2_list, k3_list):
        rec = torch.stack([
            torch.as_tensor(k1, device=device),
            torch.as_tensor(k2, device=device),
            torch.as_tensor(k3, device=device)
        ], dim=-1)
        with torch.no_grad():
            sim_xy, sim_z = data_generator.calc_q_points(rec)
        sim_xy = sim_xy.squeeze(0).to(device)
        sim_z  = sim_z.squeeze(0).to(device)

        if feature_config.use_specular_scan:
            sim_mask = (sim_xy > 0.0) | (sim_z > 0.0)
        else:
            sim_mask = sim_xy > feature_config.specular_cutoff

        sim_q = torch.stack([sim_xy[sim_mask], sim_z[sim_mask]], dim=1)
        rel   = torch.as_tensor(used_laue_indices, device=device)[sim_mask]

        sim_q_list.append(sim_q)
        rel_list.append(rel)
        sim_counts.append(sim_q.size(0))

    # 3) early‐exit if no obs or any phase has zero sim
    if N_obs == 0 or any(c == 0 for c in sim_counts):
        plot_dict = {
            "test_qxy": txy.detach().cpu().numpy(),
            "test_qz":  tz.detach().cpu().numpy(),
            "sim_q":    [sq.detach().cpu().numpy() for sq in sim_q_list],
            "indices_match": [
                torch.empty((0,7), device=device).detach().cpu().numpy()
                for _ in range(num_phases)
            ],
            "plot_title": ("no obs q" if N_obs==0 else "no simulated q")
        }
        return (
            [torch.empty((0,7), device=device) for _ in range(num_phases)],
            plot_dict,
            sim_counts
        )

    # 4) compute per‐phase distance matrices
    dist_list = [
        torch.norm(test_q.unsqueeze(1) - sim_q.unsqueeze(0), dim=2)
        for sim_q in sim_q_list
    ]

    # 6) now loop per‐phase & dispatch
    im_list = []
    for phase_idx in range(num_phases):
        dists = dist_list[phase_idx]
        sim_q = sim_q_list[phase_idx]
        rel   = rel_list[phase_idx]
        matches = []

        if method == 'observed':
            for i in range(N_obs):
                md, j   = torch.min(dists[i], dim=0)
                if md <= q_cutoff:
                    entry = torch.empty(7, device=device, dtype=rel.dtype)
                    entry[0:3] = rel[j].float()
                    entry[3]   = i
                    entry[4:6] = test_q[i]
                    entry[6]   = 1.0
                    matches.append(entry)

        elif method == 'simulated':
            for j in range(sim_q.size(0)):
                md, i = torch.min(dists[:,j], dim=0)
                if md <= q_cutoff:
                    entry = torch.empty(7, device=device, dtype=rel.dtype)
                    entry[0:3] = rel[j].float()
                    entry[3]   = i
                    entry[4:6] = test_q[i]
                    entry[6]   = 1.0
                    matches.append(entry)

        elif method == 'one2one':
            cost = dists.cpu().numpy()
            row, col = linear_sum_assignment(cost)
            keep = cost[row, col] <= q_cutoff
            for i, j in zip(row[keep], col[keep]):
                entry = torch.empty(7, device=device, dtype=rel.dtype)
                entry[0:3] = rel[j].float()
                entry[3]   = i
                entry[4:6] = test_q[i]
                entry[6]   = 1.0
                matches.append(entry)

        elif method == 'simulated_fractional':
            min_vals, obs_idx = torch.min(dists, dim=0)
            sim_mask = (min_vals <= q_cutoff)
            counts   = torch.bincount(obs_idx[sim_mask],
                                      minlength=N_obs).float()
            weights  = torch.zeros(sim_q.size(0), device=device)
            weights[sim_mask] = 1.0 / counts[obs_idx[sim_mask]]

            for j in range(sim_q.size(0)):
                if not sim_mask[j]:
                    continue
                i   = obs_idx[j]
                entry = torch.empty(7, device=device, dtype=rel.dtype)
                entry[0:3] = rel[j].float()
                entry[3]   = i
                entry[4:6] = test_q[i]
                entry[6]   = weights[j]
                matches.append(entry)
        # stack (or empty)
        im_list.append(torch.stack(matches, dim=0)
                       if matches else torch.empty((0,7), device=device))

    # 7) build plot_dict if requested
    if return_plot_dict:
        # you can adapt this to show all phases’ sim_q if you like:
        a,b,c,α,β,γ = GixdUtility.reciprocal_lattice_to_unit_cell_deg(
            k1_list[0], k2_list[0], k3_list[0]
        )
        title = f"[{a:.3f} {b:.3f} {c:.3f} {α:.3f} {β:.3f} {γ:.3f}]"
        plot_dict = {
            "test_qxy": txy.detach().cpu().numpy(),
            "test_qz":  tz .detach().cpu().numpy(),
            "sim_q":    [sq.detach().cpu().numpy() for sq in sim_q_list],
            "indices_match": [
                im.detach().cpu().numpy() for im in im_list
            ],
            "plot_title": title
        }
    else:
        plot_dict = {"return_plot_dict": return_plot_dict}

    return im_list, plot_dict, sim_counts

def match_laue_indices_nphase_old(used_laue_indices, k1_list, k2_list, k3_list,
                                        test_qxy, test_qz,
                                        data_generator: GixdDataGenerator,
                                        q_cutoff, return_plot_dict:bool=False):
    device = data_generator.device
    test_qxy = torch.as_tensor(test_qxy, device=device)
    test_qz = torch.as_tensor(test_qz, device=device)
    mask = (test_qxy > 0.01) | (test_qz > 0.0)
    test_q = torch.stack([test_qxy[mask], test_qz[mask]], dim=1)

    #k1, k2, k3 = [torch.as_tensor(x, device=device) for x in (k1, k2, k3)]
    num_phases = len(k1_list)
    sim_q_list = []
    used_list = []
    rel_list = []
    for i in range(num_phases):
        rec = torch.stack([k1_list[i], k2_list[i], k3_list[i]], dim=-1)
        with torch.no_grad():
            sim_xy, sim_z = data_generator.calc_q_points(rec)
        sim_xy = sim_xy.to(device).squeeze(0)
        sim_z = sim_z.to(device).squeeze(0)
        mask2 = (sim_xy > 0.01) | (sim_z > 0.0)
        sim_q_list.append(torch.stack([sim_xy[mask2], sim_z[mask2]], dim=1))
        used_list.append(torch.as_tensor(used_laue_indices, device=device))
        rel_list.append(used_list[i][mask2])

    # handle no simulated or no observed points
    if test_q.size(0) == 0 or sim_q_list[0].size(0) == 0 or sim_q_list[1].size(0)== 0: #TODO: this is hardcoded for two phases :(
        plot_dict = {
            "test_qxy": test_qxy.detach().cpu().numpy(),  # 1D tensor (N,)
            "test_qz": test_qz.detach().cpu().numpy(),    # 1D tensor (N,)
            'sim_q': [sim_q.detach().cpu().numpy() for sim_q in sim_q_list],
            "indices_match": num_phases*[torch.empty((0, 6), device=device).detach().cpu().numpy()],  # tensor of shape (num_matches, 6)
            "plot_title": "no obs q" if test_q.size(0)==0 else "no simulated q"
        }
        return torch.empty((0, 7), device=device), plot_dict, [sim_q.size(0) for sim_q in sim_q_list]

    dist_list = [torch.norm(test_q.unsqueeze(1) - sim_q.unsqueeze(0), dim=2) for sim_q in sim_q_list]
    matches_list = []
    for i in range(num_phases):
        matches_list.append([])

    for i in range(test_q.size(0)):
        md = 1e3
        idx = None
        best_p = None
        for p in range(num_phases):
            md_p, idx_p = torch.min(dist_list[p][i], dim=0)
            if md_p < md:
                md = md_p
                idx = idx_p
                best_p = p
        if md <= q_cutoff and idx.item():
            laue = rel_list[best_p][idx].float()
            matches_list[best_p].append(torch.cat([
                laue,
                torch.tensor([i, test_q[i, 0], test_q[i, 1]], device=device)
            ]))
            #usedi.add(idx.item())
    im_list = [torch.stack(matches, dim=0) if matches else torch.empty((0, 6), device=device) for matches in matches_list]
    if return_plot_dict:
        #a_val, b_val, c_val, alpha_val, beta_val, gamma_val = GixdUtility.reciprocal_lattice_to_unit_cell_deg(
        #    k1.cpu().numpy(), k2.cpu().numpy(), k3.cpu().numpy())
        #plot_title = f"[{a_val:.3f} {b_val:.3f} {c_val:.3f} {alpha_val:.3f} {beta_val:.3f} {gamma_val:.3f}]"
        plot_dict = {
            "test_qxy": test_qxy.detach().cpu().numpy(),  # 1D tensor (N,)
            "test_qz": test_qz.detach().cpu().numpy(),    # 1D tensor (N,)
            'sim_q': [sim_q.detach().cpu().numpy() for sim_q in sim_q_list],        # tensor of shape (N_sim_valid, 2)
            "indices_match": [im.detach().cpu().numpy() for im in im_list],  # tensor of shape (num_matches, 6)
            "plot_title": "title not implemented"
        }
    else:
        plot_dict = {"return_plot_dict": return_plot_dict}
    return im_list, plot_dict, [sim_q.size(0) for sim_q in sim_q_list]

def direct_inversion_pytorch_nphase(first_predicted_cell_deg: list, #list of n phases [[a,b,c,...],[a,b,c,...]]
                             observed_qxy, observed_qz, #unchanged
                             trial_top_layer, #list on n 3-tuples [(h,k,l), (h,k,l)]
                             data_generator: GixdDataGenerator,
                             method: str = 'cg', #good default, don't change for now
                             loss_type: str = 'l2', #l2 is much better than l1
                             q_cutoff: float = 0.1, #0.1 is probably good but should be optimized later
                             maxiter: int = 50, #don't change
                             outer_loops: int = 10, #10 should be good, but could be increased to 20 for two phases
                             return_uncertainties: bool = False #False, we will work on this later, no priority now
                             ) -> Tensor:
    assert first_predicted_cell_deg.device.type == data_generator.device.type, \
        f"Device mismatch: cell on {first_predicted_cell_deg.device}, generator on {data_generator.device}"
    num_phases = len(first_predicted_cell_deg)
    assert num_phases == len(trial_top_layer), f"there must be one top_layer for each predicted_cell_deg"
    best_x_list = [GixdUtility.normalize_cell_deg(cell).to(data_generator.device) for cell in first_predicted_cell_deg]

    for outer in range(outer_loops):
        x_t  = torch.concat(best_x_list).clone().detach()
        #with torch.no_grad():
        if True:
            k1_list, k2_list, k3_list = [], [], []
            for best_x in best_x_list:
                denorm = GixdUtility.denormalize_cell_torch(best_x)
                v1, v2, v3 = data_generator.calc_real_vectors_from_unit_cell_rad(denorm)
                R = data_generator.calc_rotation_matrix(
                    v1, v2, v3,
                    torch.tensor([trial_top_layer], device=data_generator.device, dtype=torch.int64)
                )
                v1r, v2r, v3r = data_generator.rotate_cell_by_matrix(v1, v2, v3, R)
                k1, k2, k3 = data_generator.calc_reciprocal_cell(v1r, v2r, v3r)
                k1_list.append(k1[0])
                k2_list.append(k2[0])
                k3_list.append(k3[0])

            if outer==0:
                indices_match, init_plot_dict, init_num_sim = match_laue_indices_nphase(
                    data_generator.laue_indices_tensor,
                    k1, k2, k3,
                    observed_qxy, observed_qz,
                    data_generator,
                    q_cutoff,
                    return_plot_dict=True
                )
            else:
                indices_match, _, _ = match_laue_indices_nphase(
                    data_generator.laue_indices_tensor,
                    k1, k2, k3,
                    observed_qxy, observed_qz,
                    data_generator,
                    q_cutoff,
                    return_plot_dict=False
                )
            if len(indices_match[0]) == 0 and len(indices_match[1]) == 0: #  matching failed
                return {"refined_predicted_cell_deg": denorm,
                        "refined_uncertainty_cell_deg": torch.zeros_like(denorm),
                        "convergence_status": False,
                        "init_plot_dict": init_plot_dict,
                        "refined_plot_dict": init_plot_dict,
                        "refined_q_loss": 1e3,
                        "refined_num_matches": 0,
                        "refined_num_sim_peaks": init_num_sim}

        def loss_fun(x_concat):
            return direct_inversion_loss_nphase(x_concat, indices_match, data_generator, trial_top_layer, loss_type)

        if method in ["cg"]:
            return_uncertainties = False

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
        best_x_list = [result.x[i*6:(i+1)*6].clone.detach() for i in range(num_phases)]

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

    refined_indices_match, refined_plot_dict, refined_num_sim_peaks = match_laue_indices_nphase(
        data_generator.laue_indices_tensor,
        k1, k2, k3,
        observed_qxy, observed_qz,
        data_generator,
        q_cutoff,
        return_plot_dict=True
    )
    
    refined = GixdUtility.denormalize_cell_deg(best_x)
    
    uncertainty_cell_deg = torch.zeros_like(refined)

    return_dict = {
        "refined_predicted_cell_deg": refined,
        "refined_uncertainty_cell_deg": uncertainty_cell_deg,
        "convergence_status": result.status,
        "init_plot_dict": init_plot_dict,
        "refined_plot_dict": refined_plot_dict,
        "refined_q_loss": result.fun,
        "refined_num_matches": len(refined_indices_match),
        "refined_num_sim_peaks": refined_num_sim_peaks
    }
    return return_dict


def ls_res(x_np, data_generator, trial_top_layer, obs_xy, obs_z, q_cutoff):
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
        q_cutoff
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