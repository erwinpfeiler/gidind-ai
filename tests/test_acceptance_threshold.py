from types import SimpleNamespace
import numpy as np
import pytest
import torch

@pytest.mark.parametrize('loss,coverage,sigma,accepted', [
    (9.99e-4, 1.0, 0.01, True),
    (1e-3, 1.0, 0.01, False),
    (1.001e-3, 1.0, 0.01, False),
    (float('nan'), 1.0, 0.01, False),
    (float('inf'), 1.0, 0.01, False),
    (1e-4, 0.9, 0.01, True),
    (1e-4, 0.8, 0.01, False),
    (1e-4, 1.0, 0.11, False),
])
@pytest.mark.parametrize('only_converged', [False, True])
def test_real_case_acceptance_boundary(monkeypatch, loss, coverage, sigma, accepted, only_converged):
    import gixd_regressor as module
    plot={'indices_match':np.zeros((int(10*coverage),7)), 'test_qxy':np.zeros(10)}
    result={'refined_uncertainty_cell_deg':torch.full((6,),sigma,dtype=torch.float64),
            'refined_q_loss':loss,'refined_plot_dict':plot,'init_plot_dict':plot,
            'refined_predicted_cell_deg':torch.tensor([5.,6.,7.,90.,90.,90.])}
    monkeypatch.setattr(module,'direct_inversion_pytorch',lambda **kwargs:result)
    output=module._unknown_refine_worker(
        {'sample_index':0,'q_xy':np.zeros(10),'q_z':np.zeros(10),
         'pred_cell_deg':[[5.,6.,7.,90.,90.,90.]]},
        data_generator=SimpleNamespace(),
        feature_config=SimpleNamespace(miller_indices=[(0,0,1)]),
        device=torch.device('cpu'),q_cutoff_1=0.06,q_cutoff_2=0.06,
        refinement_backend='serial',return_only_converged=only_converged)
    if only_converged:
        assert ('is_converged0' in output)==accepted
    else:
        assert output['is_converged0']==accepted
