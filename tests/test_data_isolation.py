"""Data preparation and development/evaluation boundaries, without model downloads."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from read_crystal_data_2 import (
    CELL_COLUMNS,
    REDUCED_COLUMNS,
    checked_niggli_cell,
    remove_duplicates_from_df,
)


def frame(cells):
    return pd.DataFrame(cells, columns=CELL_COLUMNS)


def test_equivalent_permuted_cells_are_removed_using_reduced_keys():
    cells = frame([[5, 7, 9, 90, 90, 90], [9, 5, 7, 90, 90, 90],
                   [5.0003, 7, 9, 90, 90, 90], [5.0008, 7, 9, 90, 90, 90]])
    result = remove_duplicates_from_df(cells)
    assert len(result) == 2
    np.testing.assert_allclose(result.iloc[0][REDUCED_COLUMNS].to_numpy(float), [5, 7, 9, 90, 90, 90])
    assert result.iloc[1]['a'] == 5.0008  # keys rounded; retained data are not quantized


def test_deduplication_is_global_across_systems():
    seen = set()
    assert len(remove_duplicates_from_df(frame([[5, 7, 9, 90, 90, 90]]), seen_keys=seen)) == 1
    assert len(remove_duplicates_from_df(frame([[7, 9, 5, 90, 90, 90]]), seen_keys=seen)) == 0


@pytest.mark.parametrize('cell', [[0, 7, 9, 90, 90, 90], [5, 7, 9, 1, 1, 179],
                                 [np.nan, 7, 9, 90, 90, 90]])
def test_invalid_cells_do_not_become_reduction_failure_sentinels(cell):
    with pytest.raises(ValueError):
        checked_niggli_cell(cell)


def test_fresh_split_saves_reduced_partitions_and_cannot_overlap(tmp_path, monkeypatch):
    from unit_cell_generator import UnitCellGenerator
    monkeypatch.chdir(tmp_path)
    generator = object.__new__(UnitCellGenerator)
    generator.lattice_types = ['orthorhombic', 'triclinic']
    generator.seed = 42
    first = [[5+i/10, 7, 9, 90, 90, 90] for i in range(20)]
    second = [[7, 9, 5+i/10, 90, 90, 90] for i in range(5)] + [
        [5+i/10, 7, 9, 80, 85, 95] for i in range(10)]
    generator.load_data({'orthorhombic': frame(first), 'triclinic': frame(second)})
    train, test = set(), set()
    for system in generator.lattice_types:
        for folder, keys in [('training_unit_cells', train), ('test_unit_cells', test)]:
            cells = np.load(Path(folder) / f'{system}_niggli.npy')
            keys.update(map(tuple, np.round(cells, 3)))
    assert not train & test
    assert len(train | test) == 30
    manifest = json.loads(Path('training_unit_cells/split_manifest.json').read_text())
    assert manifest['deduplication_scope'] == 'global_before_any_split'
    before = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    generator.load_data({})  # reuse cached data without requiring COD again
    assert all(p.read_bytes() == content for p, content in before.items())
    Path('test_unit_cells/triclinic.npy').unlink()
    with pytest.raises(ValueError, match='Incomplete cached'):
        generator.load_data({})


def test_validation_requests_larger_than_pool_return_correct_size():
    from unit_cell_generator import UnitCellGenerator
    generator = object.__new__(UnitCellGenerator)
    generator.validation_crystals = {'cubic': np.array([[5, 5, 5, 90, 90, 90.]])}
    generator.train_crystals = {'cubic': np.array([[10, 10, 10, 90, 90, 90.]])}
    generator.test_crystals = {'cubic': np.array([[20, 20, 20, 90, 90, 90.]])}
    generator.rng = np.random.default_rng(42)
    generator.laue_indices = [[1, 0, 0]]
    generator.calc_intensities = False
    generator._filter_sample = lambda x: x
    generator._convert_to_niggli = lambda x, n: x
    cells, mask = generator.get_validation_unit_cells(7, 'cubic')
    assert cells.shape == (7, 6) and mask.shape == (7, 1)
    assert np.all(cells[:, 0] == 5)


@pytest.mark.parametrize('split', [None, 'testing', 'training'])
def test_refinement_tuning_rejects_test_or_unlabelled_csv(tmp_path, split):
    from gixd_regressor import GixdRegressor
    generator = object.__new__(GixdRegressor)
    rows = {'sample_index': [0]}
    if split is not None:
        rows['data_split'] = [split]
    source = tmp_path / 'predictions.csv'
    pd.DataFrame(rows).to_csv(source, index=False)
    with pytest.raises(ValueError, match='validation-only'):
        generator.optimize_refinement_hyperparameters(input_csv=str(source), n_trials=1)


def test_default_tuning_generates_validation_predictions(tmp_path):
    from gixd_regressor import GixdRegressor
    generator = object.__new__(GixdRegressor)
    generator.path_config = SimpleNamespace(root_path=str(tmp_path))
    class StopAfterProvenance(Exception):
        pass
    def predict(**kwargs):
        assert kwargs['data_split'] == 'validation'
        raise StopAfterProvenance
    generator.predict_and_save_csv = predict
    with pytest.raises(StopAfterProvenance):
        generator.optimize_refinement_hyperparameters(n_trials=1)


def test_final_training_metrics_never_open_test_loader():
    from gixd_regressor import GixdRegressor
    def forbidden():
        raise AssertionError('test loader was opened during training')
    model = SimpleNamespace(train_dataloader=lambda: iter([None]), val_dataloader=lambda: iter([None]),
                            test_dataloader=forbidden,
                            summarize_single_batch=lambda batch, stage: {
                                f'{stage}_{key}': 0 for key in ['loss','mae','min_mae','max_weight_mae','std']})
    GixdRegressor._print_final_swa_batch_metrics(SimpleNamespace(gixd_model=model))


def test_training_manifest_update_count_matches_actual_accumulation():
    from gidind_ai.training import ACCUMULATE_GRAD_BATCHES, _expected_optimizer_steps
    assert ACCUMULATE_GRAD_BATCHES == 1
    assert _expected_optimizer_steps(100000, 256) == 391


def test_cod_parser_reads_cif_and_keeps_provenance(tmp_path):
    from read_crystal_data_2 import load_cod_to_pickle
    (tmp_path/'123.cif').write_text('''data_123
_symmetry_Int_Tables_number 1
_cell_length_a 9.0(2)
_cell_length_b 5
_cell_length_c 7
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
''')
    load_cod_to_pickle(tmp_path, tmp_path/'parsed')
    data = pd.read_pickle(tmp_path/'parsed.pkl')
    assert data.iloc[0]['cod_id'] == '123'
    assert data.iloc[0]['a'] == 9
    np.testing.assert_allclose(data[REDUCED_COLUMNS].to_numpy(), [[5, 7, 9, 90, 90, 90]])


def test_cubic_uniform_sampler_broadcasts_for_arbitrary_batch_size():
    from unit_cell_generator import UnitCellGenerator
    generator = object.__new__(UnitCellGenerator)
    generator.rng = np.random.default_rng(42)
    generator.lower_bounds_arr = np.array([3,3,3,60,60,60])
    generator.upper_bounds_arr = np.array([30,30,30,120,120,120])
    generator.return_niggli = False
    cells = generator.sample_unit_cells_uniform(11, 'cubic')
    assert cells.shape == (11, 6)
    np.testing.assert_array_equal(cells[:, 0], cells[:, 1])
    np.testing.assert_array_equal(cells[:, 1], cells[:, 2])


def test_unknown_orientation_tuning_uses_validation_and_repeats_same_rng_draws(tmp_path):
    from gixd_regressor import GixdRegressor
    regressor = object.__new__(GixdRegressor)
    regressor.path_config = SimpleNamespace(root_path=str(tmp_path))
    validation = SimpleNamespace(rng=np.random.default_rng(100),
                                 unit_cell_generator=SimpleNamespace(rng=np.random.default_rng(200)))
    regressor.validation_data_generators = {'uniform': validation}
    draws = []
    def benchmark(**kwargs):
        assert kwargs['data_split'] == 'validation'
        draws.append((validation.rng.random(), validation.unit_cell_generator.rng.random()))
        return .8, 2
    regressor.predict_and_refine_unknown_benchmark = benchmark
    study = regressor.optimize_unknown_refinement(num_trials=2, num_samples=2)
    assert len(draws) == 2 and draws[0] == draws[1]
    assert study.best_value == pytest.approx(79.8)


def test_gmm_fit_reserves_validation_without_using_test_cells(monkeypatch):
    from sklearn.mixture import GaussianMixture

    from unit_cell_generator import UnitCellGenerator
    generator = object.__new__(UnitCellGenerator)
    generator.lattice_types = ['cubic']
    generator.train_crystals = {'cubic': np.array([[float(i)]*3 + [90]*3 for i in range(4, 24)])}
    generator.test_crystals = {'cubic': np.array([[1000.]*3 + [90]*3])}
    generator.seed = 42
    generator.optimize_gmm = False
    generator.gaussian_mixture_models = {}
    generator.BEST_NUM_GMM_COMPONENTS = {'cubic': 1}
    fit_inputs = []
    original_fit = GaussianMixture.fit
    def fit(model, values, *args, **kwargs):
        fit_inputs.append(values.copy())
        return original_fit(model, values, *args, **kwargs)
    monkeypatch.setattr(GaussianMixture, 'fit', fit)
    generator.learn_crystal_distribution(1)
    validation = generator.validation_crystals['cubic'][:, 0]
    for fitted in fit_inputs:
        assert not set(fitted[:, 0]) & set(validation)
        assert 1000 not in fitted
    # Integer random_state makes sklearn repeat every same-size sample. A
    # persistent RandomState instead yields different draws and supports resume.
    gmm = generator.gaussian_mixture_models['cubic']
    assert not np.array_equal(gmm.sample(8)[0], gmm.sample(8)[0])


def test_tetragonal_coordinates_retain_legacy_min_max_mapping():
    from unit_cell_generator import UnitCellGenerator
    cells = [[10,10,5,90,90,90], [5,10,10,90,90,90], [10,5,10,90,90,90]]
    result = UnitCellGenerator._gmm_coordinates(cells, 'tetragonal')
    np.testing.assert_array_equal(result, [[5,10]]*3)
    assert result.dtype == np.float32


def test_monoclinic_coordinates_retain_legacy_length_order_and_angle_selection():
    from unit_cell_generator import UnitCellGenerator
    cells = [[5,7,9,90,105,90], [7,9,5,105,90,90], [9,5,7,90,90,105],
             [5,7,9,105,75,90], [5,7,9,105,90,75]]
    result = UnitCellGenerator._gmm_coordinates(cells, 'monoclinic')
    np.testing.assert_array_equal(result, [[5,7,9,105], [7,9,5,105], [9,5,7,105],
                                           [5,7,9,75], [5,7,9,75]])
    assert result.dtype == np.float32


def test_contaminated_cached_split_is_rejected_without_repair(tmp_path, monkeypatch):
    from unit_cell_generator import UnitCellGenerator
    monkeypatch.chdir(tmp_path)
    generator = object.__new__(UnitCellGenerator)
    generator.lattice_types = ['triclinic']
    generator.seed = 42
    Path('training_unit_cells').mkdir()
    Path('test_unit_cells').mkdir()
    np.savetxt('training_unit_cells/triclinic.npy', [[5,7,9,90,90,90]])
    path = Path('test_unit_cells/triclinic.npy')
    np.savetxt(path, [[9,5,7,90,90,90]])
    before = path.read_bytes()
    with pytest.raises(ValueError, match='train/test overlap'):
        generator.load_data({})
    assert path.read_bytes() == before


def test_gmm_sampling_replays_from_checkpointed_generator_state():
    from copy import deepcopy

    from sklearn.mixture import GaussianMixture

    from unit_cell_generator import UnitCellGenerator
    generator = object.__new__(UnitCellGenerator)
    generator.lattice_types = ['cubic']
    generator.return_niggli = False
    generator.rng = np.random.default_rng(42)
    gmm = GaussianMixture(1, random_state=42).fit(np.arange(5., 15.).reshape(-1, 1))
    generator.gaussian_mixture_models = {'cubic': gmm}
    state = deepcopy(generator.rng.bit_generator.state)
    first = generator.sample_unit_cells_from_gmm(8, 'cubic')
    second = generator.sample_unit_cells_from_gmm(8, 'cubic')
    assert not np.array_equal(first, second)
    generator.rng.bit_generator.state = state
    np.testing.assert_array_equal(first, generator.sample_unit_cells_from_gmm(8, 'cubic'))
    assert gmm.random_state == 42
