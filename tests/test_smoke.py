import numpy as np
import pytest


@pytest.mark.smoke
def test_inference_and_refinement_modules_import_without_models_or_external_data(
    monkeypatch, tmp_path
):
    """The supported core modules must import without model weights or input data."""
    monkeypatch.chdir(tmp_path)

    import gixd_regressor
    import polishing_fit
    from config_loader import FeatureConfig
    from gixd_utility import GixdUtility

    assert callable(gixd_regressor.GixdRegressor)
    assert callable(polishing_fit.direct_inversion_pytorch)
    assert FeatureConfig().image_size > 0

    cell = np.array([8.0, 9.0, 10.0, 90.0, 100.0, 110.0])
    normalized = GixdUtility.normalize_cell_deg(cell)
    np.testing.assert_allclose(GixdUtility.denormalize_cell_deg(normalized), cell)
