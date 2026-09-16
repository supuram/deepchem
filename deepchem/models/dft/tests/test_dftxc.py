import pytest
import os
import numpy as np
try:
    import torch
except:
    pass
from deepchem.models.dft.dftxc import DFTXC, XCModel, ExpM1Activation, _construct_nn_model


@pytest.mark.torch
def test_dftxc_water_atomization():
    from deepchem.feat.dft_data import DFTEntry
    nn = torch.nn.Sequential(
        torch.nn.Linear(2, 4),
        torch.nn.Linear(4, 1, bias=False)
    ).to(torch.double)

    model = DFTXC(xcstr="lda_x", nnmodel=nn, aweight0=0.0)

    systems = [
        {'moldesc': 'O 0.0 0.0 0.0; H 0.0 1.43 1.11; H 0.0 -1.43 1.11', 'basis': 'sto-3g', 'spin': 0, 'number': 1},
        {'moldesc': 'O 0.0 0.0 0.0', 'basis': 'sto-3g', 'spin': 2, 'number': 1},
        {'moldesc': 'H 0.0 0.0 0.0', 'basis': 'sto-3g', 'spin': 1, 'number': 2}
    ]
    entry = DFTEntry.create(e_type='ae', true_val='0.5', systems=systems)

    output = model([entry])

    assert len(output) == 1
    val = output[0]
    assert isinstance(val, torch.Tensor)
    assert not torch.isnan(val).any(), "SCF produced NaN energy"
    assert not torch.isinf(val).any(), "SCF produced Infinite energy"
    
@pytest.mark.torch
def test_xcmodel():
    from deepchem.data.data_loader import DFTYamlLoader
    import deepchem
    dc_root = os.path.dirname(deepchem.__file__)
    asset_path = os.path.join(dc_root, "models", "tests", "assets", "test_dftxcdata.yaml")
    if not os.path.exists(asset_path):
        asset_path = os.path.abspath("deepchem/models/tests/assets/test_dftxcdata.yaml")
        if not os.path.exists(asset_path):
            pytest.skip(f"Asset file not found at {asset_path}")
    data = DFTYamlLoader()
    dataset = data.create_dataset(asset_path)
    dataset.get_shape()
    model = XCModel("lda_x", batch_size=1)
    loss = model.fit(dataset, nb_epoch=1, checkpoint_interval=1)
    
    assert len(dataset) > 0
    assert loss is not None
    assert not np.isnan(loss)


@pytest.mark.torch
def test_ExpM1Activation():
    from deepchem.models.dft.dftxc import ExpM1Activation
    model = ExpM1Activation()
    x = torch.tensor(2.5)
    output = model(x)
    
    assert output is not None
    assert not np.isnan(output)


@pytest.mark.torch
def test_construct_nn_model():
    model_type1 = _construct_nn_model(input_size=2, hidden_size=8, n_layers=2, modeltype=1)
    model_type2 = _construct_nn_model(input_size=2, hidden_size=8, n_layers=2, modeltype=2)

    x = torch.randn(5, 2)
    out1 = model_type1(x)
    out2 = model_type2(x)

    assert out1.shape == (5, 1)
    assert out2.shape == (5, 1)
