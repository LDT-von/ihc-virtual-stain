"""CPU interface checks for the real pinned SMP adapter; no training jobs."""
import copy
import importlib.util
import sys
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import train_semifinal_v7 as v7
from compare_semifinal_v7 import require_matched_training
from src.data.roi_manifest import MARKERS
from src.models.marker_smp_unet import MarkerSMPUNet, SMP_VERSION
from src.train_marker_context import (build_reconstruction_model, environment,
                                      load_model, predict, update_ema)


@pytest.fixture
def smp_available():
    if importlib.util.find_spec('segmentation_models_pytorch') is None:
        pytest.skip('Optional SMP dependency is not installed')


@pytest.fixture(autouse=True)
def single_cpu_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_real_four_marker_gradients_and_rectangular_tta_without_downloads(smp_available):
    with mock.patch('torch.hub.load_state_dict_from_url',
                    side_effect=AssertionError('Weights must not be downloaded')):
        model = build_reconstruction_model({
            'architecture': 'smp_resnet34_unet', 'markers': 4, 'width': 4,
            'smp_version': SMP_VERSION,
        })
    assert model.network.encoder.conv1.in_channels == 1
    image = torch.rand(1, 1, 17, 23)
    prediction = model(image)
    assert prediction.shape == (1, 4, 17, 23)
    assert torch.isfinite(prediction).all()
    assert ((prediction >= 0) & (prediction <= 1)).all()
    loss = v7.reconstruction_loss(prediction, torch.rand_like(prediction),
                                  torch.ones(4), 'normalized')
    loss.backward()
    assert torch.isfinite(loss)
    for weight in (model.network.encoder.conv1.weight,
                   model.network.segmentation_head[0].weight):
        assert weight.grad is not None
        assert torch.isfinite(weight.grad).all()
        assert weight.grad.abs().sum() > 0
    model.eval()
    with torch.no_grad():
        averaged = predict(model, image, tta=8)
    assert averaged.shape == prediction.shape
    assert torch.isfinite(averaged).all()


def test_marker_intensities_are_independent_instead_of_softmax_classes(smp_available):
    model = MarkerSMPUNet(markers=4, width=4).eval()
    with torch.no_grad():
        model.network.segmentation_head[0].weight.zero_()
        model.network.segmentation_head[0].bias.zero_()
        prediction = model(torch.rand(1, 1, 32, 37))
    torch.testing.assert_close(prediction, torch.full_like(prediction, 0.5))
    torch.testing.assert_close(prediction.sum(dim=1), torch.full((1, 32, 37), 2.0))


def test_single_marker_checkpoint_roundtrip_and_dependency_guard(tmp_path, smp_available):
    model = MarkerSMPUNet(markers=1, width=4).eval()
    config = {'architecture': 'smp_resnet34_unet', 'markers': 1, 'width': 4,
              'smp_version': SMP_VERSION}
    run = {'markers': ['CD68'], 'model_config': config, 'environment': environment()}
    path = tmp_path / 'single.pt'
    torch.save({'format': 1, 'run': run, 'ema': model.state_dict()}, path)
    restored, checkpoint = load_model(path, torch.device('cpu'))
    assert checkpoint['run']['markers'] == ['CD68']
    image = torch.rand(1, 1, 35, 47)
    with torch.no_grad():
        torch.testing.assert_close(restored(image), model(image), rtol=0, atol=0)
    damaged = copy.deepcopy(run)
    damaged['environment']['source_sha256']['smp/decoders/unet/model.py'] = 'different'
    torch.save({'format': 1, 'run': damaged, 'ema': model.state_dict()}, path)
    with pytest.raises(ValueError, match='dependency mismatch'):
        load_model(path, torch.device('cpu'))


def test_ema_copies_batchnorm_statistics_and_averages_parameters():
    model = torch.nn.BatchNorm2d(4)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    before = ema.weight.clone()
    with torch.no_grad():
        model.weight.fill_(3)
    model(torch.rand(2, 4, 8, 8) + 4)
    assert not torch.equal(ema.running_mean, model.running_mean)
    update_ema(ema, model, 0.75)
    torch.testing.assert_close(ema.weight, before * 0.75 + model.weight * 0.25)
    for name, buffer in model.named_buffers():
        torch.testing.assert_close(dict(ema.named_buffers())[name], buffer, rtol=0, atol=0)


def test_constructor_input_and_optional_dependency_contracts(smp_available):
    for config in ({'width': 0}, {'width': True}, {'markers': 0},
                   {'markers': 1.5}, {'smp_version': 'other'}):
        with pytest.raises(ValueError):
            MarkerSMPUNet(**config)
    with mock.patch.dict(sys.modules, {'segmentation_models_pytorch': None}):
        with pytest.raises(ImportError, match='requirements-smp-unet.txt'):
            MarkerSMPUNet()
    with mock.patch.dict(sys.modules, {
        'segmentation_models_pytorch': SimpleNamespace(__version__='0.0.0'),
    }):
        with pytest.raises(RuntimeError, match='Expected segmentation'):
            MarkerSMPUNet()
    model = MarkerSMPUNet(width=4)
    for image in (torch.rand(1, 3, 32, 32), torch.rand(1, 1, 15, 32),
                  torch.rand(1, 32, 32)):
        with pytest.raises(ValueError):
            model(image)
    with pytest.raises(TypeError):
        model(torch.zeros(1, 1, 32, 32, dtype=torch.uint8))


def test_v7_cli_and_configs_without_starting_fit(smp_available):
    args = SimpleNamespace(architecture='smp_resnet34_unet', width=16)
    for markers in (MARKERS, ('CD68',)):
        config = v7.model_config_from_args(args, markers)
        assert config['markers'] == len(markers)
        assert config['smp_version'] == SMP_VERSION
        assert 'context' not in config
    arguments = ['train_semifinal_v7.py', 'fit', '--architecture', 'smp_resnet34_unet',
                 '--target-marker', 'CD68', '--data-root', 'not-read',
                 '--manifest', 'not-read', '--output', 'not-created', '--fold', '0']
    with mock.patch.object(sys, 'argv', arguments), mock.patch.object(v7, 'fit') as fit:
        v7.main()
    assert fit.call_args.args[0].architecture == 'smp_resnet34_unet'
    assert fit.call_args.args[0].target_marker == 'CD68'
    hashes = v7.source_hashes()
    assert 'src/models/marker_smp_unet.py' in hashes
    assert 'requirements-smp-unet.txt' in hashes
    assert 'smp/encoders/resnet.py' in environment()['source_sha256']


def test_matched_training_family_width_and_recipe_guards():
    baseline = {'architecture': 'context', 'width': 96, 'lr': 3e-4, 'loss': 'normalized'}
    candidate = {**baseline, 'architecture': 'smp_resnet34_unet', 'width': 16}
    require_matched_training(baseline, candidate)
    with pytest.raises(ValueError, match='lr'):
        require_matched_training(baseline, {**candidate, 'lr': 1e-4})
    with pytest.raises(ValueError, match='width'):
        require_matched_training(candidate, {**candidate, 'width': 32})
    with pytest.raises(ValueError, match='width'):
        require_matched_training(baseline, {**baseline, 'width': 48})
