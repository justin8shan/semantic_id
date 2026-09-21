import torch

from semantic_id.model.losses import compute_loss
from semantic_id.model.rqvae import RQVAE, Decoder, Encoder, RQVAEConfig


def _small_config(**overrides):
    base = dict(d_in=16, W=8, L=2, d=4, beta=0.25, gamma=0.9, dead_code_reset_threshold=0.05)
    base.update(overrides)
    return RQVAEConfig(**base)


def test_encoder_decoder_shapes():
    encoder = Encoder(d_in=16, d=4)
    decoder = Decoder(d=4, d_out=16)
    x = torch.randn(5, 16)
    z = encoder(x)
    assert z.shape == (5, 4)
    x_hat = decoder(z)
    assert x_hat.shape == (5, 16)


def test_forward_shapes_and_code_indices_in_range():
    config = _small_config()
    model = RQVAE(config)
    x = torch.randn(10, 16)
    output = model(x, training=False)
    assert output.x_hat.shape == x.shape
    assert len(output.code_indices_per_level) == config.L
    for idx in output.code_indices_per_level:
        assert idx.shape == (10,)
        assert idx.min() >= 0 and idx.max() < config.W


def test_gradient_flows_through_straight_through_to_encoder():
    config = _small_config()
    model = RQVAE(config)
    x = torch.randn(32, 16)

    output = model(x, training=True)
    loss = compute_loss(x, output, config.beta)
    loss.total.backward()

    encoder_grads = [p.grad for p in model.encoder.parameters()]
    assert all(g is not None for g in encoder_grads)
    assert any(g.abs().sum().item() > 0 for g in encoder_grads)

    # Codebooks are EMA buffers, not autograd parameters — they must not
    # appear among the trainable parameters the optimizer would step.
    trainable_names = {name for name, p in model.named_parameters()}
    assert not any("codebook" in name for name in trainable_names)


def test_ema_update_moves_codebook_toward_assigned_data():
    config = _small_config(W=4, gamma=0.5)
    model = RQVAE(config)
    level = model.quantizer.levels[0]
    initial_codebook = level.codebook.clone()

    # Push a batch of clearly-clustered data through in training mode.
    x = torch.randn(64, config.d_in) + 5.0
    model(x, training=True)

    assert not torch.allclose(level.codebook, initial_codebook)
    assert level.ema_cluster_size.sum().item() > 0


def test_kmeans_init_reduces_quantization_error_vs_random_init():
    config = _small_config(W=4, d=4)
    torch.manual_seed(0)

    # Clustered data in latent space: 4 tight clusters matching W=4.
    centers = torch.randn(4, config.d) * 5
    data = centers.repeat_interleave(20, dim=0) + torch.randn(80, config.d) * 0.05

    model_random = RQVAE(config)
    dist_random = torch.cdist(data, model_random.quantizer.levels[0].codebook)
    error_random = dist_random.min(dim=1).values.mean()

    model_kmeans = RQVAE(config)
    model_kmeans.quantizer.levels[0].kmeans_init(data, n_iters=10)
    dist_kmeans = torch.cdist(data, model_kmeans.quantizer.levels[0].codebook)
    error_kmeans = dist_kmeans.min(dim=1).values.mean()

    assert error_kmeans < error_random


def test_dead_code_reset_reseeds_unused_codes():
    from semantic_id.model.rqvae import EMAVectorQuantizerLevel

    level = EMAVectorQuantizerLevel(codebook_size=4, code_dim=4, gamma=0.9, dead_code_reset_threshold=0.1)

    # Force all usage onto code 0 by making it the only near vector, then run
    # a training forward pass so epoch_usage_count reflects that skew.
    level.codebook.data[0] = torch.zeros(4)
    level.codebook.data[1:] = torch.ones(3, 4) * 100
    residual = torch.zeros(50, 4)
    level(residual, training=True)

    usage_frac = level.epoch_usage_count / level.epoch_usage_count.sum()
    assert (usage_frac < 0.1).any()

    high_error_pool = torch.randn(10, 4) * 50
    codebook_before = level.codebook.clone()
    n_reset = level.dead_code_reset(high_error_pool)

    assert n_reset > 0
    assert not torch.allclose(level.codebook, codebook_before)
