#!/usr/bin/env python3
"""
Test script for REINFORCE-based experimental design components.

Tests:
1. GPT2GenerativePolicy - generation and log-probability computation
2. Gradient flow through log-probabilities (REINFORCE requirement)
3. MonteCarloFisherObjective - Fisher estimation and rewards
4. Full optimization loop (with mock embeddings to avoid SD overhead)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_gpt2_policy_generation():
    """Test that GPT2GenerativePolicy generates text with valid log-probs."""
    logger.info("=" * 60)
    logger.info("Test 1: GPT2GenerativePolicy generation")
    logger.info("=" * 60)

    from components.gpt2_generative_policy import GPT2GenerativePolicy

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Create policy
    policy = GPT2GenerativePolicy(
        model_name="gpt2",
        lora_rank=4,  # Small rank for testing
        lora_alpha=8.0,
        device=device,
    )

    # Generate text
    text, log_prob = policy.generate_with_log_prob(
        prompt_prefix="A beautiful",
        max_new_tokens=10,
        temperature=1.0,
    )

    logger.info(f"Generated text: '{text}'")
    logger.info(f"Log probability: {log_prob.item():.4f}")
    logger.info(f"Log prob requires_grad: {log_prob.requires_grad}")

    # Verify log_prob is a valid tensor with gradients
    assert isinstance(log_prob, torch.Tensor), "log_prob should be a tensor"
    assert log_prob.requires_grad, "log_prob should require gradients"
    assert not torch.isnan(log_prob), "log_prob should not be NaN"
    assert not torch.isinf(log_prob), "log_prob should not be Inf"

    logger.info("TEST 1 PASSED: GPT2GenerativePolicy generates valid text with log-probs")
    return True


def test_gradient_flow():
    """Test that gradients flow from a loss through log_prob to LoRA parameters."""
    logger.info("=" * 60)
    logger.info("Test 2: Gradient flow through log-probability")
    logger.info("=" * 60)

    from components.gpt2_generative_policy import GPT2GenerativePolicy

    device = "cuda" if torch.cuda.is_available() else "cpu"

    policy = GPT2GenerativePolicy(
        model_name="gpt2",
        lora_rank=4,
        lora_alpha=8.0,
        device=device,
    )

    # Generate and get log_prob
    text, log_prob = policy.generate_with_log_prob(
        prompt_prefix="The",
        max_new_tokens=5,
    )

    # Simulate REINFORCE: loss = -reward * log_prob
    reward = torch.tensor(2.5, device=device)  # Positive reward
    loss = -reward * log_prob

    logger.info(f"Generated: '{text}'")
    logger.info(f"Log prob: {log_prob.item():.4f}")
    logger.info(f"Reward: {reward.item():.2f}")
    logger.info(f"Loss: {loss.item():.4f}")

    # Backward pass
    loss.backward()

    # Check gradients exist on LoRA parameters
    has_grad = False
    for name, param in policy.model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_norm = param.grad.norm().item()
            if grad_norm > 0:
                has_grad = True
                if "lora" in name.lower():
                    logger.info(f"  {name}: grad_norm = {grad_norm:.6e}")

    assert has_grad, "Expected gradients on LoRA parameters"
    logger.info("TEST 2 PASSED: Gradients flow to LoRA parameters")
    return True


def test_fisher_objective():
    """Test MonteCarloFisherObjective with mock embeddings."""
    logger.info("=" * 60)
    logger.info("Test 3: Monte Carlo Fisher Information objective")
    logger.info("=" * 60)

    from components.monte_carlo_fisher import MonteCarloFisherObjective

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create objective
    objective = MonteCarloFisherObjective(
        lambda_reg=1.0,
        embedding_dim=768,
        device=device,
    )

    # Create mock embeddings (K=4 policies, N=8 samples each)
    K, N, d = 4, 8, 768
    all_embeddings = []
    for q in range(K):
        embeddings_q = [
            torch.randn(d, device=device) for _ in range(N)
        ]
        # Normalize like CLIP
        embeddings_q = [e / e.norm() for e in embeddings_q]
        all_embeddings.append(embeddings_q)

    # Compute Fisher Information
    I_hat = objective.compute_fisher_info(all_embeddings)
    logger.info(f"Fisher Info shape: {I_hat.shape}")
    logger.info(f"Fisher Info trace: {I_hat.trace().item():.4f}")

    # Compute objective (log det)
    obj = objective.compute_objective(all_embeddings)
    logger.info(f"Objective (log det): {obj.item():.4f}")

    # Compute rewards
    rewards = objective.compute_rewards(all_embeddings, baseline="per_policy")
    logger.info(f"Rewards shape: {len(rewards)} policies, {len(rewards[0])} samples each")
    logger.info(f"Sample rewards (policy 0): {[r if isinstance(r, float) else r.item() for r in rewards[0][:3]]}")

    # Verify rewards are centered (baseline subtraction)
    mean_reward = sum(r if isinstance(r, float) else r.item() for r in rewards[0]) / len(rewards[0])
    logger.info(f"Mean reward for policy 0 (should be ~0): {mean_reward:.6f}")

    assert I_hat.shape == (d, d), "Fisher matrix should be d×d"
    assert not torch.isnan(obj), "Objective should not be NaN"

    logger.info("TEST 3 PASSED: Fisher objective computes correctly")
    return True


def test_reinforce_loss():
    """Test REINFORCE loss computation with mock data."""
    logger.info("=" * 60)
    logger.info("Test 4: REINFORCE loss computation")
    logger.info("=" * 60)

    from components.monte_carlo_fisher import MonteCarloFisherObjective

    device = "cuda" if torch.cuda.is_available() else "cpu"

    objective = MonteCarloFisherObjective(
        lambda_reg=1.0,
        device=device,
    )

    # Mock embeddings
    K, N, d = 2, 4, 64  # Smaller for speed
    all_embeddings = [
        [torch.randn(d, device=device) / (d**0.5) for _ in range(N)]
        for _ in range(K)
    ]

    # Mock log_probs (with gradients, simulating GPT-2 output)
    all_log_probs = [
        [torch.tensor(-5.0 + torch.randn(1).item(), device=device, requires_grad=True) for _ in range(N)]
        for _ in range(K)
    ]

    # Compute REINFORCE loss
    loss = objective.compute_reinforce_loss(all_embeddings, all_log_probs, baseline="per_policy")

    logger.info(f"REINFORCE loss: {loss.item():.4f}")
    logger.info(f"Loss requires_grad: {loss.requires_grad}")

    # Backward
    loss.backward()

    # Check gradients on log_probs
    has_grad = False
    for q in range(K):
        for j in range(N):
            if all_log_probs[q][j].grad is not None:
                grad = all_log_probs[q][j].grad.item()
                if abs(grad) > 1e-10:
                    has_grad = True
                    logger.info(f"  log_prob[{q}][{j}].grad = {grad:.6e}")

    assert has_grad, "Expected gradients on log_probs"
    logger.info("TEST 4 PASSED: REINFORCE loss computes with gradients")
    return True


def test_optimization_step():
    """Test a single optimization step with mock embeddings."""
    logger.info("=" * 60)
    logger.info("Test 5: Full optimization step (mock embeddings)")
    logger.info("=" * 60)

    from components.gpt2_generative_policy import MultiPolicyManager
    from components.monte_carlo_fisher import MonteCarloFisherObjective
    import torch.optim as optim

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create policies (K=2 for speed)
    K = 2
    policy_manager = MultiPolicyManager(
        num_policies=K,
        model_name="gpt2",
        lora_rank=4,
        lora_alpha=8.0,
        device=device,
    )

    # Create optimizer
    optimizer = optim.AdamW(policy_manager.get_all_trainable_parameters(), lr=1e-3)

    # Create Fisher objective
    fisher = MonteCarloFisherObjective(lambda_reg=1.0, device=device)

    # Track initial parameters
    initial_params = [
        p.clone().detach() for p in policy_manager.get_all_trainable_parameters()
    ]

    # Simulate one optimization step
    N = 4  # Samples per policy
    d = 768  # Embedding dim

    # Generate samples and get log_probs
    all_samples = policy_manager.generate_samples(
        num_samples_per_policy=N,
        prompt_prefix="A",
        max_new_tokens=5,
        temperature=1.0,
    )

    all_log_probs = [[log_prob for _, log_prob in samples_q] for samples_q in all_samples]

    # Mock embeddings (in real usage, these come from SD+CLIP)
    all_embeddings = [
        [torch.randn(d, device=device) / (d**0.5) for _ in range(N)]
        for _ in range(K)
    ]

    # Compute loss
    loss = fisher.compute_reinforce_loss(all_embeddings, all_log_probs, baseline="per_policy")
    logger.info(f"Loss before step: {loss.item():.4f}")

    # Optimization step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Check parameters changed
    current_params = list(policy_manager.get_all_trainable_parameters())
    params_changed = False
    for init_p, curr_p in zip(initial_params, current_params):
        if not torch.allclose(init_p, curr_p):
            params_changed = True
            break

    assert params_changed, "Parameters should change after optimization step"
    logger.info("TEST 5 PASSED: Optimization step updates parameters")
    return True


def main():
    """Run all tests."""
    logger.info("Starting REINFORCE component tests")
    logger.info("=" * 60)

    all_passed = True

    tests = [
        ("GPT2 Policy Generation", test_gpt2_policy_generation),
        ("Gradient Flow", test_gradient_flow),
        ("Fisher Objective", test_fisher_objective),
        ("REINFORCE Loss", test_reinforce_loss),
        ("Optimization Step", test_optimization_step),
    ]

    for test_name, test_fn in tests:
        try:
            if not test_fn():
                all_passed = False
                logger.error(f"Test '{test_name}' FAILED")
        except Exception as e:
            logger.error(f"Test '{test_name}' FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False

    logger.info("=" * 60)
    if all_passed:
        logger.info("ALL TESTS PASSED!")
    else:
        logger.error("SOME TESTS FAILED!")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
