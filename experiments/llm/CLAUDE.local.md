# Local Notes - REINFORCE Experiments

## Current Best Config (2026-01-27)

```bash
python reinforce_word_level.py \
    --model magicprompt \
    --samples 8 \
    --horizon 8 \
    --iterations 60 \
    --lambda-reg 1 \
    --lr 1e-3 \
    --baseline moving_avg \
    --optimizer sgd \
    --seed 123 \
    --init-noise 0 \
    --joint-update \
    --sum-trajectories \
    -T 8
```

**Key settings:**
- `M=8` (--samples): Number of Fisher samples for gradient estimation
- `T=8` (-T): Number of trajectories summed for Fisher
- `--baseline moving_avg`: Within-batch mean centering (essential - prevents mode collapse)
- `--lr 1e-3`: Higher LR needed because baseline centering reduces gradient magnitude ~40x
- `--sum-trajectories`: Sum T trajectory Fishers (aligns train with eval objective)
- `--joint-update`: Update all K=4 policies together
- `--lambda-reg 1`: Balanced regularization

**Results:**
- Training and eval objectives match (~25-27)
- Shows improving trend: 25.5 → 26.7 over iterations
- Requires A100 80GB for M=8, T=8

## AWS Instance (p4de.24xlarge - 8x A100 80GB)

**SSH:**
```bash
ssh -i ~/atom/a100-key.pem ubuntu@52.73.147.169
```

**Environment:** `/opt/pytorch` virtualenv (Python 3.12 with PyTorch, transformers, omegaconf)

**Run experiment:**
```bash
ssh -i ~/atom/a100-key.pem ubuntu@52.73.147.169 \
  "cd /home/ubuntu/experiment-design-mdp/experiments/llm && \
   source /opt/pytorch/bin/activate && \
   python -u reinforce_word_level.py \
     --model magicprompt --samples 8 --horizon 8 --iterations 60 \
     --lambda-reg 1 --lr 1e-3 --baseline moving_avg --optimizer sgd \
     --seed 123 --init-noise 0 --joint-update --sum-trajectories -T 8"
```

**Background run (with nohup):**
```bash
ssh -i ~/atom/a100-key.pem ubuntu@52.73.147.169 \
  "cd /home/ubuntu/experiment-design-mdp/experiments/llm && \
   source /opt/pytorch/bin/activate && \
   nohup python -u reinforce_word_level.py \
     --model magicprompt --samples 8 --horizon 8 --iterations 60 \
     --lambda-reg 1 --lr 1e-3 --baseline moving_avg --optimizer sgd \
     --seed 123 --init-noise 0 --joint-update --sum-trajectories -T 8 \
   > reinforce_run.log 2>&1 &"
```

**Check output:**
```bash
ssh -i ~/atom/a100-key.pem ubuntu@52.73.147.169 \
  "tail -50 /home/ubuntu/experiment-design-mdp/experiments/llm/reinforce_run.log"
```
