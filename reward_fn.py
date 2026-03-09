"""
Reward Functions for GRPO (RLVR phase)

Purpose:
  These functions are called by GRPOTrainer to score each model completion.
  The reward signal must be programmatically verifiable — no human judgment needed.

Two rewards:
  1. format_reward:   Does the completion have <think> and <answer> tags?
                      Ensures the model doesn't abandon structured reasoning.
  2. accuracy_reward: Is the predicted label correct?
                      The core signal that drives actual claim verification ability.

Why separate rewards?
  - If we only used accuracy_reward, the model might learn to output a label
    without any reasoning (reward hacking). Format reward prevents this.
  - If we only used format_reward, the model could write long <think> blocks
    that say nothing useful. Accuracy reward ensures content quality.

Total reward = format_reward + accuracy_reward  (max = 1.2)
"""

import re

VALID_LABELS = {"true", "partially true", "false"}


def format_reward(completions: list[str], **kwargs) -> list[float]:
    """
    Reward for structurally correct output format.

    Checks:
      - Response contains <think>...</think> block  (model showed reasoning)
      - Response contains <answer>...</answer> block (model gave a verdict)

    Returns 0.2 per completion if both tags are present, else 0.0.

    Why 0.2 (not higher)?
      Format reward should be a soft encouragement, not the main signal.
      Accuracy reward (max 1.0) is 5x larger, keeping the priority correct.
    """
    rewards = []
    for c in completions:
        has_think  = bool(re.search(r"<think>.*?</think>", c, re.DOTALL))
        has_answer = bool(re.search(r"<answer>.*?</answer>", c, re.DOTALL | re.IGNORECASE))
        rewards.append(0.2 if (has_think and has_answer) else 0.0)
    return rewards


def accuracy_reward(completions: list[str], ground_truth: list[str], **kwargs) -> list[float]:
    """
    Reward for predicting the correct label.

    Extracts text inside <answer>...</answer>, normalises it, and compares
    to the ground truth label.  Returns 1.0 for correct, 0.0 otherwise.

    Ground truth labels: "true" | "partially true" | "false"

    Why binary reward (not partial credit)?
      - The task has an unambiguous ground truth — a claim is either
        correctly verified or not.
      - Binary reward gives cleaner training signal for GRPO advantage
        computation; partial credit can make the advantage noisy.
    """
    rewards = []
    for completion, gt in zip(completions, ground_truth):
        m = re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL | re.IGNORECASE)
        if not m:
            rewards.append(0.0)
            continue
        predicted = m.group(1).strip().lower()
        # Accept minor whitespace variants
        predicted = re.sub(r"\s+", " ", predicted)
        if predicted not in VALID_LABELS:
            rewards.append(0.0)
        else:
            rewards.append(1.0 if predicted == gt.strip().lower() else 0.0)
    return rewards


def combined_reward(completions: list[str], ground_truth: list[str], **kwargs) -> list[float]:
    """
    Combined reward = format_reward + accuracy_reward.

    This is the function registered with GRPOTrainer.
    Maximum possible reward per completion: 1.2
    """
    fmt = format_reward(completions, **kwargs)
    acc = accuracy_reward(completions, ground_truth=ground_truth, **kwargs)
    return [f + a for f, a in zip(fmt, acc)]
