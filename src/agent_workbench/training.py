"""Framework-neutral rollout records, not a substitute for a token-level trainer."""

import copy
from .types import Trajectory


def rollout_record(trace: Trajectory, *, task_id: str, group_id: str,
                   reward: float) -> dict:
    """Preserve role provenance; gold answers must remain outside model messages.

    A trainer must attach actual prompt/action token IDs, sampling logprobs,
    template and policy versions. Retokenizing these strings is insufficient to
    claim exact on-policy training. See docs/rl-roadmap.md.
    """
    if not task_id or not group_id:
        raise ValueError("task and rollout group IDs are required")
    if trace.status != "completed":
        raise ValueError("incomplete trajectories need an explicit failure reward policy")
    return {"schema_version": "1", "task_id": task_id, "group_id": group_id,
            "reward": float(reward), "messages": copy.deepcopy(trace.messages),
            "trainable_roles": ["assistant"], "status": trace.status,
            "metadata": copy.deepcopy(trace.metadata),
            "token_alignment": "not_recorded", "trainer_ready": False}
