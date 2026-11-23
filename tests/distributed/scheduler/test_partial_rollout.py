import asyncio

import pytest
import torch

from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.scheduler.user_defined_rollout_loop import (
    UserDefinedRolloutLoop,
    postprocess_paused_data,
)

PROMPT_LENGTH = 8
SEQUENCE_LENGTH = 32
PAD_TOKEN_ID = 0
EOS_TOKEN_ID = 2
PROMPT = list(range(1, PROMPT_LENGTH + 1))
MISSING = object()


def make_request(**meta_info):
    batch = {
        "input_ids": torch.tensor([PROMPT]),
        "attention_mask": torch.ones(1, PROMPT_LENGTH, dtype=torch.long),
        "position_ids": torch.arange(PROMPT_LENGTH).view(1, PROMPT_LENGTH),
    }
    return DataProto.from_dict(batch, meta_info={"generation_config": {"num_return_sequences": 1}, **meta_info})


def make_response(finish_reason, token_ids, routed_experts=MISSING):
    meta_info = {
        "finish_reasons": [finish_reason],
        "output_token_ids": [token_ids],
        "output_logprobs": [[-0.5] * len(token_ids)],
        "eos_token_id": [EOS_TOKEN_ID, PAD_TOKEN_ID],
        "pad_token_id": PAD_TOKEN_ID,
    }
    if routed_experts is not MISSING:
        meta_info["routed_experts"] = routed_experts
    return DataProto(meta_info=meta_info)


def assert_equal(actual, expected):
    torch.testing.assert_close(actual, torch.tensor(expected, dtype=actual.dtype), rtol=0, atol=0)


@pytest.mark.parametrize("router_value", [MISSING, None], ids=["router_key_missing", "router_key_none"])
def test_resume_paused_response_without_router_data(router_value):
    paused = make_response("abort", [11, 12, 13], routed_experts=router_value)

    result = postprocess_paused_data(make_request(), paused, SEQUENCE_LENGTH, PROMPT_LENGTH)

    assert "routed_experts" not in result.batch.keys()
    assert all(isinstance(result.batch[key], torch.Tensor) for key in result.batch.keys())
    assert_equal(result.batch["input_ids"], [PROMPT + [11, 12, 13]])
    assert_equal(result.batch["attention_mask"], [[1] * (PROMPT_LENGTH + 3)])
    assert_equal(result.batch["init_input_ids"], [PROMPT])
    assert result.meta_info["pre_output_token_ids"] == [[11, 12, 13]]
    assert result.meta_info["pre_output_logprobs"] == [[-0.5, -0.5, -0.5]]
    assert result.meta_info["max_new_tokens"] == 21


def test_pause_without_tokens_returns_original_request():
    request = make_request()

    result = postprocess_paused_data(
        request, DataProto(meta_info={"finish_reasons": ["abort"]}), SEQUENCE_LENGTH, PROMPT_LENGTH
    )

    assert result is request


def test_paused_response_preserves_router_metadata():
    # Router accumulation across multiple partial responses is outside the scope of this test.
    routed_experts = [torch.zeros(PROMPT_LENGTH + 2, 2, 2, dtype=torch.long)]
    paused = make_response("abort", [11, 12, 13], routed_experts=routed_experts)

    result = postprocess_paused_data(make_request(), paused, SEQUENCE_LENGTH, PROMPT_LENGTH)

    assert "routed_experts" not in result.batch.keys()
    assert result.meta_info["pre_routed_experts"] is routed_experts


class PauseThenFinishContext:
    def __init__(self):
        self.sequence_length = SEQUENCE_LENGTH
        self.prompt_length = PROMPT_LENGTH
        self.is_val = True
        self.pipeline_config = None
        self.generate_calls = 0
        self.responses = [
            make_response("abort", [11, 12, 13], routed_experts=None),
            make_response("stop", [14, EOS_TOKEN_ID], routed_experts=None),
        ]

    async def generate(self, req, domain):
        self.generate_calls += 1
        return self.responses.pop(0)

    async def compute_rewards(self, req, domain):
        return DataProto.from_dict({"scores": torch.zeros(req.batch.batch_size[0])})


def test_resume_and_complete_keeps_token_order():
    # Exercise the real pause/resume path without starting Ray or vLLM.
    context = PauseThenFinishContext()

    results = asyncio.run(
        UserDefinedRolloutLoop()._generate_and_reward_impl(
            context=context, req=make_request(collect_unfinished=True), domain="math"
        )
    )

    assert context.generate_calls == 2
    assert len(results) == 1
    batch = results[0].batch
    assert "routed_experts" not in batch.keys()
    generated = [11, 12, 13, 14, EOS_TOKEN_ID]
    response_length = SEQUENCE_LENGTH - PROMPT_LENGTH
    assert_equal(batch["responses"][0], generated + [PAD_TOKEN_ID] * (response_length - len(generated)))
    used = PROMPT_LENGTH + len(generated)
    assert_equal(batch["attention_mask"][0], [1] * used + [0] * (SEQUENCE_LENGTH - used))
    assert_equal(
        batch["response_mask"][0], [0] * PROMPT_LENGTH + [1] * len(generated) + [0] * (SEQUENCE_LENGTH - used)
    )
    assert_equal(batch["infer_logprobs"][0, PROMPT_LENGTH - 1 : used - 1], [-0.5] * len(generated))
