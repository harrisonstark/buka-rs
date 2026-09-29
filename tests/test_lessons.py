"""Answer-key checks. Notebooks re-implement the same functions."""

import torch

from buka_rs.attention import bidirectional_attention
from buka_rs.block import mean_pool, rms_norm, swiglu
from buka_rs.config import OneConfig, tiny
from buka_rs.data import clean_state, encode_state, load_examples, parse_bool
from buka_rs.decisions import choice_answer, confidence, fit_temperature, noul_prob, score_answer
from buka_rs.infer import response_payload
from buka_rs.model import OneModel
from buka_rs.train import run


def test_tiny_is_about_a_million_params():
    model = OneModel(tiny())
    count = model.num_parameters()
    assert 700_000 < count < 1_300_000


def test_confidence_and_noul():
    certain = confidence(torch.tensor([0.0, 1.0]))
    uniform = confidence(torch.tensor([0.5, 0.5]))
    assert float(certain) > 0.99
    assert float(uniform) < 0.01
    assert abs(float(noul_prob(torch.tensor(0.0))) - 0.5) < 1e-5


def test_choice_and_score_answers():
    choice = choice_answer(torch.tensor([0.0, 0.0, 5.0]), ("chat", "ops", "learning"))
    assert choice["choice"] == "learning"
    assert abs(sum(choice["probabilities"].values()) - 1) < 1e-4
    score = score_answer(torch.tensor([0.0, 0.0, 8.0]), ("calm", "soon", "now"))
    assert score["score"] > 1.9
    assert score["legend"] == ["calm", "soon", "now"]


def test_bytes_and_clean():
    ids, mask = encode_state("hi", max_len=4, pad_id=256)
    assert ids[:2] == [ord("h"), ord("i")]
    assert ids[2:] == [256, 256]
    assert mask == [1.0, 1.0, 0.0, 0.0]
    tail, tail_mask = encode_state("abcdef", max_len=3)
    assert tail == [ord("d"), ord("e"), ord("f")]
    assert tail_mask == [1.0, 1.0, 1.0]
    assert clean_state("  hey   there ") == "hey there"
    assert clean_state("https://example.com/a") is None


def test_attention_sees_the_whole_state():
    q = torch.zeros(1, 1, 3, 2)
    k = torch.zeros(1, 1, 3, 2)
    v = torch.tensor([0.0, 0.0, 1.0, 1.0, 2.0, 2.0]).view(1, 1, 3, 2)
    out = bidirectional_attention(q, k, v)
    assert abs(float(out[0, 0, 0, 0]) - 1.0) < 1e-4
    masked = bidirectional_attention(q, k, v, key_mask=torch.tensor([[1.0, 1.0, 0.0]]))
    assert abs(float(masked[0, 0, 0, 0]) - 0.5) < 1e-4


def test_norm_swiglu_pool():
    weight = torch.ones(2)
    y = rms_norm(torch.tensor([[3.0, 4.0]]), weight, eps=1e-6)
    inv = 1.0 / (12.5 ** 0.5)
    assert abs(float(y[0, 0]) - 3.0 * inv) < 1e-4
    hidden = swiglu(torch.tensor([0.0, 1.0]), torch.tensor([2.0, 3.0]))
    assert abs(float(hidden[0])) < 1e-5
    pooled = mean_pool(
        torch.tensor([[[1.0, 1.0], [3.0, 3.0], [9.0, 9.0]]]),
        torch.tensor([[1.0, 1.0, 0.0]]),
    )
    assert torch.allclose(pooled, torch.tensor([[2.0, 2.0]]))


def test_temperature_search_returns_a_grid_value():
    logits = torch.tensor([[5.0, 0.0], [0.1, 0.1]])
    labels = torch.tensor([0, 1])
    chosen = fit_temperature(logits, labels, grid=[0.5, 1.0, 2.0])
    assert chosen == 0.5


def test_bool_and_bom(tmp_path):
    assert parse_bool("false") is False
    assert parse_bool("true") is True
    path = tmp_path / "rows.jsonl"
    path.write_bytes(
        b'\xef\xbb\xbf{"state": "hey", "topic": "chat", "urgency": 0, "needs_reply": "false"}\n'
    )
    rows = load_examples(path, ("chat", "ops", "learning"), 3)
    assert rows[0].needs_reply is False


def test_payload_shape():
    body = response_payload({"choice": "ops"}, {"score": 1.2}, {"noul": 0.9})
    assert set(body["answers"]) == {"topic", "urgency", "needs_reply"}


def test_loss_drops_on_repeated_states():
    config = OneConfig(
        name="micro",
        d_model=32,
        n_layers=1,
        n_heads=4,
        d_ff=64,
        max_seq_len=32,
    )
    examples = []
    from buka_rs.data import Example

    for _ in range(12):
        examples.append(Example("server is down tonight?", "ops", 2, True))
        examples.append(Example("lol fair", "chat", 0, False))
    _model, metrics = run(
        config,
        examples,
        steps=20,
        batch_size=8,
        seq_len=24,
        lr=3e-3,
        seed=3,
        device=torch.device("cpu"),
        log_every=0,
    )
    assert metrics["last_loss"] < metrics["first_loss"] - 0.05
    assert metrics["topic_acc"] > 0.9
