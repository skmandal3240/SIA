#!/usr/bin/env python3
"""Integration tests for SIA core modules.

Covers: memory (TokenCake, EpisodicStore, GraphRAG), reasoner (SIRConfig,
SIRReasoner, Router, Governor, DeepPath), shell (Dispatcher, tags, loop),
swarm, safety (privacy, audit, consent, encryption), and the end-to-end
run_sia pipeline.

Run: python3 sia-lab/tests/test_integration.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # repo root
SIA_LAB = ROOT / "sia-lab"
sys.path.insert(0, str(SIA_LAB))
sys.path.insert(0, str(SIA_LAB / "reasoner"))
sys.path.insert(0, str(SIA_LAB / "eval"))

import torch  # noqa: E402

# -- Memory ------------------------------------------------------------------
from memory import TokenCake, EpisodicStore, GraphRAGStub  # noqa: E402


def test_tokencake_eviction():
    cake = TokenCake(budget=100, reserve=10)
    cake.add("system", "You are SIA.", tokens=20, pin=True)
    cake.add("user", "msg1", tokens=30)
    cake.add("user", "msg2", tokens=40)
    cake.add("user", "msg3", tokens=30)  # should evict msg1
    msgs = cake.to_messages()
    assert "msg1" not in [m["content"] for m in msgs]
    assert any(m["content"] == "You are SIA." for m in msgs)  # pinned survived
    print("ok test_tokencake_eviction")


def test_tokencake_roundtrip():
    cake = TokenCake(budget=256, reserve=32)
    cake.add("user", "hello", tokens=10)
    cake.add("assistant", "hi there", tokens=15)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        cake.save(f.name)
    loaded = TokenCake.from_file(f.name, budget=256, reserve=32)
    assert len(loaded.to_messages()) == 2
    print("ok test_tokencake_roundtrip")


def test_episodic_ttl():
    store = EpisodicStore()
    store.add("temp", "temporary record", ttl_seconds=0.01)
    store.add("perm", "permanent record", ttl_seconds=0)
    import time
    time.sleep(0.05)
    assert store.get("temp") is None
    assert store.get("perm") is not None
    print("ok test_episodic_ttl")


def test_graphrag_multihop():
    g = GraphRAGStub()
    g.add_triple("Saurabh", "lives_in", "Patna")
    g.add_triple("Patna", "capital_of", "Bihar")
    paths = g.multi_hop("Saurabh", hops=2)
    assert len(paths) >= 1
    assert any(p[-1]["object"] == "Bihar" for p in paths)
    print("ok test_graphrag_multihop")


# -- Reasoner ----------------------------------------------------------------
from reasoner import SIRConfig, SIRReasoner  # noqa: E402


def test_reasoner_forward_backward():
    cfg = SIRConfig(vocab_size=64, dim=64, n_heads=2, n_layers=1, n_experts=2, act_max_steps=3)
    model = SIRReasoner(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    logits, info = model(x, targets=x)
    assert info["loss"] is not None
    info["loss"].backward()
    assert info["spectral_radius"] < 1.0
    print("ok test_reasoner_forward_backward")


def test_reasoner_generate():
    cfg = SIRConfig(vocab_size=32, dim=32, n_heads=2, n_layers=1, n_experts=2, act_max_steps=3)
    model = SIRReasoner(cfg)
    x = torch.randint(0, cfg.vocab_size, (1, 4))
    gen = model.generate(x, max_new=5)
    assert gen.shape == (1, 9)
    print("ok test_reasoner_generate")


from router import SIRRouter  # noqa: E402
from governor_client import GovernorClient  # noqa: E402


def test_router_fast_path():
    router = SIRRouter()
    d = router.route("set alarm")
    assert d.path == "fast"
    print("ok test_router_fast_path")


def test_router_deep_path():
    router = SIRRouter()
    d = router.route("Which state capital does Rahul live in?", ["ctx1", "ctx2"])
    assert d.path == "deep"
    print("ok test_router_deep_path")


def test_router_hot_thermal():
    router = SIRRouter()
    d = router.route("why is the sky blue?", thermal_budget={"hot": True, "deep_allowed": False})
    assert d.path == "fast"
    print("ok test_router_hot_thermal")


def test_governor_modes():
    for mode in ("cool", "normal", "hot"):
        g = GovernorClient(mode)
        b = g.budget()
        assert "max_loops" in b
        assert "act_max_steps" in b
    assert GovernorClient("hot").budget()["hot"] is True
    assert GovernorClient("cool").budget()["max_loops"] == 16
    print("ok test_governor_modes")


# -- Shell -------------------------------------------------------------------
from shell import (  # noqa: E402
    CaptureStub, StreamingSTTStub, parse_action_tags, ShellLoop,
)


def test_shell_loop_with_memory():
    cake = TokenCake(budget=256, reserve=32)
    cake.add("user", "My name is Saurabh.", tokens=20)
    loop = ShellLoop(
        capture=CaptureStub(),
        stt=StreamingSTTStub(["What is my name?"]),
        memory=cake,
    )
    turn = loop.run_once([b"audio"])
    assert "Saurabh" in turn.context
    # 2 pre-existing + 2 from this turn = 4
    assert len(cake.to_messages()) >= 3
    print("ok test_shell_loop_with_memory")


def test_action_tag_edge_cases():
    # Malformed tag should be skipped, not crash
    tags = parse_action_tags("[POINT:abc,def] not a tag")
    assert len(tags) == 0
    # Multiple tags
    tags = parse_action_tags("[POINT:1,2:a:screen0] [CLICK:3,4]")
    assert len(tags) == 2
    print("ok test_action_tag_edge_cases")


# -- Swarm -------------------------------------------------------------------
from swarm.swarm import SwarmRuntime, SwarmNode, SwarmTask  # noqa: E402


def test_swarm_consensus():
    nodes = [
        SwarmNode("a", "precise", lambda p, c: "Patna"),
        SwarmNode("b", "cautious", lambda p, c: "Patna"),
    ]
    result = SwarmRuntime(nodes).run(SwarmTask("t", "Where?", []))
    assert result["consensus"] == "Patna"
    print("ok test_swarm_consensus")


def test_swarm_disagreement():
    nodes = [
        SwarmNode("a", "precise", lambda p, c: "Patna"),
        SwarmNode("b", "cautious", lambda p, c: "Bihar"),
    ]
    result = SwarmRuntime(nodes).run(SwarmTask("t", "Where?", []))
    # Majority vote: both get 1 vote, first wins
    assert result["consensus"] in ("Patna", "Bihar")
    print("ok test_swarm_disagreement")


# -- Safety ------------------------------------------------------------------
from safety.audit import AuditLog  # noqa: E402
from safety.consent import ConsentRecord, check_access  # noqa: E402
from safety.encryption import EncryptionAtRest  # noqa: E402


def test_audit_log_roundtrip():
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        log = AuditLog(f.name)
        log.record_egress("api.example.com", 443, allowed=False)
        log.record_action("set_alarm")
        events = log.read()
        assert len(events) == 2
        log.erase()
        events = log.read()
        assert len(events) == 1  # erasure event itself
    print("ok test_audit_log_roundtrip")


def test_consent_lifecycle():
    c = ConsentRecord(data_classification="PII", retention_days=30)
    assert check_access(c)
    c.withdraw()
    assert not check_access(c)
    print("ok test_consent_lifecycle")


def test_encryption_roundtrip():
    enc = EncryptionAtRest("test-key")
    data = b"SIA private data: Rahul lives in Patna."
    ct = enc.encrypt(data)
    assert ct != data
    assert enc.decrypt(ct) == data
    print("ok test_encryption_roundtrip")


# -- Multi-hop eval ----------------------------------------------------------
from multi_hop import MultiHopBench, trivial_solver  # noqa: E402


def test_multihop_bench():
    bench = MultiHopBench()
    result = bench.evaluate(trivial_solver)
    assert result["accuracy"] == 1.0
    print("ok test_multihop_bench")


# -- End-to-end --------------------------------------------------------------
from run_sia import SIARuntime  # noqa: E402


def test_e2e_sia_run():
    rt = SIARuntime(mode="normal", use_swarm=True)
    trace = rt.run_turn("Set an alarm for 7 am", [])
    assert trace.route in ("fast", "deep")
    assert trace.spoken_chunks > 0
    assert len(rt.turns) == 1
    print("ok test_e2e_sia_run")


def test_e2e_sia_hot_mode():
    rt = SIARuntime(mode="hot", use_swarm=False)
    trace = rt.run_turn("Which state capital does Rahul live in?",
                        ["Rahul lives in Patna.", "Patna is the capital of Bihar."])
    assert trace.route == "fast"  # hot mode forces fast
    print("ok test_e2e_sia_hot_mode")


ALL_TESTS = [
    test_tokencake_eviction,
    test_tokencake_roundtrip,
    test_episodic_ttl,
    test_graphrag_multihop,
    test_reasoner_forward_backward,
    test_reasoner_generate,
    test_router_fast_path,
    test_router_deep_path,
    test_router_hot_thermal,
    test_governor_modes,
    test_shell_loop_with_memory,
    test_action_tag_edge_cases,
    test_swarm_consensus,
    test_swarm_disagreement,
    test_audit_log_roundtrip,
    test_consent_lifecycle,
    test_encryption_roundtrip,
    test_multihop_bench,
    test_e2e_sia_run,
    test_e2e_sia_hot_mode,
]


if __name__ == "__main__":
    for fn in ALL_TESTS:
        fn()
        print(f"  {fn.__name__}")
    print(f"\nall {len(ALL_TESTS)} integration tests passed")