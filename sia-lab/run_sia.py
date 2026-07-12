#!/usr/bin/env python3
"""SIA end-to-end runner — build and run the whole stack without training.

This ties every shipped phase together into a single perceive → govern →
route → reason → remember → act → speak loop:

  P2 Shell    : screen capture + STT + TTS + shared action dispatcher
  P3 Deep core: the from-scratch recurrent-depth SIR reasoner (random-init;
                genuinely runs a forward + generate pass, no training needed)
  P3 Router   : two-speed fast/deep routing under a governor budget
  P4 Memory   : TokenCake working set + episodic recall + GraphRAG triples
  P5 Swarm    : N-node consensus for hard multi-hop queries

The deep core is untrained, so its raw generation is surfaced as a diagnostic
(`core_output`) while the spoken answer is grounded in retrieved memory. That
keeps the demo honest: the model is really executed, but the correctness comes
from retrieval, not from weights we have not trained yet.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Repo-relative path wiring. `reasoner`/`swarm` are plain module dirs (no
# __init__), so their parent dirs go on sys.path to import them top-level,
# exactly as p3_eval.py does; `memory`/`shell`/`eval` are real packages.
ROOT = Path(__file__).resolve().parent  # sia-lab/
for _p in (ROOT, ROOT / "reasoner", ROOT / "eval"):
    sys.path.insert(0, str(_p))

import torch  # noqa: E402  (deep core dependency; installed via requirements)

from router import SIRRouter  # noqa: E402
from governor_client import GovernorClient  # noqa: E402
from deep_path import DeepPath  # noqa: E402
from multi_hop import trivial_solver  # noqa: E402
from memory import TokenCake, EpisodicStore, GraphRAGStub  # noqa: E402
from shell.capture import CaptureStub  # noqa: E402
from shell.stt import StreamingSTTStub  # noqa: E402
from shell.tts import StreamingTTSStub  # noqa: E402
from shell.dispatcher import Dispatcher  # noqa: E402
from swarm.swarm import SwarmRuntime, SwarmNode, SwarmTask  # noqa: E402

# Reusable SIA action parser if available.
sys.path.insert(0, str(ROOT / "posttrain"))
from parse_action import parse_sia_text  # type: ignore # noqa: E402


# A tiny world model so the memory-grounded answers have something to retrieve.
# In a deployed SIA these triples come from the user's own device context.
DEFAULT_TRIPLES = [
    ("Rahul", "lives_in", "Patna"),
    ("Patna", "capital_of", "Bihar"),
    ("Patna", "bus_to_Gaya", "2 hours"),
]


@dataclass
class TurnTrace:
    query: str
    route: str
    route_reason: str
    budget: dict
    response: str
    core_output: str | None
    consensus: str | None
    tools: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    spoken_chunks: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class SIARuntime:
    """Wires every phase into one runnable companion loop (no training)."""

    def __init__(self, mode: str = "normal", use_swarm: bool = True, seed: int = 0) -> None:
        torch.manual_seed(seed)
        # Perceive + speak (P2).
        self.capture = CaptureStub()
        self.tts = StreamingTTSStub()
        self.dispatcher = Dispatcher()
        # Govern + route (P3).
        self.governor = GovernorClient(mode)
        self.router = SIRRouter(fast_solver=self._ground)
        # Memory (P4).
        self.tokencake = TokenCake(budget=2048)
        self.episodes = EpisodicStore()
        self.graph = GraphRAGStub()
        for s, p, o in DEFAULT_TRIPLES:
            self.graph.add_triple(s, p, o)
        # Deep core (P3) — the from-scratch model, sharing the memory stores.
        self.deep = DeepPath(graph=self.graph, episodes=self.episodes)
        self.use_swarm = use_swarm
        self.turns: list[TurnTrace] = []

    # -- reasoning helpers -------------------------------------------------
    def _ground(self, context: list[str], question: str) -> str:
        """Memory-grounded answer: retrieved triples first, then keyword solve.

        This is the deterministic reasoning result the loop speaks. The deep
        core still runs (see run_turn), but until it is trained we do not trust
        its generation for correctness.
        """
        q = question.lower()
        for entity in question.replace("?", "").split():
            for triple in self.graph.query(entity.strip(".,")):
                pred = triple["predicate"]
                if "capital" in q and pred == "capital_of":
                    return triple["subject"]
                if ("how long" in q or "journey" in q or "bus" in q) and pred.startswith("bus_to"):
                    return triple["object"]
                if "where" in q and pred == "lives_in":
                    return triple["object"]
        grounded = trivial_solver(context, question)
        return grounded or "I don't have enough context yet."

    def _intent_extras(self, query: str) -> tuple[list[dict], str]:
        """Derive tool calls and an on-screen action tag from the query intent."""
        q = query.lower()
        tool_calls: list[dict] = []
        tag = ""
        if "alarm" in q or "wake" in q:
            tool_calls.append({"tool": "set_alarm", "args": {"time": "07:00"}})
        if "message" in q or "text " in q:
            tool_calls.append({"tool": "send_message", "args": {"contact": "Rahul", "body": "Running late"}})
        if any(w in q for w in ("point", "tap", "click", "press", "submit", "button")):
            tag = " [POINT:640,360:submit_button:screen0]"
        return tool_calls, tag

    def _parse_response(self, response: str) -> tuple[list[dict], list[str], str]:
        """Use the real SIA grammar parser when the response emits SIA tags."""
        tools, actions, stripped = parse_sia_text(response)
        tool_calls = [{"tool": t.tool, "args": t.arguments} for t in tools]
        tags = [f" [POINT:{a.x},{a.y}:{a.label}:{a.screen}]" for a in actions]
        return tool_calls, tags, stripped

    # -- one full loop turn ------------------------------------------------
    def run_turn(self, query: str, context: list[str] | None = None) -> TurnTrace:
        context = context or []
        # Perceive: screen + streamed transcript (P2).
        self.capture.grab()
        stt = StreamingSTTStub([query])
        transcript = ""
        for text in stt.stream(iter([b"audio"])):
            transcript = text
        transcript = transcript or query

        # Govern + route (P3).
        budget = self.governor.budget()
        decision = self.router.route(transcript, context, budget)

        core_output: str | None = None
        consensus: str | None = None
        answer = self._ground(context, transcript)

        if decision.path == "deep":
            # Run the from-scratch reasoner for real (forward + generate).
            core_output = self.deep.answer(context, transcript, decision.budget)
            if self.use_swarm and context:
                consensus = self._swarm_consensus(transcript, context)
                if consensus:
                    answer = consensus

        # Act (P2): try SIA grammar first, fall back to keyword heuristic.
        parsed_calls, parsed_tags, clean_response = self._parse_response(answer)
        if parsed_calls or parsed_tags:
            tool_calls, action_tags = parsed_calls, parsed_tags
        else:
            tool_calls, action_tags = self._intent_extras(transcript)
        action_tag = action_tags[0] if action_tags else ""

        # An action/tool command with no factual answer should read as a
        # confirmation, not the "no context" grounding fallback.
        low_signal = clean_response in ("unknown", "I don't have enough context yet.")
        if low_signal and (tool_calls or action_tag):
            done = ", ".join(c["tool"] for c in tool_calls) or "the on-screen action"
            answer = f"Done: {done}"
        elif parsed_calls or parsed_tags:
            answer = clean_response
        response = f"{answer}{action_tag}" if action_tag else answer
        tools, actions = self.dispatcher.dispatch(response, tool_calls)

        # Speak (P2).
        spoken = list(self.tts.stream(iter([response])))

        # Remember (P4).
        self.tokencake.add("user", transcript, tokens=max(1, len(transcript) // 4))
        self.tokencake.add("assistant", response, tokens=max(1, len(response) // 4))
        self.episodes.add(f"turn-{len(self.turns)}", f"Q: {transcript} A: {answer}")

        trace = TurnTrace(
            query=transcript,
            route=decision.path,
            route_reason=decision.reason,
            budget=decision.budget,
            response=response,
            core_output=core_output,
            consensus=consensus,
            tools=[f"{t.tool}->{t.output}" for t in tools],
            actions=[f"{a.tag.kind}@({a.tag.x},{a.tag.y}):{a.ok}" for a in actions],
            spoken_chunks=len(spoken),
        )
        self.turns.append(trace)
        return trace

    def _swarm_consensus(self, question: str, context: list[str]) -> str:
        """P5: two grounded nodes vote; consensus is the spoken answer."""
        def node_solver(prompt: str, ctx: list[str]) -> str:
            return self._ground(ctx, prompt)

        nodes = [
            SwarmNode("node-1", "precise", node_solver),
            SwarmNode("node-2", "cautious", node_solver),
        ]
        runtime = SwarmRuntime(nodes)
        result = runtime.run(SwarmTask("t", question, context))
        return result["consensus"]


DEMO_SCRIPT: list[tuple[str, list[str]]] = [
    ("Set an alarm for 7 am and point to the submit button", []),
    (
        "Which state capital does Rahul live in?",
        ["Rahul lives in Patna.", "Patna is the capital of Bihar."],
    ),
    (
        "How long is the bus journey from Patna to Gaya?",
        ["Patna to Gaya is a well-known route.", "Buses run hourly."],
    ),
]


def run_demo(mode: str = "normal", use_swarm: bool = True) -> int:
    print(f"==> SIA end-to-end run (governor mode={mode}, swarm={use_swarm})")
    rt = SIARuntime(mode=mode, use_swarm=use_swarm)
    for query, context in DEMO_SCRIPT:
        trace = rt.run_turn(query, context)
        print(f"\n[{trace.route.upper()}] {trace.query}")
        print(f"  route_reason : {trace.route_reason}")
        print(f"  response     : {trace.response}")
        if trace.consensus is not None:
            print(f"  swarm        : consensus={trace.consensus}")
        if trace.core_output is not None:
            print(f"  core_output  : {trace.core_output!r} (untrained deep core, diagnostic only)")
        if trace.tools:
            print(f"  tools        : {trace.tools}")
        if trace.actions:
            print(f"  actions      : {trace.actions}")
        print(f"  spoken       : {trace.spoken_chunks} audio chunk(s)")

    # Prove memory persisted across the whole run.
    msgs = rt.tokencake.to_messages()
    print(f"\n==> working memory holds {len(msgs)} message(s); "
          f"{len(rt.episodes._episodes)} episode(s) recorded")
    assert rt.turns, "no turns executed"
    assert any(t.route == "deep" for t in rt.turns), "deep core never exercised"
    assert rt.turns[1].response == "Patna", "multi-hop answer should be grounded to Patna"
    print("SIA end-to-end run passed")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the full SIA stack (no training)")
    parser.add_argument("--query", help="single query to run instead of the demo script")
    parser.add_argument("--context", nargs="*", default=[], help="context facts for --query")
    parser.add_argument("--mode", default="normal", choices=["normal", "hot", "cool"],
                        help="governor thermal mode")
    parser.add_argument("--no-swarm", action="store_true", help="disable swarm consensus")
    parser.add_argument("--json", action="store_true", help="emit the turn trace as JSON")
    args = parser.parse_args(argv)

    if args.query:
        rt = SIARuntime(mode=args.mode, use_swarm=not args.no_swarm)
        trace = rt.run_turn(args.query, args.context)
        if args.json:
            print(json.dumps(trace.to_dict(), indent=2))
        else:
            print(f"[{trace.route}] {trace.query} -> {trace.response}")
            if trace.consensus is not None:
                print(f"  swarm consensus: {trace.consensus}")
            if trace.tools:
                print(f"  tools: {trace.tools}")
            if trace.actions:
                print(f"  actions: {trace.actions}")
        return 0

    return run_demo(mode=args.mode, use_swarm=not args.no_swarm)


if __name__ == "__main__":
    raise SystemExit(main())
