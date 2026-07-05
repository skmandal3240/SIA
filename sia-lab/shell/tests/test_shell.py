"""Tests for SIA Shell embodiment."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from shell import (
    CaptureStub,
    StreamingSTTStub,
    StreamingTTSStub,
    parse_action_tags,
    ActionTag,
    Dispatcher,
    ShellLoop,
)


def test_capture_stub_dimensions():
    cap = CaptureStub(1920, 1080)
    shot = cap.grab()
    assert shot["width"] == 1920
    assert shot["height"] == 1080
    assert len(shot["buffer"]) == 1920 * 1080 * 4


def test_capture_stub_save():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        cap = CaptureStub(10, 10)
        path = cap.save(Path(tmp) / "shot.raw")
        assert path.exists()
        assert path.stat().st_size == 10 * 10 * 4


def test_stt_stream():
    stt = StreamingSTTStub(["hello", "world"])
    chunks = [b"a", b"b", b"", b"c"]
    assert list(stt.stream(iter(chunks))) == ["hello", "world"]


def test_tts_stream():
    tts = StreamingTTSStub(chunk_size=64)
    chunks = list(tts.stream(iter(["hi", "", "there"])))
    assert len(chunks) == 2
    assert all(len(c) == 64 for c in chunks)


def test_parse_point_tag():
    tags = parse_action_tags("Tap [POINT:120,240:submit:screen0] now")
    assert len(tags) == 1
    assert tags[0] == ActionTag(
        kind="POINT", x=120.0, y=240.0, label="submit", screen=0, raw="[POINT:120,240:submit:screen0]"
    )


def test_parse_click_without_screen():
    tags = parse_action_tags("[CLICK:10,20]")
    assert len(tags) == 1
    assert tags[0].screen == 0
    assert tags[0].label is None


def test_parse_type_and_scroll():
    text = "[TYPE:100,200:hello] then [SCROLL:300,400:0,-3:screen1]"
    tags = parse_action_tags(text)
    assert len(tags) == 2
    assert tags[0] == ActionTag(
        kind="TYPE", x=100.0, y=200.0, label="hello", screen=0, raw="[TYPE:100,200:hello]"
    )
    assert tags[1] == ActionTag(
        kind="SCROLL", x=300.0, y=400.0, label="scroll:0,-3", screen=1, raw="[SCROLL:300,400:0,-3:screen1]"
    )


def test_dispatcher_runs_tools_and_actions():
    def runner(tag: ActionTag) -> str:
        return f"ran {tag.kind}"

    disp = Dispatcher(action_runner=runner, tool_runner=lambda t, a: f"{t}={a}")
    tools, actions = disp.dispatch("[POINT:1,2:a]", [{"tool": "search", "args": {"q": "x"}}])
    assert len(tools) == 1
    assert tools[0].output == "search={'q': 'x'}"
    assert len(actions) == 1
    assert actions[0].ok and actions[0].message == "ran POINT"


def test_shell_loop_default_turn():
    loop = ShellLoop()
    turn = loop.run_once([b"user audio"])
    assert turn.screenshot["width"] == 1280
    assert turn.transcript == ""
    assert "[POINT:100,200:submit_button:screen0]" in turn.response
    assert len(turn.actions) == 1
    assert turn.actions[0].ok
    assert len(turn.spoken) == 1


def test_shell_loop_custom_reason():
    loop = ShellLoop(
        reason=lambda shot, text: (f"ok {text} [CLICK:5,6]", []),
        stt=StreamingSTTStub(["namaste"]),
    )
    turn = loop.run_once([b"audio"])
    assert turn.transcript == "namaste"
    assert "[CLICK:5,6]" in turn.response
    assert any(a.tag.kind == "CLICK" for a in turn.actions)


def test_action_runner_exception_surfaces():
    disp = Dispatcher(action_runner=lambda tag: (_ for _ in ()).throw(RuntimeError("boom")))
    _, actions = disp.dispatch("[POINT:0,0]")
    assert len(actions) == 1
    assert not actions[0].ok
    assert "boom" in actions[0].message


if __name__ == "__main__":
    for fn in [
        test_capture_stub_dimensions,
        test_capture_stub_save,
        test_stt_stream,
        test_tts_stream,
        test_parse_point_tag,
        test_parse_click_without_screen,
        test_parse_type_and_scroll,
        test_dispatcher_runs_tools_and_actions,
        test_shell_loop_default_turn,
        test_shell_loop_custom_reason,
        test_action_runner_exception_surfaces,
    ]:
        fn()
        print(f"ok {fn.__name__}")
    print("all tests passed")
