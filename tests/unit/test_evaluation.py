import hashlib
import json
from pathlib import Path

from magnetoconvective_control.evaluation import build_plan, freeze_snapshots


def _checkpoint(path: Path, time: int) -> None:
    header = (
        f"#std 4  7  7  7       4300       4300  {float(time):20.13E}"
        "         0      0      1 UPS01       0.0000000E+00 F"
    )
    path.write_bytes(header.ljust(132).encode("ascii") + b"payload")


def test_freeze_and_plan_are_exact_and_counterbalanced(tmp_path):
    source = tmp_path / "source"
    frozen = tmp_path / "frozen"
    source.mkdir()
    for index, time in zip(range(30, 40), range(310, 401, 10)):
        _checkpoint(source / f"blanket0.f{index:05d}", time)

    manifest_path = freeze_snapshots(source, frozen)
    manifest = json.loads(manifest_path.read_text())
    assert [row["time"] for row in manifest["snapshots"]] == list(range(310, 401, 10))
    assert manifest["snapshots"][0]["sha256"] == hashlib.sha256(
        (frozen / "blanket0.f00030").read_bytes()
    ).hexdigest()

    model = tmp_path / "policy.zip"
    model.write_bytes(b"model")
    plan = build_plan(frozen, model, seed=5489)
    assert len(plan["runs"]) == 20
    first = plan["runs"][::2]
    assert sum(row["controller"] == "feedback" for row in first) == 5
    assert plan == build_plan(frozen, model, seed=5489)

