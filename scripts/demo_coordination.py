#!/usr/bin/env python3
"""Phase 3 acceptance demonstration (no cameras needed): negotiation -> adaptation -> fusion outcome -> gesture -> safe cancellation.

    python scripts/demo_coordination.py                 # all scenarios, narrated
    python scripts/demo_coordination.py 2_occlusion     # one scenario
    python scripts/demo_coordination.py --gesture       # gesture trace only
JSONL audit logs are written to logs/demo/<scenario>.jsonl (numbers/ids/reason codes only; never frames).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from host.app.coordination import simulation as S                       # noqa: E402
from host.app.gesture.agent import GestureAgent, HandSample, InteractionObject   # noqa: E402
from host.app.gesture.config import load_gesture_config                 # noqa: E402

SHOW = {"TASK_REQUEST", "OFFER_REJECTED", "SET_SELECTION", "TASK_AWARD", "LEASE_EXPIRED", "FUSION_EVALUATED", "ADAPTATION", "DEGRADED",
        "TASK_REJECT", "SENSOR_STATUS_IGNORED"}


def narrate(name: str) -> None:
    log_path = ROOT / "logs" / "demo" / f"{name}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.unlink(missing_ok=True)
    print(f"\n{'=' * 100}\nSCENARIO {name}\n{'=' * 100}")
    h = S.SCENARIOS[name]()
    seen_sel = 0
    for e in h.audit.entries:
        if e["event_type"] not in SHOW:
            continue
        t = f"+{e['ts_ms'] - S.T0:5d} ms"
        et = e["event_type"]
        if et == "TASK_REQUEST":
            print(f"{t}  ANNOUNCE   {e['task_id']} target={e['target']} reason={e['reason']} attempt={e['attempt']} deadline=+{e['deadline_ms'] - e['ts_ms']} ms "
                  f"asking={e['expected_sensors']} unavailable={e['unavailable']}")
        elif et == "OFFER_REJECTED":
            extra = f" (age {e['observation_age_ms']:.0f} ms)" if e.get("observation_age_ms") is not None else ""
            print(f"{t}  REJECT     {e['task_id']} {e['sensor_id']}: {e['reason']}{extra} {e.get('detail', '')}")
        elif et == "SET_SELECTION":
            iq = ", ".join(f"{k}={v:.3f}" for k, v in sorted(e["individual_quality"].items(), key=lambda kv: -kv[1]))
            print(f"{t}  SELECT     {e['task_id']} offers: {iq}")
            for c in sorted(e["candidates"], key=lambda c: -(c["score"] or -1)):
                tag = f"score={c['score']:.3f}" if c["feasible"] else "REJECTED " + ",".join(c["codes"])
                ang = f" angle={c['angle_deg']:.1f}deg" if c.get("angle_deg") is not None else ""
                print(f"                         {'+'.join(c['set']):<20} {tag}{ang}")
            print(f"                         => {e['outcome']} {e['chosen'] or ''}  [{e['reason']}]")
        elif et == "TASK_AWARD":
            print(f"{t}  AWARD      {e['award_id']} sensors={e['sensors']} mode={e['mode']} trusted_3d={e['trusted_3d']} lease={e['lease_ms']} ms")
        elif et == "LEASE_EXPIRED":
            print(f"{t}  LEASE END  {e['award_id']} {e['sensors']} reason={e['reason']}")
        elif et == "FUSION_EVALUATED":
            rms = "" if e.get("rms_px") is None else f" rms={e['rms_px']:.2f}px"
            print(f"{t}  FUSION     {e['verdict']} ({e['fusion_status']}){rms} reason={e['reason']} consecutive_poor={e['consecutive_poor']}")
        elif et == "ADAPTATION":
            print(f"{t}  ADAPT      {e['reason']} {e.get('failed_set') or e.get('sensor_id') or ''} {e.get('detail') or ''}")
        elif et in ("DEGRADED", "TASK_REJECT"):
            print(f"{t}  {et:<10} {e['reason']} {e.get('detail', '')[:100]}")
    print(f"final state={h.coord.state.value} lease={h.lease_sensors()} counters="
          f"{ {k: v for k, v in h.coord.counters.items() if k.startswith(('awards', 'offers_', 'task_rejected', 'degraded', 'fusion_verdict'))} }")
    import json
    with log_path.open("w", encoding="utf-8") as fh:
        for e in h.audit.entries:
            fh.write(json.dumps(e, separators=(",", ":")) + "\n")


def gesture_demo() -> None:
    print(f"\n{'=' * 100}\nGESTURE TRACE (HOVER -> GRABBED -> tracking lost -> safe cancellation)\n{'=' * 100}")
    clock = S.SimClock(10_000)
    cfg = load_gesture_config()
    agent = GestureAgent("sess_demo0001", cfg, clock=clock)
    obj = [InteractionObject("cube_1", (0.0, 0.0, 0.0))]
    seq = [0]

    def step(label, xyz=(0.05, 0.0, 0.0), ratio=0.8, valid=True):
        clock.advance(60)
        seq[0] += 1
        s = HandSample("sess_demo0001", clock.now_ms() - 10, valid, xyz if valid else None, 0.9 if valid else None, ratio, seq[0], "board_frame")
        out = agent.update(s, obj, clock.now_ms())
        for ev in out.events:
            pos = "" if ev.hand_target_xyz_m is None else f" pos={tuple(round(v, 3) for v in ev.hand_target_xyz_m)}"
            why = ev.cancel_reason.value if ev.cancel_reason else ev.reason
            print(f"+{clock.now_ms() - 10_000:5d} ms {label:<22} {ev.event_type.value:<15} state={ev.gesture_state.value:<8} reason={why}{pos}")
        return out

    for _ in range(3):
        step("open hand in reach", ratio=0.9)
    for _ in range(5):
        step("pinching", ratio=0.1)
    for i in range(3):
        out = step("holding, moving", xyz=(0.05 + 0.01 * i, 0, 0), ratio=0.1)
        for m in out.moves:
            print(f"+{clock.now_ms() - 10_000:5d} ms {'(continuous stream)':<22} {m.event_type.value:<15} pos={tuple(round(v, 3) for v in m.hand_target_xyz_m)}")
    for _ in range(9):
        step("fusion INVALID", valid=False)
    for _ in range(3):
        clock.advance(60)
        for ev in agent.tick(clock.now_ms()).events:
            print(f"+{clock.now_ms() - 10_000:5d} ms {'tick':<22} {ev.event_type.value}")
    print(f"final gesture state = {agent.state.value}; counters = { {k: v for k, v in agent.counters.items() if k.startswith(('event:', 'transition:', 'tracking'))} }")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--gesture" not in sys.argv[1:]:
        for n in (args or list(S.SCENARIOS)):
            narrate(n)
    gesture_demo()
