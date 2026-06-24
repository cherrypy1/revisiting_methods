"""cfgbench CLI.

``run`` (autonomous orchestrator), ``status``, ``events``, ``health``, ``worker`` (one shard),
``prompts`` (inspect a benchmark). ``report`` arrives in P4.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_campaign
from .core import manifest as M
from .core.events import EventLog
from .core.layout import CampaignPaths


def _paths_for(campaign_arg) -> CampaignPaths:
    p = Path(campaign_arg)
    if p.is_dir():
        return CampaignPaths(p)
    return CampaignPaths(load_campaign(p).out_root)


def cmd_status(args) -> None:
    spec = load_campaign(args.campaign)
    paths = CampaignPaths(spec.out_root)
    refs = M.load_manifest(paths)
    if not refs:
        from .benchmarks.registry import get_benchmark
        items = M.expand(spec, lambda b: get_benchmark(b).prompts())
        refs = [it.ref for it in items]
    st = M.scan_status(paths, refs)
    print(f"campaign={spec.name} out={paths.root}")
    print(f"  total={st.total} gen_done={st.gen_done} eval_done={st.eval_done} "
          f"pending_gen={len(st.pending_gen)} pending_eval={len(st.pending_eval)} failed={st.failed}")

    import time
    from .core.layout import read_json
    if paths.heartbeats.is_dir():
        now = time.time()
        active = [h for h in (read_json(p) for p in sorted(paths.heartbeats.glob("*.json")))
                  if h and now - h.get("ts", 0) < 180]
        if active:
            rate = sum(h.get("rate", 0) or 0 for h in active)
            print(f"  workers: {len(active)} active, ~{rate:.3f} img/s")
            for h in active:
                print(f"    {h.get('shard')}: {h.get('done')}/{h.get('total')} [{h.get('phase')}]")


def cmd_events(args) -> None:
    evs = EventLog(_paths_for(args.campaign)).read()
    tail = evs[-args.n:] if args.n else evs
    for ev in tail:
        print(EventLog._human(ev))


def cmd_worker(args) -> None:
    from .core.worker import run_shard
    run_shard(args.shard)


def cmd_prompts(args) -> None:
    from collections import Counter
    from .benchmarks.registry import get_benchmark
    ps = get_benchmark(args.bench).prompts()
    cnt = Counter(p.category for p in ps)
    print(f"{args.bench}: {len(ps)} prompts")
    for k, v in sorted(cnt.items()):
        print(f"  {k}: {v}")
    for p in ps[:args.n]:
        print(f"  [{p.category}] {p.id}: {p.text[:70]}")


def cmd_run(args) -> None:
    from .core.orchestrator import Orchestrator
    spec = load_campaign(args.campaign)
    if args.stage is not None:
        spec.stage = args.stage
    if args.max_jobs is not None:
        spec.pool["max_jobs"] = args.max_jobs
    if args.verbose:
        spec.verbose = True
    Orchestrator(spec, poll=args.poll).run()


def cmd_health(args) -> None:
    from .core.health import check_once
    raise SystemExit(check_once(args.campaign))


def cmd_report(args) -> None:
    from .report import build
    md, path = build(load_campaign(args.campaign))
    print(md)
    print(f"\nwrote {path} and {path.with_name('report.csv')}")


def cmd_notify_test(args) -> None:
    from .core import notify
    if not notify.available():
        print("no notify creds — set TELEGRAM_BOT_TOKEN/CHAT_ID in ~/.config/cfgbench/notify.env")
        raise SystemExit(2)
    ok = notify.send(args.text)
    print("sent OK" if ok else "send FAILED")
    raise SystemExit(0 if ok else 1)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="cfgbench")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("status", help="show campaign progress")
    s.add_argument("campaign", help="campaign yaml or out dir")
    s.set_defaults(func=cmd_status)

    e = sub.add_parser("events", help="print event log")
    e.add_argument("campaign", help="campaign yaml or out dir")
    e.add_argument("-n", type=int, default=0, help="tail last N")
    e.set_defaults(func=cmd_events)

    w = sub.add_parser("worker", help="run one shard (internal; also manual)")
    w.add_argument("--shard", required=True, help="shard json file")
    w.set_defaults(func=cmd_worker)

    pr = sub.add_parser("prompts", help="inspect a benchmark's prompts")
    pr.add_argument("bench", help="geneval | oneig | dpg")
    pr.add_argument("-n", type=int, default=0, help="show first N prompts")
    pr.set_defaults(func=cmd_prompts)

    r = sub.add_parser("run", help="run a campaign autonomously (orchestrator)")
    r.add_argument("campaign", help="campaign yaml")
    r.add_argument("--poll", type=float, default=20, help="scheduler poll seconds")
    r.add_argument("--max-jobs", type=int, default=None, dest="max_jobs")
    r.add_argument("--stage", choices=["generate", "validate", "run"], default=None,
                   help="override campaign stage")
    r.add_argument("--verbose", action="store_true")
    r.set_defaults(func=cmd_run)

    h = sub.add_parser("health", help="one-shot health check (exit!=0 on anomaly)")
    h.add_argument("campaign", help="campaign yaml or out dir")
    h.set_defaults(func=cmd_health)

    rp = sub.add_parser("report", help="build markdown+csv report from summaries")
    rp.add_argument("campaign", help="campaign yaml")
    rp.set_defaults(func=cmd_report)

    nt = sub.add_parser("notify-test", help="send a test Telegram/email notification")
    nt.add_argument("--text", default="cfgbench test ping")
    nt.set_defaults(func=cmd_notify_test)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
