#!/usr/bin/env python3
"""AAB human-study API client (standard library only; no pip installs needed).

Setup:
    export AAB_API_URL="<the URL your coordinator gave you>"
    export AAB_TOKEN="<your personal token>"

Usage:
    python3 aab_client.py budget
    python3 aab_client.py submit --name my_method --code run.py --paper paper.json
    python3 aab_client.py status <run_id>
    python3 aab_client.py findings
    python3 aab_client.py resume
    python3 aab_client.py cancel <run_id>

`paper.json` is a JSON object with the mini-paper fields (see README section 7 and examples/paper.json).
"""
import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error

POLL_SECONDS = 30


def _cfg():
    url = os.environ.get("AAB_API_URL")
    tok = os.environ.get("AAB_TOKEN")
    if not url or not tok:
        sys.exit("ERROR: set AAB_API_URL and AAB_TOKEN in your environment (see README section 9).")
    return url.rstrip("/"), tok


def _call(method, path, body=None):
    url, tok = _cfg()
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url + path, data=data, method=method,
        headers={"x-token": tok, "content-type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            return r.status, json.loads(r.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        # the API returns useful JSON error bodies (400 malformed, 401 bad token, 404, 409 in-flight, ...)
        try:
            return e.code, json.loads(e.read().decode() or "{}")
        except Exception:
            return e.code, {"detail": str(e)}
    except Exception as e:
        return 0, {"detail": f"network error: {e}"}


def _print_scores(rec):
    """Pretty-print a finished submission's composite (headline + per-dim closed% + gates)."""
    comp = rec.get("composite") or {k: v for k, v in rec.items()
                                     if k in ("headline_pct", "petri_score", "per_benchmark", "filter_detail")}
    print("  headline_pct :", comp.get("headline_pct"), " (the number you maximize)")
    print("  petri_score  :", comp.get("petri_score"), " (overall safety level; not the objective)")
    print("  valid        :", rec.get("valid"), " (True = passed ALL gates and counts)")
    fd = comp.get("filter_detail") or {}
    if isinstance(fd, dict) and fd:
        failed = [k for k, v in fd.items() if isinstance(v, dict) and v.get("passed") is False]
        print("  gates failed :", failed or "none (all gates passed)")


def cmd_budget(_):
    st, r = _call("GET", "/budget")
    if st == 200:
        print("active_hours   : %.3f" % r.get("active_hours", 0))
        print("remaining_hours: %.3f  (of %sh)" % (r.get("remaining_hours", 0), r.get("active_budget_h", 6)))
        print("methods        : %s used / %s max  (%s remaining)" % (
            r.get("methods_used", 0), r.get("max_methods", 5), r.get("methods_remaining", "?")))
    else:
        print("ERROR %s: %s" % (st, r.get("detail") or r))


def cmd_status(args):
    st, r = _call("GET", "/status/%s" % args.run_id)
    if st != 200:
        print("ERROR %s: %s" % (st, r.get("detail") or r)); return
    phase = r.get("status")
    print("run_id :", args.run_id)
    print("phase  :", phase, "-", r.get("message", ""))
    if phase == "done":
        _print_scores(r)
    elif phase == "train_failed":
        print("--- FULL TRACEBACK ---\n" + (r.get("traceback") or ""))
    elif phase == "eval_failed":
        print("reason:", r.get("reason"), "| stage:", r.get("eval_stage"))


def cmd_findings(_):
    st, r = _call("GET", "/findings")
    if st != 200:
        print("ERROR %s: %s" % (st, r.get("detail") or r)); return
    fs = r.get("findings") or []
    print("your findings (%d):" % len(fs))
    best = None
    for f in fs:
        comp = f.get("composite") or {}
        h = comp.get("headline_pct")
        valid = f.get("valid")
        print("  %-28s | %-12s | headline=%s | valid=%s" % (
            (f.get("idea_name") or "?")[:28], f.get("phase"), h, valid))
        if valid and h is not None and (best is None or h > best[0]):
            best = (h, f.get("idea_name"))
    print("BEST VALID: %s (%.2f%%)" % (best[1], best[0]) if best else "BEST VALID: none yet")


def cmd_resume(_):
    st, r = _call("POST", "/resume")
    print("resumed. active_hours = %.3f" % r.get("active_hours", 0) if st == 200
          else "ERROR %s: %s" % (st, r.get("detail") or r))


def cmd_cancel(args):
    st, r = _call("POST", "/cancel/%s" % args.run_id)
    print(json.dumps(r, indent=2) if st == 200 else "ERROR %s: %s" % (st, r.get("detail") or r))


def cmd_submit(args):
    try:
        code = open(args.code).read()
    except Exception as e:
        sys.exit("cannot read --code %r: %s" % (args.code, e))
    try:
        paper = json.load(open(args.paper))
    except Exception as e:
        sys.exit("cannot read/parse --paper %r (must be a JSON object): %s" % (args.paper, e))

    print("submitting %r ..." % args.name)
    st, r = _call("POST", "/submit", {"idea_name": args.name, "paper": paper, "code": code})
    if st != 200:
        print("ERROR %s: %s" % (st, r.get("detail") or r)); return
    if not r.get("approved"):
        print("REJECTED by the monitor. Fix the issues below and resubmit. Reasons:")
        for v in (r.get("violations") or []):
            print("  [%s] %s" % (v.get("desideratum"), v.get("explanation") or v.get("rationale")))
        if r.get("error"):
            print("  error:", r.get("error"))
        return
    run_id = r.get("run_id")
    print("APPROVED. run_id = %s" % run_id)
    print("training now (this is on your clock), then evaluating (which is not); this can take a while ...")
    if args.no_wait:
        print("not waiting (--no-wait). Poll with:  python3 aab_client.py status %s" % run_id); return

    last = None
    while True:
        time.sleep(POLL_SECONDS)
        st, r = _call("GET", "/status/%s" % run_id)
        if st != 200:
            print("  (status check error %s: %s ; will retry)" % (st, r.get("detail") or r)); continue
        phase = r.get("status")
        if phase != last:
            print("  [%s] %s" % (time.strftime("%H:%M:%S"), phase))
            if phase == "evaluating":
                print("  >> PAUSE your own timer now: evaluation does not count toward your 6 hours (README section 8).")
            last = phase
        if phase in ("done", "train_failed", "eval_failed", "rejected", "cancelled"):
            print("=== %s ===" % phase.upper())
            if phase == "done":
                _print_scores(r)
                print(">> Evaluation is done. Resume your own timer whenever you get back to work.")
                print("review all your results with:  python3 aab_client.py findings")
            elif phase == "train_failed":
                print("--- FULL TRACEBACK ---\n" + (r.get("traceback") or ""))
            elif phase == "eval_failed":
                print("reason:", r.get("reason"), "| stage:", r.get("eval_stage"))
            return


def main():
    p = argparse.ArgumentParser(description="AAB human-study API client")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("budget").set_defaults(func=cmd_budget)
    sub.add_parser("findings").set_defaults(func=cmd_findings)
    sub.add_parser("resume").set_defaults(func=cmd_resume)
    s = sub.add_parser("submit"); s.set_defaults(func=cmd_submit)
    s.add_argument("--name", required=True, help="a short name for this method")
    s.add_argument("--code", required=True, help="path to your run.py")
    s.add_argument("--paper", required=True, help="path to your mini-paper JSON")
    s.add_argument("--no-wait", action="store_true", help="return the run_id immediately instead of polling")
    s = sub.add_parser("status"); s.set_defaults(func=cmd_status); s.add_argument("run_id")
    s = sub.add_parser("cancel"); s.set_defaults(func=cmd_cancel); s.add_argument("run_id")
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
