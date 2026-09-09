#!/usr/bin/env python
"""Work out which experiments still need running, and optionally resume from there.

A full suite run takes the better part of a day, and a run that dies part way
through used to leave no usable record of how far it got. Restarting from
experiment 1 would then throw away hours of correct results. This script asks a
simpler question than "what did the log say": for each experiment, is there a
result file on disk that is newer than the cutoff and was not produced by a
reduced-scale run?

Usage:
    python resume_suite.py                     # report only
    python resume_suite.py --since 2026-09-08T17:20
    python resume_suite.py --launch            # report, then run what is missing
"""
import argparse
import ast
import json
import os
import subprocess
import sys
from datetime import datetime

from config import Config
from run_all_experiments import EXPERIMENTS

RESULTS_SUBDIR = 'advanced_experiments'


def _result_names():
    """Map experiment id -> the basename its method saves.

    Read out of the source rather than hardcoded, so renaming an experiment's
    output cannot silently desynchronise this script from the suite.
    """
    method_to_name = {}
    exp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiments')
    for fname in os.listdir(exp_dir):
        if not fname.endswith('.py'):
            continue
        with open(os.path.join(exp_dir, fname), encoding='utf-8') as fh:
            tree = ast.parse(fh.read())
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            for sub in ast.walk(node):
                if (isinstance(sub, ast.Call)
                        and isinstance(sub.func, ast.Attribute)
                        and sub.func.attr == '_save_experiment_results'
                        and sub.args
                        and isinstance(sub.args[0], ast.Constant)):
                    method_to_name[node.name] = sub.args[0].value

    out = {}
    for eid, (_, method_name) in EXPERIMENTS.items():
        if method_name in method_to_name:
            out[eid] = method_to_name[method_name] + '_results.json'
    return out


def classify(exp_ids, cutoff, results_dir):
    """Return (done, todo, notes) for the requested experiments."""
    names = _result_names()
    done, todo, notes = [], [], {}

    for eid in exp_ids:
        label = EXPERIMENTS.get(eid, ('Unknown',))[0]
        fname = names.get(eid)
        if fname is None:
            todo.append(eid)
            notes[eid] = f'{label}: no known output file, will rerun'
            continue

        path = os.path.join(results_dir, RESULTS_SUBDIR, fname)
        if not os.path.exists(path):
            todo.append(eid)
            notes[eid] = f'{label}: no result file'
            continue

        mtime = datetime.fromtimestamp(os.path.getmtime(path))
        if cutoff and mtime < cutoff:
            todo.append(eid)
            notes[eid] = (f'{label}: stale, written {mtime:%m-%d %H:%M} '
                          f'before cutoff {cutoff:%m-%d %H:%M}')
            continue

        # A reduced-scale run writes a real file with real-looking numbers. It
        # is not a publication result, so treat it as work still to do.
        reduced = False
        try:
            with open(path, encoding='utf-8') as fh:
                payload = json.load(fh)
            if isinstance(payload, dict):
                reduced = bool(payload.get('provenance', {}).get('is_reduced_run'))
        except (json.JSONDecodeError, OSError) as exc:
            todo.append(eid)
            notes[eid] = f'{label}: unreadable ({type(exc).__name__}), will rerun'
            continue

        if reduced:
            todo.append(eid)
            notes[eid] = f'{label}: REDUCED RUN on disk, not publishable'
        else:
            done.append(eid)
            notes[eid] = f'{label}: complete, written {mtime:%m-%d %H:%M}'

    return done, todo, notes


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--exp', type=int, nargs='+', default=sorted(EXPERIMENTS),
                    help='experiments the run was meant to cover (default: all 20)')
    ap.add_argument('--since', default=None,
                    help='ISO timestamp; results older than this count as stale. '
                         'Defaults to the started_at in suite_progress.json if present.')
    ap.add_argument('--results-dir', default=Config.RESULTS_DIR)
    ap.add_argument('--launch', action='store_true',
                    help='actually run the missing experiments')
    ap.add_argument('--ids-only', action='store_true',
                    help='print just the incomplete experiment ids, for scripts')
    args = ap.parse_args(argv)

    quiet = args.ids_only

    def say(*a, **kw):
        if not quiet:
            print(*a, **kw)

    cutoff = None
    if args.since:
        cutoff = datetime.fromisoformat(args.since)
    else:
        progress = os.path.join(args.results_dir, RESULTS_SUBDIR, 'suite_progress.json')
        if os.path.exists(progress):
            try:
                with open(progress, encoding='utf-8') as fh:
                    cutoff = datetime.fromisoformat(json.load(fh)['started_at'])
                print(f'cutoff from suite_progress.json: {cutoff:%Y-%m-%d %H:%M}')
            except (json.JSONDecodeError, KeyError, ValueError, OSError):
                pass
    if cutoff is None:
        print('no cutoff given: judging only on whether a result file exists')

    done, todo, notes = classify(args.exp, cutoff, args.results_dir)

    if args.ids_only:
        print(' '.join(str(e) for e in todo))
        return 0

    print(f"\n{'=' * 64}\nSUITE STATE\n{'=' * 64}")
    for eid in args.exp:
        mark = 'done' if eid in done else 'TODO'
        print(f'  {eid:2d}. [{mark}] {notes[eid]}')
    print(f'\n  {len(done)} complete, {len(todo)} to run')

    if not todo:
        print('\nNothing to do.')
        return 0

    cmd = [sys.executable, 'run_all_experiments.py', '--exp'] + [str(e) for e in todo]
    print('\nResume with:\n  ' + ' '.join(cmd))

    if args.launch:
        print('\nLaunching...\n', flush=True)
        return subprocess.call(cmd)
    return 0


if __name__ == '__main__':
    sys.exit(main())
