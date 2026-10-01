"""Run the RIS experiment package with ``python -m experiments``."""

from .cli import run_all_experiments, run_journal_experiments, run_new_experiments


def main():
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--new-only":
        run_new_experiments()
    elif len(sys.argv) > 1 and sys.argv[1] == "--journal":
        run_journal_experiments()
    else:
        run_all_experiments()


if __name__ == "__main__":
    main()
