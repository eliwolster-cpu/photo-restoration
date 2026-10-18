"""CLI entry point. Currently only ingests the file (plan step 4)."""
import argparse
import sys

from restoration.ingest import IngestError, ingest


def main(argv=None):
    parser = argparse.ArgumentParser(description="Conservatively restore a scanned photo.")
    parser.add_argument("image")
    parser.add_argument("--storage", default="storage")
    args = parser.parse_args(argv)
    try:
        result = ingest(args.image, args.storage)
    except IngestError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    note = " (duplicate, already ingested)" if result.duplicate else ""
    print(f"job {result.job_id}{note}: original stored at {result.original_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
