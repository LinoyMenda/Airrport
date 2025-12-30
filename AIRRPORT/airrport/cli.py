from pathlib import Path
import argparse

def parse_args():
    p = argparse.ArgumentParser(
        description="Match DB AA sequences as substrings in translated dedup nucleotide reads (bulk or single-cell)."
    )
    p.add_argument("db_parquet", type=Path, help="Parquet containing AA sequences (e.g., CDR3.aa column)")
    p.add_argument("dedup_path", type=Path, help="Parquet file or directory with input parquet(s)")

    p.add_argument("--db_col", default="CDR3.aa", help="AA sequence column in DB parquet")

    # BULK
    p.add_argument("--dedup_seq_col", default="sequence", help="Nucleotide sequence column (bulk)")
    p.add_argument("--dedup_id_col", default="read_id", help="Read ID column name (bulk)")

    # SINGLE-CELL
    p.add_argument("--mode", choices=["bulk", "singlecell"], default="bulk",
                   help="Input flavor: bulk (original) or singlecell (seq_id/CB/UMI/sequence)")
    p.add_argument("--sc_seq_col", default="sequence", help="Nucleotide sequence column (single-cell)")
    p.add_argument("--sc_id_col", default="seq_id", help="Read/record ID column (single-cell)")
    p.add_argument("--sc_cb_col", default="CB", help="Cell barcode column (single-cell)")
    p.add_argument("--sc_umi_col", default="UMI", help="UMI column (single-cell)")

    # performance / IO
    p.add_argument("--workers", type=int, default=4, help="Parallel worker count")
    p.add_argument("--batch_size", type=int, default=200_000, help="Arrow batch size for reading input")
    p.add_argument("--out_root", default="matches_output", help="Root output directory")
    p.add_argument("--write_mode", choices=["single", "shards"], default="single",
                   help="Write a single output parquet or write per-batch shards and merge")
    p.add_argument("--keep_shards", action="store_true",
                   help="Keep shard files after merge (only with --write_mode=shards). Default: delete.")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-analyze files even if a final matches parquet already exists (default: skip existing).")
    return p.parse_args()
