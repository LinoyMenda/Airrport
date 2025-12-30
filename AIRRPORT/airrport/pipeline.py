from pathlib import Path
from multiprocessing import get_context
from tqdm import tqdm
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import os, sys

from .db import load_aa_list, build_automaton
from .process import worker_init, process_chunk_bulk, process_chunk_sc
from .tables import make_table_bulk, make_table_sc, empty_table_bulk, empty_table_sc
from .io import discover_inputs, make_scanner

def run(args):
    OUTPUT_ID_COL = "read_id" if args.mode == "bulk" else args.sc_id_col

    print(f"[+] Loading DB AA sequences from {args.db_parquet}")
    aa_list = load_aa_list(args.db_parquet, args.db_col)
    print(f"[+] {len(aa_list):,} unique AA sequences from DB ({args.db_col})")

    print("[+] Building Aho-Corasick automaton...")
    automaton = build_automaton(aa_list)

    dedup_files = discover_inputs(args.dedup_path)
    total_files = len(dedup_files)
    print(f"[+] Found {total_files} parquet file(s) under {args.dedup_path}")

    processed_ok = 0
    skipped_existing = 0
    failed = 0

    for dedup_file in dedup_files:
        final_out_path = Path(args.out_root) / f"matched_{Path(dedup_file).stem}.parquet"

        if final_out_path.exists() and not args.overwrite:
            print(f"[skip existing] {Path(dedup_file).name} → {final_out_path.name} already present")
            skipped_existing += 1
            continue

        try:
            file_out_dir = Path(args.out_root) / Path(dedup_file).stem
            file_out_dir.mkdir(parents=True, exist_ok=True)

            scanner, meta = make_scanner(dedup_file, args.mode, args)
            seq_col = meta["seq_col"]; id_col = meta["id_col"]
            has_count = meta.get("has_count", False)
            cb_col = meta.get("cb_col"); umi_col = meta.get("umi_col")

            if args.write_mode == "single":
                writer = None
                wrote_any = False

                with get_context("fork").Pool(
                    processes=args.workers, initializer=worker_init, initargs=(automaton,)
                ) as pool:
                    batch_idx = 0
                    for batch in tqdm(scanner.to_batches(), desc=f"{Path(dedup_file).name}", unit="batch"):
                        if args.mode == "bulk":
                            seqs = batch.column(seq_col).to_pylist()
                            ids = batch.column(id_col).to_pylist()
                            if has_count:
                                cnts = batch.column("count").to_pylist()
                                paired = list(zip(seqs, ids, cnts))
                            else:
                                paired = list(zip(seqs, ids))
                            chunk_size = max(1, len(paired) // max(1, args.workers))
                            sublists = [paired[i:i+chunk_size] for i in range(0, len(paired), chunk_size)]
                            futures = [pool.apply_async(process_chunk_bulk, (sub,)) for sub in sublists]
                            matches_accum = []
                            for fut in futures:
                                matches_accum.extend(fut.get())
                            table = make_table_bulk(matches_accum, has_count, OUTPUT_ID_COL)
                        else:
                            seqs = batch.column(seq_col).to_pylist()
                            ids = batch.column(id_col).to_pylist()
                            umis = batch.column(umi_col).to_pylist()
                            cbs = batch.column(cb_col).to_pylist()
                            paired = list(zip(seqs, ids, umis, cbs))
                            chunk_size = max(1, len(paired) // max(1, args.workers))
                            sublists = [paired[i:i+chunk_size] for i in range(0, len(paired), chunk_size)]
                            futures = [pool.apply_async(process_chunk_sc, (sub,)) for sub in sublists]
                            matches_accum = []
                            for fut in futures:
                                matches_accum.extend(fut.get())
                            table = make_table_sc(matches_accum, OUTPUT_ID_COL, umi_col, cb_col)

                        if table is not None:
                            if writer is None:
                                writer = pq.ParquetWriter(final_out_path, table.schema)
                            writer.write_table(table)
                            wrote_any = True
                        batch_idx += 1

                if writer is not None:
                    writer.close()
                if not wrote_any:
                    if args.mode == "bulk":
                        pq.write_table(empty_table_bulk(has_count, OUTPUT_ID_COL), final_out_path)
                    else:
                        pq.write_table(empty_table_sc(OUTPUT_ID_COL, umi_col, cb_col), final_out_path)
                processed_ok += 1

            else:
                with get_context("fork").Pool(
                    processes=args.workers, initializer=worker_init, initargs=(automaton,)
                ) as pool:
                    batch_idx = 0
                    for batch in tqdm(scanner.to_batches(), desc=f"{Path(dedup_file).name}", unit="batch"):
                        shard_path = file_out_dir / f"matches_batch_{batch_idx:06d}.parquet"
                        if shard_path.exists():
                            batch_idx += 1
                            continue

                        if args.mode == "bulk":
                            seqs = batch.column(seq_col).to_pylist()
                            ids = batch.column(id_col).to_pylist()
                            if has_count:
                                cnts = batch.column("count").to_pylist()
                                paired = list(zip(seqs, ids, cnts))
                            else:
                                paired = list(zip(seqs, ids))
                            chunk_size = max(1, len(paired) // max(1, args.workers))
                            sublists = [paired[i:i+chunk_size] for i in range(0, len(paired), chunk_size)]
                            futures = [pool.apply_async(process_chunk_bulk, (sub,)) for sub in sublists]
                            matches_accum = []
                            for fut in futures:
                                matches_accum.extend(fut.get())
                            table = make_table_bulk(matches_accum, has_count, OUTPUT_ID_COL)
                            if table is None:
                                table = empty_table_bulk(has_count, OUTPUT_ID_COL)
                        else:
                            seqs = batch.column(seq_col).to_pylist()
                            ids = batch.column(id_col).to_pylist()
                            umis = batch.column(umi_col).to_pylist()
                            cbs = batch.column(cb_col).to_pylist()
                            paired = list(zip(seqs, ids, umis, cbs))
                            chunk_size = max(1, len(paired) // max(1, args.workers))
                            sublists = [paired[i:i+chunk_size] for i in range(0, len(paired), chunk_size)]
                            futures = [pool.apply_async(process_chunk_sc, (sub,)) for sub in sublists]
                            matches_accum = []
                            for fut in futures:
                                matches_accum.extend(fut.get())
                            table = make_table_sc(matches_accum, OUTPUT_ID_COL, umi_col, cb_col)
                            if table is None:
                                table = empty_table_sc(OUTPUT_ID_COL, umi_col, cb_col)

                        pq.write_table(table, shard_path)
                        batch_idx += 1

                shard_files = sorted(file_out_dir.glob("matches_batch_*.parquet"))
                if not shard_files:
                    processed_ok += 1
                else:
                    merged = ds.dataset([str(p) for p in shard_files], format="parquet").to_table()
                    pq.write_table(merged, final_out_path)
                    if not args.keep_shards:
                        for pth in shard_files:
                            try: os.remove(pth)
                            except OSError: pass
                        try: os.rmdir(file_out_dir)
                        except OSError: pass
                    processed_ok += 1

        except Exception as e:
            print(f"[ERROR] {Path(dedup_file).name}: {e}", file=sys.stderr)
            failed += 1
            continue

    print("\n===== SUMMARY =====")
    print(f"Total parquet files found : {total_files}")
    print(f"Analyzed (this run)       : {processed_ok}")
    print(f"Skipped (existing output) : {skipped_existing} (use --overwrite to re-run)")
    print(f"Failed                    : {failed}")
