from pathlib import Path
import pyarrow.dataset as ds

def discover_inputs(dedup_path: Path):
    if dedup_path.is_file():
        if dedup_path.suffix != ".parquet":
            raise ValueError(f"File {dedup_path} is not a .parquet file")
        return [dedup_path]
    else:
        if not dedup_path.is_dir():
            raise ValueError(f"Path {dedup_path} does not exist")
        return sorted(p for p in dedup_path.iterdir() if p.suffix == ".parquet")

def make_scanner(dedup_file, mode, args):
    dset = ds.dataset(dedup_file, format="parquet")
    schema_names = dset.schema.names
    if mode == "bulk":
        seq_col = args.dedup_seq_col
        id_col = args.dedup_id_col
        if seq_col not in schema_names:
            raise ValueError(f"Sequence column '{seq_col}' not in {dedup_file}; available: {schema_names}")
        if id_col not in schema_names:
            raise ValueError(f"ID column '{id_col}' not in {dedup_file}; available: {schema_names}")
        has_count = "count" in schema_names
        project_cols = [seq_col, id_col] + (["count"] if has_count else [])
        scanner = dset.scanner(columns=project_cols, batch_size=args.batch_size)
        meta = dict(seq_col=seq_col, id_col=id_col, has_count=has_count)
    else:
        seq_col = args.sc_seq_col
        id_col = args.sc_id_col
        cb_col = args.sc_cb_col
        umi_col = args.sc_umi_col
        for col in [seq_col, id_col, cb_col, umi_col]:
            if col not in schema_names:
                raise ValueError(f"Column '{col}' not in {dedup_file}; available: {schema_names}")
        project_cols = [seq_col, id_col, umi_col, cb_col]
        scanner = dset.scanner(columns=project_cols, batch_size=args.batch_size)
        meta = dict(seq_col=seq_col, id_col=id_col, cb_col=cb_col, umi_col=umi_col, has_count=False)
    return scanner, meta
