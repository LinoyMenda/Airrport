import pyarrow as pa

def make_table_bulk(matches_accum, has_count, output_id_col):
    if not matches_accum:
        return None
    if has_count:
        rid, matched, cnt = zip(*matches_accum)
        return pa.table({output_id_col: pa.array(rid, type=pa.string()),
                         "CDR3_match": pa.array(matched, type=pa.string()),
                         "count": pa.array(cnt)})
    else:
        rid, matched = zip(*matches_accum)
        return pa.table({output_id_col: pa.array(rid, type=pa.string()),
                         "CDR3_match": pa.array(matched, type=pa.string())})

def make_table_sc(matches_accum, output_id_col, umi_col, cb_col):
    if not matches_accum:
        return None
    rid, matched, umi, cb = zip(*matches_accum)
    return pa.table({output_id_col: pa.array(rid, type=pa.string()),
                     "CDR3_match": pa.array(matched, type=pa.string()),
                     umi_col: pa.array(umi, type=pa.string()),
                     cb_col: pa.array(cb, type=pa.string())})

def empty_table_bulk(has_count, output_id_col):
    cols = {output_id_col: pa.array([], type=pa.string()),
            "CDR3_match": pa.array([], type=pa.string())}
    if has_count:
        cols["count"] = pa.array([], type=pa.int64())
    return pa.table(cols)

def empty_table_sc(output_id_col, umi_col, cb_col):
    return pa.table({output_id_col: pa.array([], type=pa.string()),
                     "CDR3_match": pa.array([], type=pa.string()),
                     umi_col: pa.array([], type=pa.string()),
                     cb_col: pa.array([], type=pa.string())})
