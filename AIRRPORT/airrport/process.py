from .frames import iter_all_frames

AUTOMATON = None

def worker_init(shared_automaton):
    global AUTOMATON
    AUTOMATON = shared_automaton

def process_chunk_bulk(rows):
    """rows: list of tuples (nuc_seq, rid[, count]) -> rows of (out_id, CDR3_match[, count])"""
    out = []
    for row in rows:
        if len(row) == 3:
            nuc_seq, rid, cnt = row
        else:
            nuc_seq, rid = row
            cnt = None
        if nuc_seq is None:
            continue
        base_id = "" if rid is None else str(rid)
        try:
            for strand, frame, prot_u in iter_all_frames(nuc_seq):
                for _, matched in AUTOMATON.iter(prot_u):
                    if '*' in matched:
                        continue
                    out_id = f"{base_id}|{strand}{frame}"
                    out.append((out_id, matched) if cnt is None else (out_id, matched, cnt))
        except Exception:
            continue
    return out

def process_chunk_sc(rows):
    """rows: list of tuples (nuc_seq, seq_id, umi, cb) -> (seq_id|frame, CDR3_match, umi, cb)"""
    out = []
    for nuc_seq, seq_id, umi, cb in rows:
        if nuc_seq is None:
            continue
        base_id = "" if seq_id is None else str(seq_id)
        try:
            for strand, frame, prot_u in iter_all_frames(nuc_seq):
                for _, matched in AUTOMATON.iter(prot_u):
                    if '*' in matched:
                        continue
                    out_id = f"{base_id}|{strand}{frame}"
                    out.append((out_id, matched, umi, cb))
        except Exception:
            continue
    return out
