import quickdna

def iter_all_frames(nuc_seq: str):
    """
    Yield (strand, frame, prot_string_upper) for up to 6 frames.
    Assumes order: F0,F1,F2,R0,R1,R2.
    """
    frames = quickdna.DnaSequence(str(nuc_seq)).translate_all_frames()
    if not frames:
        return
    mapping = [('+', 0), ('+', 1), ('+', 2), ('-', 0), ('-', 1), ('-', 2)]
    for idx, prot in enumerate(frames):
        if prot is None:
            continue
        strand, frame = mapping[idx] if idx < len(mapping) else ('?', -1)
        yield strand, frame, str(prot).upper()
