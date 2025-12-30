import pyarrow.dataset as ds

def load_aa_list(db_parquet, col):
    db_dataset = ds.dataset(db_parquet, format="parquet")
    if col not in db_dataset.schema.names:
        raise ValueError(f"DB column '{col}' not found. Available: {db_dataset.schema.names}")
    table_db = db_dataset.to_table(columns=[col])
    aa_list = sorted(set(x for x in table_db.column(col).to_pylist() if x is not None))
    return aa_list

def build_automaton(aa_list):
    import ahocorasick
    A = ahocorasick.Automaton()
    for aa in aa_list:
        A.add_word(str(aa), str(aa))
    A.make_automaton()
    return A
