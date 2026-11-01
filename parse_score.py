import os
import polars as pl
from fractions import Fraction

DS_FOLDER = os.path.expanduser("~/ds")
MOZ_FOLDER = os.path.join(DS_FOLDER, "mozart_piano_sonatas-main")

# from johentsch/ms3/src/ms3/dezrann.py
FRACTION_COLUMNS = (
    "mc_onset",
    "mn_onset",
    "quarterbeats",
    "quarterbeats_all_endings",
    "duration_qb",
    "act_dur",
    "mc_offset",
)

# from johentsch/ms3/dezrann.py
def safe_frac(s) -> Optional[Fraction]:
    """Converts a TSV cell such as '1/3' or '4.0' into a Fraction. Empty cells are returned as None.

    Raises:
        ValueError: If the value cannot be converted.
    """
    if s is None or (isinstance(s, float) and pd.isna(s)) or s is pd.NA:
        return None
    if isinstance(s, str):
        s = s.strip()
        if s == "":
            return None
    try:
        return Fraction(s)
    except (TypeError, ValueError, ZeroDivisionError):
        raise ValueError(f"Could not convert {s!r} to a fraction.")




def read_tsv(fpath, base_folder=MOZ_FOLDER):
    cur_path = os.path.join(base_folder, fpath)
    pl.read_csv(empty_string_is_null = True)

labels = pl.read_csv(os.path.join(MOZ_FOLDER, "harmonies/K279-1.harmonies.tsv"))
notes = ms3.load_tsv(os.path.join(MOZ_FOLDER, "notes/K279-1.notes.tsv"))

print(labels.columns)


