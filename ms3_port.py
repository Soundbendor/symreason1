import pandas as pd
from fractions import Fraction
from functools import cache, reduce


def mn2int(mn_series):
    """Turn a series of measure numbers parsed as strings into two integer columns 'mn' and 'volta'."""
    try:
        split = mn_series.fillna("").str.extract(r"(?P<mn>\d+)(?P<volta>[a-g])?")
    except Exception:
        mn_series = pd.DataFrame(mn_series, columns=["mn", "volta"])
        try:
            return mn_series.astype("Int64")
        except Exception:
            return mn_series
    split.mn = pd.to_numeric(split.mn)
    split.volta = pd.to_numeric(
        split.volta.map({"a": 1, "b": 2, "c": 3, "d": 4, "e": 5})
    )
    return split.astype("Int64")


def str2inttuple(tuple_string: str, strict: bool = True) -> Tuple[int]:
    tuple_string = tuple_string.strip("[](),")
    if tuple_string == "":
        return tuple()
    res = []
    for s in tuple_string.split(", "):
        try:
            res.append(int(s))
        except ValueError:
            if strict:
                print(
                    f"String value '{s}' could not be converted to an integer, "
                    f"'{tuple_string}' not to an integer tuple."
                )
                raise
            if s[0] == s[-1] and s[0] in ('"', "'"):
                s = s[1:-1]
            try:
                res.append(int(s))
            except ValueError:
                res.append(s)
    return tuple(res)


def str2keysig_dict(dict_content: str) -> Dict[int, int]:
    entry_tuples = re.findall(KEYSIG_DICT_ENTRY_REGEX, dict_content)
    return {int(mc): int(keysig) for mc, keysig in entry_tuples}


def str2timesig_dict(dict_content: str) -> Dict[int, str]:
    entry_tuples = re.findall(TIMESIG_DICT_ENTRY_REGEX, dict_content)
    return {int(mc): timesig for mc, timesig in entry_tuples}


# default values copied from marshmallow's fields.Boolean
TRUTHY_VALUES = {
    "t",
    "T",
    "true",
    "True",
    "TRUE",
    "on",
    "On",
    "ON",
    "y",
    "Y",
    "yes",
    "Yes",
    "YES",
    "1",
    1,
    True,
}
FALSY_VALUES = {
    "f",
    "F",
    "false",
    "False",
    "FALSE",
    "off",
    "Off",
    "OFF",
    "n",
    "N",
    "no",
    "No",
    "NO",
    "0",
    0,
    0.0,
    False,
}


def int2bool(s: str) -> Union[bool, str]:
    try:
        return bool(int(s))
    except Exception:
        return s


def value2bool(value: str | float | int | bool) -> bool | str | float | int:
    if value in TRUTHY_VALUES:
        return True
    if value in FALSY_VALUES:
        return False
    if isinstance(value, str):
        try:
            converted = float(value)
        except Exception:
            return value
        if converted in TRUTHY_VALUES:
            return True
        if converted in FALSY_VALUES:
            return False
    return value


def safe_frac(s: str) -> Union[Fraction, str]:
    try:
        return Fraction(s)
    except Exception:
        return s


def safe_int(s) -> Union[int, str]:
    try:
        return int(float(s))
    except Exception:
        return s


def tuple_to_list_recursive(t):
    if isinstance(t, (tuple, list)):
        return [tuple_to_list_recursive(e) for e in t]
    return t


def eval_string_to_nested_list(s):
    """Tries to parse a string encoding a nested list, returns the input if it fails."""
    try:
        nested_tuple_or_list = literal_eval(s)
        return tuple_to_list_recursive(nested_tuple_or_list)
    except Exception:
        return s


def parse_interval_index_column(df, column=None, closed="left"):
    """Turns a column of strings in the form '[0.0, 1.1)' into a :obj:`pandas.IntervalIndex`.

    Parameters
    ----------
    df : :obj:`pandas.DataFrame`
    column : :obj:`str`, optional
        Name of the column containing strings. If not specified, use the index.
    closed : :obj:`str`, optional
        On whot side the intervals should be closed. Defaults to 'left'.

    Returns
    -------
    :obj:`pandas.IntervalIndex`
    """
    iv_regex = SLICE_INTERVAL_REGEX
    if column is None:
        iv_strings = df.index
    else:
        iv_strings = df[column]
    values = iv_strings.str.extract(iv_regex).astype(float)
    iix = pd.IntervalIndex.from_arrays(values[0], values[1], closed=closed)
    return iix


TSV_COLUMN_CONVERTERS = {
    "added_tones": str2inttuple,
    "act_dur": safe_frac,
    "composed_end": safe_int,
    "composed_start": safe_int,
    "chord_tones": str2inttuple,
    "globalkey_is_minor": value2bool,
    "KeySig": str2keysig_dict,
    "localkey_is_minor": value2bool,
    "mc_offset": safe_frac,
    "mc_onset": safe_frac,
    "mn_onset": safe_frac,
    "movementNumber": safe_int,
    "next": str2inttuple,
    "nominal_duration": safe_frac,
    "quarterbeats": safe_frac,
    "quarterbeats_playthrough": safe_frac,
    "quarterbeats_all_endings": safe_frac,
    "onset": safe_frac,
    "duration": safe_frac,
    "scalar": safe_frac,
    "TimeSig": str2timesig_dict,
    "volta_mcs": eval_string_to_nested_list,
}

TSV_COLUMN_TITLES = {
    "absolute_base": "Int64",
    "absolute_root": "Absolute Root",
    "act_dur": "Actual Length",
    "added_tones": "Added Tones",
    "all_notes_qb": "Summed Note Duration",
    "alt_label": "Alternative Label",
    "barline": "Barline",
    "base": "Base",
    "bass_note": "Bass Note",
    "breaks": "Breaks",
    "cadence": "Cadence",
    "cadences_id": "Cadence ID",
    "changes": "Changes",
    "chord": "Chord",
    "chord_id": "Chord ID",
    "chord_tones": "Chord Tones",
    "chord_type": "Chord Type",
    "color_a": "Color Alpha",
    "color_b": "Color Blue",
    "color_g": "Color Green",
    "color_html": "Color HTML",
    "color_name": "Color Name",
    "color_r": "Color Red",
    "composed_end": "Latest Composition Year",
    "composed_start": "Earliest Composition Year",
    "corpus": "Corpus",
    "dont_count": "Exclude from measure count",
    "duration": "Duration",
    "duration_qb": "Duration in ♩",
    "expanded_id": "Label ID",
    "figbass": "Chord Inversion",
    "fname": "Piece identifier",
    "form": "Chord Category",
    "form_label_count": "Number of Form Annotation Labels",
    "globalkey": "Global Key",
    "globalkey_is_minor": "Global Key is Minor",
    "gracenote": "Grace Note",
    "guitar_chord_count": "Number of Guitar/Jazz Chord Labels",
    "harmonies_id": "Label ID",
    "harmony_layer": "Harmony Encoding Layer",
    "i": "Index",
    "keysig": "Key Signature",
    "KeySig": "Key Signatures",
    "label": "Label",
    "label_count": "Number of DCML Labels",
    "label_type": "Label Type",
    "last_mc": "Highest Measure Count",
    "last_mc_unfolded": "Highest Measure Count (unfolded)",
    "last_mn": "Highest Measure Number",
    "last_mn_unfolded": "Highest Measure Number (unfolded)",
    "leftParen": "Left Parenthesis",
    "length_qb": "Quarter Length",
    "length_qb_unfolded": "Quarter Length (unfolded)",
    "localkey": "Local Key",
    "localkey_is_minor": "Local Key is Minor",
    "marker": "Marker",
    "mc": "Measure Count",
    "mc_offset": "Offset of Encoded Measure",
    "mc_onset": "Offset within Encoded Measure",
    "mc_playthrough": "Measure Count Unfolded",
    "midi": "MIDI pitch",
    "mn": "Measure Number",
    "mn_onset": "Offset within Logical Measure",
    "movementNumber": "Movement Number",
    "name": "Name",
    "nashville": "Nashville",
    "next": "Next Measure Counts",
    "n_onsets": "Number of Note Onsets",
    "n_onset_positions": "Number of Unique Onset Positions (slices)",
    "nominal_duration": "Nominal Duration",
    "notes_id": "Notes ID",
    "numbering_offset": "Numbering Offset",
    "numeral": "Roman Numeral",
    "octave": "Octave",
    "offset:x": "Horizontal Offset",
    "offset:y": "Vertical Offset",
    "offset_x": "Horizontal Offset",
    "offset_y": "Vertical Offset",
    # 'onset':
    "pedal": "Pedal Point",
    "phraseend": "Phrase Annotation",
    "piece": "Piece identifier",
    # 'playthrough':
    "quarterbeats": "Offset from Beginning (leaving out alternative endings)",
    "quarterbeats_playthrough": "Offset from the beginning, including all repeats (in unfolded tables)",
    "quarterbeats_all_endings": "Offset from Beginning (counting through alternative "
    "endings as if they were adjacent bars)",
    "regex_match": "Regular Expression Match",
    "relativeroot": "Relative Root",
    "repeats": "Repeats",
    "rightParen": "Right Parenthesis",
    "root": "Root",
    "rootCase": "Root Case",
    "scalar": "Scalar",
    "slur": "Slur",
    "special": "Special Label",
    "staff": "Staff",
    "tied": "Tied Note",
    "timesig": "Time Signature",
    "TimeSig": "Time Signatures",
    "tpc": "Tonal Pitch Class",
    "voice": "Notational Layer",
    # 'voices': 'Voices',
    "volta": "Volta",
    "volta_mcs": "Volta Measure Counts",
}

TSV_COLUMN_DESCRIPTIONS = {
    "absolute_base": "MuseScore encoding the bass pitch class of an absolute chord label, as MuseScore tonal pitch "
    "class such that C=14, G=15, etc.",
    "absolute_root": "MuseScore encoding the root pitch class of an absolute chord label, as MuseScore tonal pitch "
    "class such that C=14, G=15, etc.",
    "act_dur": "How long a measure actually lasts, which can deviate from the time signature. Relevant, for example, "
    "for pickup measures, split measures, cadenzas.",
    "added_tones": "Chord tones considered as added, expressed as fifth intervals relative to the local tonic.",
    "all_notes_qb": "The summed duration of all notes in a given piece, measured in ♩.",
    "alt_label": "Another interpretation of the same chord which the annotator finds equally or slightly less "
    "convincing.",
    "barline": "Name of non-default barline.",
    "base": "Base",
    "bass_note": "Loweste note designated by the chord label, expressed as fifth intervals relative to the local "
    "tonic.",
    "breaks": 'Can be "section" or "line".',
    "cadence": "PAC, IAC, EC, DC, PC, HC, and HC subtypes such as HC.SIM",
    "cadences_id": "Row in the cadence table.",
    "changes": "Alterations, suspensions, additions, and omissions, written within parentheses in the chord label.",
    "chord": "The part of an annotation label that corresponds to the actual chord label.",
    "chord_id": "Row in the chords table.",
    "chord_tones": "Tones expressed by the label, as fifth intervals relative to the local tonic.",
    "chord_type": "'M', 'm', 'o', '+', 'mm7', 'Mm7', 'MM7', 'mM7', 'o7', '%7', '+7', '+M7'",
    "color_a": "Value between 0 and 255",
    "color_b": "Value between 0 and 255",
    "color_g": "Value between 0 and 255",
    "color_html": "A hexadicmal RGB value, e.g. #FF0000",
    "color_name": "Can be a CSS name or one of the names for MuseScore colors stored in ms3.utils.MS3_HTML",
    "color_r": "Value between 0 and 255",
    "composed_end": "Latest possible year of composition.",
    "composed_start": "Earliest possible year of composition.",
    "corpus": "Name of a group of pieces",
    "dont_count": "Is 1 if this MC does not increase the measure number counter, according to the MuseScore setting.",
    "duration": "As fraction of a whole note.",
    "duration_qb": "A float corresponding to duration * 4",
    "expanded_id": "Row in the expanded table.",
    "figbass": "7, 65, 43, 2, 64, 6 or empty for root position.",
    "fname": "Name identifier (filename without suffixes) of a piece",
    "form": "%, o, +, M, +M",
    "form_label_count": "Number of form annotation labels in a piece.",
    "globalkey": "The key of the entire piece as note name, lowercase designating a minor key.",
    "globalkey_is_minor": "Boolean that is 1 if the piece is in minor and 0 if it is in major.",
    "gracenote": 'Name given to a type of grace note in the MuseScore encoding, e.g. "grace16"',
    "guitar_chord_count": "Number of guitar/jazz/absolute chord labels in a piece.",
    "harmonies_id": "Row in the expanded table.",
    "harmony_layer": "0: Simple string (does not begin with a note name, otherwise MS3 will turn it into type 3; "
    "prevent through leading dot);\n"
    "1: MuseScore's Roman Numeral Annotation format;\n"
    "2: MuseScore's Nashville Number format;\n"
    "3: Absolute chord encoded by MuseScore.",
    "i": "An integer serving as row ID",
    "keysig": "Positive integer for number of sharps, negative integer for number of flats.",
    "KeySig": "All key signatures in a piece and where they occur. Format: '<MC>: <keysig>, <MC>: <keysig>, ...'"
    "Corresponds to a dictionary without outer curly braces.",
    "label": "String corresponding to the entire annotation label.",
    "label_count": "Number of chord annotation labels that match the regex of the DCML annotation standard.",
    "label_type": 'Previous name of what now is called "regex_match".',
    "last_mc": "The highest Measure Count value in the piece. Corresponds to the number of <Measure> tags.",
    "last_mc_unfolded": "The amount of <Measure> tags of a full playthrough, i.e. when all repeats are unfolded. "
    "Equals 'last_mc' if piece has no repeat signs.",
    "last_mn": "The highest Measure Number in the piece. Corresponds roughly to the number of complete bars.",
    "last_mn_unfolded": "The last Measure Number when all repeats are unfolded. Equals 'last_mn' if piece has no "
    "repeat signs.",
    "leftParen": "Pertaining to MuseScore encoding.",
    "length_qb": "The length of a piece in quarter notes.",
    "length_qb_unfolded": "The length of full playthrough in quarter notes, i.e. when all repeats are unfolded. "
    "Equals 'length_qb' if piece has no repeat signs.",
    "localkey": "The key that a Roman numeral is relative to, expressed as a Roman numeral relative to the global key.",
    "localkey_is_minor": "Boolean that is 1 if the local key is minor and 0 if it is major.",
    "marker": "Pertaining to MuseScore encoding.",
    "mc": "Running count of encoded <Measure> tags which do not necessarily express a full measure (e.g. in case of "
    "an anacrusis).",
    "mc_offset": "Distance of a <Measure> from the beginning of the logical measure, expressed as fraction of a whole "
    "note. Relevant only for split and anacrusis measures, 0 otherwise.",
    "mc_onset": "Distance of an event from the beginning of the <Measure> tag.",
    "mc_playthrough": 'Has the function of the "mc" column in tables with unfolded repeats where "mc" is not unique.',
    "midi": "Value between 0 and 127 where 60 = C4, 61 = C#4/Db4, etc.",
    "mn": "Measure number as printed in the score, computed from mc, dont_count and numbering_offset.",
    "mn_onset": "Distance from the beginning of the logical measure. Relevant, for example, to compute the metric "
    "position of an event.",
    "movementNumber": "Metadata field for the number of a movement. Should be specified as integer, not as a Roman "
    "number.",
    "n_onsets": "Number of onsets in a piece. Different from the number of entries in the 'notes' table, because "
    "that one corresponds to the number of note heads; but notes being tied to do not represent onsets.",
    "n_onset_positions": "Number of unique onset positions in a piece. Corresponds to the number of slices when "
    "performing full expansion.",
    "name": "",
    "nashville": 'Numbering system that specifies the root of a chord as scale degree of the local key, e.g. "1", '
    '"b3", "#5", "b7".',
    "next": 'The "mc" values of all <Measure> tags that can follow this one, specified as a tuple of integers.',
    "nominal_duration": "The duration corresponding to a note/rest value without applying any dots or n-tuplets, "
    'as fraction of a whole note. Multiplied with "scalar" to yield the actual duration of the '
    "note/rest.",
    "notes_id": "Row in the notes table.",
    "numbering_offset": "An integer to be added to the measure number count, as specified in MuseScore.",
    "numeral": "The Roman numeral part of a DCML label, lowercase for all chords with a minor third.",
    "octave": "Octave number where 4 is the middle octave (' in Helmholtz notation).",
    "offset:x": "Pertaining to MuseScore encoding.",
    "offset:y": "Pertaining to MuseScore encoding.",
    "offset_x": "Pertaining to MuseScore encoding.",
    "offset_y": "Pertaining to MuseScore encoding.",
    # 'onset':
    "pedal": "Specified as Roman numeral.",
    "phraseend": "The phrase annotation part of a DCML label, can be {, }, or }{. In an older version, the only label "
    "was \\\\",
    "piece": "Name identifier (filename without suffixes) of a piece",
    # 'playthrough':
    "quarterbeats": "Distance from the piece's beginning. By default, only second endings are taken into "
    "account to reflect the proportions of a simply playthrough without repeats.",
    "quarterbeats_playthrough": "Distance from the piece's beginning, including all repeats. This column is present "
    "in unfolded representations that correspond to a full 'playthrough' including all repetitions.",
    "quarterbeats_all_endings": "Distance from the piece's beginning, taking all endings into account for "
    "addressability purposes.",
    "regex_match": "The name of the first registered regular expression matching a label. By default, these include "
    '"dcml" and "form_labels".',
    "relativeroot": "The Roman numeral following the / in applied chords. Can itself include a /. For example, "
    "V/V if the label pertains to the major scale on the second scale degree.",
    "repeats": 'Can be "start", "end", "firstMeasure", "lastMeasure"',
    "rightParen": "Pertaining to MuseScore encoding.",
    "root": "Pertaining to MuseScore encoding.",
    "rootCase": "Pertaining to MuseScore encoding.",
    "scalar": "Decimal value reflecting all dots and n-tuplets modifying the duration of a note/rest. Yields the "
    'actual duration when multiplied with "nominal_duration".',
    "slur": "IDs of active slurs that a chord falls under, as a tuple of integers.",
    "special": 'Column where special labels such as "Fr6" or "Ger65" are stored which are converted internally to '
    "Roman numerals.",
    "staff": "Number of the staff where an event occurs, 1 designating the top staff.",
    "tied": "1 if a note is tied to the following one, -1 if it is being tied to by the previous one, 0 if both.",
    "timesig": 'Given as string, e.g. "4/4".',
    "TimeSig": "All time signatures in a piece and where they occur. Format: '<MC>: <timesig>, <MC>: <timesig>, ...' "
    "Corresponds to a dictionary without outer curly braces.",
    "tpc": "Specified on the line of fifths such that 0 = C, 1 = G, -1 = F, etc.",
    "voice": "A number between 1-4 where 1 is MuseScore's default layer (blue), 2 the second layer in green with "
    "downward stems, etc.",
    # 'voices': 'Voices',
    "volta": "Number of an ending bracket, given as integer.",
    "volta_mcs": 'For each group of alternative endings, a nested list where each inner list contains the "mc" values '
    "of the <Measure> tags grouped into one ending.",
}

TSV_COLUMN_DTYPES = {
    "absolute_base": "Int64",
    "absolute_root": "Int64",
    "all_notes_qb": float,
    "alt_label": "string",
    "barline": "string",
    "base": "Int64",
    "bass_note": "Int64",
    "breaks": "string",
    "cadence": "string",
    "cadences_id": "Int64",
    "changes": "string",
    "chord": "string",
    "chord_id": "Int64",
    "chord_type": "string",
    "color_name": "string",
    "color_html": "string",
    "color_r": "Int64",
    "color_g": "Int64",
    "color_b": "Int64",
    "color_a": "Int64",
    "corpus": str,
    "dont_count": "Int64",
    "duration_qb": float,
    "expanded_id": "Int64",
    "figbass": "string",
    "fname": str,
    "form": "string",
    "form_label_count": int,
    "globalkey": str,
    "gracenote": "string",
    "guitar_chord_count": int,
    "harmonies_id": "Int64",
    "harmony_layer": "Int64",
    "i": int,
    "keysig": "Int64",
    "label": "string",
    "label_count": int,
    "label_type": "string",
    "last_mc": int,
    "last_mc_unfolded": "Int64",
    "last_mn": int,
    "last_mn_unfolded": "Int64",
    "leftParen": "string",
    "length_qb": float,
    "length_qb_unfolded": float,
    "localkey": str,
    "marker": "string",
    "mc": "Int64",
    "mc_playthrough": "Int64",
    "midi": "Int64",
    "mn": str,
    "name": "string",
    "offset:x": "string",
    "offset_x": "string",
    "offset:y": "string",
    "offset_y": "string",
    "n_onsets": int,
    "n_onset_positions": int,
    "nashville": "Int64",
    "notes_id": "Int64",
    "numbering_offset": "Int64",
    "numeral": "string",
    "octave": "Int64",
    "pedal": "string",
    "piece": str,
    "phraseend": "string",
    "playthrough": "Int64",
    "regex_match": "string",
    "relativeroot": "string",
    "repeats": "string",
    "rightParen": "string",
    "root": "Int64",
    "rootCase": "Int64",
    "SID": "Int64",
    "slur": "string",
    "special": "string",
    "staff": "Int64",
    "tied": "Int64",
    "timesig": str,
    "tpc": "Int64",
    "voice": "Int64",
    # 'voices': 'Int64',
    "volta": "Int64",
}


def load_tsv(
    path, index_col=None, sep="\t", converters={}, dtype={}, stringtype=False, **kwargs
) -> Optional[pd.DataFrame]:
    """Loads the TSV file `path` while applying correct type conversion and parsing tuples.

    Parameters
    ----------
    path : :obj:`str`
        Path to a TSV file as output by format_data().
    index_col : :obj:`list`, optional
        By default, the first two columns are loaded as MultiIndex.
        The first level distinguishes pieces and the second level the elements within.
    converters, dtype : :obj:`dict`, optional
        Enhances or overwrites the mapping from column names to types included the constants.
    stringtype : :obj:`bool`, optional
        If you're using pandas >= 1.0.0 you might want to set this to True in order
        to be using the new `string` datatype that includes the new null type `pd.NA`.
    """

    global TSV_COLUMN_CONVERTERS, TSV_COLUMN_DTYPES

    if converters is None:
        conv = None
    else:
        conv = dict(TSV_COLUMN_CONVERTERS)
        conv.update(converters)

    if dtype is None:
        types = None
    elif isinstance(dtype, str):
        types = dtype
    else:
        types = dict(TSV_COLUMN_DTYPES)
        types.update(dtype)

    if stringtype:
        types = {col: "string" if typ == str else typ for col, typ in types.items()}
    try:
        df = pd.read_csv(
            path, sep=sep, index_col=index_col, dtype=types, converters=conv, **kwargs
        )
    except EmptyDataError:
        return
    if "mn" in df:
        mn_volta = mn2int(df.mn)
        df.mn = mn_volta.mn
        if mn_volta.volta.notna().any():
            if "volta" not in df.columns:
                df["volta"] = pd.Series(pd.NA, index=df.index).astype("Int64")
            df.volta.fillna(mn_volta.volta, inplace=True)
    if "interval" in df:
        try:
            iv_index = parse_interval_index_column(df, "interval")
            df.index = iv_index
            df = df.drop(columns="interval")
        except Exception:
            pass
    return df


@cache
def tsv_column2csvw_datatype() -> Dict[str, str | Dict[str, str]]:
    mapping = defaultdict(lambda: "string")
    mapping.update(
        {
            "Int64": "integer",
            str: "string",
            "string": "string",
            float: "float",
            int: "integer",
            int2bool: "boolean",
            value2bool: "boolean",
            safe_frac: {"base": "string", "format": r"-?\d+(?:\/\d+)?"},
            safe_int: "integer",
            str2inttuple: {
                "base": "string",
                "format": r"^[([]?(?:-?\d+\s*,?\s*)*[])]?$",
            },
        }
    )
    column2datatype = {
        col: mapping[dtype] for col, dtype in TSV_COLUMN_CONVERTERS.items()
    }
    column2datatype.update(
        {col: mapping[dtype] for col, dtype in TSV_COLUMN_DTYPES.items()}
    )
    return column2datatype

