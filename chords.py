

triads = {
        'major': [0,4,7],
        'minor': [0,3,7],
        'aug': [0,4,8],
        'dim': [0,3,6]
        }

major_triad_diatonic = {
        'I': (0, 'major'),
        'ii': (2, 'minor'),
        'iii': (4, 'minor'),
        'IV': (5, 'major'),
        'V': (7, 'major'),
        'vi': (9, 'minor'),
        'vii^o': (11, 'dim')
        }


minor_triad_harmonic = {
        'i': (0, 'minor'),
        'ii^o': (2, 'dim'),
        'III': (3, 'major'),
        'iv': (5, 'minor'),
        'V': (7, 'major'),
        'VI': (8, 'major'),
        'vii^o': (11, 'dim')
        }


minor_triad_diatonic = {
        'i': (0, 'minor'),
        'ii^o': (2, 'dim'),
        'III': (3, 'major'),
        'iv': (5, 'minor'),
        'v': (7, 'minor'),
        'VI': (8, 'major'),
        'VII': (10, 'major')
        }

chord_cat = {
        'tonic': {'major': ["I", "vi"], 'minor': ["i", "VI"]},
        'predom': {'major': ["IV", "ii"], "minor": ["iv", "ii^o"]},
        'dom': {"major": ["V", "vii^o"], "minor": ["V", "vii^o"]}
        }

chord_cat_abbrev = {
        "t": "tonic",
        "p": "predom",
        "d": "dom"
        }

"""
chord_cat_prog = [
        ["t", "p", "d", "t"],
        ["t", "t", "t", "t"],
        ["t", "d", "t", "t"],
        ["t", "t", "d", "t"],
        ["t", "t", "p", "t"],
        ["p", "d", "t", "t"]
        ]
"""

chord_cat_prog = ["tpdt", "tttt", "tdtt", "ttdt", "ttpt", "pdtt"]


def invert_triad(triad, inv):
    if inv == 1:
        return (triad[1:] + [triad[0] + 12])
    elif inv == 2:
        return ([triad[-1]] + [triad[0] + 12, triad[1] + 12])
    else:
        return triad



