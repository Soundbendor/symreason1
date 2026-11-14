

triads = {
        'major': [0,4,7],
        'minor': [0,3,7],
        'aug': [0,4,8],
        'dim': [0,3,6]
        }

major_triad_diatonic = {
        'I': (0, 0, 'major'),
        'ii': (2, 1, 'minor'),
        'iii': (4, 2, 'minor'),
        'IV': (5, 3, 'major'),
        'V': (7, 4, 'major'),
        'vi': (9, 5, 'minor'),
        'vii^o': (11, 6, 'dim')
        }


minor_triad_harmonic = {
        'i': (0, 0, 'minor'),
        'ii^o': (2, 1, 'dim'),
        'III': (3, 2, 'major'),
        'iv': (5, 3, 'minor'),
        'V': (7, 4, 'major'),
        'VI': (8, 5, 'major'),
        'vii^o': (11, 6, 'dim')
        }


minor_triad_diatonic = {
        'i': (0, 0,'minor'),
        'ii^o': (2, 1, 'dim'),
        'III': (3, 2, 'major'),
        'iv': (5, 3, 'minor'),
        'v': (7, 4, 'minor'),
        'VI': (8, 5, 'major'),
        'VII': (10, 6, 'major')
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

