
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
        'vii_o': (11, 'dim')
        }


minor_triad_harmonic = {
        'i': (0, 'minor'),
        'ii_o': (2, 'dim'),
        'III': (3, 'major'),
        'iv': (5, 'minor'),
        'V': (7, 'major'),
        'VI': (8, 'major'),
        'vii_o': (11, 'dim')
        }


minor_triad_diatonic = {
        'i': (0, 'minor'),
        'ii_o': (2, 'dim'),
        'III': (3, 'major'),
        'iv': (5, 'minor'),
        'v': (7, 'minor'),
        'VI': (8, 'major'),
        'VII': (10, 'major')
        }

def invert_triad(triad, inv):
    if inv == 1:
        return (triad[1:] + [triad(0) + 12])
    elif inv == 2:
        return (triad[-1] + [triad[0] + 12, triad[1] + 12])
    else:
        return triad



