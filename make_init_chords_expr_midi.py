import mido, os
import chords as CH
from itertools import product

# silberstein groups all sounding notes per step=16th note
# writes one row of csv per step, carries over previous step's chord if no new chord

# same as silberstein
_ticks_per_beat = 480
_velocity = 64
_bpm = 120
_tempo_microsec = mido.bpm2tempo(_bpm)

_middle_c = 60
_c3 = 48
_subdiv = 1
_sustain = 1.0
_dur = 1
_padding = 0

MIDI_DIR = "midi3"
def notedur_to_ticks(dur, subdiv = 1, ticks_per_beat = 480, sustain=1.0):
    on_dur = ticks_per_beat
    if dur != 4:
        on_dur *= 4./dur
    if subdiv != 1:
        on_dur /= subdiv
    off_dur = 0
    if sustain < 1.:
        orig = on_dur
        on_dur = int(min(sustain * orig, orig))
        off_dur = int(max(orig - on_dur, 0))
    else:
        on_dur = int(on_dur)
    return (on_dur, off_dur)

on_dur, off_dur = notedur_to_ticks(_dur, subdiv = _subdiv, ticks_per_beat = _ticks_per_beat, sustain = _sustain)

prog_dict = {}
for mode in ['major', 'minor']:
    chord_dict = CH.major_triad_diatonic if mode == "major" else CH.minor_triad_harmonic
    for cat_prog in CH.chord_cat_prog:
        for inv in [0, 1, 2]:
            
            chord_cats_long = [CH.chord_cat_abbrev[x] for x in cat_prog]
            # get array of chords for each category and index each possibility of chord category (for naming)
            cur_idxed_roman_opts = [[(i,y) for (i,y) in enumerate(CH.chord_cat[x][mode])] for x in chord_cats_long]
            # iterate over all combinations of chords for each category
            for chord_prog in product(*cur_idxed_roman_opts):
                chord_arr = []
                idx_str = '_'.join([str(x[0]) for x in chord_prog])
                cur_romans = [x[1] for x in chord_prog]
                for cur_roman in cur_romans:
                    chord_tup = chord_dict[cur_roman]
                    cur_root = chord_tup[0]
                    cur_bass = _c3 + cur_root
                    cur_notes = CH.triads[chord_tup[1]] 
                    cur_triad = [(_middle_c + cur_root + x) for x in cur_notes]
                    note_arr = []
                    note_arr.append(cur_bass)
                    note_arr += CH.invert_triad(cur_triad, inv)
                    chord_arr.append(note_arr)
                prog_dict[f"{cat_prog}-inv{inv}-{mode}-idx{idx_str}"] = chord_arr


for cur_name, cur_notes_arr in prog_dict.items():
    mid = mido.MidiFile(type=1, ticks_per_beat=_ticks_per_beat)
    mid.tracks.append(mido.MidiTrack())
    mid.tracks[0].append(mido.MetaMessage('set_tempo', tempo = _tempo_microsec))
    # 0 = acoustic grand piano
    mid.tracks[0].append(mido.Message('program_change', program=0, channel=0))
    num_chords = len(cur_notes_arr)
    for i in range(num_chords):
        cur_start = off_dur
        cur_end = on_dur
        if i == 0:
            cur_start = 0
        for midx, _mn in enumerate(cur_notes_arr[i]):
            note_start = cur_start if midx == 0 else 0 
            mid.tracks[0].append(mido.Message('note_on', note=_mn, velocity=_velocity, time=note_start, channel=0))
        for midx, _mn in enumerate(cur_notes_arr[i]):
            note_end = cur_end if midx == 0 else 0 
            mid.tracks[0].append(mido.Message('note_off', note=_mn, velocity=_velocity, time=note_end, channel=0))
        mid.tracks[0].append(mido.MetaMessage('end_of_track', time=0))
    mid.save(os.path.join(MIDI_DIR, f"{cur_name}.mid"))

                


