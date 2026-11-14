import mido, os
import chords as CH
from itertools import product
import musicnoteconv as MNC
import polars as pl

# silberstein groups all sounding notes per step=16th note
# writes one row of csv per step, carries over previous step's chord if no new chord

# same as silberstein
_ticks_per_beat = 480
_velocity = 64
_bpm = 120
_tempo_microsec = mido.bpm2tempo(_bpm)

_subdiv = 1
_sustain = 1.0
_dur = 4
_padding = 0

MIDI_DIR = "midi"
CSV_DIR = "midi_csv"

bass_note_arr = ['c3', 'cs3', 'd3', 'ds3', 'e3', 'f3', 'fs3', 'g3', 'gs3', 'a3', 'as3', 'b3']
data_keys = ['chord_roman', 'chord_cat', 'chord_inv', 'chord_scaledeg', 'note_num_total', 'note_num_chord']
prog_dict = {}

write_csv = True

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


for mode in ['major', 'minor']:
    chord_dict = CH.major_triad_diatonic if mode == "major" else CH.minor_triad_harmonic
    for bidx, bass_note in enumerate(bass_note_arr):
        bass_midi = MNC.note_to_midi(bass_note)
        root_midi = bass_midi + 12
        for cat_prog in CH.chord_cat_prog:
            # not technically inv because bass note is root
            for inv in [0, 1, 2]:
                chord_cats_long = [CH.chord_cat_abbrev[x] for x in cat_prog]
                # get array of chords for each category and index each possibility of chord category (for naming)
                cur_idxed_roman_opts = [[(i,y) for (i,y) in enumerate(CH.chord_cat[x][mode])] for x in chord_cats_long]

                # for csv making
                data_dict = {k: [] for k in data_keys}
                note_num_total = 0
                # iterate over all combinations of chords for each category
                for chord_prog in product(*cur_idxed_roman_opts):
                    chord_arr = []
                    idx_str = '_'.join([str(x[0]) for x in chord_prog])
                    cur_romans = [x[1] for x in chord_prog]
                    for ridx, cur_roman in enumerate(cur_romans):
                        chord_tup = chord_dict[cur_roman]
                        cur_root = chord_tup[0]
                        cur_scaledeg = chord_tup[1]
                        cur_bass =  bass_midi + cur_root
                        cur_notes = CH.triads[chord_tup[-1]] 
                        cur_triad = [(root_midi + cur_root + x) for x in cur_notes]
                        note_arr = []
                        note_arr.append(cur_bass)
                        note_arr += CH.invert_triad(cur_triad, inv)
                        chord_arr.append(note_arr)
                        # append to data_dict
                        num_notes = len(note_arr)
                        data_dict['note_num_chord'] += list(range(num_notes))
                        data_dict['note_num_total'] += [(note_num_total + x) for x in range(num_notes)]
                        data_dict['chord_roman'] += ([cur_roman] * num_notes)
                        data_dict['chord_cat'] += ([cat_prog[ridx]] * num_notes)
                        data_dict['chord_inv'] += ([inv] * num_notes)
                        data_dict['chord_scaledeg'] += ([cur_scaledeg] * num_notes)

                        # bookkeeping
                        note_num_total += num_notes
                    prog_str = f"{cat_prog}-inv{inv}-{mode}-idx{idx_str}"
                    if bidx == 0 and write_csv == True:
                        cur_df = pl.DataFrame(data_dict)
                        cur_df.write_csv(os.path.join(CSV_DIR, f'{prog_str}.csv'))
                    prog_dict[f"{bass_note}-{prog_str}"] = chord_arr


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

                


