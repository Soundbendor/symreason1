import mido, os
import chords as CH


# same as silberstein
_ticks_per_beat = 480
_velocity = 64
_bpm = 120
_tempo_microsec = mido.bpm2tempo(_bpm)

_middle_c = 60
_c3 = 48
_subdiv = 1
_sustain = 1.0
_dur = 4.
_padding = 0

MIDI_DIR = "midi"
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
    for prog in CH.chord_cat_prog:
        for short_chord_cat in prog:
            chord_cat = CH.chord_cat_abbrev[short_chord_cat]
            chord_arr = []
            for cur_roman in CH.chord_cat[chord_cat][mode]:
                chord_tup = chord_dict[cur_roman]
                cur_bass = _c3 + chord_tup[0]
                cur_notes = CH.triads[chord_tup[1]] 
                cur_triad = [(_middle_c + x) for x in cur_notes]
                for inv in [0, 1]:
                    note_arr = []
                    note_arr.append(cur_bass)
                    note_arr += CH.invert_triad(cur_triad, inv)
                    chord_arr.append(note_arr)
                    prog_dict[f"{prog}-inv{inv}-{mode}"] = chord_arr


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

                


