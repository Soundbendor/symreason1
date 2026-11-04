import mido

mid = mido.MidiFile('K279-1.mid')

for i, track in enumerate(mid.tracks):
    print('Track {}: {}'.format(i, track.name))
    for msg in track:
        print(msg.type)

