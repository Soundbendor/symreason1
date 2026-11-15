import polars as pl
import os

fname_keys = ['fname', 'csv_basename', 'bass_note', 'prog', 'inv', 'mode', 'idx']
def get_dataset_df(fdir):
    midifiles = os.listdir(fdir)
    data = {k: [] for k in fname_keys}
    for f in midifiles:
        cur_split = f.split('.')[0].split('-')
        csv_basename = '-'.join(cur_split[1:])
        data['fname'].append(f)
        data['csv_basename'].append(csv_basename)
        data['bass_note'].append(cur_split[0])
        data['prog'].append(cur_split[1])
        data['inv'].append(int(cur_split[2][-1]))
        data['mode'].append(cur_split[3])
        data['idx'].append(cur_split[4][3:])
    return pl.DataFrame(data)

def get_dataset_splits(df, bass_notes,rng):
    cur_bass_notes = CH.bass_note_arr.copy()
    rng.shuffle(cur_bass_notes)
    train_bass = cur_bass_notes[:6]
    valid_bass = cur_bass_notes[6:9]
    test_bass = cur_bass_notes[9:]
    df_train = df.filter(pl.col('bass_note').is_in(train_bass))
    df_valid = df.filter(pl.col('bass_note').is_in(valid_bass))
    df_test = df.filter(pl.col('bass_note').is_in(test_bass))
    return df_train, df_valid, df_test

def get_csv_dfs(df, csvdir):
    d = {}
    for f in df['csv_basename']:
        d[f] = pl.read_csv(os.path.join(csvdir, f'{f}.csv'))
    return d

