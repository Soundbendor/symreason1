import os, pickle
import optuna
from pathlib import Path

search_space = {
        'layer_idx': list(range(17)),
        'lr_exp': [-1, -2, -3, -4],
        'batch_size': [1024,2048],
        'wd_exp': [-2,-3,-4]
        }

SAMPLER_FOLDER = os.path.join(Path(__file__).resolve().parent.parent, 'sampler')
RDB_FOLDER = os.path.join(Path(__file__).resolve().parent.parent, 'rdb')
OPT_DIRECTION = 'minimize'

def study_callback(study, trial):
    study_sampler_path = study.user_attrs['sampler_filepath']
    with open(study_sampler_path, 'wb') as f:
        pickle.dump(study.sampler, f)

def create_study_name(parser_args, seed):
    by_cat_int = int(parser_args.by_category)
    return f"bc_{by_cat_int}-tti_{parser_args.triple_token_idx}-s_{seed}-{parser_args.suffix}"

def create_or_load_study(parser_args, seed, evaluation = False):
    ret = {}

    cur_study_name = create_study_name(parser_args, seed)
    sampler_dir = SAMPLER_FOLDER
    rdb_dir = RDB_FOLDER
    sampler_filepath = os.path.join(sampler_dir, f'{cur_study_name}.pkl')
    rdb_filepath = os.path.join(rdb_dir, f'{cur_study_name}.db')
    resuming = False
    cur_sampler = None
    if evaluation == False and os.path.exists(rdb_filepath) == True and os.path.exists(sampler_filepath) == True and parser_args.restart_study == False:
        resuming = True
        cur_sampler = pickle.load(open(sampler_filepath, 'rb'))
    rdb_url = "sqlite:///" + rdb_filepath
    ret['study_name'] = cur_study_name
    ret['sampler_filepath'] = sampler_filepath
    ret['rdb_filepath'] = rdb_filepath
    ret['resuming_study'] = resuming
    ret['study_seed'] = seed

    if cur_sampler == None: 
        cur_sampler = optuna.samplers.GridSampler(search_space, seed=seed)

    if evaluation == False:
        ret['study'] = optuna.create_study(study_name=cur_study_name, sampler = cur_sampler, storage=rdb_url, direction=UC.OPT_DIRECTION, load_if_exists = (resuming == True and parser_args.restart_study == False))
        ret['study'].set_user_attr('sampler_filepath', sampler_filepath)
    return ret

