import torch, torch.utils.data as TUD
import numpy as np
from torch import nn
import optuna

from linear_probe import LinearProbe, init_probe
from probe_dataset import ProbeDataset

import util.util_data as UD
import util.util_probing as UP
import util.util_metrics as UME
import util.util_optuna as UO
import util.util_rdb as URDB
import util.util_chords as CH
from functools import partial
from distutils.util import strtobool
from pathlib import Path
import os, sys, time, argparse, copy


SEED = 39
BATCH_SIZE = 512
MIDI_FOLDER = os.path.join(Path(__file__).resolve().parent, 'midi')
MIDI_CSV_FOLDER = os.path.join(Path(__file__).resolve().parent, 'midi_csv')
GRAPH_FOLDER = os.path.join(Path(__file__).resolve().parent, 'graph')
SHARE_PATH = os.path.join(os.sep, 'nfs','hpc', 'share', 'kwand') 
ACT_FOLDER = os.path.join(SHARE_PATH, 'symreason1_acts')
MODEL_FOLDER = os.path.join(SHARE_PATH, 'symreason1_model')

def set_torch_seed(seed):
    torch.manual_seed(seed)

def shuffle_input_notes(ipt, ground_truth, generator):
    bs = ipt.shape[0]
    seqlen = ipt.shape[1]
    _r = torch.randperm(bs * seqlen, generator = generator)
    return (ipt[_r % bs, _r // bs], ground_truth[_r % bs, _r // bs])

def train_model(model, train_size, generator, opt_fn, loss_fn, train_subset, nstd = False, batch_size=64, shuffle = True, is_classification = True, device='cpu'):
    train_dl = TUD.DataLoader(train_subset, batch_size = batch_size, shuffle=shuffle, generator=generator)
    
    total_loss = 0.


    for batch_idx, data in enumerate(train_dl):
        opt_fn.zero_grad() 
        _ipt, _ground_truth = data
        # should be (batch_size, num_reduced_tokens, new_dim)
        # (batch_size, num_reduced_tokens) 
        
        # shuffle notes amongst batch
        ipt, ground_truth = shuffle_input_notes( _ipt, _ground_truth, generator)

        model_pred = model(ipt)

        loss = loss_fn(model_pred, ground_truth)
        
        loss.backward()
        opt_fn.step()
        cur_loss = loss.item()
        total_loss += (cur_loss/train_size)
    return total_loss 

def valid_test_model(model, valid_size, generator, loss_fn, valid_subset, batch_size=64, shuffle = True, device='cpu'):
    valid_dl = TUD.DataLoader(valid_subset, batch_size = batch_size, shuffle=shuffle, generator=generator)
    
    total_loss = 0.
    # for accumulating ground truths and predictions
    truths = None
    preds = None
    
    model.eval()

    with torch.no_grad():
        for batch_idx, data in enumerate(valid_dl):
            
            _ipt, _ground_truth = data

            # shuffle notes amongst batch
            ipt, ground_truth = shuffle_input_notes( _ipt, _ground_truth, generator)
            
            model_pred = model(ipt)
           
            if loss_fn != None:
                loss = loss_fn(model_pred, ground_truth)
                
                cur_loss = loss.item()
                total_loss += (cur_loss/valid_size)

            truths, preds = UP.accumulate_truths_preds(truths, ground_truth, preds, model_pred, batch_idx)

    return total_loss, truths, preds



def _objective(trial, train_ds = None, valid_ds = None, triple_token_idx=0, by_category = False, suffix=1, device='cpu'):


    params = {
            'early_stopping_check_interval': 1,
            'early_stopping_patience': 25,
            'num_epochs': 250,
            'batch_size': BATCH_SIZE,
            'patience_delta': 10**(-5),
            'train_size': len(train_ds),
            'valid_size': len(valid_ds),
            'triple_token_idx': triple_token_idx,
            'by_category': by_category,
            'trial_number': trial.number
            }

    layer_idx = trial.suggest_categorical('layer_idx', list(range(17)))
    lr_exp = trial.suggest_int('lr_exp', -3, -2, step=1)
    wd_exp = trial.suggest_int('wd_exp', -3,-2,step=1)
   
    print(f'trial {trial.number}: layer {layer_idx}, lr_exp {lr_exp}, wd_exp {wd_exp}')
    # other init
    using_early_stopping =  params['early_stopping_check_interval'] > 0

    train_ds.set_layer_idx(layer_idx)
    valid_ds.set_layer_idx(layer_idx)

    model = init_probe(by_category, triple_token_idx, [])

    opt_fn = None
    if wd_exp < -2:
        opt_fn = torch.optim.Adam(model.parameters(), lr=10**(lr_exp), weight_decay = 10**(wd_exp)) 
    else:
        opt_fn = torch.optim.Adam(model.parameters(), lr=10**(lr_exp)) 

    
    # outer prequential loop
    valid_nlls = []
    train_avg_nlls = []
    actual_training_epochs = float('inf')

    
    # init rng
    torch_gen = torch.Generator(device=device)
    torch_gen.manual_seed(SEED)
    set_torch_seed(SEED)

    train_loss_fn = nn.CrossEntropyLoss(reduction='sum')
    valid_loss_fn = nn.CrossEntropyLoss(reduction='sum')
   

    patience = 0
        
    best_loss = float('inf')

    best_model_dict = None

    for epoch_idx in range(params['num_epochs']):
        # train/valid
        train_avg_loss = train_model(model, params['train_size'], torch_gen, opt_fn, train_loss_fn, train_ds, batch_size=params['batch_size'], shuffle = True, device=device)
        valid_avg_loss, valid_truths, valid_preds = valid_test_model(model, params['valid_size'], torch_gen, valid_loss_fn, valid_ds, batch_size=params['batch_size'], shuffle = True, device=device)
        valid_nlls.append(valid_avg_loss)
        train_avg_nlls.append(train_avg_loss)
        print(f'trial {trial.number}, epoch {epoch_idx}: train {train_avg_loss}, valid {valid_avg_loss}')
        trial.report(valid_avg_loss, step = epoch_idx)
        # early stopping
        if using_early_stopping == False:
            best_loss = valid_avg_loss
        else:
            last_epoch = epoch_idx == (params['num_epochs'] - 1)
            if epoch_idx % params['early_stopping_check_interval'] == 0:
                if valid_avg_loss < max(best_loss - params['patience_delta'], 0.):
                    best_loss = valid_avg_loss
                    patience = 0
                    best_model_dict = copy.deepcopy(model.state_dict())
                else:
                    patience += 1
            if patience >= params['early_stopping_patience'] or last_epoch == True:
                actual_training_epochs = epoch_idx + 1
                model.load_state_dict(best_model_dict)
                break
            elif last_epoch == True:
                # end of training, just report what you have
                actual_training_epochs = epoch_idx + 1
                best_loss = valid_avg_loss
                best_model_dict = copy.deepcopy(model.state_dict())


    # bookkeeping

    by_cat_int = int(params["by_category"])
    params['best_loss'] = best_loss
    params['actual_training_epochs'] = actual_training_epochs
    params['train_avg_nlls'] = train_avg_nlls
    params['model_savename'] = f"l_{layer_idx}-lrexp_{lr_exp}-wdexp_{wd_exp}-bs_{params['batch_size']}-tti_{triple_token_idx}-bc_{by_cat_int}-trial_{trial.number}-{suffix}"
    for k,v in params.items():
        trial.set_user_attr(key=k, value=v)
 
    torch.save(best_model_dict, os.path.join(MODEL_FOLDER, f"{params['model_savename']}.pt"))
    return best_loss


if __name__ == "__main__":
    #### arg parsing
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-sf", "--suffix", type=int, default=1, help="suffix")
    parser.add_argument("-rs", "--restart_study", type=strtobool, default=False, help="force restart of optuna study")
    parser.add_argument("-bc", "--by_category", type=strtobool, default=False, help="by category")
    parser.add_argument("-cu", "--use_cuda", type=strtobool, default=True, help="use cuda")
    parser.add_argument("-ev", "--eval", type=strtobool, default=False, help="eval")
    parser.add_argument("-sg", "--study_graph", type=strtobool, default=False, help="graph study")
    parser.add_argument("-db", "--debug", type=strtobool, default=False, help="debug")
    parser.add_argument("-tti", "--triple_token_idx", type=int, default=-1, help="triple token index")
    args = parser.parse_args()

    #### some initialization
    device = 'cpu'
    if args.use_cuda == True and torch.cuda.is_available() == True:
        device = 'cuda'
        torch.cuda.empty_cache()
        torch.set_default_device(device)
        set_torch_seed(SEED)
    
    np_rng = np.random.default_rng(SEED)

    if args.debug == True:
        exit()
    data_df = UD.get_dataset_df(ACT_FOLDER)
    # do splits by key with 6/3/3
    train_df, valid_df, test_df = UD.get_dataset_splits(data_df, CH.bass_note_arr, np_rng)

    train_csvs = UD.get_csv_dfs(train_df, MIDI_CSV_FOLDER)
    valid_csvs = UD.get_csv_dfs(valid_df, MIDI_CSV_FOLDER)
    train_ds = ProbeDataset(data_df=train_df, csv_dfs=train_csvs, layer_idx=0, from_dir = ACT_FOLDER , triple_token_idx = args.triple_token_idx, by_category = args.by_category, device=device)
    valid_ds = ProbeDataset(data_df=valid_df, csv_dfs=valid_csvs, layer_idx=0, from_dir = ACT_FOLDER , triple_token_idx = args.triple_token_idx, by_category = args.by_category, device=device)

    # optuna stuff
    study_dict = UO.create_or_load_study(args, SEED, evaluation = args.eval)
    if args.eval == False and args.study_graph == False:
        objective = partial(_objective,  train_ds = train_ds, valid_ds = valid_ds, triple_token_idx=args.triple_token_idx, by_category = args.by_category, suffix=args.suffix, device=device)
        callback_arr = [UO.study_callback]
        study_dict['study'].optimize(objective, timeout = None, n_trials = None, n_jobs=1, gc_after_trial = True, callbacks=callback_arr)
        cur_studyname = study_dict['study_name']
    if args.study_graph == True:
        # run if eval if false
        fig = optuna.visualization.plot_intermediate_values(study_dict['study'])
        fig.write_image(os.path.join(GRAPH_FOLDER, f"{cur_studyname}.webp"))
    if args.eval == True:
        for layer_idx in list(range(17)):
            param_dict, best_dict, study_attr_dict, trial_attr_dict = URDB.get_best_params_of_layer_idx(study_dict, layer_idx)
            eval_dict = URDB.make_eval_param_dict(param_dict, best_dict)
            model_savename = trial_attr_dict['model_savename']
            model = init_probe(trial_attr_dict['by_category'], trial_attr_dict['triple_token_idx'], []) 
            model.load_state_dict(torch.load(
                os.path.join(MODEL_FOLDER, f'{model_savename}.pt'),
                map_location=device,
                weights_only = False))
            # init rng
            torch_gen = torch.Generator(device=device)
            torch_gen.manual_seed(SEED)
            set_torch_seed(SEED)

            test_loss_fn = nn.CrossEntropyLoss(reduction='sum')

            test_csvs = UD.get_csv_dfs(test_df, MIDI_CSV_FOLDER)
            test_ds = ProbeDataset(data_df=test_df, csv_dfs=test_csvs, layer_idx=layer_idx, from_dir = ACT_FOLDER , triple_token_idx = args.triple_token_idx, by_category = args.by_category, device=device)
            
            test_avg_loss, test_truths, test_preds = valid_test_model(model, len(test_ds), torch_gen, test_loss_fn, test_ds, batch_size=BATCH_SIZE, shuffle = True, device=device)
            param_str = f"Layer {layer_idx}, BS = {BATCH_SIZE}, LR_EXP = {param_dict['lr_exp']}, WD_EXP = {param_dict['wd_exp']}"
            res = UME.get_classification_metrics(test_truths, test_preds, layer_idx = layer_idx, by_category = trial_attr_dict['by_category'], param_str = param_str, save_name = model_savename, make_cm = True)
