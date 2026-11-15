import torch, torch.utils.data as TUD
import numpy as np
from torch import nn

from linear_probe import LinearProbe
from probe_dataset import ProbeDataset

import util.util_data as UD
from functools import partial
from distutils.util import strtobool
import os, sys, time, argparse, copy


SEED = 39
MIDI_FOLDER = os.path.join(Path(__file__).resolve().parent, 'midi')
SHARE_PATH = os.path.join(os.sep, 'nfs','hpc', 'share', 'kwand') 
ACT_PATH = os.path.join(SHARE_PATH, 'symreason1')

def set_torch_seed(seed):
    torch.manual_seed(seed)

def train_model(model, mean, stdev, train_size, generator, opt_fn, loss_fn, train_subset, nstd = False, batch_size=64, shuffle = True, is_classification = True, device='cpu'):
    train_dl = TUD.DataLoader(train_subset, batch_size = batch_size, shuffle=shuffle, generator=generator)
    
    total_loss = 0.


    for batch_idx, data in enumerate(train_dl):
        opt_fn.zero_grad() 
        _ipt, ground_truth = data
        ipt = None
        if nstd == False:
            ipt = (_ipt - mean)/stdev
        else:
            ipt = (_ipt - mean)


        model_pred = model(ipt)

        loss = None
        if is_classification == True:
            loss = loss_fn(model_pred, ground_truth)
        else:
            loss = loss_fn(model_pred.flatten(), ground_truth.flatten())
        
        loss.backward()
        opt_fn.step()
        cur_loss = loss.item()
        total_loss += (cur_loss/train_size)
    return total_loss 

def valid_test_model(model, mean, stdev, generator, loss_fn, valid_subset, nstd = False, batch_size=64, shuffle = True, is_classification = True, device='cpu'):
    valid_dl = TUD.DataLoader(valid_subset, batch_size = batch_size, shuffle=shuffle, generator=generator)
    
    total_loss = 0.
    # for accumulating ground truths and predictions
    truths = None
    preds = None
    
    model.eval()

    with torch.no_grad():
        for batch_idx, data in enumerate(valid_dl):
            
            _ipt, ground_truth = data
            ipt = None
            if  nstd == False:
                ipt = (_ipt - mean)/stdev
            else:
                ipt = (_ipt - mean)



            model_pred = model(ipt)
           
            if loss_fn != None:
                loss = None
                if is_classification == True:
                    loss = loss_fn(model_pred, ground_truth)
                else:
                    loss = loss_fn(model_pred.flatten(), ground_truth.flatten())
                
                cur_loss = loss.item()
                total_loss += cur_loss

            truths, preds = UP.accumulate_truths_preds(truths, ground_truth, preds, model_pred, batch_idx, is_classification)

    return total_loss, truths, preds



def _objective(trial, datadict, subsetdict, configdict, wandbdict, device='cpu'):

    
    trial_number = trial.number

    layer_idx = trial.suggest_categorical('layer_idx', list(range(configdict['model_num_layers'])))
     
   

    # other init
    using_early_stopping =  configdict['early_stopping_check_interval'] > 0
      
    cur_mean = torch.from_numpy(UP.load_mean(configdict, layer_idx)).to(device)
    cur_stdev = torch.from_numpy(UP.load_std(configdict, layer_idx)).to(device)
    # wandbstuff
    cur_run = None
    run_name = None
    short_name = None
    if configdict['use_wandb'] == True:
        param_dict = {'learning_rate_exp': np.log10(configdict['learning_rate']), 'batch_size': configdict['batch_size'], 'data_norm': True, 'layer_idx': layer_idx, 'seed': configdict['seed']}
        run_name, short_name = UO.get_run_and_short_names(configdict, layer_idx, param_dict) 
        cur_run = UW.init(wandbdict, {'id': run_name, 'name': short_name})
        UW.add_to_summary(cur_run, param_dict)

    num_steps = subsetdict['num_preq_steps']


    # now for the actual train/valid loops
    cur_test_subset = subsetdict['test_subset']
    cur_test_subset.dataset.set_layer_idx(layer_idx)
    
    # outer prequential loop
    accum_metrics = []
    valid_nlls = []
    test_nll = -1. 
    train_avg_nlls = []
    actual_training_epochs = []

    preq_all_size = subsetdict['preq_all_size']

    for preq_idx in  range(num_steps+1):

        # init rng
        torch_gen = torch.Generator(device=device)
        torch_gen.manual_seed(configdict['seed'])
        set_seed(configdict['seed'])

        # init model
        model = MLPProbe(in_dim =configdict['model_dim'], out_dim = datadict['num_classes'], hidden_dims = configdict['probe_hidden_dims'])

        # init optimizer and loss
        opt_fn = torch.optim.Adam(model.parameters(), lr=configdict['learning_rate'])

        train_loss_fn = None
        valid_loss_fn = None
        if datadict['is_classification'] == True:
            train_loss_fn = nn.CrossEntropyLoss(reduction='sum')
            valid_loss_fn = nn.CrossEntropyLoss(reduction='sum')
        else:
            train_loss_fn = nn.MSELoss(reduction='sum')
            valid_loss_fn = nn.MSELoss(reduction='sum')



        # setting subsets
        full_train = False
        if preq_idx == num_steps:
            full_train = True
        cur_train_subset = None
        cur_valid_subset = None
        cur_train_size = None
        cur_valid_size = None
        if full_train == False:
            cur_train_subset = subsetdict['preq'][preq_idx]['train_subset']
            cur_valid_subset = subsetdict['preq'][preq_idx]['encode_subset']
            cur_train_size = subsetdict['preq'][preq_idx]['train_size']
            cur_valid_size = subsetdict['preq'][preq_idx]['encode_size']
        else:
            cur_train_subset = subsetdict['preq_all_subset']
            cur_valid_subset = subsetdict['valid_subset']
            cur_train_size = subsetdict['preq_all_size']
            cur_valid_size = subsetdict['valid_size']
        
        cur_train_subset.dataset.set_layer_idx(layer_idx)
        cur_valid_subset.dataset.set_layer_idx(layer_idx)

        train_pct = 100. * (cur_train_size/preq_all_size)
        valid_pct = 100. * (cur_valid_size/preq_all_size)

        print(f'layer/preq ({layer_idx},{preq_idx}): train/valid ({cur_train_size},{cur_valid_size}) ({train_pct:.4f},{valid_pct:.4f})')

        patience = 0
        
        best_loss = float('inf')

        best_model_dict = None

        for epoch_idx in range(configdict['num_epochs']):
            # train/valid
            train_avg_loss = train_model(model, cur_mean, cur_stdev, cur_train_size, torch_gen, opt_fn, train_loss_fn, cur_train_subset, nstd=configdict['nonstandard'], batch_size=configdict['batch_size'], shuffle = configdict['dataloader_shuffle'], is_classification = datadict['is_classification'], device=device)
            valid_loss, valid_truths, valid_preds = valid_test_model(model, cur_mean, cur_stdev, torch_gen, valid_loss_fn, cur_valid_subset, nstd=configdict['nonstandard'], batch_size=configdict['batch_size'], shuffle = configdict['dataloader_shuffle'], is_classification = datadict['is_classification'], device=device)

            train_avg_nlls.append(train_avg_loss)
            # early stopping
            if using_early_stopping == False:
                best_loss = valid_loss
            else:
                last_epoch = epoch_idx == (configdict['num_epochs'] - 1)
                if epoch_idx % configdict['early_stopping_check_interval'] == 0:
                    if valid_loss < np.max([(best_loss - configdict['patience_delta']), 0.]):
                        best_loss = valid_loss
                        patience = 0
                        best_model_dict = copy.deepcopy(model.state_dict())
                    else:
                        patience += 1
                if patience >= configdict['early_stopping_patience'] or last_epoch == True:
                    actual_training_epochs = epoch_idx + 1
                    model.load_state_dict(best_model_dict)
                    break
                elif last_epoch == True:
                    # end of training, just report what you have
                    actual_training_epochs = epoch_idx + 1
                    best_loss = valid_loss
        # training on full training set, should not encode actual validation set
        if full_train == False:
            valid_nlls.append(best_loss)
        else:
            test_nll = best_loss

        test_loss, test_truths, test_preds = valid_test_model(model, cur_mean, cur_stdev, torch_gen, valid_loss_fn, cur_test_subset, nstd = configdict['nonstandard'], batch_size=configdict['batch_size'], shuffle = configdict['dataloader_shuffle'], is_classification = datadict['is_classification'], device=device)
        test_metrics = UME.get_metrics(test_truths, test_preds, valid_nlls, preq_idx, layer_idx, trial_number, datadict, subsetdict, configdict, save_to_csv = True, make_cm = True)
        accum_metrics.append(test_metrics)

    # bookkeeping

    metric_dict = UME.accum_metrics_to_metric_dict(accum_metrics)
    for k,v in metric_dict.items():
        trial.set_user_attr(key=k, value=v)
    
    final_mdl = metric_dict['online_mdl'][-1]
    trial.set_user_attr(key='valid_nlls', value = valid_nlls)
    trial.set_user_attr(key='test_nll', value = test_nll)
    trial.set_user_attr(key='final_mdl', value = final_mdl)
    trial.set_user_attr(key='train_avg_nlls', value = train_avg_nlls)
    trial.set_user_attr(key='actual_training_epochs', value = actual_training_epochs)
    
    # naming
    trial.set_user_attr(key='run_name', value=run_name)
    trial.set_user_attr(key='short_name', value=short_name)
    # wandb stuff
    if configdict['use_wandb'] == True:
        UW.log_array(cur_run, 'train_avg_nll', train_avg_nlls)
        UW.finish_run(cur_run)
    return final_mdl


if __name__ == "__main__":
    #### arg parsing
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-ds", "--dataset", type=str, default="polyrhythms", help="dataset")

    args = parser.parse_args()

    #### some initialization
    device = 'cpu'
    if args.use_cuda == True and torch.cuda.is_available() == True:
        device = 'cuda'
        torch.cuda.empty_cache()
        torch.set_default_device(device)
        set_torch_seed(SEED)
    
    np_rng = np.random.default_rng(SEED)
    data_df = UD.get_dataset_df(ACT_PATH)
    df_train, df_valid, df_test = UD.get_dataset_splits(data_df, CH.bass_note_arr,rng)
    # do splits by key with 6/3/3
        
