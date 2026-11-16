from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import os
import torch
from pathlib import Path

# 4 qtr note chords x 4 notes/chord = 51 tokens (at 120 bpm, 60 bpm)

MIDI_FOLDER = os.path.join(Path(__file__).resolve().parent, 'midi')
SHARE_PATH = os.path.join(os.sep, 'nfs','hpc', 'share', 'kwand') 
ACT_PATH = os.path.join(SHARE_PATH, 'symreason1')

model = AutoModelForCausalLM.from_pretrained(
    "loubb/aria-medium-base",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(
    "loubb/aria-medium-base",
    trust_remote_code=True,
)

vocab_dict = {int(v):k for (k,v) in tokenizer.get_vocab().items()}
#print(vocab_dict)
# returns tensors keyed by 'input_ids' (int) and 'attention_masks' (bool)
#print(prompt['input_ids'].shape, prompt['attention_masks'].shape)
#model.config.output_hidden_states = True

for mfile in os.listdir(MIDI_FOLDER):
    cur_name = mfile.split('.')[0]
    midipath = os.path.join(MIDI_FOLDER, mfile)
    prompt = tokenizer.encode_from_file(midipath, return_tensors='pt')
    outputs = model.forward(
            input_ids = prompt['input_ids'],
            output_hidden_states = True,
            output_attentions = True,
            return_dict = True
            )
    # first dim is batch_size and we're doing one at a time
    cur_ids = prompt['input_ids'].squeeze(0)
    
    # dimension (layers, batch_size, seqlen, dim)
    # so squeeze first dim to get (layers, seqlen, dim)
    cur_hidden = torch.stack(outputs['hidden_states']).squeeze(1)
    torch.save(cur_ids, os.path.join(ACT_PATH, f'{cur_name}-tokens.pt'))
    torch.save(cur_hidden, os.path.join(ACT_PATH, f'{cur_name}-hidden.pt'))

