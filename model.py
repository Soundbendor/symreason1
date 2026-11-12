from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import os
import torch


# 4 qtr note chords x 4 notes/chord = 51 tokens (at 120 bpm, 60 bpm)

MIDI_FOLDER = 'midi'
MIDI_FOLDER2 = 'midi2'
MIDI_FOLDER3 = 'midi3'

FOLDERS = [MIDI_FOLDER, MIDI_FOLDER2, MIDI_FOLDER3]
model = AutoModelForCausalLM.from_pretrained(
    "loubb/aria-medium-base",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(
    "loubb/aria-medium-base",
    trust_remote_code=True,
)


# returns tensors keyed by 'input_ids' (int) and 'attention_masks' (bool)
#print(prompt['input_ids'].shape, prompt['attention_masks'].shape)
#model.config.output_hidden_states = True

prev = None
first_time = True
for mfile in os.listdir(MIDI_FOLDER):
    for fol in FOLDERS:
        midipath = os.path.join(fol, mfile)
        prompt = tokenizer.encode_from_file(midipath, return_tensors='pt')
        # don't pass attention masks since non_causal?
        """
        outputs = model.forward(
                input_ids = prompt['input_ids'],
                output_hidden_states = True,
                output_attentions = True,
                return_dict = True
                )
        """
        cur_ids = prompt['input_ids'].flatten()
        if first_time == True:
            prev = cur_ids.clone()
            first_time = False
        else:
            cur_isin = torch.isin(prev, cur_ids)
            prev = prev[cur_isin].unique()
        #print(cur_ids[:3], cur_ids[-3:])
        print(fol, '0', torch.where(cur_ids == 0)[0])
        print(fol, '17227', torch.where(cur_ids == 17227)[0])
        print(fol, '17327', torch.where(cur_ids == 17327)[0])
#print(outputs.keys())

print(prev)
"""
for k,v in tokenizer.get_vocab().items():
    print(f"{k[0]},{k[1]},{v}")
"""
#print(prompt['input_ids'].shape, prompt2['input_ids'].shape, prompt3['input_ids'].shape)
#print(torch.stack(outputs['hidden_states']).shape, torch.stack(outputs2['hidden_states']).shape, torch.stack(outputs3['hidden_states']).shape)
#print(type(model))
#print(dir(model.config))
#print(type(tokenizer))
