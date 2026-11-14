from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import os
import torch


# 4 qtr note chords x 4 notes/chord = 51 tokens (at 120 bpm, 60 bpm)

MIDI_FOLDER = 'midi'
MIDI_FOLDER2 = 'midi2'
MIDI_FOLDER3 = 'midi3'

FOLDERS = [MIDI_FOLDER, MIDI_FOLDER2, MIDI_FOLDER3]
FOLDERS = [MIDI_FOLDER3]
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

prev = None
first_time = True
for mfile in os.listdir(MIDI_FOLDER):
    for fol in FOLDERS:
        midipath = os.path.join(fol, mfile)
        prompt = tokenizer.encode_from_file(midipath, return_tensors='pt')
        # don't pass attention masks since non_causal?
        outputs = model.forward(
                input_ids = prompt['input_ids'],
                output_hidden_states = True,
                output_attentions = True,
                return_dict = True
                )
        cur_ids = prompt['input_ids'].flatten()
        """
        if first_time == True and fol != MIDI_FOLDER:
            prev = cur_ids.clone()
            first_time = False
        elif fol != MIDI_FOLDER:
            cur_isin = torch.isin(prev, cur_ids)
            prev = prev[cur_isin].unique()
        """
        if first_time == True:
            cur = [(x.item(), vocab_dict[x.item()]) for x in cur_ids]
            print(cur)
            first_time = False
            break

        # output is of size (num_layers, batch_size, seqlen, dimension)
        print(torch.stack(outputs['hidden_states']).shape)
        #print(cur_ids[:3], cur_ids[-3:])
        #print(fol, '0', torch.where(cur_ids == 0)[0])
        #print(fol, '17227', torch.where(cur_ids == 17227)[0])
        #print(fol, '17327', torch.where(cur_ids == 17327)[0])
        #print(fol, '5', torch.where(cur_ids == 5)[0])
#print(outputs.keys())

#print(prev)
"""
print('bos_token', tokenizer.bos_token)
print('eos_token', tokenizer.eos_token)
print('unk_token', tokenizer.unk_token)
print('sep_token',tokenizer.sep_token)
print('pad_token', tokenizer.pad_token)
print('cls_token',tokenizer.cls_token)
print('mask_token', tokenizer.mask_token)
print('extra_special_tokens',tokenizer.extra_special_tokens)
print('split_special_tokens',tokenizer.split_special_tokens)
"""
"""
for k,v in tokenizer.get_vocab().items():
    print(f"{k[0]},{k[1]},{v}")
"""
#print(prompt['input_ids'].shape, prompt2['input_ids'].shape, prompt3['input_ids'].shape)
#print(torch.stack(outputs['hidden_states']).shape, torch.stack(outputs2['hidden_states']).shape, torch.stack(outputs3['hidden_states']).shape)
#print(type(model))
#print(dir(model.config))
#print(type(tokenizer))
