from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import os
import torch


# 4 qtr note chords x 4 notes/chord = 51 tokens (at 120 bpm, 60 bpm)

MIDI_FOLDER = 'midi'
MIDI_FOLDER2 = 'midi2'
MIDI_FOLDER3 = 'midi3'
midipath = os.path.join(MIDI_FOLDER, 'tpdt-inv0-major-idx0_0_0_0.mid')
midipath2 = os.path.join(MIDI_FOLDER2, 'tpdt-inv0-major-idx0_0_0_0.mid')
midipath3 = os.path.join(MIDI_FOLDER3, 'tpdt-inv0-major-idx0_0_0_0.mid')

model = AutoModelForCausalLM.from_pretrained(
    "loubb/aria-medium-base",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(
    "loubb/aria-medium-base",
    trust_remote_code=True,
)

prompt = tokenizer.encode_from_file(midipath, return_tensors='pt')
prompt2 = tokenizer.encode_from_file(midipath2, return_tensors='pt')
prompt3 = tokenizer.encode_from_file(midipath3, return_tensors='pt')

# returns tensors keyed by 'input_ids' (int) and 'attention_masks' (bool)
#print(prompt['input_ids'].shape, prompt['attention_masks'].shape)
#model.config.output_hidden_states = True

# don't pass attention masks since non_causal?
outputs = model.forward(
        input_ids = prompt['input_ids'],
        output_hidden_states = True,
        output_attentions = True,
        return_dict = True
        )
outputs2 = model.forward(
        input_ids = prompt2['input_ids'],
        output_hidden_states = True,
        output_attentions = True,
        return_dict = True
        )
outputs3 = model.forward(
        input_ids = prompt3['input_ids'],
        output_hidden_states = True,
        output_attentions = True,
        return_dict = True
        )


#print(outputs.keys())

print(prompt['input_ids'].shape, prompt2['input_ids'].shape, prompt3['input_ids'].shape)
print(torch.stack(outputs['hidden_states']).shape, torch.stack(outputs2['hidden_states']).shape, torch.stack(outputs3['hidden_states']).shape)
torch.save(prompt['input_ids'], 'tpdt-inv0-major-idx0_0_0_0-tok1.pt')
torch.save(prompt2['input_ids'], 'tpdt-inv0-major-idx0_0_0_0-tok2.pt')
torch.save(prompt3['input_ids'], 'tpdt-inv0-major-idx0_0_0_0-tok3.pt')
torch.save(torch.stack(outputs['hidden_states']), 'tpdt-inv0-major-idx0_0_0_0-hidden1.pt')
torch.save(torch.stack(outputs2['hidden_states']), 'tpdt-inv0-major-idx0_0_0_0-hidden2.pt')
torch.save(torch.stack(outputs3['hidden_states']), 'tpdt-inv0-major-idx0_0_0_0-hidden3.pt')

#print(type(model))
#print(dir(model.config))
#print(type(tokenizer))
