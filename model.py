from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer


model = AutoModelForCausalLM.from_pretrained(
    "loubb/aria-medium-base",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(
    "loubb/aria-medium-base",
    trust_remote_code=True,
)

model.config.output_hidden_states = True
print(type(model))
print(dir(model.config))
print(type(tokenizer))
