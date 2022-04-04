import transformers 
import torch as T

class BoostedModel(transformers.modeling_utils.PreTrainedModel):
    def __init__(self, base_model, k, alpha_long, alpha_short, *args, **kwargs):
        super().__init__(config=base_model.config)
        self.base_lm = base_model
        self.k = k
        self.alpha_long = alpha_long
        self.alpha_short = alpha_short
        
    def forward(self, *args, **kwargs):
        x_long = self.base_lm.forward(*args, **kwargs)

        kwargs['input_ids'] = kwargs['input_ids'][...,-self.k:]
        x_short = self.base_lm.forward(*args, **kwargs)
       
        x_long.logits[...,-1,:] = self.alpha_long*x_long.logits[...,-1,:] - self.alpha_short*x_short.logits[...,-1,:]
        
        return x_long
    
if __name__ == '__main__':

    device = T.device('cuda')

    tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2')
    base_model = transformers.AutoModelForCausalLM.from_pretrained('gpt2-large').to(device)
    base_model.eval()

    boosted_model = BoostedModel(base_model, k=8, alpha_long=1.5, alpha_short=-0.5)

    ins = T.LongTensor([tokenizer.encode('Once upon a midnight dreary,')]).to(device)

    outputs = boosted_model.generate(input_ids=ins, do_sample=True, max_length=100, top_p=0.95)

    print(tokenizer.decode(outputs[0]))
