import torch
import torch.nn as nn
from transformers import BertModel, AlbertModel, RobertaModel


class StyleExtractor(nn.Module):
   def __init__(self, model_type='bert'):
       """
       model_type: one of 'bert', 'albert', 'roberta'
       """
       super().__init__()

       if model_type == 'bert':
           self.model = BertModel.from_pretrained("bert-base-uncased", output_hidden_states=True)
       elif model_type == 'albert':
           self.model = AlbertModel.from_pretrained("albert-base-v2", output_hidden_states=True)
       elif model_type == 'roberta':
           self.model = RobertaModel.from_pretrained("roberta-base", output_hidden_states=True)
       else:
           raise ValueError("model_type must be one of: bert, albert, roberta")

       self.model_type = model_type

   def forward(self, input):
       attention_mask = (input != 0).float()
       outputs = self.model(input, attention_mask=attention_mask)
       # Stack hidden states - note that for ALBERT architecture,
       # all layers share parameters so hidden states may be less meaningful
       hidden_states = torch.stack(outputs[2], dim=1)

       first_hidden_states = hidden_states[:,:,0,:]

       return first_hidden_states
