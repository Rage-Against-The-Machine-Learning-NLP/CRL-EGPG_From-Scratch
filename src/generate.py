from src.utils import Specials, get_bert_tokenizer, pkl_load, resolve_relpath
from src.parse_config import ModelConfig, load_config_from_json
config = load_config_from_json(resolve_relpath("config.json"))
# config.model_save_dir = "models/default"
import nltk
nltk.download('punkt_tab')
import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from src.dataset import OurDataset
from src.modules.seq2seq import Seq2Seq
from src.modules.style import StyleExtractor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def convert_ids_to_words(idx2word, id_tensor):
    return [idx2word.get(id.item(), '<UNK>') for id in id_tensor if id.item() != 0]

def _get_trained_models(
    config: ModelConfig, device: torch.device
) -> tuple[Seq2Seq, StyleExtractor]:
    seq2seq = Seq2Seq(config=config, device=device)
    seq2seq.load_state_dict(
        torch.load(
            os.path.join(config.model_save_dir, "seq2seq.pkl"),
            weights_only=True,
            map_location=device,
        )
    )
    seq2seq.eval()
    seq2seq.decoder.set_mode("infer")

    style_extractor = StyleExtractor().to(device)
    style_extractor.load_state_dict(
        torch.load(
            os.path.join(config.model_save_dir, "style_extractor.pkl"),
            weights_only=True,
            map_location=device,
        )
    )
    style_extractor.eval()

    return seq2seq, style_extractor

def pack_input(src):
    src_tokennized = [nltk.word_tokenize(line) for line in src]
    src_idx = [[word2idx[word] if word in word2idx else word2idx['UNK'] for word in words] for words in src_tokennized]
    src_idx = src_idx[0]
    in_len = len(src_idx)
    src_out = torch.zeros(15, dtype=torch.long)

    if in_len > 15:
        src_out[0:15] = torch.tensor(src_idx[0:15])
        in_len = 15
    else:
        src_out[0:in_len] = torch.tensor(src_idx)

    src_out = src_out.unsqueeze(0)
    length = torch.tensor([in_len])
    return src_out, length

def pack_sim(sim):
    bert_sim = torch.zeros(15 + 2, dtype=torch.long)
    bert_sim[0:min(len(sim), 15 + 2)] = torch.tensor(sim[0:min(15 + 2, len(sim))])
    return bert_sim

idx2word = pkl_load(os.path.join(config.dataset_dir, "index_to_word.pkl"))
word2idx = pkl_load(os.path.join(config.dataset_dir, "word_to_index.pkl"))
seq2seq, style_extractor = _get_trained_models(config, device)
bert_src = pkl_load(os.path.join(config.dataset_dir, "test_src_bert_ids.pkl"))
bert_src = bert_src[:5]
tokenizer = get_bert_tokenizer()
print("Exempelar sentences: ")
for sent in bert_src:
  words = tokenizer.decode(sent, skip_special_tokens=True)
  print(words)
print("*"*20)
sentences = [
    "how do i develop good project management skills ?",
    "which is the best anime to watch ?",
    "do you want to kiss teddy ?",
    "if you are at the lowest point of your life , what do you do ?",
    "when did you first realize that you were gay lesbian bi ?",
    "he believed his son had died in a terrorist attack .",
    "it is hard for me to imagine where they could be hiding it underground .",
    "i had a strange call from a woman ."
]

def generate_variations(input_sentence):
  print(f"\nOriginal sentence: {input_sentence}")
  print("-" * 50)
  src_idx, in_len = pack_input([input_sentence])
  with torch.no_grad():
          temp_arr = []
          src_idx = src_idx.to(device)
          in_len = in_len.to(device)
          for style in bert_src:
              bert_sim = pack_sim(style)
              bert_sim = bert_sim.unsqueeze(0)
              bert_sim = bert_sim.to(device)
              style_emb = style_extractor(bert_sim)
              id_arr, _ = seq2seq.forward(src_idx, in_len, style_emb)
              temp_arr.append(id_arr)

          # Print variations
          for i, bat in enumerate(temp_arr):
              for sequence in bat:
                  words = convert_ids_to_words(idx2word, sequence)
                  print(f"Variation {i+1}: {' '.join(words)}")
                  break

for sentence in sentences:
    generate_variations(sentence)
