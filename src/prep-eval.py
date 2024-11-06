import os

import json
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from src.dataset import OurDataset
from src.modules.seq2seq import Seq2Seq
from src.modules.style import StyleExtractor
from src.parse_config import ModelConfig, load_config_from_json
from src.utils import Specials, get_bert_tokenizer, pkl_load, resolve_relpath


def evaluate(config: ModelConfig) -> None:
    test_set = OurDataset(config.dataset_dir, 15, "test")
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seq2seq = Seq2Seq(config, device=device).to(device)
    seq2seq.load_state_dict(
        torch.load(
            os.path.join(config.model_save_dir, "seq2seq.pkl"),
            weights_only=True,
        )
    )
    seq2seq.eval()

    style_extractor = StyleExtractor().to(device)
    style_extractor.load_state_dict(
        torch.load(
            os.path.join(config.model_save_dir, "style_extractor.pkl"),
            weights_only=True,
        )
    )
    style_extractor.eval()

    seq2seq.decoder.set_mode("infer")

    sim = pkl_load(os.path.join(config.dataset_dir, "test_similarity.pkl"))
    idx2word = pkl_load(os.path.join(config.dataset_dir, "index_to_word.pkl"))
    bert_opt = pkl_load(os.path.join(config.dataset_dir, "test_trg_bert_ids.pkl"))
    normal_opt = pkl_load(os.path.join(config.dataset_dir, "test_trg.pkl"))

    exemplar_list = []
    generated_list = []

    with torch.no_grad():
        for idx, (
            src,
            src_len,
            content_trg,
            content_len,
            trg,
            trg_input,
            bert_src,
            bert_trg,
            bert_exmp,
        ) in enumerate(tqdm(test_loader, desc=f"testing model")):
            src: torch.Tensor
            src_len: torch.Tensor
            content_trg: torch.Tensor
            content_len: torch.Tensor
            trg: torch.Tensor
            trg_input: torch.Tensor
            bert_src: torch.Tensor
            bert_trg: torch.Tensor
            bert_exmp: torch.Tensor

            (
                src,
                src_len,
                content_trg,
                content_len,
                trg,
                trg_input,
                bert_src,
                bert_trg,
                bert_exmp,
            ) = (
                src.to(device),
                src_len.to(device),
                content_trg.to(device),
                content_len.to(device),
                trg.to(device),
                trg_input.to(device),
                bert_src.to(device),
                bert_trg.to(device),
                bert_exmp.to(device),
            )

            predicted_targets = []
            for similar_sent_idx in sim[idx]:
                bert_sent_opt = bert_opt[similar_sent_idx]
                SENT_LEN = min(config.training.max_sent_len + 2, len(bert_sent_opt))

                bert_sim_sent = torch.zeros(
                    config.training.max_sent_len + 2, dtype=torch.long
                )
                bert_sim_sent[:SENT_LEN] = torch.tensor(bert_sent_opt[:SENT_LEN])
                bert_sim_sent = bert_sim_sent.unsqueeze(0).to(device)

                style_embedding = style_extractor(bert_sim_sent)
                predicted_trg, _ = seq2seq.forward(
                    src, src_len, style_embedding, decoder_input=trg_input
                )
                predicted_targets.append(predicted_trg)

            def extract_coverage(items, trg_opt) -> float:
                _, elements = items  # the first item is the index from enumerate
                intersection_size = len(
                    set(trg_opt) & {item.item() for item in elements[0]}
                )
                return intersection_size / len(trg_opt)

            max_coverage_ind, _ = max(
                enumerate(predicted_targets),
                key=lambda x: extract_coverage(x, normal_opt[idx]),
            )

            exemplar_list.append(sim[idx][max_coverage_ind])
            generated_list.append(predicted_targets[max_coverage_ind])

    generated_file = os.path.join(config.results_dir, "trg_gen.txt")
    if os.path.exists(generated_file):
        os.remove(generated_file)

    with open(generated_file, "a") as f:
        for bat in generated_list:
            for sentence_id in bat:
                words = [
                    idx2word.get(id.item(), Specials.UNK)
                    for id in sentence_id
                    if id.item() != 0
                ]
                f.write(" ".join(words) + "\n")

    exemplar_file = os.path.join(config.results_dir, "trg_exmp.txt")
    if os.path.exists(exemplar_file):
        os.remove(exemplar_file)
    tokenizer = get_bert_tokenizer()
    with open(exemplar_file, "a") as f:
        for sentence_id in exemplar_list:
            words = tokenizer.decode(bert_opt[sentence_id][1:-2])
            f.write(words + "\n")


if __name__ == "__main__":
    config = load_config_from_json(resolve_relpath("config.json"))
    evaluate(config)
