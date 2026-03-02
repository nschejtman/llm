import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(
    txt,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0,
):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    return dataloader


def main():
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

        batch_size = 8  # amount of phrases per batch
        max_length = 4  # lenght of a phrase

        data_loader = create_dataloader_v1(
            raw_text, batch_size=batch_size, max_length=max_length, stride=1
        )

        data_iter = iter(data_loader)
        inputs, targets = next(data_iter)

        # To embedding space
        vocab_size = tiktoken.get_encoding("gpt2").n_vocab
        output_dim = 256

        token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
        token_embeddings = token_embedding_layer(inputs)

        context_length = max_length
        pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
        pos_embeddings = pos_embedding_layer(torch.arange(context_length))

        input_embeddings = token_embeddings + pos_embeddings


if __name__ == "__main__":
    main()
