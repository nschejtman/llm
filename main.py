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

        columns = 8
        rows = 4

        data_loader = create_dataloader_v1(
            raw_text, batch_size=rows, max_length=columns, stride=1
        )
        data_iter = iter(data_loader)
        print("Total batches:", len(data_iter))
        first_batch = next(data_iter)
        print("Input")
        print(first_batch[0])
        print("Target")
        print(first_batch[1])

        vocab_size = tiktoken.get_encoding("gpt2").n_vocab
        print("Vocab size", vocab_size)
        output_dim = 3
        torch.manual_seed(123)
        embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
        print(embedding_layer.weight)

        print(embedding_layer(first_batch[0]))


if __name__ == "__main__":
    main()
