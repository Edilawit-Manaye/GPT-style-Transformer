"""
Little Shakespeare dataset: character-level tokenization and chunked sequences.
Matches the spec: block_size=128, 90/10 train/val split, target = input shifted by 1.
"""
import torch
from torch.utils.data import Dataset


# Character vocabulary: all unique chars in the data (typically ~65 for Shakespeare)
def get_vocab(text: str):
    """Build character-level vocab (stoi, itos)."""
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


class ShakespeareDataset(Dataset):
    """
    Chunked character-level dataset.
    Each item: (input block of length block_size, target block = input shifted by 1).
    """

    def __init__(self, text: str, block_size: int, stoi: dict):
        self.block_size = block_size
        self.stoi = stoi
        # Encode entire text to token indices
        self.data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

    def __len__(self):
        # Number of non-overlapping blocks (or overlapping: len - block_size)
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y


def load_and_split(
    url: str = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    block_size: int = 128,
    train_frac: float = 0.9,
):
    """Download text, build vocab, create train/val datasets."""
    import urllib.request
    with urllib.request.urlopen(url) as f:
        text = f.read().decode("utf-8")
    stoi, itos = get_vocab(text)
    n = len(text)
    split_idx = int(n * train_frac)
    train_text, val_text = text[:split_idx], text[split_idx:]
    train_ds = ShakespeareDataset(train_text, block_size, stoi)
    val_ds = ShakespeareDataset(val_text, block_size, stoi)
    return train_ds, val_ds, stoi, itos
