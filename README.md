# GPT from Scratch — Little Shakespeare

Final task: implement and train a GPT-style Transformer on the Little Shakespeare dataset.

## Run on Google Colab

1. **Upload the notebook:** In Colab, go to **File → Upload notebook** and choose `gpt_little_shakespeare.ipynb`.
2. **Enable GPU:** **Runtime → Change runtime type → T4 GPU** (optional but faster).
3. **Run all cells:** **Runtime → Run all** (or run cells top to bottom).

The notebook is self-contained: dataset download, model, training, and sampling are in one place. No need to upload `dataset.py`, `model.py`, or `train.py` unless you prefer to run the script version.

## Run locally (optional)

```bash
pip install -r requirements.txt
python train.py
```

Checkpoints are saved every 500 steps as `gpt_shakespeare_ckpt.pt`.

## What this solution covers (evaluation criteria)

- **Understanding of architecture:** Markdown in the notebook explains embeddings, causal self-attention, multi-head attention, FFN, add & norm, decoder stack, and next-token loss.
- **Implementation quality:** Modular classes (`CausalSelfAttention`, `FeedForward`, `DecoderBlock`, `GPT`), clear naming, correct causal masking and loss.
- **Training process:** Character-level data prep, 90/10 split, block size 128; AdamW, warmup + cosine LR, gradient clipping; initial loss at step 0, train/val loss logging every 200 steps; loss plot (train and val curves); checkpointing every 500 steps.
