from pathlib import Path

import numpy as np
import torch
import clip


def main():
    imagenet21k_vocab_file = Path("imagenet21k_wordnet_lemmas.txt")
    lines = imagenet21k_vocab_file.read_text().splitlines()

    cat_labels = [line.split(",")[0] for line in lines]
    cat_labels = list(set(cat_labels))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device)
    text_inputs = torch.cat([clip.tokenize(f"a {c}") for c in cat_labels]).to(device)
    print(f"{text_inputs.shape = }")
    chunks = torch.split(text_inputs, 100, dim=0)
    chunk_features = []
    for i, chunk in enumerate(chunks):
        chunk_feature = model.encode_text(chunk).detach()
        chunk_feature /= chunk_feature.norm(dim=-1, keepdim=True)
        chunk_feature = chunk_feature.cpu().numpy()
        chunk_features.append(chunk_feature)
    text_features = np.concatenate(chunk_features, axis=0)
    print(f"{text_features.shape = }")

    save_path = "imagenet21k_clip_a+cname.npy"
    np.save(save_path, text_features)


if __name__ == '__main__':
    main()
