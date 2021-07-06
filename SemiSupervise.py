import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset
from tqdm.auto import tqdm

def get_pseudo_labels(dataset, model, batch_size=64, device="cuda"):
    loader = DataLoader(dataset, batch_size=batch_size *
                        3, shuffle=False, pin_memory=True)
    pseudo_labels = []
    for batch in tqdm(loader):
        img, _ = batch
        with torch.no_grad():
            logits = model(img.to(device))
            pseudo_labels.append(logits.argmax(dim=-1).detach().cpu())
    pseudo_labels = torch.cat(pseudo_labels)
    for idx, ((img, _), pseudo_label) in enumerate(zip(dataset.samples, pseudo_labels)):
        dataset.samples[idx] = (img, pseudo_label.item())
    return dataset
