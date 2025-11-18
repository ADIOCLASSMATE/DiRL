from torch.utils.data import Dataset, DataLoader

class NormalDataset(Dataset):
    def __init__(self, prompts, responses):
        self.prompts = prompts
        self.responses = responses

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return (
            self.prompts[idx],
            self.responses[idx]
        )
        
class TrainDataset(Dataset):
    def __init__(self, extended_input_ids, p_mask, tok_idx_ext, labels):
        self.extended_input_ids = extended_input_ids
        self.p_mask = p_mask
        self.tok_idx_ext = tok_idx_ext
        self.labels = labels

    def __len__(self):
        return len(self.extended_input_ids)

    def __getitem__(self, idx):
        return (
            self.extended_input_ids[idx],
            self.p_mask[idx],
            self.tok_idx_ext[idx],
            self.labels[idx]
        )
