import torch

class Dataset:
    def __init__(self, reviews, targets):
        self.reviews = reviews
        self.targets = targets

    def __len__(self):
        return len(self.reviews)
        
    def __getitem__(self, item):
        return {
            "review" : torch.tensor(review, dtype=torch.long),
            "target" : torch.tensor(review, dtype=torch.float)
        }