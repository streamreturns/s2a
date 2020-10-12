import torch
from torch.utils.data import Dataset


class NsmcDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.document = dataframe.document
        self.targets = self.data.label
        self.max_length = max_length

    def __len__(self):
        return len(self.document)

    def __getitem__(self, index):
        document = str(self.document[index])
        document = ' '.join(document.split())

        inputs = self.tokenizer.encode_plus(document, None, add_special_tokens=True, max_length=self.max_length, truncation=True, padding='max_length', return_token_type_ids=True)
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.long)
        }
