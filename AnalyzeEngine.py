import numpy, pandas, torch
from sklearn import metrics
import torch
from torch.utils.data import Dataset, DataLoader  # , RandomSampler, SequentialSampler
from transformers import AutoTokenizer, AutoModel

# Sections of config

# Defining some key variables that will be used later on in the training
MAX_LEN = 256
TRAIN_BATCH_SIZE = 28
VALID_BATCH_SIZE = 4
EPOCHS = 20
LEARNING_RATE = 1e-05


def encode_onehot(index, size):
    onehot = numpy.zeros(size, dtype=numpy.int8)
    onehot[index] = 1
    return onehot


class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.document = dataframe.document
        self.targets = self.data.label
        self.max_len = max_len

    def __len__(self):
        return len(self.document)

    def __getitem__(self, index):
        document = str(self.document[index])
        document = ' '.join(document.split())

        inputs = self.tokenizer.encode_plus(
            document,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.long)
        }


# Creating the customized model, by adding a drop out and a dense layer on top of distil bert to get the final output for the model.
class SentimentBinaryClassification(torch.nn.Module):
    def __init__(self, model_path=None):
        super(SentimentBinaryClassification, self).__init__()

        if model_path is None:  # load pretrained model
            self.l1 = AutoModel.from_pretrained('bert-base-multilingual-cased', cache_dir='transformers-cached')
        else:
            self.l1 = AutoModel.from_pretrained(model_path)
        self.l2 = torch.nn.Dropout(0.5)
        self.l3 = torch.nn.Linear(768, 2)  # classes: 2

    def forward(self, ids, mask, token_type_ids):
        _, output_1 = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output


class BinarySentimentModel:
    def __init__(self, model_path=None, pytorch_device=None):
        self.model_path = model_path

        if pytorch_device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = pytorch_device
        print('`device`:', self.device)

        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased', cache_dir='transformers-cached')  # , local_files_only=True))
        self.model = SentimentBinaryClassification(model_path=self.model_path)
        self.model.to(self.device)

        self.train_parameters = {
            'batch_size': TRAIN_BATCH_SIZE,
            'shuffle': True,
            'num_workers': 0
        }

        self.test_parameters = {
            'batch_size': VALID_BATCH_SIZE,
            'shuffle': True,
            'num_workers': 0
        }

    @staticmethod
    def load_df(file_name):
        df = pandas.read_csv(file_name, index_col='id', sep='\t')
        df['label'] = df['label'].apply(lambda x: encode_onehot(x, 2))

        return df

    @staticmethod
    def loss_function(outputs, targets):
        return torch.nn.BCEWithLogitsLoss()(outputs, targets)

    def train(self):
        train_df = self.load_df('ratings_train.txt')
        print('<Head of `Train Dataset`>')
        print(train_df.head())

        print('Training Dataset: {}'.format(train_df.shape))
        training_dataset = CustomDataset(train_df.reset_index(drop=True), self.tokenizer, MAX_LEN)

        optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=5e-05)

        for epoch in range(EPOCHS):
            self.model.train()

            for i, data in enumerate(DataLoader(training_dataset, **self.train_parameters), 0):
                ids = data['ids'].to(self.device, dtype=torch.long)
                mask = data['mask'].to(self.device, dtype=torch.long)
                token_type_ids = data['token_type_ids'].to(self.device, dtype=torch.long)
                targets = data['targets'].to(self.device, dtype=torch.float)

                outputs = self.model(ids, mask, token_type_ids)

                loss = self.loss_function(outputs, targets)
                if i % 1000 == 0:
                    print(i, f'Epoch: {epoch}, Loss:  {loss.item()}')

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # torch.save(self.model.state_dict(), './s2a_binary_sentiment_nsmc.model')
            self.model.save_pretrained('asdf')

    def validation(self, data_loader):
        self.model.eval()
        fin_targets = []
        fin_outputs = []

        with torch.no_grad():
            for _, data in enumerate(data_loader, 0):
                ids = data['ids'].to(self.device, dtype=torch.long)
                mask = data['mask'].to(self.device, dtype=torch.long)
                token_type_ids = data['token_type_ids'].to(self.device, dtype=torch.long)
                targets = data['targets'].to(self.device, dtype=torch.float)
                outputs = self.model(ids, mask, token_type_ids)
                fin_targets.extend(targets.cpu().detach().numpy().tolist())
                fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

        return fin_outputs, fin_targets

    def test(self):
        test_df = self.load_df('ratings_test.txt')
        print('<Head of `Test Dataset`')
        print(test_df.head())

        print('Testing Dataset: {}'.format(test_df.shape))
        testing_dataset = CustomDataset(test_df.reset_index(drop=True), self.tokenizer, MAX_LEN)

        for epoch in range(1):  # range(EPOCHS):
            outputs, targets = self.validation(DataLoader(testing_dataset, **self.test_parameters))
            outputs = numpy.array(outputs) >= 0.5
            accuracy = metrics.accuracy_score(targets, outputs)
            f1_score_micro = metrics.f1_score(targets, outputs, average='micro')
            f1_score_macro = metrics.f1_score(targets, outputs, average='macro')

            print(f'Accuracy: {accuracy}')
            print(f'F1 Score (Micro): {f1_score_micro}')
            print(f'F1 Score (Macro): {f1_score_macro}')

    def infer(self, document):
        encoded = self.tokenizer.encode_plus(
            document,
            None,
            add_special_tokens=True,
            max_length=MAX_LEN,
            truncation=True,
            padding='max_length',
            return_token_type_ids=True
        )

        ids = torch.tensor([encoded['input_ids']], dtype=torch.long).to(self.device, dtype=torch.long)
        mask = torch.tensor([encoded['attention_mask']], dtype=torch.long).to(self.device, dtype=torch.long)
        token_type_ids = torch.tensor([encoded['token_type_ids']], dtype=torch.long).to(self.device, dtype=torch.long)

        outputs = self.model(ids, mask, token_type_ids)
        print(outputs)


if __name__ == '__main__':
    # for Train
    binarySentimentModel = BinarySentimentModel(model_path=None)
    binarySentimentModel.train()

    # for Test
    binarySentimentModel = BinarySentimentModel(model_path='./s2a_binary_sentiment_nsmc.model')
    binarySentimentModel.test()
