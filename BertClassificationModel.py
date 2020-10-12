import torch, transformers


# Creating the customized model, by adding a drop out and a dense layer on top of distil bert to get the final output for the model.
class BertClassificationModel(torch.nn.Module):
    def __init__(self, num_classes=2, pretrained_model='bert-base-multilingual-cased'):
        super(BertClassificationModel, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained(pretrained_model)
        self.l2 = torch.nn.Dropout(0.5)
        self.l3 = torch.nn.Linear(768, int(num_classes))

    def forward(self, ids, mask, token_type_ids):
        _, output_1 = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output
