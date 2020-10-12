import torch, os
from flask import Flask, json, request, redirect
from flask_cors import CORS
from transformers import BertTokenizer
from BertClassificationModel import BertClassificationModel

# settings
MAX_LEN = 256  # must be matched with fine-tune training parameters
MODEL_PATH = 'epoch26.model'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
print('use Torch Device `{}`'.format(device))

# deploy model
model = BertClassificationModel(num_classes=2, pretrained_model='bert-base-multilingual-cased')
model.to(device)

# load BertClassification model
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

print('Fine-tuned BertClassificationModel `{}` loaded.'.format(MODEL_PATH))
print('> Location: `{}`'.format(os.path.abspath(MODEL_PATH)))

# load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

s2a_analyze_api = Flask(__name__)

# set CORS origins
s2a_analyze_api.config['CORS_HEADERS'] = 'Content-Type'
cors = CORS(s2a_analyze_api, resources={r'/*': {'origins': '*'}})

# set max content length
s2a_analyze_api.config['MAX_CONTENT_LENGTH'] = 8192 * 1024 * 1024


def infer(text):
    tokenized_document = tokenizer.encode_plus(text, None, add_special_tokens=True, max_length=MAX_LEN, truncation=True, padding='max_length', return_token_type_ids=True)

    ids = torch.tensor([tokenized_document['input_ids']], dtype=torch.long).to(device, dtype=torch.long)
    mask = torch.tensor([tokenized_document['attention_mask']], dtype=torch.long).to(device, dtype=torch.long)
    token_type_ids = torch.tensor([tokenized_document['token_type_ids']], dtype=torch.long).to(device, dtype=torch.long)
    outputs = model(ids, mask, token_type_ids)

    sentiment = 'positive' if int(outputs.argmax().data) == 1 else 'negative'
    raw = torch.squeeze(outputs).tolist()
    softmax = torch.squeeze(torch.softmax(outputs, 1)).tolist()

    return {
        'sentiment': sentiment,
        'raw': raw,
        'softmax': softmax
    }


@s2a_analyze_api.route('/sentiment', methods=['POST'])
def sentiment():
    print('POST `sentiment()`')
    if request.method == 'POST':
        # print('[POST]', request.values)

        if 'text' in request.values:
            text = request.values['text']
        else:
            text = ''

        inferred = infer(text)
        print(inferred)

        response = s2a_analyze_api.response_class(
            response=json.dumps(inferred),
            status=200,
            mimetype='application/json',
        )

        response.headers['Access-Control-Allow-Origin'] = '*'

        return response


if __name__ == '__main__':
    s2a_analyze_api.run(host='0.0.0.0', port=1128, debug=False)
