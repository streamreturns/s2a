# S2A
Deep Learning based Sentiment Analysis from Surprise to Anger (S2A)

## Usage (Python 3)
### Import Library
```
import requests
```

### Set API Host
```
api_ip = 'searchlight.kr'
s2a_port = '1128'
```

### [Example] Positive Sentiment
- **Request**
```
text = '봄날에 입고가기 좋아요'
r = requests.post('http://{api_ip}:{s2a_port}/sentiment'.format(api_ip=api_ip, s2a_port=int(s2a_port)), data={'text': text})
```

- **Response**
```
{'raw': [-3.826526641845703, 3.8188369274139404],
 'sentiment': 'positive',
 'softmax': [0.00047802767949178815, 0.9995219707489014]}
```

### [Example] Negative Sentiment
- **Request**
```
text = '재질이 거칠어요'
r = requests.post('http://{api_ip}:{s2a_port}/sentiment'.format(api_ip=api_ip, s2a_port=int(s2a_port)), data={'text': text})
```

- **Response**
```
{'raw': [2.857180118560791, -2.8349578380584717],
 'sentiment': 'negative',
 'softmax': [0.9966388940811157, 0.0033610411919653416]}
```

### [Example] Bulk Inference
This example analyzes `text` column in the `target_df`.

- **Define Bulk Inference Method `analyze_sentiment`**
```
def analyze_sentiment(row):
    text = prettify_string(row['text'])
    
    r = requests.post('http://{api_ip}:{s2a_port}/sentiment'.format(api_ip=api_ip, s2a_port=int(s2a_port)),
                      data={
                          'text': text
                      })
    
    if r.ok:
        resp = r.json()
        row['sentiment'] = resp['sentiment']
        row['raw'] = numpy.max(resp['raw'])
        row['softmax'] = numpy.max(resp['softmax'])
    else:
        row['sentiment'] = 'API error'
        row['raw'] = 'API error'
        row['softmax'] = 'API error'    
        
    return row
```

- **Bulk Inference**
```
target_df = target_df.apply(lambda x: analyze_sentiment(x), axis=1)
```

### Supported Languages

Korean (한국어)

---

- **Maintainer**
Byeongho Kang (byeongho.kang@yahoo.com)
- **GitHub**
https://github.com/streamreturns
