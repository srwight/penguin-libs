import random, json, langdetect

shuffle_size = 20
with open('nlp/yelp_academic_dataset_review.json', encoding='utf8') as js:
    with open('nlp/shuffled.json', 'a') as js2:
        shuf = []
        for i, line in enumerate(js):
            ln = {k:dict(json.loads(line))[k] for k in ['text', 'stars']}
            try:
                if langdetect.detect(ln['text']) == 'en':
                    if len(shuf) < shuffle_size:
                        shuf.append(ln)
                        continue
                    else:
                        random.shuffle(shuf)
                        x = shuf.pop()
                        print(f'\rWriting line {i}', end='')
                        js2.write(f'{json.dumps(x)}\n')
            except langdetect.lang_detect_exception.LangDetectException:
                print(f'\rError on line {i}:\n', ln)
                with open('nlp/emojis.json', 'a') as emoji:
                    emoji.write(f'{json.dumps(ln)}\n')
        for line in shuf:
            js2.write(f'{json.dumps(line)}\n')