def runStanza(nlp, sentence_text):
    doc = nlp(sentence_text)
    data = {}
    data['sentences'] = []
    sentID = 0
    for sentence in doc.sentences:
        newSent = {}
        newSent['tokens'] = []
        newSent['index'] = sentID
        sentID += 1
        for token in sentence.tokens:
            if len(token.words) == 1:
                newToken = {}
                newToken['index'] = token.id[0]
                newToken['originalText'] = token.text
                newToken['word'] = token.text
                newToken['featuresText'] = "_"
                if hasattr(token, "feats"):
                    newToken['featuresText'] = token.feats
                newToken['characterOffsetBegin'] = token.start_char
                newToken['characterOffsetEnd'] = token.end_char
                newToken['isMultiwordToken'] = False
                newToken['isMultiwordFirstToken'] = False
                for t in token.words:
                    newToken['pos'] = t.xpos
                    newToken['ud_pos'] = t.upos
                    newToken['lemma'] = t.lemma
                newSent['tokens'].append(newToken)
            else:
                first = True
                for t in token.words:
                    newToken = {}
                    newToken['index'] = t.id
                    newToken['originalText'] = token.text
                    newToken['word'] = t.text
                    newToken['characterOffsetBegin'] = token.start_char
                    newToken['characterOffsetEnd'] = token.end_char
                    newToken['featuresText'] = "_"
                    if hasattr(t, "feats"):
                        newToken['featuresText'] = t.feats
                    newToken['isMultiwordToken'] = True
                    newToken['isMultiwordFirstToken'] = first
                    newToken['pos'] = t.xpos
                    newToken['ud_pos'] = t.upos
                    newToken['lemma'] = t.lemma
                    first = False
                    newSent['tokens'].append(newToken)

        data['sentences'].append(newSent)
    return data

def runTint(tint_url, sentence_text):
    # requires args['tint-url']
    myobj = {'text' : sentence_text.strip()}
    x = requests.post(tint_url, data = myobj)
    data = json.loads(x.text)
    return data

