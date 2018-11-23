# -*- coding: utf-8 -*-
def BeginStemmer(text):
    result=""
    if text != "":
        import EgyptionStemmer
        stemmer = EgyptionStemmer.EgyptionStemmer()
        #print text
        for t in text.split():
            r=stemmer.stemming(t)
            result+= r+" "
    return result



def textNormalizeRemain(text):
    print("Normalize text begin")
    import re
    """patternArabWords= r'\w+'
    match = re.search(patternArabWords,text,flags=re.UNICODE)
    if not match:
        return  ""
    """
    # replace  hamza
    patternLink1 = r'\xd8\xa4|\xd8\xa6'
    match = re.search(patternLink1, text)
    if match:
        text = re.sub(patternLink1, "\xd8\xa1", text)
    print("hamza replaced")
    # replace  alf
    patternLink1 = r'\xd8\xa3|\xd8\xa5|\xd8\xa2'
    match = re.search(patternLink1, text)
    if match:
        text = re.sub(patternLink1, "\xd8\xa7", text)
    print("alf replaced")
    #replace lam alf ﻷ
    patternLink1 = r'\xef\xbb\xb7'
    match = re.search(patternLink1, text)
    if match:
        text = re.sub(patternLink1, "\xd9\x84\xd8\xa7", text)
    print("lam alf replaced")
    # replace heh
    patternLink1 = r'\xd8\xa9'
    match = re.search(patternLink1, text)
    if match:
        text = re.sub(patternLink1, "\xd9\x87", text)
    print("heh replaced")
    # replace yah
    patternLink1 = r'\xd9\x89'
    match = re.search(patternLink1, text)
    if match:
        text = re.sub(patternLink1, "\xd9\x8a", text)
    print("yah replaced")
    import pyarabic.araby as araby
    text=araby.strip_tashkeel(text)
    text = strip_tashkeel(text)
    text=araby.strip_harakat(text)
    text=araby.strip_lastharaka(text)
    text=araby.strip_shadda(text)
    print("tashkeel removed")
    """text=strip_tatweel(text)
    print("tatweel removed")"""
    text = removeNoArabicWords(text)
    print("no arabic removed")
    return text


def removeNoArabicWords(text):
    print("no arabic removed begin")
    texts = text.split()
    result = ""
    import pyarabic.araby as araby
    for w in texts:
        try:
            w = araby.strip_tatweel(w.decode("utf8"))
            if araby.is_arabicword(w):
                result = result + w + " "
        except:
            print("Error")
    return result


def strip_tashkeel(text):
    import re
    pattern = r'\xd9\x92'
    match = re.search(pattern, text)
    if match:
        text = re.sub(pattern, "", text)

    pattern = r'\xd9\x8d'
    match = re.search(pattern, text)
    if match:
        text = re.sub(pattern, "", text)

    pattern = r'\xd9\x90'
    match = re.search(pattern, text)
    if match:
        text = re.sub(pattern, "", text)

    pattern = r'\xd9\x8e'
    match = re.search(pattern, text)
    if match:
        text = re.sub(pattern, "", text)

    pattern = r'\xd9\x8b'
    match = re.search(pattern, text)
    if match:
        text = re.sub(pattern, "", text)

    pattern = r'\xd9\x8f'
    match = re.search(pattern, text)
    if match:
        text = re.sub(pattern, "", text)

    pattern = r'\xd9\x8c'
    match = re.search(pattern, text)
    if match:
        text = re.sub(pattern, "", text)

    pattern = r'\xd9\x91'
    match = re.search(pattern, text)
    if match:
        text = re.sub(pattern, "", text)
    return text


"""def strip_tatweel(text):
    import re
    pattern=r'\xd9\x80'
    match = re.search(pattern,text)
    if match:
        text = re.sub(pattern,"",text)
    return text
"""


def removeRedundantChar(text):
    #text = text.encode('utf8')
    texts = text.split()
    newText = " "
    for w in texts:
        word = ""
        x = 0
        while (x < len(w)):
            count = reduncdanceCount(w[x:x + 2], x, w)
            # print(count)
            if (count == 2):
                # print(2)
                word += w[x:x + 2] + w[x:x + 2]
                x += 4
            elif (count == 1):
                # print(1)
                word += w[x:x + 2]
                x += 2
            else:
                # print(count)
                word += w[x:x + 2]
                x += (count * 2)
        newText += word + " "
    return newText


def reduncdanceCount(ch, index, text):
    count = 0
    x = index
    while (x < len(text)):
        if (text[x:x + 2] == ch):
            count += 1
            x += 2
        else:
            return count
    return count

def removeStopWords(text):
    import FileManging
    File=FileManging.FileManger("C:\\Users\\Marwa\\PycharmProjects\\LoadDS","127StopWords.txt")
    StopWords = File.readFile()
    result=" "
    try:
        text=text.encode('utf_8')
    except:
        text = text
    for t in text.split():
        if t not in StopWords.split():
            result+=t+" "
    return result
