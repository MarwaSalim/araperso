# -*- coding: utf-8 -*-
# clean text from noise take text and return cleaning text
def textCleaning(text):
    print("cleaning text begin")
    import re

    # to remove any email from tweet
    pattern = r"([\w\.-_]+)@([\w\.-]+)(\.[\w\.]+)"
    match = re.search(pattern, text)
    if match:
        text = re.sub(pattern, " ", text)
    print("emails removed")

    # to remove any url from tweet
    pattern = r'https?:(\/){2}\S+'
    match = re.search(pattern, text)
    if match:
        text = re.sub(pattern, " ", text)
    print("url removed")

    # to remove @username
    pattern = r'@\S+'
    match = re.search(pattern, text)
    if match:
        text = re.sub(pattern, " ", text)
    print("@username removed")

    # to remove any digits
    pattern = r'[0-9]+'
    match = re.search(pattern, text)
    if match:
        text = re.sub(pattern, " ", text)
    print("English digit removed")

    # remove noise Marks
    pattern = r"[<,>\.\?/;:\'\"\{\[\]\}|\\~`!@#%$^&*\(\)_=\+-]"
    match = re.search(pattern, text)
    if match:
        text = re.sub(pattern, " ", text)
    print("noise marks removed")

    # replace arabic marks
    text = removeArabicMarks(text)
    print("Arabic marks removed")
    text=removeEmotions(text)
    print("emotions removed removed")
    # get english words
    pattern = r"[A-Za-z][A-Za-z\s]*"
    match = re.search(pattern, text)
    if match:
        english = re.findall(pattern, text)
        # translate english to arabic
        from textblob import TextBlob
        for engWords in english:
            # modifyFile("E:\\master\\GOGOGO\\train Stem","arabic.txt",engWords+'\n')
            try:
                blob = TextBlob(engWords)
                arabWord = blob.translate(to="ar")
                """totalWord = " "
                for w1 in arabWord.words:
                    print w1
                    totalWord = totalWord + w1 + ' '
                """
                # modifyFile("E:\\master\\GOGOGO\\train Stem","arabic.txt",totalWord+'\n')
                text = text.replace(engWords, str(arabWord)+" ")
            except:
                text = text.replace(engWords, ' ')
    print("English words translated")
    text=removeNoArabicWords(text)
    print("Text cleaned")
    return text

def removeArabicMarks(text):
    import re
    patternLPoint=[r'\xe2\x80\xa6',r'\xd8\x9f',r'\xd8\x8c',r'\xe2\x80\x99',r'\xe2\x80\x98',r'\xd8\x9b',r'\xc3\xb7',r'\xc3\x97']
    for x in patternLPoint:
        match = re.search(x, text)
        if match:
            text = re.sub(x, " ", text)
    return text
def removeEmotions(text):
    import re
    patternLPoint = [r'\xf0\x9f\x98\x82',r'\xf0\x9f\x90\xb8', r'\xe2\x9c\x8b', r'\xf0\x9f\x98\x8c', r'\xf0\x9f\x98\x8e',r'\xf0\x9f\x92\x99',
                     r'\xf0\x9f\x98\x8b',r'\xf0\x9f\x92\x94',r'\xf0\x9f\x92\x9e', r'\xe2\x80\x94',r'\xf0\x9f\x92\x9c',
                     r'\xf0\x9f\x8c\xb7', r'\xf0\x9f\x8e\x8b',r'\xf0\x9f\x91\x8a', r'\xf0\x9f\x91\x8d',r'\xf0\x9f\xa4\x94',
                     r'\xf0\x9f\x98\x94',r'\xf0\x9f\x92\x83',r'\xf0\x9f\x92\x96',r'\xf0\x9f\xa4\xa6',r'\xe2\x81\xa6',
                     r'\xe2\x81\xa9',r'\xe2\x9d\xa4',r'\xf0\x9f\x98\x8d',r'\xc2\xab',r'\xc2\xbb',r'\xf0\x9f\x8c\xbb',
                     r'\xf0\x9f\x98\xad',r'\xf0\x9f\x8c\x9a',r'\xf0\x9f\xa4\x97',r'\xf0\x9f\xa4\xb8',r'\xf0\x9f\x8f\xbb',
                     r'\xe2\x80\x8d',r'\xe2\x99\x80',r'\xef\xb8\x8f',r'\xf0\x9f\x92\x95',r'\xf0\x9f\x92\x86',
                     r'\xf0\x9f\x8c\x9e', r'\xf0\x9f\xa6\x89',r'\xf0\x9f\x92\x98',r'\xf0\x9f\x99\x88',
                     r'\xf0\x9f\x8c\xba',r'\xf0\x9f\x98\x8f', r'\xf0\x9f\x92\xab', r'\xe2\x9c\x8a', r'\xf0\x9f\x98\xa2', r'\xf0\x9f\x8c\xbc',
                     r'\xf0\x9f\x99\x84', r'\xf0\x9f\xa4\xb7',r'\xf0\x9f\x9a\xb6', r'\xf0\x9f\x99\x82', r'\xf0\x9f\x98\x84', r'\xf0\x9f\x8e\x88',
                     r'\xf0\x9f\x92\x9a', r'\xf0\x9f\xa4\x90', r'\xf0\x9f\x8c\xb8',r'\xf0\x9f\x8e\xb6', r'\xf0\x9f\x8e\x89', r'\xef\xb8\x8f', r'\xf0\x9f\x8f\x83',
                     r'\xf0\x9f\x8c\x99', r'\xf0\x9f\xa4\xb0', r'\xf0\x9f\x9a\xac',r'\xf0\x9f\x91\xa9',
                     r'\xf0\x9f\x8d\xb3', r'\xf0\x9f\x98\x8a',
                     r'\xf0\x9f\x92\xb0',r'\xf0\x9f\x8c\xb9',r'\xf0\x9f\x98\x87', r'\xf0\x9f\x98\xb6', r'\xf0\x9f\x92\x85', r'\xe2\x98\x94',
                     r'\xf0\x9f\x90\xa0',r'\xf0\x9f\x93\xa2', r'\xf0\x9f\x98\x81',r'\xef\xb7\xba',r'\xef\xb4\xbf', r'\xf0\x9f\x91\x90',
                     r'\xf0\x9f\x92\x93',r'\xf0\x9f\x92\xaa', r'\xe2\x99\x89',r'\xf0\x9f\x99\x87',
                     r'\xf0\x9f\x99\x8c', r'\xf0\x9f\x99\x8b',r'\xe2\x9c\x94', r'\xf0\x9f\x91\xbb',
                     r'\xf0\x9f\x91\xbb',r'\xf0\x9f\x91\x8c',r'\xf0\x9f\x92\x81',
                     r'\xf0\x9f\x98\x9e',r'\xf0\x9f\x91\x89',r'\xf0\x9f\x91\x88',r'\xf0\x9f\x99\x86',
                     r'\xf0\x9f\x98\x98',r'\xf0\x9f\x98\x89',r'\xf0\x9f\x98\x85',r'\xf0\x9f\x95\xba',
                     r'\xf0\x9f\x8f\xbd'
                     ]
    for x in patternLPoint:
        match = re.search(x, text)
        if match:
            text = re.sub(x, " ", text)
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


