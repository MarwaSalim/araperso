import os
import FileManging
from PreprocessingStage import removeStopWords,BeginStemmer,removeRedundantChar,textNormalizeRemain
from CleaningStage import textCleaning
def makeThisOptions(options,path):
    for filename in os.listdir(path):
        print(filename)
        IPath=path
        Opath=path
        IF = FileManging.FileManger(IPath, filename)
        text = IF.readFile()
        if options[0]==1:
            text=removeStopWords(text)
            Opath += "_StopWords"
        if options[1]==1:
            text=BeginStemmer(text)
            Opath += "_Stem"
        OF = FileManging.FileManger(Opath, filename)
        try:
            OF.writeFile(text)
        except:
            os.mkdir(Opath)
            OF.writeFile(text)
def makeDefaultOptions(Path):
    for filename in os.listdir(Path):
        Output_Path = Path+"_Clean"
        IF = FileManging.FileManger(Path, filename)
        text=IF.readFile()
        text = textCleaning(text)
        OF = FileManging.FileManger(Output_Path, filename)
        try:
            OF.writeFile(text)
        except:
            os.mkdir(Output_Path)
            OF.writeFile(text)
        text = textNormalizeRemain(OF.readFile())
        Output_Path += "_Normalize"
        OF = FileManging.FileManger(Output_Path, filename)
        try:
            OF.writeFile(text)
        except:
            os.mkdir(Output_Path)
            OF.writeFile(text)
        text = removeRedundantChar(OF.readFile())
        Output_Path += "_Redundant"
        OF = FileManging.FileManger(Output_Path, filename)
        try:
            OF.writeFile(text)
        except:
            os.mkdir(Output_Path)
            OF.writeFile(text)
print "cleaning Normalize Redundacy end"