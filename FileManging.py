# -*- coding: utf-8 -*-
class FileManger:
    def __init__(self, directory, name):
        self.directory=directory
        self.name=name



    #read data from file take directory and file name and return data readed
    def readFile(self):
        import os
        os.chdir(self.directory)
        fileNeed=open(self.name,"r")
        self.text=fileNeed.read()
        fileNeed.close()
        #print("file " + name + " readed")
        return self.text

    #write data in file take directory and file name
    def writeFile(self,text):
        import os
        os.chdir(self.directory)
        fileNeed = open(self.name,"w")
        try:
            fileNeed.write(text.encode("utf8"))
        except:
            fileNeed.write(text)
        fileNeed.close()
        #print("data writen in " + name)

    #modify data in file take directory and file name
    def modifyFile(self,text):
        fileNeed = open(self.name,"a")
        fileNeed.write(text)
        fileNeed.close()
        #print("data updated in " + name)

    #clear file from any data
    def clearFile(self):
        fileNeed = open(self.name,"w").close()
        #print("file " + name+" cleared")
