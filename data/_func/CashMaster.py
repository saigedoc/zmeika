#import dill
from dill import loads, dumps
#import os
from os import path, mkdir, listdir, remove, rmdir
#import sys
from sys import getsizeof
#import numpy as np
#import random

class CASH_MASTER:
    def __init__(self, Mb = 500000000, Direstory = 'data\cash', Traceback = True):
        self.max_bytes = Mb
        if Direstory == '':
            1 / 0
        self.PATH = Direstory
        self.output = Traceback

        pp = ''
        for p in self.PATH.split('\%s' %('')):
            pp += p + '\%s' %('')
            if not(path.exists(pp)):
                mkdir(pp)


        self.extension = '.cm'


    def Call(self, arr_name):
        path =  self.PATH + r"\%s" %(arr_name)

        f = open(path + self.extension, 'rb')
        #size = dill.load(f.read())
        arr = bytearray(0)
        arr += f.read()
        f.close()




        #size = arr[0]
        #arr = bytes(bytearray(arr)[1:])
        i=1
        while True:
            try:
                arr += open(path + '-part%s'%i + self.extension, 'rb').read()
                i+=1
            except:
                break
            

        #arr = dill.loads(arr)
        arr = loads(arr)

        return arr

        

    def Cash(self, arr_now, arr_name):
        
        
        path = self.PATH + r"\%s" %arr_name

        #f = open(path + self.extension, 'wb')
        #arr_now = dill.dumps(arr_now)
        arr_now = dumps(arr_now)

        size = getsizeof(arr_now)
        #size = sys.getsizeof([size, arr_now])
        #print(size, self.max_bytes)
        

        idx = 0
            
        #size = bytes(size)
        #arr_now[:0] = [len(size)]
            
        for i in range(round((size / self.max_bytes) + 0.5)):
            #open((path + '@%s' %size + '@' + '$%s' %i), 'wb').write(arr_now[idx:idx+(max_bytes)])
            #idx += (max_bytes)
            if i == 0:
                file_path = path
            else:
                file_path = path + '-part%s' %i
                
            f = open(file_path + self.extension, 'wb')
                

            f.write(arr_now[idx:idx+self.max_bytes])

            f.close()
                
            idx +=self.max_bytes

    def newpath(self, NewDirestory, Save = True):
        if Save:
            self.save()
        if NewDirestory == '':
            1 / 0
        self.PATH = NewDirestory
    def delete(self, Dir_too = False):
        files = listdir(self.PATH)
        p0 = 0
        for f in files:
            try:
                remove(self.PATH + '\%s' %f)
            except:
                if self.output:
                    print('Deleting was failed.')
            if self.output:
                p = round(files.index(f) / len(files) * 100)
                if p0 != p and p < 100:
                    print(p, '%')
                    p0 = p
        if self.output:
            print('100 %')
        if Dir_too:
            rmdir(self.PATH)
            self.__init__(Mb = self.max_bytes, Direstory = self.PATH, Traceback = self.output)