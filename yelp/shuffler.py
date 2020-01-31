# Created by Stephen Wight
from random import shuffle
import os

def shuf(filein:str, fileout:str='', kernel_size:int=20) -> str:
    '''
    shuf lazy-loads and randomizes the lines of a text file by using random.shuffle to 
    randomize the contents of a rolling kernel. It will return the name of the output file.

    args:
    filein: the filename of the file to be randomized
    fileout: the output file
    kernel_size: how many items to be randomized at a time
    '''

    ### Set fileout if it was not passed
    if fileout == '':
        if '.' in filein:
            fileout = f'{filein.rsplit(".",1)[0]}_shuf.{filein.rsplit(".")[1]}'
        else:
            fileout = f'{filein}_shuf'
    
    ## Ensure that the destination file is empty
    open(fileout, 'w+t').close()

    with open(filein, 'rt') as fl_in:
        with open(fileout, 'at') as fl_out:
            shuffler = []
            for i, line in enumerate(fl_in):
                if len(shuffler) < kernel_size:
                    shuffler.append(line)
                    continue
                shuffle(shuffler)
                fl_out.write(f'{shuffler.pop()}')
                shuffler.append(line)
            while len(shuffler) > 0:
                shuf(shuffler)
                fl_out.write(f'{shuffler.pop()}')
    
    return fileout
