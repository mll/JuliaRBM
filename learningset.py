#!/usr/bin/python

import cPickle
import gzip
import os
import sys
import time
import png


def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)

    print '... loading data'

    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an numpy.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #numpy.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.


    test_set_x, test_set_y = test_set
    valid_set_x, valid_set_y = valid_set
    train_set_x, train_set_y = train_set

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval

def stringFromDigit(digit):
   out = ""
   for i in range(0,10):
     if i == digit:
       out += "1"
     else:
       out += "0"
     if i!= 9:
       out+=" "
   return out

if __name__ == '__main__':
    retval = load_data("mnist.pkl.gz")
    
    learningSetFile = open("learn-1.dat","w")
    learningSetFile.write(str(len(retval[0][0]))+" "+str(len(retval[0][0][0]))+" "+str(10)+" \n\n")
    for i in xrange(0,len(retval[0][1])):
      im = retval[0][0][i]
      learningSetFile.write(" ".join(map(str,im)))
      learningSetFile.write("\n")
      learningSetFile.write(stringFromDigit(retval[0][1][i])+"\n\n")
      
      print i,"/",len(retval[0][0])
    learningSetFile.close()
        
      
      
#      print "Image: ",str(im),len(im)," target: ",retval[0][1][i]

      #plik = open("imis"+str(i)+"-"+str(retval[0][1][i])+".png","wb")
      #structure = []
      #for b in xrange(0,28):
#         row = []
#         for j in xrange(0,28):
#            row.append( im[j+b*28]*256)
#         structure.append(row)
#      s = structure
#      w = png.Writer(len(s[0]), len(s), greyscale=True,bitdepth=8)
#      w.write(plik,s)
#      plik.close()
        
      
      
      
    
    
    
