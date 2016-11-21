import openface
import time
import argparse

class modelInferance:
    def __init__(self,modelpath='/home/avail/code/torch-projs/openface/models/openface/nn4.small2.v1.t7',
                 indim=96):
        self.net = openface.TorchNeuralNet(model=modelpath,
                                           imgDim=indim,cuda=True)

        print('initialized')

    def getFeatures(self,wfaces):
            reps = []
            start = time.clock()
            for wf in wfaces:
                reps.append(self.net.forward(wf))
            return reps

    def kill(self):
        print('attempting to kill lua server')
        self.net.__del__()

def main():
    mi = modelInferance()
    mi.test()
    
if __name__ == '__main__':
    main()
