import collections
class window:
    def __init__(self,windowsize=15):
        self.ws = windowsize
        self.fifoqueue = collections.deque([],self.ws)

    def push(self,obj):
        self.fifoqueue.appendleft(obj)

    def getobjs(self):
        return self.fifoqueue

    def isfull(self):
        return (len(self.fifoqueue) == self.ws) 
