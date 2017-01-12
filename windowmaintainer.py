import collections
# This class will be the storage class for a single sliding window of features
class window:
    def __init__(self,windowsize=15):
        # -- windowsize : number of objects in sliding window
        self.ws = windowsize
        # -- use fifo deque to allow old data to fall off the end
        self.fifoqueue = collections.deque([],self.ws)

    # add a new object in the newest location
    def push(self,obj):
        self.fifoqueue.appendleft(obj)

    # get the objects in the queue
    def getobjs(self):
        return self.fifoqueue

    #test if the queue has met its quota
    def isfull(self):
        return (len(self.fifoqueue) == self.ws) 

# This class will be the storage class for a set of windows with unique ids
class IdWindows:
    #initialize a storage object and declare the size of windows
    #that will be used uniformly
    def __init__(self,windowsize=15):
        self.windows = {}
        self.ws = windowsize

    # Method will push an object onto the storage object
    # using the given ID
    def push(self,id,obj):
        if id in self.windows:
            self.windows[id].push(obj)
        else:
            self.windows[id] = window(self.ws)
            self.windows[id].push(obj)
    # Get objects for a specified ID
    def getobjs(self,id):
        return self.windows[id].getobjs()

    # Check if the collection with given ID is full
    def isfull(self,id):
        return self.windows[id].isfull()
