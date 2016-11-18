from demo import outfacing
from svmclassifier import train
from datacache import feature_cacher
from tkinter import Tk,Button,Entry,StringVar

from multiprocessing import Process, Lock, Pipe



class mygui:
    def __init__(self):
        self.window = Tk()
        self.demoButton = Button(self.window,text='RunDemo',command=self.democlick )
        self.stop = Button(self.window, text="Stop",command=self.stopclick )
        self.supervise = Button(self.window, text='supervise', command=self.superviseclick )
        self.input = Entry(self.window)
        self.stop['state'] ='disabled'
        self.demoButton['state'] = 'disabled'
        #processes
        self.caching = None
        self.training = None
        self.running = None

        #communication pipes
        self.cachingpipe = None
        self.trainingpipe = None
        self.runningpipe = None
        
    def buildgui(self):
        self.demoButton.place(x=1,y=25)
        self.supervise.place(x=1,y=50)
        self.input.place(x=1,y=75)
        self.stop.place(x=1,y=100)
        self.window.mainloop()
        
    def democlick(self):
        print('democlick')
        self.stop['state'] = 'normal'
        self.supervise['state'] = 'disabled'
        self.demoButton['state'] = 'disabled'
        #other stuff
        
    def stopclick(self):
        print('stopclick')
        self.stop['state'] = 'disabled'
        #self.demoButton['state'] = 'normal'
        self.supervise['state'] = 'normal'
        #other actions...
        if self.caching is not None and self.caching.exitcode is not None:
            self.cachingpipe.send('STOP')
            self.caching.join()
        
        
    def superviseclick(self):
        print('superviseclick')
        self.stop['state'] = 'normal'
        self.supervise['state'] = 'disabled'
        self.demoButton['state'] = 'disabled'
        #other actions
        lbl = self.input.get()
        if lbl is not None and len(lbl) > 0:
            cacher = feature_cacher()
        else:
            cacher = None
            self.stopclick()
            return
        self.cachingpipe,childpipe = Pipe()
        self.caching = Process(target=cacher.cacheprocess, args=(lbl,childpipe))
        print('code:',self.caching.exitcode)
        self.caching.start()
        

    

if __name__ == '__main__':
    gui = mygui()
    gui.buildgui()


