from demo import mainloop
from svmclassifier import train
from datacache import feature_cacher
from tkinter import Tk,Button,Entry,StringVar

from multiprocessing import Process, Lock, Pipe



class mygui:
    def __init__(self):
        self.window = Tk()
        self.demoButton = Button(self.window,text='RunRecognizer',command=self.democlick )
        self.stop = Button(self.window, text="Stop",command=self.stopclick )
        self.supervise = Button(self.window, text='Learn Face', command=self.superviseclick )
        self.input = Entry(self.window)
        self.stop['state'] ='disabled'
        
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
        self.runningpipe,childpipe = Pipe()
        self.running = Process(target=mainloop,args=(childpipe,))
        self.running.start()
        
    def stopclick(self):
        print('stopclick')
        self.stop['state'] = 'disabled'
        self.demoButton['state'] = 'normal'
        self.supervise['state'] = 'normal'
        #other actions...
        if self.caching is not None and self.caching.exitcode is None:
            self.cachingpipe.send('STOP')
            self.caching.join()
        if self.running is not None and self.running.exitcode is None:
            self.runningpipe.send('STOP')
            self.running.join()
        
    def superviseclick(self):
        print('superviseclick')
        self.stop['state'] = 'normal'
        self.supervise['state'] = 'disabled'
        self.demoButton['state'] = 'disabled'
        #other actions
        lbl = self.input.get()
        if lbl is None or len(lbl) == 0:
            self.stopclick()
            return
        self.cachingpipe,childpipe = Pipe()
        def runner():
            feature_cacher().cacheprocess(lbl,childpipe)
        self.caching = Process(target=runner)
        print('code:',self.caching.exitcode)
        self.caching.start()
        

    

if __name__ == '__main__':
    gui = mygui()
    gui.buildgui()
