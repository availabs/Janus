from demo import mainloop
from svmclassifier import train
from datacache import feature_cacher
from tkinter import Tk,Button,Entry,StringVar

from multiprocessing import Process, Lock, Pipe



class mygui:
    def __init__(self):
        self.window = Tk()
        self.stop = Button(self.window, text="Stop Recording",command=self.stopclick )
        self.record = Button(self.window, text='Record', command=self.recordclick )
        self.terminate = Button(self.window, text="Exit",command=self.exitclick)
        self.learn =  Button(self.window, text='Learn', command=self.learnclick)
        self.input = Entry(self.window)
        self.stop['state'] ='disabled'
        
        #processes
        self.training = None
        self.running = None

        #communication pipes
        self.trainingpipe = None
        self.runningpipe = None
        
        self.runningpipe,childpipe = Pipe()
        self.running = Process(target=mainloop,args=(childpipe,))
        self.running.start()
        
    def pipelistener(self):
        print('running pipelistener')
        if self.runningpipe.poll(0.001) is not None:
            signal = self.runningpipe.recv()
            print(signal)
            if signal == 'Done Thinking':
                self.learn['state']='normal'

        
    def buildgui(self):
        self.record.place(x=1,y=50)
        self.input.place(x=1,y=75)
        self.stop.place(x=1,y=100)
        self.learn.place(x=1,y=125)
        self.terminate.place(x=1,y=150)
        

        self.window.mainloop()

    def learnclick(self):
        print('learnclick')
        #self.learn['state'] = 'disabled'
        self.runningpipe.send('LEARN')
        
    def exitclick(self):
        print('stopclick')
        self.stop['state'] = 'disabled'
        self.record['state'] = 'normal'
        #other actions...
        if self.running is not None and self.running.exitcode is None:
            self.runningpipe.send('STOP')
            self.running.join()
        self.window.quit()
            

    def stopclick(self):
        print('stoprecording')
        self.stop['state'] = 'disabled'
        self.record['state'] = 'normal'
        self.runningpipe.send('STOPCACHE')
        print ('STOPCACHE')
            
    def recordclick(self):
        print('recordclick')

        #other actions
        lbl = self.input.get()
        if lbl is None or len(lbl) == 0:
            return
        self.stop['state'] = 'normal'
        self.record['state'] = 'disabled'
        print('sending pipe')
        self.runningpipe.send('label:'+lbl)
        
        

    

if __name__ == '__main__':
    gui = mygui()
    gui.buildgui()
