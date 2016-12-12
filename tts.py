import os
from gtts import gTTS
import time

def genfile(text):
    tts = gTTS(text=text,lang='en')
    tts.save('temp.mp3')


def playfile():
    os.system('play temp.mp3')


def gentts(text):
    genfile(text)
    playfile()
    

def mptts(ttsque):
    while True:
        message = ttsque.get()
        if message == 'KILL':
            break
        else:
            print(message)
        gentts(message)

        
    
    
if __name__ == '__main__':
    for quote in ['this is a test', 'of the emergency' , 'alert system']:
        genfile(quote)
        playfile()
