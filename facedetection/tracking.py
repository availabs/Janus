import dlib
import numpy as np
class multitracker:
    def __init__(self,objs=None,frame=None):
        self.trackers ={}
        self.idIndex = 0
        if objs is not None and frame is not None:
            self.buildtrackers(frame,objs)

    def isempty(self):
        return len(self.trackers.keys()) <= 0
    
    def buildtracker(self,frame,objs):
        for box in objs:
            currtracker = dlib.correlation_tracker()
            self.trackers[str(self.idIndex)] = currtracker
            currtracker.start_track(frame,box)
            self.idIndex += 1

    def trackobjs(self,frame):
        for key in self.trackers.keys():
            if self.trackers[key] is not None:
                self.trackers[key].update(frame)

    def getupdates(self):
        updates = []
        for k in self.trackers.keys():
            if self.trackers[k] is not None:
                pos = self.trackers[k].get_position()
                updates.append((k,drec2rec(pos)))
        return updates

    def supplement(self,frame,poss):
        self.trackobjs(frame)
        updates = self.getupdates()
        tupdates = []
        
        Tix = np.zeros(len(updates),dtype=np.uint8)
        for pos in poss:
            isknown = False
            for tix,(k,fpos) in enumerate(updates):
                ivu = iou(pos,fpos)
                if ivu > 0.5:
                    isknown = True
                    Tix[tix] = 1
            if not isknown:
                self.buildtracker(frame,[pos])
                
                    
        for ind,(k,fpos) in zip(list(Tix),updates):
            if ind == 0:
                self.trackers[k] = None
            else:
                tupdates.append((k,fpos))
                
                    
                
                
def iou(pos,fpos):
    maxleft = max(pos.left(),fpos.left())
    minright = min(pos.right(),fpos.right())
    maxtop = max(pos.top(),fpos.top())
    minbot = min(pos.bottom(),fpos.bottom())
    iwid = max(minright-maxleft,0)
    ihei = max(minbot - maxtop,0)
    inter = iwid * ihei
    union = pos.area() + fpos.area() - inter
    return inter/union

def drec2rec(drec):
    x1,y1 = round(drec.left()),round(drec.top())
    x2,y2 = round(drec.right()),round(drec.bottom())
    return dlib.rectangle(left=x1,top=y1,right=x2,bottom=y2)
