import random

class Melody:
    def __init__(self, pitch=0, duration=0, offset=0, velocity=0):
        '''
        'pitch': 0~127숫자값,
        'duration': 0~elementsPerMeasure,
        'offset': 0~elementsPerMeasure,
        'velocity': 0~127숫자값
        '''

        self.pitch = pitch
        self.duration = duration
        self.offset = offset
        self.velocity = velocity
        pass

    @classmethod
    def createRandom(cls, measureInSec=None, elementsPerMeasure=None):
        if measureInSec is None:
            measureInSec = 5
        if elementsPerMeasure is None:
            elementsPerMeasure = 32

        pitch = random.randrange(0, 128)
        duration = random.randrange(0, elementsPerMeasure)
        offset = random.randrange(0, elementsPerMeasure)
        velocity = random.randrange(0, 128)
        melody = Melody(pitch=pitch, duration=duration, offset=offset, velocity=velocity)

        return melody

    def toJSON(self):
        return {
            'pitch': self.pitch,
            'duration': self.duration,
            'offset': self.offset,
            'velocity': self.velocity,
        }