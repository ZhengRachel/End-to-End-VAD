import os
import librosa

PATHROOT = '/home/rczheng/TaLData/TaL80/core/'
WRITE_PATH = '/home/rczheng/End-to-End-VAD/E2EVAD/'

files = []
with open('E2EVAD/test.list') as fp:
    lines = fp.readlines()
    for l in lines:
        sp, wf = l.strip().split()[0], l.strip().split()[1]
        wavpath = os.path.join(PATHROOT, sp, wf)
        t = librosa.get_duration(filename=wavpath)
        if t>6 and t<6.25:
            wav, _ = librosa.load(wavpath, sr=16000)
            files.append([sp, wf])
    print(len(files))
    fp.close()

with open('E2EVAD/train.list') as fp:
    lines = fp.readlines()
    for l in lines:
        sp, wf = l.strip().split()[0], l.strip().split()[1]
        wavpath = os.path.join(PATHROOT, sp, wf)
        t = librosa.get_duration(filename=wavpath)
        if t>6 and t<6.25:
            files.append([sp, wf])
    print(len(files))
    fp.close()

with open('E2EVAD/all.list', 'w') as fp:
    for f in files:
        fp.write('%s\t%s\n' % (f[0], f[1]))
    fp.close()
