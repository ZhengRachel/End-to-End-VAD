from cProfile import label
import os
import shutil
import librosa
import random
random.seed('1234')
import numpy as np
from tqdm import tqdm
import tempfile
import cv2

def make_video(path):
    # input video 240 (H) x 320 (W) 
    in_path = path
    files = os.listdir(in_path)
    files = list(filter(lambda file: file.find('.jpg') != -1, files))
    files = sorted(files, key=lambda file: int(os.path.splitext(file)[0]))
    array = [cv2.imread(os.path.join(in_path, file), cv2.IMREAD_GRAYSCALE) for file in files]
    array = [cv2.resize(a, (128, 64)) for a in array]
    ims = list(filter(lambda im: not im is None, array))
    ims = np.array(ims)
    return ims

PATHROOT = '/home/rczheng/TaLData/TaL80/core/'
FRAME_RATE = 25
MIN_FRAMES = 150

with open('E2EVAD/all.list') as fp:
    lines = fp.readlines()
    fp.close()

train_index = [i for i in range(0, len(lines))]
test_index = sorted(random.sample(train_index, 150))
for i in test_index:
    train_index.remove(i)
train_index = sorted(train_index)


train_cnt = 0
if os.path.exists('E2EVAD/train'):
    shutil.rmtree('E2EVAD/train')
if not os.path.exists('E2EVAD/train'):
    os.mkdir('E2EVAD/train')
for i in tqdm(train_index):
    wavpath = os.path.join(PATHROOT, lines[i].strip().split('\t')[0], lines[i].strip().split('\t')[1])
    shutil.copy(wavpath, os.path.join('E2EVAD/train', '%s.wav'%train_cnt))
    shutil.copy(wavpath.replace('.wav', '.mp4'), os.path.join('E2EVAD/train', '%s.avi'%train_cnt))

    p = tempfile.mkdtemp()
    frame_rate = FRAME_RATE
    cmd = 'ffmpeg  -i \'{}\' -qscale:v 2 -r {} \'{}/%d.jpg\' -loglevel quiet'.format(
                wavpath.replace('.wav', '.mp4'),
                frame_rate,
                p)
    os.system(cmd)
    lip_video = make_video(p)
    shutil.rmtree(p)
    lip_video = lip_video[:MIN_FRAMES,:,:]
    np.save(os.path.join('E2EVAD/train', '%s_video.npy'%train_cnt), lip_video)
    
    wav, _ = librosa.load(wavpath, sr=16000)
    wavt, index = librosa.effects.trim(wav, top_db=25)
    begin, end = int(float(index[0])/16000*25), min(int(float(index[1])/16000*25), MIN_FRAMES)
    labels = np.zeros(MIN_FRAMES)
    labels[begin:end+1] = 1
    # print(labels)
    np.save(os.path.join('E2EVAD/train', '%s_labels.npy'%train_cnt), labels)

    with open(os.path.join('E2EVAD/train', '%s.txt'%train_cnt), 'w') as fp:
        fp.write('%s\t%s\n'%(lines[i].strip().split('\t')[0], lines[i].strip().split('\t')[1]))
        fp.close()
    train_cnt += 1


test_cnt = 0
if os.path.exists('E2EVAD/test'):
    shutil.rmtree('E2EVAD/test')
if not os.path.exists('E2EVAD/test'):
    os.mkdir('E2EVAD/test')
for i in tqdm(test_index):
    wavpath = os.path.join(PATHROOT, lines[i].strip().split('\t')[0], lines[i].strip().split('\t')[1])
    shutil.copy(wavpath, os.path.join('E2EVAD/test', '%s.wav'%test_cnt))
    shutil.copy(wavpath.replace('.wav', '.mp4'), os.path.join('E2EVAD/test', '%s.avi'%test_cnt))

    p = tempfile.mkdtemp()
    frame_rate = FRAME_RATE
    cmd = 'ffmpeg  -i \'{}\' -qscale:v 2 -r {} \'{}/%d.jpg\' -loglevel quiet'.format(
                wavpath.replace('.wav', '.mp4'),
                frame_rate,
                p)
    os.system(cmd)
    lip_video = make_video(p)
    shutil.rmtree(p)
    np.save(os.path.join('E2EVAD/test', '%s_video.npy'%test_cnt), lip_video)
    
    wav, _ = librosa.load(wavpath, sr=16000)
    wavt, index = librosa.effects.trim(wav, top_db=25)
    begin, end = int(float(index[0])/16000*25), int(float(index[1])/16000*25)
    labels = np.zeros(lip_video.shape[0])
    labels[begin:end+1] = 1
    np.save(os.path.join('E2EVAD/test', '%s_labels.npy'%test_cnt), labels)

    with open(os.path.join('E2EVAD/test', '%s.txt'%test_cnt), 'w') as fp:
        fp.write('%s\t%s\n'%(lines[i].strip().split('\t')[0], lines[i].strip().split('\t')[1]))
        fp.close()
    test_cnt += 1

