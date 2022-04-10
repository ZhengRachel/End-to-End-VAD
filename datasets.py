import os
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class VideoDataset(Dataset):
    """Dataset Class for Loading video files"""
    def __init__(self, DataDir, timeDepth, is_train):
        """
        Args:
        DataDir (string): Directory with all the data.
        timeDepth: Number of frames to be loaded in a sample
        is_train(bool): Is train or test dataset
        """
        self.DataDir = DataDir
        self.is_train = is_train
        self.timeDepth = timeDepth
        self.video_duration_in_frames = 150
        self.raw_video = sorted([x for x in os.listdir(self.DataDir) if ".avi" in x])

    def __len__(self):
        return len(self.raw_video)*(self.video_duration_in_frames-self.timeDepth)

    def __getitem__(self, idx):
        vid_idx = int(idx / (self.video_duration_in_frames - self.timeDepth))
        frame_idx = idx % (self.video_duration_in_frames - self.timeDepth)

        # print(vid_idx, frame_idx)
        videos = np.load(os.path.join(self.DataDir, '%s_video.npy'%vid_idx))
        sample = videos[frame_idx:frame_idx+self.timeDepth, :, :]

        labels = np.load(os.path.join(self.DataDir, '%s_labels.npy'%vid_idx))
        sample_label = labels[frame_idx+self.timeDepth]

        return sample, sample_label


if __name__ == '__main__':
    train_dataset = VideoDataset(DataDir='E2EVAD/train/', timeDepth = 15, is_train=True)
    print('%s samples found.' %len(train_dataset))
    train_loader = DataLoader(
        train_dataset,
        batch_size=2, shuffle=True,
        num_workers=0, pin_memory=False,
        drop_last=True)
    print('%s batches found.'%(len(train_loader)))
    for i, data in enumerate(train_loader):
        input, target = data
        print(input.shape, target)