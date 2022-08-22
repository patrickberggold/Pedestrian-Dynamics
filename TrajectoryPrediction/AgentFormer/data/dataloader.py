from data.nuscenes_pred_split import get_nuscenes_pred_split
import os, random, numpy as np, copy

from .preprocessor import preprocess
from .ethucy_split import get_ethucy_split
from .floorplan_simple_split import get_flooplan_simple_split
from utils.utils import print_log


class data_generator(object):

    def __init__(self, parser, log, split='train', phase='training'):
        self.past_frames = parser.past_frames
        self.min_past_frames = parser.min_past_frames
        self.frame_skip = parser.get('frame_skip', 1)
        self.phase = phase
        self.split = split
        assert phase in ['training', 'testing'], 'error'
        assert split in ['train', 'val', 'test'], 'error'

        if parser.dataset == 'nuscenes_pred':
            data_root = parser.data_root_nuscenes_pred           
            seq_train, seq_val, seq_test = get_nuscenes_pred_split(data_root)
            self.init_frame = 0
        elif parser.dataset in {'eth', 'hotel', 'univ', 'zara1', 'zara2'}:
            # data_root = parser.data_root_ethucy
            data_root = os.path.join('TrajectoryPrediction','AgentFormer','datasets','eth_ucy')
            seq_train, seq_val, seq_test = get_ethucy_split(parser.dataset)
            self.init_frame = 0
        elif parser.dataset == 'floorplan_simple':
            data_root = 'C:\\Users\\Remotey\\Documents\\Datasets\\SIMPLE_FLOORPLANS\\CSV_SIMULATION_DATA_numAgents_50'
            img_root = 'C:\\Users\\Remotey\\Documents\\Datasets\\SIMPLE_FLOORPLANS\\HDF5_INPUT_IMAGES_resolution_800_800'
            [seq_train, seq_val, seq_test], [img_train, img_val, img_test] = get_flooplan_simple_split(data_root, img_root)
            self.init_frame = 0
        else:
            raise ValueError('Unknown dataset!')

        process_func = preprocess
        self.data_root = data_root

        print_log("\n-------------------------- loading %s data --------------------------" % split, log=log)
        if self.split == 'train':  self.sequence_to_load, self.img_to_load = seq_train, img_train
        elif self.split == 'val':  self.sequence_to_load, self.img_to_load = seq_val, img_val
        elif self.split == 'test': self.sequence_to_load, self.img_to_load = seq_test, img_test
        else:                      assert False, 'error'

        self.num_total_samples = 0
        self.num_sample_list = []
        self.sequence = []
        for id_seq, seq_name in enumerate(self.sequence_to_load[:5]):
            print_log("loading sequence {} ...".format(seq_name), log=log)
            preprocessor = process_func(data_root, seq_name, parser, log, img_name=self.img_to_load[id_seq], split=self.split, phase=self.phase)
            # 1 seq = (min_past_frames + min_future_frames - 1) frames
            num_seq_samples = preprocessor.num_fr - (parser.min_past_frames + parser.min_future_frames - 1) * self.frame_skip
            self.num_total_samples += num_seq_samples
            self.num_sample_list.append(num_seq_samples)
            self.sequence.append(preprocessor)
        
        self.sample_list = list(range(self.num_total_samples)) # total number of samples = total number of sequences
        self.index = 0
        print_log(f'total num samples: {self.num_total_samples}', log)
        print_log("------------------------------ done --------------------------------\n", log=log)

    def shuffle(self):
        random.shuffle(self.sample_list)
        
    def get_seq_and_frame(self, index):
        index_tmp = copy.copy(index)
        for seq_index in range(len(self.num_sample_list)):    # 0-indexed
            if index_tmp < self.num_sample_list[seq_index]:
                frame_index = index_tmp + (self.min_past_frames - 1) * self.frame_skip + self.sequence[seq_index].init_frame     # from 0-indexed list index to 1-indexed frame index (for mot)
                return seq_index, frame_index
            else:
                index_tmp -= self.num_sample_list[seq_index]

        assert False, 'index is %d, out of range' % (index)

    def is_epoch_end(self):
        if self.index >= self.num_total_samples:
            self.index = 0      # reset
            return True
        else:
            return False

    def next_sample(self):
        # get data from frame in corresponding sequence 
        sample_index = self.sample_list[self.index] # sequence index
        seq_index, frame = self.get_seq_and_frame(sample_index)
        seq = self.sequence[seq_index]
        self.index += 1
        
        data = seq(frame)
        return data      

    def __call__(self):
        return self.next_sample()
