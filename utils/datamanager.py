import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

from utils.general_class import DatamanagerPlugin

import numpy as np
import random

class DspritesManager(DatamanagerPlugin):
    ''' https://github.com/deepmind/dsprites-dataset/blob/master/dsprites_reloading_example.ipynb'''
    def __init__(self, dataset_zip):
        self.image = np.expand_dims(dataset_zip['imgs'], axis=-1).astype(float)
        self.latents_values = dataset_zip['latents_values']
        self.latents_classes = dataset_zip['latents_classes']
        self.metadata = dataset_zip['metadata'][()]
        self.latents_sizes = self.metadata['latents_sizes']
        self.nlatent = len(self.latents_sizes) #6
        self.latents_bases = np.concatenate((self.latents_sizes[::-1].cumprod()[::-1][1:], np.array([1,])))
        super().__init__(ndata=self.image.shape[0])

    def print_shape(self):
        print("Image shape : {}({}, max = {}, min = {})".format(self.image.shape, self.image.dtype, np.amax(self.image), np.amin(self.image)))
        print("Latent size : {}".format(self.latents_sizes))

    def normalize(self, nmin, nmax):
        cmin = np.amin(self.image)
        cmax = np.amax(self.image)
        slope = (nmax-nmin)/(cmax-cmin)

        self.image = slope*(self.image-cmin) + nmin
        self.print_shape()

    def latent2idx(self, latents):
        return np.dot(latents, self.latents_bases).astype(int)

    def next_batch_latent_random(self, batch_size):
        samples = np.zeros([batch_size, self.nlatent])
        for lat_i, lat_size in enumerate(self.latents_sizes):
            samples[:, lat_i] = np.random.randint(lat_size, size=batch_size)
        return samples

    def next_batch_latent_fix(self, batch_size, latent_idx, latent_value): 
        samples = self.next_batch_latent_random(batch_size)
        samples[:, latent_idx] = latent_value
        return self.image[self.latent2idx(samples)]

    def next_batch_latent_fix_idx(self, batch_size, latent_idx, latent_value): 
        samples = self.next_batch_latent_random(batch_size)
        samples[:, latent_idx] = latent_value
        return self.latent2idx(samples)

    def next_batch(self, batch_size):
        subidx = self.sample_idx(batch_size)
        return self.image[subidx], self.latents_classes[subidx]


class Shapes3DManager(DatamanagerPlugin):
    def __init__(self, dataset_zip):
        self.image = dataset_zip['images'][:]  # array shape [480000,64,64,3], uint8 in range(256)
        self.latents_classes = dataset_zip['labels']  # array shape [480000,6], float64
        self.nlatent = self.latents_classes.shape[1]
        self.latents_sizes = np.array([10, 10, 10, 8, 4, 15]).astype(np.int32)

        # self.latents_values = dataset_zip['latents_values']
        # self.latents_classes = dataset_zip['latents_classes']
        # self.metadata = dataset_zip['metadata'][()]
        # self.latents_sizes = self.metadata['latents_sizes']
        # self.nlatent = len(self.latents_sizes) #6
        # self.latents_bases = np.concatenate((self.latents_sizes[::-1].cumprod()[::-1][1:], np.array([1,])))

        self._FACTORS_IN_ORDER = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape', 'orientation']
        self._NUM_VALUES_PER_FACTOR = {'floor_hue': 10, 'wall_hue': 10, 
                'object_hue': 10, 'scale': 8, 'shape': 4, 'orientation': 15}
        super().__init__(ndata=self.image.shape[0])

    def print_shape(self):
        print("Image shape : {}({}, max = {}, min = {})".format(self.image.shape, self.image.dtype, np.amax(self.image), np.amin(self.image)))
        print("Latent size : {}".format(self.latents_sizes))

    def normalize(self, nmin, nmax):
        cmin = np.amin(self.image)
        cmax = np.amax(self.image)
        slope = (nmax-nmin)/(cmax-cmin)

        self.image = slope*(self.image-cmin) + nmin
        self.print_shape()

    # def latent2idx(self, latents):
        # return np.dot(latents, self.latents_bases).astype(int)

    def latent2idx(self, latents):
        '''
        latents: np array shape [batch_size, 6]
        '''
        indices = 0
        base = 1
        for factor, name in reversed(list(enumerate(self._FACTORS_IN_ORDER))):
            indices += latents[:, factor] * base
            base *= self._NUM_VALUES_PER_FACTOR[name]
        return indices.astype(int)

    def next_batch_latent_random(self, batch_size):
        samples = np.zeros([batch_size, self.nlatent])
        for lat_i, lat_size in enumerate(self.latents_sizes):
            samples[:, lat_i] = np.random.randint(lat_size, size=batch_size)
        return samples

    def next_batch_latent_fix(self, batch_size, latent_idx, latent_value): 
        samples = self.next_batch_latent_random(batch_size)
        samples[:, latent_idx] = latent_value
        indices = self.latent2idx(samples)
        # ims = []
        # for ind in indices:
            # im = self.image[ind]
            # im = np.asarray(im)
            # ims.append(im)
        # ims = np.stack(ims, axis=0)
        # ims = (ims / 255.).astype(np.float32)
        # return ims.reshape([batch_size, 64, 64, 3])
        return (self.image[indices] / 255.).astype(float)

    def next_batch_latent_fix_idx(self, batch_size, latent_idx, latent_value): 
        samples = self.next_batch_latent_random(batch_size)
        samples[:, latent_idx] = latent_value
        return self.latent2idx(samples)

    def next_batch(self, batch_size):
        subidx = self.sample_idx(batch_size)
        # ims = []
        # for ind in subidx:
            # im = self.image[ind]
            # im = np.asarray(im)
            # ims.append(im)
        # ims = np.stack(ims, axis=0)
        # ims = (ims / 255.).astype(np.float32)
        # return ims.reshape([batch_size, 64, 64, 3]), None
        # return self.image[subidx], self.latents_classes[subidx]
        return (self.image[subidx] / 255.).astype(float), None
