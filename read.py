import numpy as np
import binvox_rw as bvx 
import tarfile
import re
from io import BytesIO
from PIL import Image

class ShapeNetLoader(object):

    def __init__(self, n_imgs=1, desired_imgs=['04'], input_img_sz=(200,200,1), input_voxel_sz=(32,32,32)):
        self.n_images = n_imgs
        self.desired_imgs = desired_imgs
        self.voxel_set = []
        self.image_set = []
        self.input_img_sz = input_img_sz
        self.input_voxel_sz = input_voxel_sz

    def load_image(self, filebuffer):
        img = Image.open(BytesIO(bytearray(filebuffer.read())))
        img = img.convert('L').resize(self.input_img_sz[:2])
        return np.reshape(np.asarray(img, dtype='float32'), self.input_img_sz)

    def load_data(self, bucket='gs://shapenet', target_obj_set=['04256520']):
        self.data_path = bucket
        self.object_set = target_obj_set
        with tarfile.open(self.data_path+'/ShapenetVox32.tar', 'r') as voxtar, tarfile.open(self.data_path+'/ShapeNetRendering.tar','r') as imgtar:
            for obj in self.object_set:
                print(obj)
                regex_voxel = re.compile('.*/'+obj+'/.*/model\.binvox')
                names = list(filter(regex_voxel.match, voxtar.getnames()))
                pic_locs = ['ShapeNetRendering/'+x.partition('/')[-1].rpartition('/')[0]+'/rendering/' for x in names]
                for i,j in enumerate(names):
                    self.voxel_set.append(np.reshape(np.array(bvx.read_as_3d_array(voxtar.extractfile(j)).data, dtype='float32'), self.input_voxel_sz))
                    loaded_imgs = []
                    for m in self.desired_imgs:
                        loaded_imgs.append(self.load_image(imgtar.extractfile(pic_locs[i]+m+'.png')))
                    self.image_set.append(loaded_imgs)
        self.image_set = np.asarray(self.image_set)
        self.voxel_set = np.asarray(self.voxel_set)

    def get_shapes(self):
        return self.image_set.shape, self.voxel_set.shape

    def get_batch_iterator(self, batch_size=5):
        l = len(self.image_set)
        for ndx in range(0, l, batch_size):
            yield self.image_set[ndx:min(ndx + batch_size, l)], self.voxel_set[ndx:min(ndx + batch_size, l)]
