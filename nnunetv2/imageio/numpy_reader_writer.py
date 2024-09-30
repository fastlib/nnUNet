#    Copyright 2021 HIP Applied Computer Vision Lab, Division of Medical Image Computing, German Cancer Research Center
#    (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from typing import Tuple, Union, List
import numpy as np
from nnunetv2.imageio.base_reader_writer import BaseReaderWriter


class NumpyIO(BaseReaderWriter):
    """
    ONLY SUPPORTS 1D timeseries
    """

    supported_file_endings = [
        '.npy'
    ]

    def read_images(self, image_fnames: Union[List[str], Tuple[str, ...]]) -> Tuple[np.ndarray, dict]:
        images = []
        for f in image_fnames:
            npy_img = np.load(f)
            assert npy_img.ndim == 1 or npy_img.ndim == 2, "Only 1D timeseries with one or more channels supported"
            if npy_img.ndim == 2:
                # channel to front, add additional dim so that we have shape (c, 1, 1, X)
                images.append(npy_img.transpose((1, 0))[:, None, None])
            elif npy_img.ndim == 1:
                # grayscale image
                images.append(npy_img[None, None, None])

        if not self._check_all_same([i.shape for i in images]):
            print('ERROR! Not all input images have the same shape!')
            print('Shapes:')
            print([i.shape for i in images])
            print('Image files:')
            print(image_fnames)
            raise RuntimeError()
        return np.vstack(images, dtype=np.float32, casting='unsafe'), {'spacing': (999, 999, 1)}

    def read_seg(self, seg_fname: str) -> Tuple[np.ndarray, dict]:
        return self.read_images((seg_fname, ))

    def write_seg(self, seg: np.ndarray, output_fname: str, properties: dict) -> None:
        np.save(output_fname, seg[0].astype(np.uint8, copy=False))

if __name__ == '__main__':
    images = ('/Users/lukasarts/Dropbox/UU/ASRA/nnUNet/nnUNet_raw/Dataset0011_test/imagesTr/case_0_0000.npy','/Users/lukasarts/Dropbox/UU/ASRA/nnUNet/nnUNet_raw/Dataset0011_test/imagesTr/case_1_0000.npy')
    segmentation = '/Users/lukasarts/Dropbox/UU/ASRA/nnUNet/nnUNet_raw/Dataset0011_test/labelsTr/case_0_0000.npy'
    imgio = NumpyIO()
    img, props = imgio.read_images(images)
    seg, segprops = imgio.read_seg(segmentation)