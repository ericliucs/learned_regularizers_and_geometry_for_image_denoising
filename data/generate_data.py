import os
import shutil
from PIL import Image

import urllib.request
import tarfile


def download():
    """
    Downloads the training and testing data
    """
    bsds_url = 'http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz'
    tar_name = bsds_url.split('/')[-1]
    print('start downloading training data from "{}"'.format(bsds_url))
    urllib.request.urlretrieve(bsds_url, tar_name)
    print('downloading done')

    print('start extracting tar file')
    with tarfile.open(tar_name, 'r:gz') as tar_file:
        tar_file.extractall('./tmp/')
    print('extracting done')


def save_training_data(training_dir: str = os.path.join(os.getcwd(),'data', 'BSDS400')):
    """Saves training data to data directory

    Parameters
    ----------
    training_dir: (str) - Directory to store the BSDS training data in
    """
    if not os.path.exists(training_dir):
        os.makedirs(training_dir)
    data_source_base_dir = './tmp/BSR/BSDS500/data/images'
    for src in ['test', 'train']:
        src_dir = os.path.join(data_source_base_dir, src)
        files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f)) and
                 os.path.splitext(f)[1] == '.jpg']
        for file in files:
            full_file_name = os.path.join(src_dir, file)
            with Image.open(full_file_name) as im:
                img = im.load()
                file = file.replace('jpg', 'png')
                im.save(os.path.join(training_dir, file))
    print('training data done')


def save_testing_data(test_dir: str = os.path.join(os.getcwd(),'data', 'BSDS68')):
    """Saves test data to data directory

    Parameters
    ----------
    test_dir: (str) - Directory to store the BSDS test data in
    """
    data_source_base_dir = './tmp/BSR/BSDS500/data/images'
    print('generating test data')
    test_set_list_url = 'http://www.visinf.tu-darmstadt.de/media/visinf/vi_data/foe_test.txt'
    test_list = urllib.request.urlopen(test_set_list_url).read().decode('utf-8').split('\n')[:-1]
    src_dir = os.path.join(data_source_base_dir, 'val')
    for i, file in enumerate(test_list):
        with Image.open(os.path.join(src_dir, file)) as im:
            img = im.load()
            im.save(os.path.join(test_dir, '{}.png'.format(i + 1)))
    print('test data done')


def cleanup():
    """
    Cleans up tmp directory where data is initially stored
    """
    bsds_url = 'http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz'
    tar_name = bsds_url.split('/')[-1]
    print('cleaning up')
    shutil.rmtree('./tmp')
    os.remove(tar_name)
    print('cleaning up done')


if __name__ == '__main__':
    download()
    save_training_data()
    save_testing_data()
    cleanup()
