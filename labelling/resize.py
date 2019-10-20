from __future__ import print_function
import argparse
import cv2
import os


def resize_images(in_path, out_path):
    count = 0

    for subdir, dirs, files in os.walk(in_path):
        for file in files:
            # get input image from path given
            file = os.path.join(in_path, file)
            in_img = cv2.imread(file)

            # resize
            dim = (512, 512)
            out_img = cv2.resize(in_img, dim)

            # write to output path
            out_file = str(count) + '.jpg'
            out_img_path = os.path.join(out_path, out_file)
            cv2.imwrite(out_img_path, out_img)

            count += 1
            if count % 1000 == 0:
                print('Finished', count, 'images')


parser = argparse.ArgumentParser()
parser.add_argument('directory', type=str, help='The path to the image folder')
parser.add_argument('--out_dir', type=str, default='')
args = parser.parse_args()

resize_images(args.directory, args.out_dir)
