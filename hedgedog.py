#!/usr/bin/env python3

import cv2
import numpy as np 
import sys 
import random 
import getopt 
from multiprocessing import Pool

class Segment:
    def __init__(self, start, end):
        self.start = start
        self.end = end 


class Ray:
    def __init__(self, pos, d):
        """ pos: numpy 2d array
            d: direction, numpy 2d array
        """
        self.pos = pos
        self.d = d

    def at(self, r):
        return self.pos + r * self.d 

    def segment(self, r1, r2):
        return Segment(self.at(r1), self.at(r2))


def score(points, ray, cutoff, seg_cutoff):
    """ points: np nx3 array, (x,y,value)
    return: segment
    """
    
    relative_pos = points[:, 0:2] - ray.pos

    distance_h = np.dot(relative_pos, ray.d)
    distance_o = (relative_pos - np.array([dh * ray.d for dh in distance_h]))**2

    select_idx = distance_o[:,0]**2 * distance_o[:,1]**2 < cutoff * cutoff 

    #neighbor = points[select_idx]
    used_distance_h = distance_h[select_idx]

    # if the ray is selected? (d->any points < cutoff?)
    if len(used_distance_h) <= 5:
        return None

    # if so, arrange points from large to small, choose first segment (min-min2)
    seg_end = -1
    used_distance_h = np.sort(used_distance_h)
    for i in range(len(used_distance_h) - 1):
        if used_distance_h[i+1] - used_distance_h[i] > 2 * seg_cutoff:
            seg_end = i
            break
        elif used_distance_h[i+1] - used_distance_h[i] > seg_cutoff and random.random() < 0.8:
            seg_end = i
            break
        elif used_distance_h[i+1] - used_distance_h[i] > seg_cutoff * 0.5 and random.random() < 0.2:
            seg_end = i
            break

    if seg_end == 0:
        return None

    # return segment
    return ray.segment(used_distance_h[0], used_distance_h[seg_end])


def img2list(img, cutoff):
    
    points = []
    for (i, j), v in np.ndenumerate(img):
        if v > cutoff:
            points.append(np.array([j, i, v]))

    return np.array(points)


def segments2img(segments, imgsize, color):
    
    img = np.zeros(imgsize)
    for seg in segments:
        cv2.line(img, (int(seg.start[0]), int(seg.start[1])), (int(seg.end[0]), int(seg.end[1])), color)

    return img


def main(argv):
    
    help_str = "Usage: hedgedog (-BbCcsn arg) image"

    # img parameters
    contrast = 1.25
    bright = 20
    black_cutoff = 150

    # ray parameters
    ray_number = 400
    ray_cutoff = 5.0
    seg_cutoff = 15.0

    opts, args = getopt.getopt(argv[1:], 'B:b:C:c:s:n:h')
    for opt, arg in opts:
        if opt == '-B':
            bright = int(arg)
        elif opt == '-b':
            black_cutoff = int(arg)
        elif opt == '-C':
            contrast = float(arg)
        elif opt == '-c':
            ray_cutoff = float(arg)
        elif opt == '-n':
            ray_number = int(arg)
        elif opt == '-s':
            seg_cutoff = float(arg)
        elif opt == '-h':
            print(help_str)
            return

    if len(args) < 1:
        print('Error: No enough args')
        return

    # read img (black-white)
    img = cv2.imread(args[0])
    cv2.imshow('picture', img)

    img_sc = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_sc = 255 - img_sc

    # img preprocessing
    img_sc = contrast * (img_sc - np.average(img_sc)) + np.average(img_sc) + bright

    cv2.imshow('picture2', img_sc)

    # get points
    points = img2list(img_sc, black_cutoff)

    # randomly generate segments
    random_coords = np.random.random((ray_number, 2))
    random_angles = np.random.random(ray_number) * 2 * np.pi
    random_directions = np.column_stack([np.cos(random_angles), np.sin(random_angles)])

    random_coords[:, 0] *= img_sc.shape[0]
    random_coords[:, 1] *= img_sc.shape[1]

    task_pool = Pool(3)
    segment_queue = []

    for i in range(ray_number):
        ray = Ray(
            random_coords[i], random_directions[i]
        )

        segment_queue.append(task_pool.apply_async(score, args=(points, ray, ray_cutoff, seg_cutoff)))

    task_pool.close()
    task_pool.join()

    segments = [s.get() for s in segment_queue]

    img2 = 255 - segments2img([s for s in segments if s], img_sc.shape, 255)

    # inverse img2

    cv2.imshow('sampled', img2)

    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)