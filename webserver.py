import os
import cv2
import time
import cPickle
import datetime
import logging
import flask
import werkzeug
import optparse
import tornado.wsgi
import tornado.httpserver
import numpy as np
import pandas as pd
from PIL import Image
import cStringIO as StringIO
import urllib
import exifutil
from tools.axademo import detect_cni
from flask import Flask, redirect, url_for, request, session, abort, render_template, flash
import os
import caffe
import glob
import SOAPpy
from io import BytesIO
import requests
import caffe

CAFFE_ROOT ='/home/cuong-nguyen/2016/Workspace/brexia/Septembre/Codesource/caffe-master'
REPO_DIR = os.path.abspath(os.path.dirname(__file__))
MODELE_DIR = os.path.join(REPO_DIR, 'models/googlenet')
DATA_DIR = os.path.join(REPO_DIR, 'data/googlenet')
UPLOAD_FOLDER = '/tmp/caffe_demos_uploads'
ALLOWED_IMAGE_EXTENSIONS = set(['png', 'bmp', 'jpg', 'jpe', 'jpeg', 'gif'])


class ImagenetClassifier(object):
    default_args = {
        'model_def_file': (
            '{}/deploy.prototxt'.format(MODELE_DIR)),
        'pretrained_model_file': (
            '{}/train_val.caffemodel'.format(DATA_DIR)),
        'mean_file': (
            '{}/python/caffe/imagenet/ilsvrc_2012_mean.npy'.format(CAFFE_ROOT)),
        'class_labels_file': (
            '{}/synset_words.txt'.format(MODELE_DIR)),
        'bet_file': (
            '{}/data/ilsvrc12/imagenet.bet.pickle'.format(CAFFE_ROOT)),
    }
    for key, val in default_args.iteritems():
        if not os.path.exists(val):
            raise Exception(
                "File for {} is missing. Should be at: {}".format(key, val))
    default_args['image_dim'] = 256
    default_args['raw_scale'] = 255.

    def __init__(self, model_def_file, pretrained_model_file, mean_file,
                 raw_scale, class_labels_file, bet_file, image_dim, gpu_mode):
        logging.info('Loading net and associated files...')
        if gpu_mode:
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()
        net = caffe.Net(model_def_file, pretrained_model_file, caffe.TEST)
        self.net = net
        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2,0,1))
        # transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))
        transformer.set_raw_scale('data', 255)
        transformer.set_channel_swap('data', (2,1,0)) # if using RGB instead of BGR
        self.transformer = transformer

        with open(class_labels_file) as f:
            labels_df = pd.DataFrame([
                {
                    'synset_id': l.strip().split(' ')[0],
                    'name': ' '.join(l.strip().split(' ')[1:]).split(',')[0]
                }
                for l in f.readlines()
            ])
        self.labels = labels_df.sort('synset_id')['name'].values

    def classify_image(self, image):
        try:
            net = self.net
            net.blobs['data'].data[...] = self.transformer.preprocess('data', image)

            starttime = time.time()
            # scores = self.net.predict([image], oversample=True).flatten()
            out = net.forward()
            proba = out['prob'][0]
            scores = net.blobs['fc8'].data[0]
            endtime = time.time()

            indices = (-proba).argsort()[:3]
            predictions = self.labels[indices]

            # In addition to the prediction text, we will also produce
            # the length for the progress bar visualization.
            meta_proba = [
                (p, '%.5f' % proba[i])
                for i, p in zip(indices, predictions)
            ]

            score = [
                (p, '%.5f' % scores[i])
                for i, p in zip(indices, predictions)
            ]
            logging.info('result: %s', str(meta_proba))

            return (True, score, meta_proba, '%.3f' % (endtime - starttime))

        except Exception as err:
            logging.info('Classification error: %s', err)
            return (False, 'Something went wrong when classifying the '
                           'image. Maybe try another one?')


def ocr(urlpath):
	response = requests.get(urlpath)
	img = Image.open(BytesIO(response.content))
	im=img.save("temp.png")
	cwd = os.getcwd()
	print cwd
	path=os.path.join(cwd, 'temp.png')
	string_buffer = StringIO.StringIO(urllib.urlopen(urlpath).read())
	image = caffe.io.load_image(string_buffer)
	parser = optparse.OptionParser()
	parser.add_option('-d', '--debug', help="enable debug mode", action="store_true", default=False)
	parser.add_option(
        '-p', '--port',
        help="which port to serve content on",
        type='int', default=5000)
	parser.add_option(
        '-g', '--gpu',
        help="use gpu mode",
        action='store_true', default=False)
	opts, args = parser.parse_args()
	ImagenetClassifier.default_args.update({'gpu_mode': opts.gpu})
	classifier=ImagenetClassifier(**ImagenetClassifier.default_args)
	classifier.net.forward()
	result= classifier.classify_image(image)
	cnis, preproc_time, roi_file_images=detect_cni(path)
	return cnis, result



server = SOAPpy.SOAPServer(("127.0.0.1",5000))
server.registerFunction(ocr)
server.serve_forever()

