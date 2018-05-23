#!/usr/bin/env python

from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
import SocketServer
import urlparse
import json
import numpy as np
import os
import sys
import argparse
import glob
import time
import urllib
import urllib2

from PIL import Image
from StringIO import StringIO
import caffe

import boto3
import botocore
import time


def getS3Bucket():
    try:
        s3_bucket_name = os.environ["S3_BUCKET"]
        aws_access_key_id = os.environ["AWS_ACCESS_KEY_ID"]
        aws_secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]
        print s3_bucket_name
        print aws_access_key_id
        print aws_secret_access_key

        s3 = boto3.resource('s3')

        try:
            bucket = s3.Bucket(s3_bucket_name)
            print bucket
            for key in bucket.objects.all():
                print(key.key)

            # bucket.download_file(KEY, 'my_local_image.jpg')
            return bucket, s3_bucket_name
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                print("The object does not exist.")
            else:
                raise
            return None, None

    except KeyError:
        print "S3 env vars not set"
        return None, None


def getGenderScore(image, image_data):
    print 'Getting gender score for image:' + image

    print 'gender_net', gender_net
    print 'gender_list', gender_list

    img_data_rs = resize_image(image_data, sz=(256, 256))
    input_image = caffe.io.load_image(StringIO(img_data_rs))

    prediction = gender_net.predict([input_image])
    gender = gender_list[prediction[0].argmax()]
    print 'Predicted gender:', gender

    return gender


def initGenderClassifier():
    print 'Init nsfw clasifier'

    # Load the mean image
    mean_filename = './gender_model/gender_mean.binaryproto'
    proto_data = open(mean_filename, "rb").read()
    a = caffe.io.caffe_pb2.BlobProto.FromString(proto_data)
    mean = caffe.io.blobproto_to_array(a)[0]

    # Load the gender network
    gender_net_pretrained = './gender_model/gender_net.caffemodel'
    gender_net_model_file = './gender_model/gender_deploy.prototxt'
    gender_net = caffe.Classifier(gender_net_model_file, gender_net_pretrained,
                                  mean=mean,
                                  channel_swap=(2, 1, 0),
                                  raw_scale=255,
                                  image_dims=(256, 256))
    # labels
    gender_list = ['Male', 'Female']
    return gender_net, gender_list


def getNSFWScore(image, image_data):

    print 'Getting nsfw score for image:' + image

    print 'caffe_transformer', caffe_transformer
    print 'nsfw_net', nsfw_net

    scores = caffe_preprocess_and_compute(
        image_data, caffe_transformer=caffe_transformer, caffe_net=nsfw_net, output_layers=['prob'])

    # Scores is the array containing SFW / NSFW image probabilities
    # scores[1] indicates the NSFW probability
    print "NSFW score:  ", scores
    return scores[1]


def initNSFWClassifier():
    print 'Init nsfw clasifier'

    model_def = './nsfw_model/nsfw_deploy.prototxt'
    pretrained_model = './nsfw_model/resnet_50_1by2_nsfw.caffemodel'

    # Pre-load caffe model.
    nsfw_net = caffe.Net(model_def, pretrained_model, caffe.TEST)

    # Load transformer
    # Note that the parameters are hard-coded for best results
    caffe_transformer = caffe.io.Transformer(
        {'data': nsfw_net.blobs['data'].data.shape})
    # move image channels to outermost
    caffe_transformer.set_transpose('data', (2, 0, 1))
    # subtract the dataset-mean value in each channel
    caffe_transformer.set_mean('data', np.array([104, 117, 123]))
    # rescale from [0, 1] to [0, 255]
    caffe_transformer.set_raw_scale('data', 255)
    caffe_transformer.set_channel_swap(
        'data', (2, 1, 0))  # swap channels from RGB to BGR

    # return vars
    return caffe_transformer, nsfw_net


def resize_image(data, sz=(256, 256)):
    img_data = str(data)
    im = Image.open(StringIO(img_data))
    if im.mode != "RGB":
        im = im.convert('RGB')
    imr = im.resize(sz, resample=Image.BILINEAR)
    fh_im = StringIO()
    imr.save(fh_im, format='JPEG')
    fh_im.seek(0)
    return bytearray(fh_im.read())


def caffe_preprocess_and_compute(pimg, caffe_transformer=None, caffe_net=None,
                                 output_layers=None):
    if caffe_net is not None:

        # Grab the default output names if none were requested specifically.
        if output_layers is None:
            output_layers = caffe_net.outputs

        img_data_rs = resize_image(pimg, sz=(256, 256))
        image = caffe.io.load_image(StringIO(img_data_rs))

        H, W, _ = image.shape
        _, _, h, w = caffe_net.blobs['data'].data.shape
        h_off = max((H - h) / 2, 0)
        w_off = max((W - w) / 2, 0)
        crop = image[h_off:h_off + h, w_off:w_off + w, :]
        transformed_image = caffe_transformer.preprocess('data', crop)
        transformed_image.shape = (1,) + transformed_image.shape

        input_name = caffe_net.inputs[0]
        all_outputs = caffe_net.forward_all(blobs=output_layers,
                                            **{input_name: transformed_image})

        outputs = all_outputs[output_layers[0]][0].astype(float)
        return outputs
    else:
        return []


def getImageData(image, bucket, s3_bucket_name):
    print 'Image:' + image
    print bucket, s3_bucket_name
    useS3 = False
    if bucket is not None and s3_bucket_name is not None:
        if s3_bucket_name + '.s3.amazonaws.com' in image or 'amazonaws.com/' + s3_bucket_name in image:
            useS3 = True
    if useS3 == True:
        try:
            print 'Pulling image from S3'
            file_name = image[image.rfind("/") + 1:]
            print file_name
            bucket.download_file(file_name, "/tmp/" + file_name)
            image_data = open('/tmp/' + file_name).read()
            return image_data, None
        except Exception as e:
            print e.__doc__
            error = e.message
            print error
            return None, error

    elif 'http://' in image or 'https://' in image:
        try:
            print 'Downloading image', image
            req = urllib2.Request(
                image, headers={'User-Agent': "Magic Browser"})
            con = urllib2.urlopen(req)

            if con.getcode() is 200:
                image_data = con.read()
                return image_data, None
            else:
                error = "Server responded with " + \
                    str(con.getcode()) + " when attempting to download image "

                print error
                return None, error
        except Exception as e:
            print e
            print e.__doc__
            error = e.message
            print error
            return None, error

    else:
        return None, 'File is not a downloadable resource'


class web_handler(BaseHTTPRequestHandler):
    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/json')
        self.end_headers()

    def do_GET(self):
        self._set_headers()
        print self
        print self.path
        o = urlparse.urlparse(self.path)
        params = urlparse.parse_qs(o.query)
        print params
        if 'image' in params:
            image = params['image'][0]
            t1 = time.time()
            image_data, error = getImageData(image, bucket, s3_bucket_name)

            if error is not None:
                print error
                data = {
                    "error": "true",
                    "msg": error
                }
                self.wfile.write(json.dumps(data))
            else:
                t2 = time.time()
                gender = getGenderScore(image, image_data)
                t3 = time.time()
                nsfw = getNSFWScore(image, image_data)
                t4 = time.time()

                data = {
                    "image": image + "?" + str(t4),
                    "gender": gender,
                    "nsfw": float(format(nsfw, ".2f")),
                    "timings": {
                        "download": float(format(t2 - t1, ".2f")),
                        "gender": float(format(t3 - t2, ".2f")),
                        "nsfw": float(format(t4 - t3, ".2f")),
                        "total": float(format(t4 - t1, ".2f"))
                    }
                }
                self.wfile.write(json.dumps(data))
        else:

            error = 'Unknown path & params: ' + self.path
            print error
            data = {
                "error": "true",
                "msg": error
            }
            self.wfile.write(json.dumps(data))


def runServer(bucket, s3_bucket_namecant, server_class=HTTPServer, handler_class=web_handler, port=80):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print 'Starting httpd...'

    print 'caffe_transformer', caffe_transformer
    print 'nsfw_net', nsfw_net

    print 'gender_net', gender_net
    print 'gender_list', gender_list

    httpd.serve_forever()


if __name__ == "__main__":
    from sys import argv

    print 'initNSFWClassifier'
    caffe_transformer, nsfw_net = initNSFWClassifier()
    print 'caffe_transformer', caffe_transformer
    print 'nsfw_net', nsfw_net

    print 'initGenderClassifier'
    gender_net, gender_list = initGenderClassifier()
    print 'gender_net', gender_net
    print 'gender_list', gender_list

    print 'getS3Bucket'
    bucket, s3_bucket_name = getS3Bucket()
    print 'bucket', bucket
    print 'starting server'
    runServer(bucket, s3_bucket_name)


""" USAGE

docker run -v $(pwd)/classifier:/workspace -v $(pwd)/monitor:/images -p 3005:80 bvlc/caffe:cpu python classifier.py

docker run -v $(pwd)/classifier:/workspace -v $(pwd)/monitor:/images -p 3005:80 -it bvlc/caffe:cpu bash

"""
