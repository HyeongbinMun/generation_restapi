# -*- coding: utf-8 -*-
import cv2
from AnalysisEngine.config import DEBUG
from AnalysisEngine.celerys import app
from celery.signals import worker_init, worker_process_init
from billiard import current_process
from utils import Logging

import sys, os
sys.path.append("/workspace/model/FaceAdapter")
print("DEBUG: sys.path is:", sys.path)

@worker_init.connect
def model_load_info(**__):
    print(Logging.i("===================="))
    print(Logging.s("Worker Analyzer Initialize"))
    print(Logging.s("===================="))

@worker_process_init.connect
def module_load_init(**__):
    global apt

    if not DEBUG:
        worker_index = current_process().index
        print(Logging.i("====================\n"))
        print(Logging.s("Worker Id: {0}".format(worker_index)))
        print(Logging.s("===================="))

    # TODO:
    #   - Add your model
    #   - You can use worker_index if you need to get and set gpu_id
    #       - ex) gpu_id = worker_index % TOTAL_GPU_NUMBER
    from model.FaceAdapter.infer_class import FaceAdapter

    apt = FaceAdapter()


@app.task(acks_late=True, queue='WebAnalyzer', routing_key='webanalyzer_tasks')
def analyzer_by_image(source_path, target_path ,model_name):
    # image = cv2.imread(file_path)
    # image = cv2.resize(image, (1280, 1280), interpolation=cv2.INTER_AREA)
    print('model name : ', model_name)
    if model_name == 'adapter':
        out_images = apt.run_inference(source_path, target_path)
    elif model_name == 'spsl':
        pass
    elif model_name == 'npr':
        pass
    return out_images