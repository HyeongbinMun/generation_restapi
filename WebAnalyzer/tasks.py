# -*- coding: utf-8 -*-
import cv2
from AnalysisEngine.config import DEBUG
from AnalysisEngine.celerys import app
from celery.signals import worker_init, worker_process_init
from billiard import current_process
from utils import Logging

import sys
sys.path.append("/workspace/DeepfakeBench/training")

@worker_init.connect
def model_load_info(**__):
    print(Logging.i("===================="))
    print(Logging.s("Worker Analyzer Initialize"))
    print(Logging.s("===================="))

@worker_process_init.connect
def module_load_init(**__):
    global ucf
    global spsl
    global npr

    if not DEBUG:
        worker_index = current_process().index
        print(Logging.i("====================\n"))
        print(Logging.s("Worker Id: {0}".format(worker_index)))
        print(Logging.s("===================="))

    # TODO:
    #   - Add your model
    #   - You can use worker_index if you need to get and set gpu_id
    #       - ex) gpu_id = worker_index % TOTAL_GPU_NUMBER
    from DeepfakeBench.training.inference.infer_ucf_class import UCF
    from DeepfakeBench.training.inference.infer_spsl_class import SPSL
    from DeepfakeBench.training.inference.infer_npr_class import NPR
    # yolov7 = YOLOv7()
    ucf = UCF()
    spsl = SPSL()
    npr = NPR()


@app.task(acks_late=True, queue='WebAnalyzer', routing_key='webanalyzer_tasks')
def analyzer_by_image(file_path, model_name):
    # image = cv2.imread(file_path)
    # image = cv2.resize(image, (1280, 1280), interpolation=cv2.INTER_AREA)
    print('model name : ', model_name)
    if model_name == 'ucf':
        results, out_images = ucf.run_inference(file_path)
    elif model_name == 'spsl':
        results, out_images = spsl.run_inference(file_path)
    elif model_name == 'npr':
        results, out_images = npr.run_inference(file_path)
    return results, out_images