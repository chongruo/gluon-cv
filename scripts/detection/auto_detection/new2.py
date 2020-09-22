import autogluon as ag

from gluoncv.auto.estimators.ssd import SSDEstimator
from gluoncv.auto.estimators.faster_rcnn import FasterRCNNEstimator
from gluoncv.auto.estimators.yolo import YOLOEstimator
from gluoncv.auto.tasks.object_detection import ObjectDetection


if __name__ == '__main__':
    args = {
        #'dataset': 'voc',
        #'dataset': 'comic',
        #'dataset': 'watercolor',
        'dataset': 'clipart',
        #'dataset': 'voc_tiny',
        'meta_arch': 'yolo3',
        #'meta_arch': 'ssd',
        #'backbone': ag.Categorical('resnet18_v1', 'resnet50_v1'),
        'backbone': 'mobilenet1.0',
        'lr': 1e-3,
        #'lr': 0,
        'epochs': 100,
        'lr_decay_epoch': "80,90",
        'data_shape': 320,
        'warmup_epochs': 4,
        'syncbn': True,
        'num_trials': 1,
        'batch_size': 64,
        'custom_model': False,
        'save_interval': 50,
    }

    if args['meta_arch'] == 'ssd':
        estimator = SSDEstimator
    elif args['meta_arch'] == 'faster_rcnn':
        estimator = FasterRCNNEstimator
    elif args['meta_arch'] == 'yolo3':
        estimator = YOLOEstimator
    elif args['meta_arch'] == 'center_net':
        estimator = CenterNetEstimator
    else:
        estimator = None

    task = ObjectDetection(args, estimator)

    detector = task.fit()
    test_map = detector.evaluate()
    print("mAP on test dataset: {}".format(test_map[-1][-1]))
    print(test_map)
    detector.save('final_model.model')
