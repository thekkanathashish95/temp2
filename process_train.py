#Importing training data
test_record_fname = '/content/valid/Food.tfrecord'
train_record_fname = '/content/train/Food.tfrecord'
label_map_pbtxt_fname = '/content/train/Food_label_map.pbtxt'


#Model Configuration
MODELS_CONFIG = 

{
    'efficientdet-d0': {
        'model_name': 'efficientdet_d0_coco17_tpu-32',
        'base_pipeline_file': 'ssd_efficientdet_d0_512x512_coco17_tpu-8.config',
        'pretrained_checkpoint': 'efficientdet_d0_coco17_tpu-32.tar.gz',
        'batch_size': 8
    }
}


chosen_model = 'efficientdet-d0'

#Number of training steps
num_steps = 5000

#Number of steps after which evaluation should occur
num_eval_steps = 500


model_name = MODELS_CONFIG[chosen_model]['model_name']
pretrained_checkpoint = MODELS_CONFIG[chosen_model]['pretrained_checkpoint']
base_pipeline_file = MODELS_CONFIG[chosen_model]['base_pipeline_file']
batch_size = MODELS_CONFIG[chosen_model]['batch_size']


#Identifying the number of classes from label map file.
def get_num_classes(pbtxt_fname):
    from object_detection.utils import label_map_util
    label_map = label_map_util.load_labelmap(pbtxt_fname)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=90, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return len(category_index.keys())
    
num_classes = get_num_classes(label_map_pbtxt_fname)

#Preparing for training
pipeline_file = '/content/models/research/deploy/pipeline_file.config'
model_dir = '/content/training/'


#Training custom object detection model
!python /content/models/research/object_detection/model_main_tf2.py \
    --pipeline_config_path={pipeline_file} \
    --model_dir={model_dir} \
    --alsologtostderr \
    --num_train_steps={num_steps} \
    --sample_1_of_n_eval_examples=1 \
    --num_eval_steps={num_eval_steps}
