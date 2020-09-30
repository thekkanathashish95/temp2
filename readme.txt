The first time we train the model, we will clone and install the Tensorflow Object Detection Dependencies and API. This import of file structure and all the associated utility and helper functions will help us to train and retrain the model as per our convenience. The first round of training will also help us to generate the pipeline configuration file and downloads the pretrained checkpoint of the base model for that we can reuse for retraining the model. 

The first task for the retraining process would be to convert and import data in the form of TFRecord. CVAT has an option to dump annotation directly as TFRecord. Else, create_pascal_tf_record.py file can be used for the same purpose. 

After conversion, codes in process_train.py file will do the retraining part of the process. We will be reusing the pipeline configuration file, label map from the initial training for this purpose. 

After successful completion of the training, we can use codes in saving_model.py file to export our new trained model to fine tuned model folder for implementation.

We can implement the rest of the implemenation using model_run.py file we built already.
