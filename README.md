# ASL-2-Text
<h1>Project Report</h1>
<h1> ASL Classification</h1>
To recreate the 2D CNN model that has been used for this tool:<br>
1. Run the create_dataset.py file in asl_classification/2d_cnn. Ensure that the data direrctory is correct. If this is executed successfully, a pickle file of the data will be created. <br>
2. Next step is Training the 2D CNN. Run the train_2D_CNN.py file. Ensure that the data directory is correct. This will train and save the model to the cnn_model_2d.p. This is the tfinal trained model. <br>
3. Use this model to detect characters in realtime by running the image_classifier_cnn.py.<br>

Similarly, to reproduce other models like randomforest and svm, run the train_svm and train_model.py.
<h1>Sentence Generation</h1>
The sentence generation code can be found int he sentence_generation folder. It comprises of 2 python notebooks. The Pegasus_Tuner007 python notebook implements the fine tuned tuner007 Pegasus model. The Pegasus+MRPC python notebook has code for fine tuning a Pegasus model on the MRPC dataset. The trained model has been saved in the pegasus-fine-tuned-model-mrpc folder. Steps to reproduce the code are mentioned in the python notebooks.


<h1> Details of implementation are included in the code as comments.</h1>
