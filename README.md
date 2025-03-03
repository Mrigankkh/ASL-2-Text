# ASL-2-Text
<h1>Project Report</h1>

![Project_Report_page-0001](https://github.com/user-attachments/assets/e1c0bbd2-eb06-42dd-9a03-e8f939082b8f)
![Project_Report_page-0002](https://github.com/user-attachments/assets/b8681af8-a3f5-4682-9022-f6fd49fbcb29)
![Project_Report_page-0003](https://github.com/user-attachments/assets/18f00b2a-6a02-47a1-a703-5c94fd240a3c)
![Project_Report_page-0004](https://github.com/user-attachments/assets/6d75f45a-6c06-4a92-b448-e52d6fbd4662)
![Project_Report_page-0005](https://github.com/user-attachments/assets/1cfd4259-da62-419f-88a4-1ddf3e62ce58)


<h1> ASL Classification</h1>
To recreate the 2D CNN model that has been used for this tool:<br>
1. Run the create_dataset.py file in asl_classification/2d_cnn. Ensure that the data direrctory is correct. If this is executed successfully, a pickle file of the data will be created. <br>
2. Next step is Training the 2D CNN. Run the train_2D_CNN.py file. Ensure that the data directory is correct. This will train and save the model to the cnn_model_2d.p. This is the tfinal trained model. <br>
3. Use this model to detect characters in realtime by running the image_classifier_cnn.py.<br>

Similarly, to reproduce other models like randomforest and svm, run the train_svm and train_model.py.
<h1>Sentence Generation</h1>
The sentence generation code can be found int he sentence_generation folder. It comprises of 2 python notebooks. The Pegasus_Tuner007 python notebook implements the fine tuned tuner007 Pegasus model. The Pegasus+MRPC python notebook has code for fine tuning a Pegasus model on the MRPC dataset. The trained model has been saved in the pegasus-fine-tuned-model-mrpc folder. Steps to reproduce the code are mentioned in the python notebooks.


<h1> Details of implementation are included in the code as comments.</h1>
