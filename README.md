# NLP_PROJECT_VISDIAL


*This repository contains 4 folders.Firstly, we require the twitter dataset in form of a csv file to be used.
We extract data from twitter and then preprocess it using MM data collection.py* 

*The training ,evaluation and the model architecture could be found in the language models folders.*

*Before training the model,you need to extract visual features from the dataset using any of the convolutional neural netowrk or faster rcnn or visual transformer based on your needs.The code sample for the same is present in visual features folder.*

*Data splitting, creating the part of speech sequences need to be done before training as well.The code is available in the pre processing folder itself.*

*The model architectures used are sequence to sequence and transformer model with modifications in each of them.Only the baseline code has been provided here.
Modifications include, with and without pre processed part of speech tags, with and without attention, and using various visual features including local and global.
For the transformer part, we have used modifications such as combining the local and the global features, using only the local features and visual transformer + textual transformer.*

*At the end you may need to clean the feedbacks by removing the redundant sentences and rows from the csv before quantitative evaluation.Use the code of Post processing folder for the same.*


*Link to dataset [https://drive.google.com/drive/u/3/folders/1OKR5RN7HH8hAivCFvdJsXshygna1ffl9]*

*Link to csv file [https://drive.google.com/file/d/1VzEcp0UK4LAah0Z2gkDRE9lz1OW2SK0w/view?usp=sharing]*

*Link to Resnet visual features [https://drive.google.com/file/d/15q-SS_Juon_uLwd4eV2jQb0Q8i3pJqX2/view?usp=sharing]*

*For visual transformer features extraction, you may need to run the following command. $pip install -q git+https://github.com/huggingface/transformers.git$*
