# Simple-GPT
This is a tutorial project for training a large language model
The project includes the following files:
1. dataprocessing.py: preprocesses downloaded text dataset and save them as .json files. If you have good internet collection the script will download dataset from huggingface-hub.
2. model.py: defines the structure and algorithm of model.
3. pretrain.py: pretrains the model with preprocessed dataset.
4. sft.py: finetune the model to make it a chatbot.
5. utils.py: includes some helper functions for example trainning functions and samplin functions.


Before pretrainning, please download text dataset from huggingface. Good text datasets include Wikipedia, Redpajama, e.t.c..
The model is finetuned with the ultrachat-200k dataset.
The pretrainning is done with a RTX 4060Ti 16G GPU and takes roughly a month to finish. Then the finetune procedure is followed. After these, you could obtain a simple chatbot.
Simple conversations with the chatbot:
```python
"""
<User>: I have problems when installing windows on my computer, what should I do?
<Agent>: <bos>Yes! Here are some problems when it comes to installing windows on your computer:

1. Unexpected delays - if you're not using a window or door, you may need to install it without the necessary equipment and software in place for your computer's door-to-door interface (DOI)
2. Lack of compatibility with other devices - If there are any issues with your WiFi network, you may need to install them without any compatibility issues such as USB/CPU compatibility
3. Not being able to access your WindowService connections - When installing windowservice connections on my computer's doorstep, I often require more than one connection at once that is easier than others. This can be due in particular to unnecessarily lowering your Windows Service connections if they accessed it by using the same programming language and software installed in your computer's doorstep

4. Inadequate maintenance or repaint: If there arisen some issues with Windows Servicecontroller (WSAC) systems that may not work for you, you may need more repairs and replacements as needed. These issues are oftentimes caused by poorly installed WindowServicecontroller systems, which can lead to delays in installations, repair problems, and replacement of windows on their own.
"""
```
