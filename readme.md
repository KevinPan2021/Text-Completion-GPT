Introduction:
	This project aims to preform Text Generation using Decoder only Transformer (GPT2). This serves as the foundation for fine-tuning GPT on Question and Answering Tasks and later few shots learning (GPT3).



Dataset: 
	https://www.kaggle.com/datasets/aladdinpersson/pascal-voc-dataset-used-in-yolov3-video/data


Build: 
	M1 Macbook Pro
	Miniforge 3 (Python 3.9)
	PyTorch version: 2.2.1

* Alternative Build:
	Windows (NIVIDA GPU)
	Anaconda 3
	PyTorch



Generate ".py" file from ".ui" file:
	1) open Terminal. Navigate to directory
	2) Type "pyuic5 -x qt_main.ui -o qt_main.py"



Core Project Structure:
	GUI.py (Run to generate a GUI)
	main.py (Run to train model)
	model_GPT2.py
	qt_main.py
	training.py
	visualization.py


Credits:
	Transformer model is referenced from "https://www.youtube.com/watch?v=kCc8FmEb1nY&ab_channel=AndrejKarpathy"
	