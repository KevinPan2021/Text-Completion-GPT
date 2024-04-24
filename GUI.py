application_name = 'Text Generation'
# pyqt packages
from PyQt5.QtWidgets import QMainWindow, QApplication

import tiktoken
import sys
import torch

from model_GPT2 import GPT2
from qt_main import Ui_Application
from main import GPU_Device, inference



class QT_Action(Ui_Application, QMainWindow):
    def __init__(self):
        # system variable
        super(QT_Action, self).__init__()
        self.setupUi(self)
        self.retranslateUi(self)
        self.setWindowTitle(application_name) # set the title
        self.mouse_pos = None
        
        # runtime variable
        self.model = None
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.output_vocab = None
        self.max_len = 64
        
        
        # load the model
        self.load_model_action()
        

            
    # linking all button/textbox with actions    
    def link_commands(self,):
        self.comboBox_model.activated.connect(self.load_model_action)
        self.toolButton_process.clicked.connect(self.process_action)
        
            
    # choosing between models
    def load_model_action(self,):
        self.model_name = self.comboBox_model.currentText()
        
        # load the model
        vocab_size = self.tokenizer.max_token_value + 1
        block_size = 128 # max sequence length
        num_embed = 384
        num_heads = 6
        num_layers = 6
        dropout = 0.0
        if self.model_name == 'GPT2':
            self.model = GPT2(vocab_size, block_size, num_embed, num_heads, num_layers, dropout)

        # loading the training model weights
        self.model.load_state_dict(torch.load(f'{self.model_name}.pth'))
            
        # move model to GPU
        self.model = self.model.to(GPU_Device())
        
        self.model.eval() # Set model to evaluation mode
    
        
    def process_action(self):
        # model inference
        out_sentence = inference(self.model, self.tokenizer, max_new_tokens=300)
        
        self.textEdit_GeneratedText.setPlainText(out_sentence)
        
        
def main():
    app = QApplication(sys.argv)
    action = QT_Action()
    action.link_commands()
    action.show()
    sys.exit(app.exec_())
    
    
if __name__ == '__main__':
    main()