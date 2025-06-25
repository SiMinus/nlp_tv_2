import torch
from torch import nn
from transformers import Trainer 

class CustomTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        # 打印输入数据的键名，查看实际接收到的是什么
        print("输入数据的键名:", list(inputs.keys()))
        
        labels = inputs.get("labels")
        # 如果尝试使用"label"键获取，看是否存在
        label_single = inputs.get("label")
        print("使用'labels'获取的值:", labels is not None)
        print("使用'label'获取的值:", label_single is not None)

        outputs = model(**inputs)
        logits = outputs.get("logits")
        logits = logits.float()

        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(self.class_weights, dtype=torch.float).to(device=self.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

    def set_class_weights(self, class_weights):
        self.class_weights = class_weights
    
    def set_device(self, device):
        self.device = device