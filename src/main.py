import bert
import os
model_dir = "../models/uncased_L-8_H-512_A-8"

bert_params = bert.params_from_pretrained_ckpt(model_dir)
l_bert = bert.BertModelLayer.from_params(bert_params, name="bert")