import random
import tqdm
import numpy as np

import torch
import torch.optim as optim
from AutoRegressiveWrapper import AutoRegressiveWrapper
from models import GPT
import Utils
import sys
import math
import os

#-----constants-------
NUM_BATCHES = int(1e6)
BATCH_SIZE = 16
GRADIENT_ACCUMULATE_EVERY = 1
LEARNING_RATE = 3e-4
VALIDATE_EVERY = 500
GENERATE_EVERY = 500
GENERATE_LENGTH = 512
SEQ_LENGTH = 1024
RESUME_TRAINING = False # set to false to start training from beginning
#----------------------

def decode_token(token): # convert token to character
    return str(chr(max(32,token)))

def decode_tokens(tokens): # convert sequence of characters to tokens
    return ''.join(list(map(decode_token, tokens)))

def count_parameters(model): # count number of trainable parameters in the model
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def configure_optimizers(gpt):
    """
    This long function is unfortunately doing something very simple and is being very defensive:
    We are seperating out all parameters of the model into two buckets: those that will experience
    wight decay for regularization and those that won't (biases, and layernorm/embedding weights).
    We are then returning the PyTorch optimizer object.
    """

    # seperate out all parameters to those that will and won't experience regularization weight decay
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear, )
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
    for mn, m in gpt.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn,pn) if mn else pn # full param name
            # random note: because named_modules and named_parameters are recursive
            # we will see the same tensors p many many times. but doing it this way
            # allows us to know which parent module any tensor p belongs to...
            if pn.endswith('bias'):
                # all biases will not be displayed
                no_decay.add(fn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                no_decay.add(fpn)

    # validate that we considered every parameters
    param_dict = {pn: p for pn, p in gpt.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
    assert len(param_dict.keys() - union_params) == 0, "parameters %s were not seperated into either decay/no_decay set!" \
        % (str(param_dict.keys() - union_params), )

    # create the pytorch optimizer object
    optim_groups = [
        {"params" : [param_dict[pn] for pn in sorted(list(decay))], "weight_decay":0.1},
        {"params" : [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay" : 0.0,}
        ]
    optimizer = torch.optim.AdamW(optim_groups, lr=LEARNING_RATE, betas=(0.9,0.95))
    return optimizer
def main():
    
    gpt = GPT(
        dim = 512, # embedding
        num_unique_tokens=256, # for character level modeling
        num_layers=8,
        heads=8,
        max_seq_len=SEQ_LENGTH,       
    )

    model = AutoRegressiveWrapper(gpt)
    model.cuda()
    pcount = count_parameters(model)
    print("count of parameters in the model = ", pcount/1e6, " million")

    train_loader, val_loader, val_dataset = Utils.get_loaders_enwiki8(SEQ_LENGTH, BATCH_SIZE)
    #optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE) # optimizer
    optim = configure_optimizers(model)

    # ------------- training ---------------
    if RESUME_TRAINING == False:
        start = 0
    else:
        checkpoint_data = torch.load('checkpoint/mingpt_model.pt')
        model.load_state_dict(checkpoint_data['state_dict'])
        optim.load_state_dict(checkpoint_data['optimizer'])
        start = checkpoint_data['epoch']

    for i in tqdm.tqdm(range(start,NUM_BATCHES), mininterval=10, desc = 'training'):
        model.train()
        total_loss = 0
        for __ in range(GRADIENT_ACCUMULATE_EVERY):
            loss = model(next(train_loader))
            loss.backward()
        if (i%100==0):
            print(f'training loss: {loss.item()} -- iteration = {i}')

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        optim.zero_grad()

        if i % VALIDATE_EVERY == 0:
            model.eval()
            total_len2 = 0
            total_loss2 = 0
            val_count = 1000  # number of validations to compute average BPC
            with torch.no_grad():
                for v in range(0, val_count):
                    loss = model(next(val_loader))
                    total_loss += loss.item()
                    loss_m = loss.mean()
                    total_loss2 += SEQ_LENGTH * loss_m.item() #loss.float().item() #seq_len
                    total_len2 += SEQ_LENGTH
                print(f'--------validation loss: {total_loss/val_count}')
                print(f'Perplexity : {math.exp(total_loss/val_count)}, BPC: {total_loss/val_count*np.log2(2.7173)}')

                bpc2 = (total_loss2/total_len2)/math.log(2)
                print("BPC 2 = ", bpc2)
                total_loss = 0

        if i % GENERATE_EVERY == 0:
            model.eval()
            inp = random.choice(val_dataset)[:-1]
            input_start_sequence = decode_tokens(inp)
            print("-------start input--------")
            print(f'%s \n\n', (input_start_sequence))
            print("-------end of start input-------")
            sample = model.generate(inp,GENERATE_LENGTH)
            output_str = decode_tokens(sample)
            print("----------generated output------------")
            print(output_str)
            print("--------end generated output---------")

            # ------------ save the latest model -----------------------
            print("----------saving model-------------")
            checkpoint_data = {
                'epoch' : i,
                'state_dict' : model.state_dict(),
                'optimizer' : optim.state_dict()
                }
            ckpt_path = os.path.join("checkpoint/mingpt_model.pt")
            torch.save(checkpoint_data, ckpt_path)
            # revert model to training mode
            model.train()


if __name__ == "__main__":
    sys.exit(int(main() or 0))
