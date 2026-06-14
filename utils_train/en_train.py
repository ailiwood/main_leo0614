from tqdm import tqdm
from utils_tools.metricsTop import MetricsTop
from utils_models.ours_model import rob_d2v_MATF
import random
import numpy as np
from utils_tools.data_loader import data_loader
from utils_tools.tools import *


class TrainConfig(object):
    """Configuration class to store the configurations of training.
    """
    def __init__(self,
                train_mode='regression',
                 # loss_weights={'M': 0.8, 'T': 0.1, 'A': 0.05, 'F': 0.05},
                # loss_weights={'M': 0.5, 'T':0.425, 'A':0.07, 'F':0.005,},
                #  loss_weights={'M': 0.4, 'T': 0.4, 'A': 0.15, 'F': 0.05},
                 loss_weights={'M': 0.357, 'T': 0.357, 'A': 0.25, 'F': 0.005},
                 # loss_weights={'M': 0.6, 'T': 0.2, 'A': 0.15, 'F': 0.05},
                 model_save_path='checkpoint/',
                 learning_rate=1e-5,
                 epochs=None,
                 dataset_name='mosi',
                 early_stop=50,
                 dropout=0.3,
                 batch_size=16,
                 multi_task=True,
                 num_hidden_layers=1,
                 tasks='MTAF',
                 context=True,
                 text_context_len=2,
                 audio_context_len=1,
                ):
        self.train_mode = train_mode
        self.loss_weights = loss_weights
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.dataset_name = dataset_name
        self.model_save_path = model_save_path
        self.early_stop = early_stop
        self.seed = 1
        self.dropout = dropout
        self.batch_size = batch_size
        self.multi_task = multi_task
        self.num_hidden_layers = num_hidden_layers
        self.tasks = tasks
        self.context = context
        self.text_context_len = text_context_len
        self.audio_context_len = audio_context_len


class EnTrainer():
    def __init__(self, config):
        self.config = config
        self.criterion = nn.L1Loss() if config.train_mode == 'regression' else nn.CrossEntropyLoss()
        self.metrics = MetricsTop(config.train_mode).getMetics(config.dataset_name)
        self.tasks = config.tasks
        
    def do_train(self, model, data_loader):    
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate)

        total_loss = 0
        # Loop over all batches.         
        for batch in tqdm(data_loader):                    
            text_inputs = batch["text_tokens"].to(device)
            text_mask = batch["text_masks"].to(device)
            text_context_inputs = batch["text_context_tokens"].to(device)
            text_context_mask = batch["text_context_masks"].to(device)
            audio_inputs = batch["audio_inputs"].to(device)
            audio_mask = batch["audio_masks"].to(device)
            audio_context_inputs = batch["audio_context_inputs"].to(device)
            audio_context_mask = batch["audio_context_masks"].to(device)

            # x_squeezed = x.squeeze(1)
            frame__embedding = batch["frame__embedding"].squeeze(1).to(device)
            targets = batch["targets"].to(device).view(-1, 1)
            optimizer.zero_grad()
            if self.config.context:
                outputs = model(text_inputs, text_mask, text_context_inputs, text_context_mask,
                                audio_inputs, audio_mask, audio_context_inputs, audio_context_mask,
                                frame__embedding)
            else:
                outputs = model(text_inputs, text_mask, audio_inputs, audio_mask)
            # Compute the training loss.
            loss = 0.0
            for m in self.tasks:
                sub_loss = self.config.loss_weights[m] * self.criterion(outputs[m], targets)
                loss += sub_loss
                total_loss += loss.item()*text_inputs.size(0)

            loss.backward()                   
            optimizer.step()                
                
        total_loss = round(total_loss / len(data_loader.dataset), 4)
#         print('TRAIN'+" >> loss: ",total_loss)
        return total_loss

    def do_test(self, model, data_loader, mode):
        model.eval()   # Put the model in eval mode.
        y_pred = []
        y_true = []
        total_loss = 0
        val_loss = {
            'M': 0,
            'T': 0,
            'A': 0,
            'F': 0,
        }

        with torch.no_grad():
            for batch in tqdm(data_loader):                                                      # Loop over all batches.
                text_inputs = batch["text_tokens"].to(device)
                text_mask = batch["text_masks"].to(device)
                text_context_inputs = batch["text_context_tokens"].to(device)
                text_context_mask = batch["text_context_masks"].to(device)

                audio_inputs = batch["audio_inputs"].to(device)
                audio_mask = batch["audio_masks"].to(device)
                audio_context_inputs = batch["audio_context_inputs"].to(device)
                audio_context_mask = batch["audio_context_masks"].to(device)
                frame__embedding = batch["frame__embedding"].to(device)

                targets = batch["targets"].to(device).view(-1, 1)

                if self.config.context:
                    outputs = model(text_inputs, text_mask, text_context_inputs, text_context_mask, audio_inputs,
                                    audio_mask, audio_context_inputs, audio_context_mask, frame__embedding)
                else:
                    outputs = model(text_inputs, text_mask, audio_inputs, audio_mask)
                
                # Compute loss.
                loss = 0.0
                for m in self.tasks:
                    sub_loss = self.config.loss_weights[m] * self.criterion(outputs[m], targets)
                    loss += sub_loss
                    val_loss[m] += sub_loss.item() * text_inputs.size(0)
                total_loss += loss.item() * text_inputs.size(0)

                # add predictions
                y_pred.append(outputs['M'].cpu())
                y_true.append(targets.cpu())

        total_loss = round(total_loss / len(data_loader.dataset), 4)
        print(mode + " >> loss: ", total_loss)

        pred, true = torch.cat(y_pred), torch.cat(y_true)
        eval_results = self.metrics(pred, true)
        print('%s: >> ' % 'Fusion' + dict_to_str(eval_results))
        eval_results['Loss'] = total_loss
        
        return eval_results


def TrainRun(config):
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.backends.cudnn.deterministic = True

    train_loader, test_loader, val_loader = data_loader(config.batch_size, config.dataset_name,
                                                        text_context_length=config.text_context_len, audio_context_length=config.audio_context_len)

    model = rob_d2v_MATF(config).to(device)
    for param in model.data2vec_model.feature_extractor.parameters():
        param.requires_grad = False

    trainer = EnTrainer(config)

    lowest_eval_loss = 100
    highest_eval_acc = 0
    best_epoch = 0
    for epoch in range(config.epochs):
        print('---------------------EPOCH: ', epoch+1, '--------------------')
        trainer.do_train(model, train_loader)
        eval_results = trainer.do_test(model, val_loader, "VAL")

        if eval_results['Loss'] < lowest_eval_loss:
            lowest_eval_loss = eval_results['Loss']
            torch.save(model, config.model_save_path+f'RH_loss_{config.dataset_name}_{config.seed}_{lowest_eval_loss}.pth')
            best_epoch = epoch
        if eval_results['Has0_acc_2'] >= highest_eval_acc:
            highest_eval_acc = eval_results['Has0_acc_2']
            # torch.save(model, config.model_save_path+f'RH_acc_{config.dataset_name}_{config.seed}_{highest_eval_acc}.pth')
        if epoch - best_epoch >= config.early_stop:
            break
    model.load_state_dict(torch.load(config.model_save_path+f'RH_acc_{config.dataset_name}_{config.seed}_{highest_eval_acc}.pth'))
    test_results_loss = trainer.do_test(model, test_loader, "TEST")
    print('%s: >> ' %('TEST (highest val acc) ') + dict_to_str(test_results_loss))

    model.load_state_dict(torch.load(config.model_save_path+f'RH_loss_{config.dataset_name}_{config.seed}_{lowest_eval_loss}.pth'))
    test_results_acc = trainer.do_test(model, test_loader, "TEST")
    print('%s: >> ' %('TEST (lowest val loss) ') + dict_to_str(test_results_acc))
