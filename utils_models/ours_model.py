import torch
from torch import nn
from transformers import RobertaModel, Data2VecAudioModel
from utils_models.attention_encoder import CMELayer, BertConfig
from utils_models.DEConv import DEConv_2
from utils_models.transformer import selfTransformer
from utils_tools.tools import device


class rob_d2v_MATF(nn.Module):
    def __init__(self, config):
        super().__init__()

        """""
        RobertaModel:
        Data2VecAudioModel:
        The data type of the output:
             "last_hidden_state": tensor([[[...]]]),
             "pooler_output": tensor([[...]]),
             "hidden_states": None,
             "attentions": None
        """""

        self.roberta_model = RobertaModel.from_pretrained('roberta-large')
        self.data2vec_model = Data2VecAudioModel.from_pretrained("facebook/data2vec-audio-large-960h")

        self.T_output_predicted_layers = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(1024 * 2, 1)
        )

        self.A_output_predicted_layers = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(1024 * 2, 1)
        )

        self.F_output_predicted_layers = nn.Sequential(
            DEConv_2(32),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.T_output_layers = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(1024 * 2, 128)
        )
        self.A_output_layers = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(1024 * 2, 128)
        )
        self.F_output_layers = nn.Sequential(
            DEConv_2(32),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )

        self.F_T_A_Transformer = selfTransformer(d_model=128 * 3)
        self.fused_output_layers = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(1024 * 4 + 384, 768),
            nn.ReLU(),
            nn.Linear(768, 1)
        )

        # cls embedding layers
        self.text_cls_emb = nn.Embedding(num_embeddings=1, embedding_dim=1024)
        self.audio_cls_emb = nn.Embedding(num_embeddings=1, embedding_dim=1024)

        # CME layers
        Bert_config = BertConfig(num_hidden_layers=config.num_hidden_layers, hidden_size=1024, intermediate_size=4096,
                                 num_attention_heads=16)
        self.CME_layers = nn.ModuleList(
            [CMELayer(Bert_config) for _ in range(Bert_config.num_hidden_layers)]
        )

    def prepend_cls(self, inputs, masks, layer_name):
        if layer_name == 'text':
            embedding_layer = self.text_cls_emb
        elif layer_name == 'audio':
            embedding_layer = self.audio_cls_emb
        index = torch.LongTensor([0]).to(device=inputs.device)
        cls_emb = embedding_layer(index)
        cls_emb = cls_emb.expand(inputs.size(0), 1, inputs.size(2))
        outputs = torch.cat((cls_emb, inputs), dim=1)

        cls_mask = torch.ones(inputs.size(0), 1).to(device=inputs.device)
        masks = torch.cat((cls_mask, masks), dim=1)
        return outputs, masks

    def forward(self, text_inputs, text_mask, text_context_inputs, text_context_mask, audio_inputs, audio_mask,
                audio_context_inputs, audio_context_mask, frame__embedding):
        # frames feature extraction
        frame__embedding = frame__embedding.float()  # Shape is [batch_size, 1024]

        # text feature extraction
        raw_output = self.roberta_model(text_inputs, text_mask, return_dict=True)
        T_hidden_states = raw_output.last_hidden_state
        input_pooler = raw_output["pooler_output"]  # Shape is [batch_size, 1024]

        # text context feature extraction
        raw_output_context = self.roberta_model(text_context_inputs, text_context_mask, return_dict=True)
        T_context_hidden_states = raw_output_context.last_hidden_state
        context_pooler = raw_output_context["pooler_output"]  # Shape is [batch_size, 1024]

        # audio feature extraction
        audio_out = self.data2vec_model(audio_inputs, audio_mask, output_attentions=True)
        A_hidden_states = audio_out.last_hidden_state
        #  average over unmasked audio tokens
        A_features = []
        audio_mask_idx_new = []
        for batch in range(A_hidden_states.shape[0]):
            layer = 0
            while layer < 12:
                try:
                    padding_idx = sum(audio_out.attentions[layer][batch][0][0] != 0)
                    audio_mask_idx_new.append(padding_idx)
                    break
                except:
                    layer += 1
            truncated_feature = torch.mean(A_hidden_states[batch][:padding_idx], 0)  # Shape is [1024]
            A_features.append(truncated_feature)
        A_features = torch.stack(A_features, 0).to(device)  # Shape is [batch_size, 1024]
        audio_mask_new = torch.zeros(A_hidden_states.shape[0], A_hidden_states.shape[1]).to(device)
        for batch in range(audio_mask_new.shape[0]):
            audio_mask_new[batch][:audio_mask_idx_new[batch]] = 1

        # audio context feature extraction
        audio_context_out = self.data2vec_model(audio_context_inputs, audio_context_mask, output_attentions=True)
        A_context_hidden_states = audio_context_out.last_hidden_state
        ## average over unmasked audio tokens
        A_context_features = []
        audio_context_mask_idx_new = []
        for batch in range(A_context_hidden_states.shape[0]):
            layer = 0
            while layer < 12:
                try:
                    padding_idx = sum(audio_context_out.attentions[layer][batch][0][0] != 0)
                    audio_context_mask_idx_new.append(padding_idx)
                    break
                except:
                    layer += 1
            truncated_feature = torch.mean(A_context_hidden_states[batch][:padding_idx], 0)
            A_context_features.append(truncated_feature)
        A_context_features = torch.stack(A_context_features, 0).to(device)
        audio_context_mask_new = torch.zeros(A_context_hidden_states.shape[0], A_context_hidden_states.shape[1]).to(
            device)
        for batch in range(audio_context_mask_new.shape[0]):
            audio_context_mask_new[batch][:audio_context_mask_idx_new[batch]] = 1

        T_features = torch.cat((input_pooler, context_pooler), dim=1)
        A_features = torch.cat((A_features, A_context_features), dim=1)

        # Perform text and speech classification
        T_output_predicted = self.T_output_predicted_layers(T_features)
        A_output_predicted = self.A_output_predicted_layers(A_features)
        F_output_predicted = self.F_output_predicted_layers(frame__embedding)

        # Second Branch Modal Fusion
        T_output = self.T_output_layers(T_features)
        A_output = self.A_output_layers(A_features)
        F_output = self.F_output_layers(frame__embedding)
        A_F_T_tensor = torch.cat([T_output, A_output, F_output], dim=1)
        A_F_T_fuse_tensor = self.F_T_A_Transformer(A_F_T_tensor)

        # CME layers
        text_inputs, text_attn_mask = self.prepend_cls(T_hidden_states, text_mask, 'text')  # add cls token
        audio_inputs, audio_attn_mask = self.prepend_cls(A_hidden_states, audio_mask_new, 'audio')  # add cls token

        text_context_inputs, text_context_attn_mask = self.prepend_cls(T_context_hidden_states, text_context_mask,
                                                                       'text')  # add cls token
        audio_context_inputs, audio_context_attn_mask = self.prepend_cls(A_context_hidden_states,
                                                                         audio_context_mask_new,
                                                                         'audio')  # add cls token

        for layer_module in self.CME_layers:
            text_inputs, audio_inputs = layer_module(text_inputs, text_attn_mask,
                                                     audio_inputs, audio_attn_mask)

        for layer_module in self.CME_layers:
            text_context_inputs, audio_context_inputs = layer_module(text_context_inputs, text_context_attn_mask,
                                                                     audio_context_inputs, audio_context_attn_mask)

        # fused features
        fused_hidden_states = torch.cat((text_inputs[:, 0, :], text_context_inputs[:, 0, :], audio_inputs[:, 0, :],
                                         audio_context_inputs[:, 0, :], A_F_T_fuse_tensor),
                                        dim=1)  # Shape is [batch_size, 1024*4]

        # last linear output layer
        fused_output = self.fused_output_layers(fused_hidden_states)  # Shape is [batch_size, 1]

        return {
            'T': T_output_predicted,
            'A': A_output_predicted,
            'F': F_output_predicted,
            'M': fused_output,
        }
