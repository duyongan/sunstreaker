# -*- coding: utf-8 -*-
# @Time    : 2022/11/26 19:35 
# @Author  : duyongan
# @Email   : 13261051171@163.com
# @phone   : 13261051171
import os
import copy
import json
import numpy as np
import jax.numpy as jnp
from sunstreaker import Model
from typing import Union, List, Dict, Optional, Tuple, Any
from sunstreaker import initializers
from sunstreaker import activations
from sunstreaker.engine.input_layer import Input
from sunstreaker.activations import Activation
from sunstreaker.layers import Add, Dense, Layer, Dropout, Embedding, LayerNormalization, Lambda, \
    MultiHeadAttention

CONFIG_NAME = "config.json"


class BertConfig:
    model_type = "bert"

    def __init__(
            self,
            vocab_size=30522,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=512,
            type_vocab_size=2,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            pad_token_id=0,
            position_embedding_type="absolute",
            use_cache=True,
            classifier_dropout=None,
            **kwargs
    ):
        self.pad_token_id = pad_token_id
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout

    @classmethod
    def _dict_from_json_file(cls, json_file: Union[str, os.PathLike]):
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return json.loads(text)

    @classmethod
    def from_json_file(cls, json_file: Union[str, os.PathLike]):
        config_dict = cls._dict_from_json_file(json_file)
        return cls(**config_dict)

    def to_json_file(self, json_file_path: Union[str, os.PathLike]):
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string())

    def to_json_string(self) -> str:
        config_dict = self.to_dict()
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

    def to_dict(self) -> Dict[str, Any]:
        output = copy.deepcopy(self.__dict__)
        if hasattr(self.__class__, "model_type"):
            output["model_type"] = self.__class__.model_type
        return output

    def update(self, config_dict: Dict[str, Any]):
        for key, value in config_dict.items():
            setattr(self, key, value)


class BertEmbedings(Layer):
    config: BertConfig
    dtype: jnp.dtype = jnp.float32

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.deterministic: bool = True

    def build(self):
        self.word_embeddings = Embedding(
            vocabulary_size=self.config.vocab_size,
            dimension=self.config.hidden_size,
            initializer="", )
        self.position_embeddings = Embedding(
            self.config.max_position_embeddings,
            self.config.hidden_size,
            initializer="", )
        self.token_type_embeddings = Embedding(
            self.config.type_vocab_size,
            self.config.hidden_size,
            initializer="", )
        self.LayerNorm = LayerNormalization(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        self.dropout = Dropout(rate=self.config.hidden_dropout_prob, deterministic=True)
        return self.config.max_position_embeddings, self.config.hidden_size

    def __call__(self, inputs, **kwargs):
        input_ids, position_ids, token_type_ids = inputs
        inputs_embeds = self.word_embeddings(input_ids.astype(jnp.int32))
        position_embeds = self.position_embeddings(position_ids.astype(jnp.int32))
        token_type_embeddings = self.token_type_embeddings(token_type_ids.astype(jnp.int32))
        hidden_states = inputs_embeds + token_type_embeddings + position_embeds
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class BertPreTrainingHeads(Layer):
    config: BertConfig
    dtype: jnp.dtype = jnp.float32
    bias_init = initializers.Zeros()

    def __init__(self,
                 shared_embedding=None,
                 with_pool=False,
                 with_nsp=False,
                 with_mlm=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.shared_embedding = shared_embedding
        self.with_pool = with_pool
        self.with_nsp = with_nsp
        self.with_mlm = with_mlm
        if self.with_nsp and not self.with_pool:
            self.with_pool = True

    def build(self):
        self.dense_pool = Dense(
            self.config.hidden_size,
            kernel_init="RandomNormal",
            dtype=self.dtype,
            activation="tanh"
        )
        self.dense = Dense(self.config.hidden_size, dtype=self.dtype, activation=self.config.hidden_act, use_bias=False)
        self.LayerNorm = LayerNormalization(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        self.decoder = Dense(self.config.vocab_size, dtype=self.dtype, use_bias=False)
        self.seq_relationship = Dense(2, dtype=self.dtype, use_bias=True)
        return ()

    def __call__(self, inputs, **kwargs):
        hidden_states = inputs
        cls_hidden_state = hidden_states[:, 0]
        pooled_output = self.dense_pool(cls_hidden_state)

        hidden_states = self.dense(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        if self.shared_embedding is not None:
            hidden_states = self.decoder.forward({"params": {"kernel": self.shared_embedding.T}}, hidden_states)
        else:
            hidden_states = self.decoder(hidden_states)
        prediction_scores = hidden_states
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class BertLayer(Layer):
    config: BertConfig
    dtype: jnp.dtype = jnp.float32

    def build(self):
        self.attention = MultiHeadAttention(heads=self.config.num_attention_heads,
                                            head_size=self.config.hidden_size // self.config.num_attention_heads)
        self.dropout = Dropout(rate=self.config.attention_probs_dropout_prob)
        self.add1 = Add()
        self.LayerNorm1 = LayerNormalization()
        self.feedForward = Dense(units=self.config.intermediate_size, activations="relu")
        self.add2 = Add()
        self.LayerNorm2 = LayerNormalization()
        return self.input_shape

    def __call__(self, inputs, **kwargs):
        xi = inputs
        x = self.attention(inputs)
        x = self.dropout(x)
        x = self.add1([x, xi])
        x = self.LayerNorm1(x)
        xi = x
        x = self.feedForward(x)
        x = self.add2([x, xi])
        x = self.LayerNorm2(x)
        return x


class BertLayerCollection(Layer):
    config: BertConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def __init__(self,
                 encoder_hidden_states: Optional[jnp.ndarray] = None,
                 encoder_attention_mask: Optional[jnp.ndarray] = None,
                 init_cache: bool = False,
                 deterministic: bool = True,
                 output_attentions: bool = False,
                 output_hidden_states: bool = False,
                 return_dict: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.encoder_hidden_states = encoder_hidden_states
        self.encoder_attention_mask = encoder_attention_mask
        self.init_cache = init_cache
        self.deterministic = deterministic
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.output_hidden_states = output_hidden_states
        self.return_dict = return_dict

    def build(self):
        self.layers = [BertLayer(self.config,
                                 name=str(i),
                                 dtype=self.dtype,
                                 ) for i in range(self.config.num_hidden_layers)]

    def call(self, inputs, **kwargs):
        hidden_states, attention_mask, head_mask = inputs
        all_attentions = () if self.output_attentions else None
        all_hidden_states = () if self.output_hidden_states else None
        all_cross_attentions = () if (self.output_attentions and self.encoder_hidden_states is not None) else None
        for i, layer in enumerate(self.layers):
            if self.output_hidden_states:
                all_hidden_states += (hidden_states,)
            layer_outputs = layer.forward(self.params, (hidden_states, attention_mask, head_mask))
            hidden_states = layer_outputs[0]
            if self.output_attentions:
                all_attentions += (layer_outputs[1],)
                if self.encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)
        if self.output_hidden_states:
            all_hidden_states += (hidden_states,)
        outputs = (hidden_states, all_hidden_states, all_attentions, all_cross_attentions)
        if not self.return_dict:
            return tuple(v for v in outputs if v is not None)
        return outputs


class BertModel(Model):
    model = None

    def build(self, rng=None):
        ...

    def call(self, inputs, trainable=True, **kwargs):
        ...

    @classmethod
    def from_pretrained(cls, path_or_name, *model_args, **model_kwargs):
        import torch
        from huggingface_hub import snapshot_download
        if not os.path.exists(path_or_name):
            path_or_name = snapshot_download(repo_id="bert-base-chinese")
        pt_state_dict = torch.load(path_or_name, map_location="cpu")
        state = {k: jnp.array(pt_state_dict[k]) for k in pt_state_dict}

        return state
