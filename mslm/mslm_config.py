import copy
import json


class MSLMConfig():
    """
    Configuration for the (M)SLM model
    """
    def __init__(
        self,
        seed=1,
        clear_cache_by_batch=False,
        preserve_case=False,
        model_dim=256,
        model_dropout=0.1,
        pretrained_embedding=None,
        encoder_type="transformer",
        transformer_mask_type='cloze',
        num_encoder_layers=1,
        encoder_dim=256,
        num_heads=4,
        feedforward_dim=256,
        attention_window=None,
        encoder_dropout=0.1,
        num_decoder_layers=1,
        max_seg_length=8,
        length_exponent=2,
        length_penalty_lambda=None,
        use_lexicon=False,
        lexicon_min_count=1,
        autoencoder=False,
        batch_size=8,
        batch_by='sequences',
        max_padding=None,
        gradient_accumulation_steps=1,
        checkpoint_interval=64,
        max_train_steps=4096,
        early_stopping=None,
        optimizer_algorithm="adam",
        num_warmup_steps=512,
        warmup="linear",
        decay="linear",
        learning_rate=0.001,
        gamma=0.8,
        gamma_steps=512,
        gradient_clip=1.0
    ):
        self.seed = seed
        self.clear_cache_by_batch = clear_cache_by_batch
        self.preserve_case = preserve_case
        self.model_dim = model_dim
        self.model_dropout = model_dropout
        self.pretrained_embedding = pretrained_embedding
        self.encoder_type = encoder_type
        self.transformer_mask_type = transformer_mask_type
        self.num_encoder_layers = num_encoder_layers
        self.encoder_dim = encoder_dim
        self.num_heads = num_heads
        self.feedforward_dim = feedforward_dim
        self.attention_window = attention_window
        self.encoder_dropout = encoder_dropout
        self.num_decoder_layers = num_decoder_layers
        self.max_seg_length = max_seg_length
        self.length_exponent = length_exponent
        self.length_penalty_lambda = length_penalty_lambda
        self.use_lexicon = use_lexicon
        self.lexicon_min_count = lexicon_min_count
        self.autoencoder = autoencoder
        self.batch_size = batch_size
        self.batch_by = batch_by
        self.max_padding = max_padding
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.checkpoint_interval = checkpoint_interval
        self.max_train_steps = max_train_steps
        self.early_stopping = early_stopping
        self.optimizer_algorithm = optimizer_algorithm
        self.num_warmup_steps = num_warmup_steps
        self.warmup = warmup
        self.decay = decay
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gamma_steps = gamma_steps
        self.gradient_clip = gradient_clip

        if self.optimizer_algorithm not in ["sgd", "adam"]:
            raise ValueError(
                f"Optimizer algorithm {self.optimizer_algorithm} is not valid"
            )
        if self.warmup not in ["flat", "linear"]:
            raise ValueError(f"Warmup pattern {self.warmup} is not valid")
        if self.decay not in ["linear", "exponential"]:
            raise ValueError(f"Decay pattern {self.decay} is not valid")

    @classmethod
    def from_dict(cls, dict_object):
        """
        Constructs a `MSLMConfig` from a Python dictionary of parameters
        """
        config = MSLMConfig()
        for (key, value) in dict_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json(cls, json_file):
        """Constructs a `MSLMConfig` from a json file of parameters"""
        with open(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        """Serializes this instance to a Python dictionary"""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def to_json_file(self, json_file):
        """Serializes this instance to a JSON file."""
        with open(json_file, "w") as writer:
            writer.write(self.to_json_string())
