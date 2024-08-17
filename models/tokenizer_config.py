from transformers import AutoTokenizer

class TokenizerConfig:
    def __init__(self, model_name='gpt2', padding_type='max_length', max_length=512, add_special_tokens=True):
        """
        Initialize the tokenizer with specific configurations.
        
        Args:
        model_name (str): Name of the model to load tokenizer for.
        padding_type (str): Type of padding to be used ('max_length', 'longest').
        max_length (int): Maximum length of the tokens.
        add_special_tokens (bool): Whether or not to encode special tokens.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.padding_type = padding_type
        self.max_length = max_length
        self.add_special_tokens = add_special_tokens

        # For models that require a specific token for padding (e.g., GPT, GPT-2)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    def tokenize_inputs(self, texts):
        """
        Tokenize a list of texts according to the initialized configurations.

        Args:
        texts (list of str): List of texts to tokenize.

        Returns:
        dict: A dictionary with input_ids and attention_mask.
        """
        return self.tokenizer(
            texts,
            padding=self.padding_type,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
            add_special_tokens=self.add_special_tokens
        )
