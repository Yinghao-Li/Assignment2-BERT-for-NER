"""
# Author: Yinghao Li
# Modified: October 16th, 2023
# ---------------------------------------
# Description: collate function for batch processing
"""

import torch
from transformers import DataCollatorForTokenClassification

from .batch import unpack_instances, Batch


class DataCollator(DataCollatorForTokenClassification):
    def __call__(self, instance_list: list[dict]):
        tk_ids, attn_masks, lbs = unpack_instances(instance_list, ["bert_tk_ids", "bert_attn_masks", "bert_lbs"])

        # Update `tk_ids`, `attn_masks`, and `lbs` to match the maximum length of the batch.
        # The updated type of the three variables should be `torch.int64``.
        # Hint: some functions and variables you may want to use: `self.tokenizer.pad()`, `self.label_pad_token_id`.
        # --- TODO: start of your code ---

        padded_inputs = self.tokenizer.pad({"input_ids": tk_ids, "attention_mask": attn_masks})
        tk_ids = torch.tensor(padded_inputs.input_ids, dtype=torch.int64)
        attn_masks = torch.tensor(padded_inputs.attention_mask, dtype=torch.int64)

        max_len = tk_ids.shape[1]

        # `padding_side` is right for distilbert
        padding_side = self.tokenizer.padding_side
        if padding_side == "right":
            lbs = torch.stack(
                [torch.cat((lb, torch.full((max_len - len(lb),), self.label_pad_token_id)), dim=0) for lb in lbs]
            )
        else:
            lbs = torch.stack(
                [torch.cat((torch.full((max_len - len(lb),), self.label_pad_token_id), lb), dim=0) for lb in lbs]
            )

        # --- TODO: end of your code ---

        return Batch(input_ids=tk_ids, attention_mask=attn_masks, labels=lbs)
