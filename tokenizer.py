from  SmartAITool.core import cprint
from tokenizers import Tokenizer, models, pre_tokenizers, decoders


def get_tokenizer(save_path = 'tokenizer.json'):

    tokenizer = Tokenizer(models.BPE())
    
    tokenizer.add_special_tokens(["▁"]) #its for CTC
    all_charactors = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ ")
    tokenizer.add_tokens(all_charactors) #character levels A to Z and space 
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space = True)
    tokenizer.decoder = decoders.ByteLevel()

    tokenizer.blank_token = "▁"
    tokenizer.blank_token_id = tokenizer.token_to_id("▁")
    tokenizer.save(save_path)

    print("running is finished")



get_tokenizer()



