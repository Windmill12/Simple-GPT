
from model import RotaryGPTLanguageModelForPFinetune, get_lora_model, RotaryGPTLanguageModel
from dataprocessing import tokenizer
from utils import *


def load_or_create_pe_finetune_model(config, model_path, if_load_model):
    # PEFT
    if if_load_model:
        try:
            model = torch.load(model_path)
            print("Model loaded")
        except FileNotFoundError:
            print("Failed to load the model, try to recreate it")
            model = RotaryGPTLanguageModelForPFinetune(config["pretrained_path"], config["num_of_sft_layers"],
                                                      config["vocab_size"], config["embed_size"], config["num_heads"],
                                                      config["dropout"], config["device"])
    else:
        print("Creating new model...")
        model = RotaryGPTLanguageModelForPFinetune(config["pretrained_path"], config["num_of_sft_layers"],
                                                  config["vocab_size"], config["embed_size"], config["num_heads"],
                                                  config["dropout"], config["device"])
    return model


def load_or_create_lora_finetune_model(config, model_path, if_load_model):
    # LoRA finetune
    if if_load_model:
        try:
            model = torch.load(model_path)
            print("Model loaded")
        except FileNotFoundError:
            print("Failed to load the model, try to recreate it")
            model = torch.load(config["pretrained_path"])
            model = get_lora_model(model, rank=16)
    else:
        print("Creating new model...")
        model = torch.load(config["pretrained_path"])
        model = get_lora_model(model, rank=16)
    return model


def load_or_create_model(config, model_path, if_load_model):
    # Full parameter finetune, Always the best choice if you have a good device
    if if_load_model:
        try:
            model = torch.load(model_path)
            print("Model loaded")
        except FileNotFoundError:
            print("Failed to load the model, try to recreate it")
            model = RotaryGPTLanguageModel(config["vocab_size"], config["embed_size"], config["num_layers"],
                                           config["num_heads"], config["dropout"], config["device"])
    else:
        print("Creating new model...")
        model = RotaryGPTLanguageModel(config["vocab_size"], config["embed_size"], config["num_layers"],
                                       config["num_heads"], config["dropout"], config["device"])
    return model


if __name__ == "__main__":
    # Define device. typically use cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = {
        "pretrained_path": "./Saved_model/model2_pretrained.pth",
        "num_layers": 12,
        "num_of_sft_layers": 1,
        "vocab_size": 50257,  # vocabulary size
        "embed_size": 768,
        "num_heads": 12,
        "dropout": 0.1,
        "device": device
    }
    model = load_or_create_model(config, "./Saved_model/model2_Full_finetune.pth", True)
    num_epochs = 0
    if num_epochs == 0:
        system_prompt = ("<User>: You are a highly knowledgeable and friendly assistant. Your goal is to understand "
                         "and respond to user inquiries with clarity. Your interactions are always respectful, "
                         "helpful, and focused on delivering the most accurate information to the user.\n")
        question_prompt = "<User>: I have problems when installing windows on my computer, what should I do?\n<Agent>: "
        eval_simplegpt_model(model, tokenizer,
                             initial_text=question_prompt,
                             repetition_penalty=1.6)
        exit(0)
    train_simplegpt_model(model, device, "./datasets/ultrachat200k_100000-200000_train.json",
                          "./datasets/ultrachat200k_0-100000_eval.json",
                          "./Saved_model/model2_Full_finetune.pth", num_epochs, batch_size=2, lr=2e-5)

