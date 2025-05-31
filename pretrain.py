
from model import SimpleGPTLanguageModel, RotaryGPTLanguageModel
from dataprocessing import tokenizer
from utils import *

# Somehow we can estimate the expressive power by looking at its information flow. The Larger the information flow
# is, the easier the model trains
# Using batch helps to converge, making the losses more stable


def load_or_create_model(config, model_path, if_load_model):
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

    # The model configuration
    config = {
        "vocab_size": 50257,  # 假设的词汇表大小
        "embed_size": 768,
        "num_layers": 12,
        "num_heads": 12,
        "dropout": 0.1,
        "device": device
    }
    # param size: 200M
    # The gpt-2 small configuration
    # Initialize models
    num_epochs = 1
    model = load_or_create_model(config, "./Saved_model/model2_pretrained.pth", True)

    if num_epochs == 0:
        # validate model
        eval_simplegpt_model(model, tokenizer, "What is a plane?", seq_len=1024)
        exit(0)

    train_simplegpt_model(model, device, "./datasets/redpajama_800000-900000_train.json"
                          , "./datasets/redpajama_800000-900000_eval.json",
                          "./Saved_model/model2_pretrained.pth", num_epochs, lr=5e-5)

