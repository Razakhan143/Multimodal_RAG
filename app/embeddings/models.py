from open_clip import create_model_and_transforms, get_tokenizer
class vis_models:
    def __init__(self):
        self.model, _, self.preprocess = create_model_and_transforms(
        "ViT-B-32",
        pretrained="laion2b_s34b_b79k"
        )
# Load OpenCLIP model
    def clip_model(self):
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = self.model.to(device)
        self.tokenizer = get_tokenizer("ViT-B-32")
        return self.model, self.preprocess, self.tokenizer



