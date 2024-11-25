from transformers import CLIPModel, CLIPProcessor

model_name = "openai/clip-vit-base-patch32"
save_directory = "models/clip-vit-base-patch32"

# Baixar e salvar o modelo
model = CLIPModel.from_pretrained(model_name)
model.save_pretrained(save_directory)

# Baixar e salvar o processador
processor = CLIPProcessor.from_pretrained(model_name)
processor.save_pretrained(save_directory)
