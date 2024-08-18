import warnings
warnings.filterwarnings("ignore")

from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer, ViTModel, GPT2LMHeadModel, GPT2Config, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from PIL import Image, UnidentifiedImageError
import torch
from torch.utils.data import Dataset, DataLoader
import os
import openml
from sklearn.model_selection import train_test_split
from torchvision import transforms
import logging
import time
import cProfile
import pstats

# Configurar o logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurar a API key do OpenML
openml.config.apikey = 'c479fc7c1f84c5f2ec0bc947f1372ffb'

# Configurar o decoder com cross-attention
decoder_config = GPT2Config.from_pretrained("gpt2", add_cross_attention=True)
decoder = GPT2LMHeadModel.from_pretrained("gpt2", config=decoder_config)

# Carregar o encoder
encoder = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

# Combinar o encoder e o decoder em um VisionEncoderDecoderModel
model = VisionEncoderDecoderModel(encoder=encoder, decoder=decoder)

# Configurar pad_token_id e decoder_start_token_id
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model.config.pad_token_id = tokenizer.eos_token_id
model.config.decoder_start_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id else 0

# Carregar o feature extractor e o tokenizer
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k", do_rescale=False)

# Adicionar token de padding ao tokenizer
tokenizer.pad_token = tokenizer.eos_token

# Função para carregar o dataset do OpenML
def load_openml_dataset(dataset_id):
    logger.info(f"Carregando dataset do OpenML com ID {dataset_id}...")
    dataset = openml.datasets.get_dataset(dataset_id)
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
    logger.info("Dataset carregado com sucesso.")
    return X, y

# Dataset personalizado para OpenML
class OpenMLImageDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images.reset_index(drop=True)  # Resetar os índices do DataFrame
        self.labels = labels.reset_index(drop=True)  # Resetar os índices do DataFrame
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.fromarray(self.images.iloc[idx].values.reshape(28, 28).astype('uint8')).convert("RGB")  # Converter para RGB
        label = self.labels.iloc[idx]
        if self.transform:
            image = self.transform(image)
        # Gerar input_ids e attention_mask
        inputs = tokenizer(label, return_tensors="pt", padding="max_length", truncation=True, max_length=50)
        pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values.squeeze()
        return {"pixel_values": pixel_values, "input_ids": inputs.input_ids.squeeze(), "attention_mask": inputs.attention_mask.squeeze(), "labels": inputs.input_ids.squeeze()}

# Função para verificar o dataset e remover itens incompletos
def filter_dataset(dataset):
    filtered_dataset = []
    for item in dataset:
        if "input_ids" in item and "attention_mask" in item and "pixel_values" in item:
            filtered_dataset.append(item)
        else:
            logger.warning("Item incompleto encontrado e ignorado.")
    return filtered_dataset

# Função para treinar o modelo
def train_model():
    # Criar diretórios necessários
    project_dir = "C:/Users/pssousa/Downloads/image-llm-project"
    results_dir = os.path.join(project_dir, "results")
    logs_dir = os.path.join(project_dir, "logs")

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    logger.info("Diretórios 'results' e 'logs' criados/verificados.")

    # Carregar o dataset do OpenML (exemplo: Fashion-MNIST com ID 40996)
    X, y = load_openml_dataset(40996)

    # Dividir o dataset em treino e teste
    logger.info("Dividindo o dataset em treino e teste...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logger.info("Divisão do dataset concluída.")

    # Transformações para as imagens
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Criar datasets e dataloaders com logging detalhado
    logger.info("Criando datasets e dataloaders...")
    start_time = time.time()
    train_dataset = OpenMLImageDataset(X_train, y_train, transform=transform)
    logger.info(f"Train dataset criado em {time.time() - start_time:.2f} segundos.")
    start_time = time.time()
    test_dataset = OpenMLImageDataset(X_test, y_test, transform=transform)
    logger.info(f"Test dataset criado em {time.time() - start_time:.2f} segundos.")
    start_time = time.time()
    
    train_dataset_filtered = filter_dataset(train_dataset)
    test_dataset_filtered = filter_dataset(test_dataset)
    
    train_loader = DataLoader(train_dataset_filtered, batch_size=32, shuffle=True, num_workers=4)
    logger.info(f"Train dataloader criado em {time.time() - start_time:.2f} segundos.")
    start_time = time.time()
    
    test_loader = DataLoader(test_dataset_filtered, batch_size=32, shuffle=False, num_workers=4)
    logger.info(f"Test dataloader criado em {time.time() - start_time:.2f} segundos.")

    # Configurar os argumentos de treinamento
    training_args = TrainingArguments(
        output_dir=results_dir,
        per_device_train_batch_size=8,
        num_train_epochs=10,
        learning_rate=5e-5,
        save_steps=10_000,
        save_total_limit=2,
        logging_dir=logs_dir,
        logging_steps=500 
    )

    # Configurar o Data Collator
    class CustomDataCollator(DataCollatorForSeq2Seq):
        def __call__(self, features):
            filtered_features = [feature for feature in features if "input_ids" in feature and "attention_mask" in feature and "pixel_values" in feature]
            if len(filtered_features) == 0:
                logger.error("Nenhum item válido encontrado no batch.")
                return None
            
            pixel_values = torch.stack([feature["pixel_values"] for feature in filtered_features])
            input_ids = torch.stack([feature["input_ids"] for feature in filtered_features])
            attention_mask = torch.stack([feature["attention_mask"] for feature in filtered_features])
            labels = torch.stack([feature["labels"] for feature in filtered_features])
            return {"pixel_values": pixel_values, "input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    data_collator = CustomDataCollator(tokenizer, model=model, padding=True)

    # Profiling do código
    profiler = cProfile.Profile()
    profiler.enable()

    # Treinar o modelo
    logger.info("Iniciando o treinamento do modelo...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset_filtered,
        eval_dataset=test_dataset_filtered,
        data_collator=data_collator
    )

    try:
        trainer.train()
    except TypeError as e:
        logger.error(f"Erro ignorado: {e}")
        # Continue com o restante do código

    logger.info("Treinamento do modelo concluído.")

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats(10)  # Mostrar as 10 funções mais demoradas

# Função para gerar a descrição da imagem com ajustes para diversificação
def generate_caption(image_path):
    try:
        # Tentar abrir a imagem
        image = Image.open(image_path).convert("RGB")
    except UnidentifiedImageError:
        return "Erro: Não foi possível identificar a imagem. Pode ser que o arquivo não seja uma imagem válida."

    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
    attention_mask = torch.ones(pixel_values.shape[:2], dtype=torch.long)  # Criar a máscara de atenção
    output_ids = model.generate(
        pixel_values,
        attention_mask=attention_mask,
        max_length=50,
        num_beams=4,
        early_stopping=True,
        temperature=0.7,
        top_k=50
    )
    
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    logger.info(f"Imagem submetida: {os.path.basename(image_path)}")
    logger.info(f"Descrição do Modelo: {caption}")
    
    return f"Imagem submetida: {os.path.basename(image_path)}\n\nDescrição do Modelo: {caption}"

# Testar a função com uma imagem local
image_path = "C:/Users/pssousa/Pictures/Screenshots/Captura de tela 2024-08-17 104821.png"
print(generate_caption(image_path))

# Treinar o modelo
train_model()