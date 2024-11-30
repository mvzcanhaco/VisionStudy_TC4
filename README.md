# Tech Challenge 4 - Sistema de Análise de Vídeo com Visão Computacional
> POS Tech FIAP - Arquitetura de Sistemas Autônomos

## 📝 Descrição
Este projeto é parte do Tech Challenge 4 da POS Tech FIAP, focado em visão computacional. O sistema realiza análise de vídeo em três principais aspectos:
- Detecção e rastreamento de pessoas
- Análise de sentimentos através de expressões faciais
- Classificação de atividades e detecção de anomalias

## 🎯 Objetivos
- Identificar e rastrear pessoas em vídeos
- Analisar expressões faciais para determinar estados emocionais
- Classificar atividades realizadas pelas pessoas
- Detectar comportamentos anômalos
- Gerar relatórios e visualizações dos resultados

## 🛠️ Tecnologias Utilizadas

### Modelos e Frameworks
- **YOLO + ByteTrack**: Para detecção e rastreamento de pessoas
- **DeepFace**: Para análise de expressões faciais e emoções
- **OpenCLIP**: Para classificação de atividades
- **PyTorch**: Como framework base de deep learning
- **OpenCV**: Para processamento de imagens e vídeo

### Bibliotecas Principais
- `ultralytics`: Implementação do YOLO
- `deepface`: Análise facial
- `open_clip`: Classificação de imagens
- `torch`: Operações de deep learning
- `opencv-python`: Processamento de imagem
- `numpy`: Operações numéricas
- `Pillow`: Manipulação de imagens

## 🏗️ Arquitetura do Sistema

### Pipeline Principal
1. **Extração de Frames**: Conversão do vídeo em frames
2. **Detecção e Tracking**: Identificação e rastreamento de pessoas
3. **Análise de Sentimentos**: Processamento de expressões faciais
4. **Classificação de Atividades**: Identificação das ações realizadas
5. **Geração de Resultados**: Criação de relatórios e visualizações

### Componentes Principais
- `Pipeline`: Orquestra todo o fluxo de processamento
- `PersonTracker`: Gerencia detecção e rastreamento de pessoas
- `SentimentAnalyzer`: Realiza análise de expressões faciais
- `ActivityClassifier`: Classifica atividades nas imagens

## 🚀 Como Executar
bash
Python 3.8+ requerido
python -m venv venv
source venv/bin/activate # Linux/Mac
ou
.\venv\Scripts\activate # Windows
Instalar dependências
pip install -r requirements.txt

```
### Execução
```bash
python main.py --video path/to/video.mp4 --fps 2 --exec-num 1
```
### Pré-requisitos

### Parâmetros Principais
- `--video`: Caminho para o arquivo de vídeo
- `--fps`: Frames por segundo para processamento
- `--exec-num`: Número da execução para organização dos resultados
- `--skip-activity`: Pula classificação de atividades
- `--skip-tracking`: Pula rastreamento de pessoas
- `--skip-sentiment`: Pula análise de sentimentos

## 📊 Resultados e Saídas

### Estrutura de Diretórios
Outputs/
└── Exec_{N}/
    ├── Frames_extracted/
    ├── Person_Tracks/
    ├── Activity_Frames/
    ├── Annotated_Frames/
    ├── Persons_by_sentiment/
    ├── complete_results.csv
    ├── complete_results.json
    └── summary.json

### Tipos de Saída
- Frames anotados com detecções
- Recortes de pessoas detectadas
- Classificações de atividades
- Análises de sentimento
- Relatórios em CSV e JSON
- Vídeo final com anotações

## 📈 Métricas e Avaliação
- Número total de pessoas detectadas
- Contagem de frames processados
- Distribuição de emoções detectadas
- Distribuição de atividades classificadas
- Taxa de detecção de faces
- Confiança das classificações

## ⚠️ Limitações Conhecidas
- Dependência de boa iluminação
- Necessidade de faces visíveis para análise de sentimento
- Consumo significativo de recursos computacionais
- Possível queda de performance em vídeos muito longos

## 🔄 Próximos Passos
- Implementação de processamento em batch
- Otimização de uso de memória
- Adição de mais classes de atividades
- Melhoria na detecção de anomalias
- Interface gráfica para visualização



## 📄 Licença
[Informações sobre licenciamento]
```
 
