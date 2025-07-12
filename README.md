# resnet50-alexnet-egg-classification

Este projeto implementa uma solução baseada em Redes Neurais Convolucionais (CNNs) para classificar ovos em duas categorias: **danificados** (`damaged`) e **não danificados** (`not_damaged`). O treinamento e avaliação são realizados utilizando os modelos **ResNet50** e **AlexNet**, ambos implementados na biblioteca `torch` (PyTorch).

## Links importantes

1. [Artigo do trabalho](https://github.com/luizvilasboas/resnet50-alexnet-egg-classification/blob/main/docs/article.pdf)
2. [Vídeo explicando](https://www.youtube.com/watch?v=Ijp6jcghPM8)

## Descrição do Projeto

A classificação de ovos é uma tarefa essencial em várias indústrias para garantir a qualidade e a segurança do produto final. Este projeto utiliza imagens de ovos como entrada e classifica cada imagem em uma das duas categorias mencionadas. 

### Funcionalidades

1. **Treinamento de Modelos**: O treinamento é realizado usando dois modelos conhecidos: ResNet50 e AlexNet.
2. **Avaliação de Performance**: Mede a precisão, perda e outras métricas para comparar os desempenhos dos modelos.

## Requisitos

- Python 3.8 ou superior
- PyTorch 2.0 ou superior
- CUDA (opcional, para aceleração com GPU)
- Bibliotecas adicionais listadas no arquivo `requirements.txt`

### Instalação

1. Clone o repositório:
   ```
   git clone https://github.com/luizvilasboas/resnet50-alexnet-egg-classification.git
   cd resnet50-alexnet-egg-classification
   ```

2. Crie um ambiente virtual (opcional, mas recomendado):
   ```
   python -m venv venv
   source venv/bin/activate
   ```

3. Instale as dependências:
   ```
   pip install -r requirements.txt
   ```

4. Certifique-se de que o diretório `dataset/` está configurado corretamente com o conjunto de dados que pode ser achado [aqui](https://www.kaggle.com/datasets/abdullahkhanuet22/eggs-images-classification-damaged-or-not).

## Instruções de Execução

Execute o script de treinamento especificando o modelo que deseja usar:
   ```
   python3 train_alexnet.py
   python3 train_resnet50.py
   ```

## Resultados e Relatórios

Os resultados do treinamento, incluindo gráficos de precisão e perda, são salvos automaticamente no diretório `output/`. Você pode visualizar os modelos treinados no mesmo diretório.

## Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para abrir uma issue ou enviar um pull request.

## Licença

Este projeto está licenciado sob a [MIT License](https://github.com/luizvilasboas/resnet50-alexnet-egg-classification/blob/main/LICENSE).