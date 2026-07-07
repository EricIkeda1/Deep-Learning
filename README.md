# Deep Learning com MATLAB

## Descrição

Este projeto demonstra a implementação de uma Rede Neural Multicamadas (MLP - Multilayer Perceptron) utilizando MATLAB e a Deep Learning Toolbox para classificação do conjunto de dados Iris.

O objetivo é apresentar conceitos fundamentais de Deep Learning, incluindo treinamento de redes neurais, classificação supervisionada e avaliação de desempenho.

---

## Objetivos

- Implementar uma Rede Neural Multicamadas (MLP).
- Classificar as espécies do conjunto de dados Iris.
- Aplicar técnicas de treinamento supervisionado.
- Avaliar o desempenho do modelo por meio de métricas e gráficos.

---

## Tecnologias

- MATLAB
- Deep Learning Toolbox
- Machine Learning
- Inteligência Artificial

---

## Estrutura do Projeto

```text
Deep-Learning/
│
├── Dados/
│   └── Iris.csv
│
├── mlp_iris.m
│
└── README.md
```

---

## Dataset

O projeto utiliza o Iris Dataset, composto por:

- 150 amostras
- 4 atributos numéricos
- 3 classes

### Atributos

- Comprimento da Sépala
- Largura da Sépala
- Comprimento da Pétala
- Largura da Pétala

### Classes

- Iris-setosa
- Iris-versicolor
- Iris-virginica

---

## Arquitetura da Rede

- Tipo: Multilayer Perceptron (MLP)
- Entradas: 4 neurônios
- Camada oculta: 10 neurônios
- Saída: 3 neurônios
- Função de treinamento: `trainscg`
- Função de desempenho: `crossentropy`

---

## Funcionalidades

O projeto realiza:

- Leitura do conjunto de dados Iris.
- Conversão das classes para One-Hot Encoding.
- Embaralhamento dos dados.
- Divisão entre treinamento, validação e teste.
- Treinamento da rede neural.
- Classificação das amostras.
- Cálculo da acurácia.
- Exibição da matriz de confusão.
- Exibição da curva de treinamento.
- Exibição da curva ROC.
- Exibição do histograma de erros.
- Predição de novas amostras.

---

## Como Executar

Clone o repositório:

```bash
git clone https://github.com/EricIkeda1/Deep-Learning.git
```

Abra o projeto no MATLAB.

Execute o arquivo:

```matlab
mlp_iris
```

---

## Exemplo de Resultado

```text
=======================================
RESULTADOS
=======================================
Número de Amostras : 150
Número de Entradas : 4
Número de Classes  : 3
Número de Acertos  : 143
Número de Erros    : 7
Acurácia           : 95.33%
=======================================
```

---

## Conceitos Aplicados

- Redes Neurais Artificiais
- Multilayer Perceptron (MLP)
- Deep Learning
- Aprendizado Supervisionado
- Backpropagation
- One-Hot Encoding
- Classificação
- Avaliação de Modelos

---

## Referências

- MATLAB Documentation
- Deep Learning Toolbox Documentation
- Iris Dataset

---

## Autor
EricIkeda1
