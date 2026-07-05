    %% ============================================================
    % DEEP LEARNING - MLP PARA CLASSIFICAÇÃO DO DATASET IRIS
    % Autor: EricIkeda1
    % =============================================================

    clc;
    clear;
    close all;

%% ============================================================
% Carregar Dataset Iris (.csv)
% =============================================================

dados = readtable("Dados/Iris.csv");

% Ignora a coluna Id
X = table2array(dados(:,2:5))';

% A espécie está na sexta coluna
classes = string(dados{:,6});

% Converter para One-Hot
T = zeros(3,length(classes));

for i = 1:length(classes)

    nome = lower(strtrim(classes(i)));

    if contains(nome,"setosa")
        T(:,i) = [1;0;0];

    elseif contains(nome,"versicolor")
        T(:,i) = [0;1;0];

    elseif contains(nome,"virginica")
        T(:,i) = [0;0;1];

    else
        error("Classe desconhecida: " + nome);
    end

end

    %% ============================================================
    % Embaralhar Dados
    % =============================================================

    rng('default')

    indices = randperm(size(X,2));

    X = X(:,indices);
    T = T(:,indices);

    %% ============================================================
    % Criar Rede Neural
    % =============================================================

    % Uma camada escondida com 10 neurônios
    net = patternnet(10);

    %% Função de treinamento
    net.trainFcn = 'trainscg';

    %% Divisão dos dados

    net.divideParam.trainRatio = 0.70;
    net.divideParam.valRatio   = 0.15;
    net.divideParam.testRatio  = 0.15;

    %% Função de desempenho
    net.performFcn = 'crossentropy';

    %% ============================================================
    % Treinamento
    % =============================================================

    [net,tr] = train(net,X,T);

    %% ============================================================
    % Teste
    % =============================================================

    Y = net(X);

    %% ============================================================
    % Classe prevista
    % =============================================================

    [~,classePrevista] = max(Y);

    %% Classe correta

    [~,classeReal] = max(T);

    %% ============================================================
    % Acurácia
    % =============================================================

    acertos = sum(classePrevista == classeReal);

    acuracia = acertos/length(classeReal)*100;

    fprintf('\n');
    fprintf('=======================================\n');
    fprintf('ACURÁCIA = %.2f %%\n',acuracia);
    fprintf('=======================================\n');

    %% ============================================================
    % Matriz de Confusão
    % =============================================================

    figure;

    plotconfusion(T,Y);

    title('Matriz de Confusão');

    %% ============================================================
    % Performance
    % =============================================================

    figure;

    plotperform(tr);

    title('Curva de Treinamento');

    %% ============================================================
    % Histograma de Erro
    % =============================================================

    figure;

    ploterrhist(T-Y);

    title('Histograma dos Erros');

    %% ============================================================
    % ROC
    % =============================================================

    figure;

    plotroc(T,Y);

    title('Curva ROC');

    %% ============================================================
    % Visualização das Classes
    % =============================================================

    figure

    gscatter(X(1,:)', X(2,:)', classes);

    xlabel('Comprimento da Sépala');
    ylabel('Largura da Sépala');
    title('Dataset Iris');
    grid on;

    %% ============================================================
    % Mostrar pesos
    % =============================================================

    disp('Pesos da primeira camada');

    disp(net.IW{1});

    disp('Bias da primeira camada');

    disp(net.b{1});

    disp('Pesos da camada de saída');

    disp(net.LW{2});

    disp('Bias da saída');

    disp(net.b{2});

    %% ============================================================
    % Fazer previsão de uma nova flor
    % =============================================================

    novaFlor = [5.1
                3.5
                1.4
                0.2];

    resultado = net(novaFlor);

    [~,classe] = max(resultado);

    fprintf('\n');

    switch classe

        case 1
            disp('Nova flor: SETOSA');

        case 2
            disp('Nova flor: VERSICOLOR');

        case 3
            disp('Nova flor: VIRGINICA');

    end

    disp('Probabilidades:')

    disp(resultado)