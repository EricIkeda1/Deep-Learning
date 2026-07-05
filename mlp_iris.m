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
    fprintf('=========== RESULTADOS ===========\n');
    fprintf('Número de Amostras      : %d\n', size(X,2));
    fprintf('Número de Entradas      : %d\n', size(X,1));
    fprintf('Número de Classes       : %d\n', size(T,1));
    fprintf('Número de Acertos       : %d\n', acertos);
    fprintf('Número de Erros         : %d\n', length(classeReal)-acertos);
    fprintf('Acurácia                : %.2f %%\n', acuracia);
    fprintf('Erro Médio              : %.6f\n', perform(net,T,Y));
    fprintf('Função de Treinamento   : %s\n', net.trainFcn);
    fprintf('Função de Desempenho    : %s\n', net.performFcn);
    fprintf('==================================\n');

    

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

    disp('==================================');
    disp('PESOS DA PRIMEIRA CAMADA');
    disp(net.IW{1});

    disp('BIAS DA PRIMEIRA CAMADA');
    disp(net.b{1});

    disp('PESOS DA CAMADA DE SAÍDA');
    disp(net.LW{2});

    disp('BIAS DA CAMADA DE SAÍDA');
    disp(net.b{2});
    disp('==================================');

    %% ============================================================
    % Fazer previsão de uma nova flor
    % =============================================================

    fprintf('\n');
    fprintf('Primeiras 10 previsões:\n\n');

    for i = 1:10

        fprintf('Amostra %3d -> ',i);

        switch classeReal(i)
            case 1
                fprintf('Real: Setosa       ');
            case 2
                fprintf('Real: Versicolor   ');
            case 3
                fprintf('Real: Virginica    ');
        end

        switch classePrevista(i)
            case 1
                fprintf(' Prevista: Setosa');
            case 2
                fprintf(' Prevista: Versicolor');
            case 3
                fprintf(' Prevista: Virginica');
        end

        fprintf('\n');

    end