####  Projeto - Prevendo o Consumo de Energia de Carros Elétricos  ####

# Configurando o diretório de trabalho
setwd("~/Desktop/DataScience/CienciaDeDados/1.Big-Data-Analytics-com-R-e-Microsoft-Azure-Machine-Learning/20.Projetos_com_Feedback/1.Prevendo_Consumo_Energia_Carros_Eletricos")
getwd()


## Machine Learning em Logística Prevendo o Consumo de Energia de Carros Elétricos ##


## Carregando Pacotes
library(readxl)         # carregar arquivos 

library(dplyr)          # manipulação de dados

library(ggplot2)        # gera gráficos
library(shiny)          # intercace gráfica

library(randomForest)   # carrega algoritimo de ML (randomForest)
library(e1071)          # carrega algoritimo de ML (SVM)
library(glmnet)         # carrega algoritimo de ML
library(xgboost)        # carrega algoritimo de ML
library(class)          # carrega algoritimo de ML
library(rpart)          # carrega algoritimo de ML


library(caret)          # cria confusion matrix

library(h2o)            # framework para construir modelos de machine learning



#  -> Seu trabalho é construir um modelo de Machine Learning capaz de prever o consumo de energia de veículos elétricos.


## Carregando dados

dados <- data.frame(read_xlsx("dataset/FEV-data-Excel.xlsx"))

# Verificando e removendo valores ausentes
colSums(is.na(dados))
dados <- dados[complete.cases(dados), ]

dim(dados)
str(dados)
summary(dados)

length(unique(dados$Car.full.name))
length(unique(dados$Make))
length(unique(dados$Model))


## Análise Exploratória

# Gráfico de dispersão (scatter plot) entre Consumo de Energia e outras variáveis importantes
plot(dados$mean...Energy.consumption..kWh.100.km., dados$Engine.power..KM., 
     xlab = "Consumo de Energia (kWh/100km)", ylab = "Potência do Motor (KM)",
     main = "Consumo de Energia vs. Potência do Motor")






#### Versão 1

## Carregando dados
dados <- data.frame(read_xlsx("dataset/FEV-data-Excel.xlsx"))
dados <- dados[complete.cases(dados), ]


# - Mantem tipos de dados originais (chr e int)
# - Cria duas novas variáveis
# - Cria 1 Tipo de Modelo utilizando todas as variáveis (RandomForest)


# Criando novas variáveis
dados$Weight.Power.Ratio <- dados$Minimal.empty.weight..kg. / dados$Engine.power..KM.  # Relação entre Peso e Potência do Motor
dados$Battery.Range.Ratio <- dados$Battery.capacity..kWh. / dados$Range..WLTP...km.    # Relação entre Capacidade da Bateria e Alcance

str(dados)
summary(dados)

## Criando Modelo

# Dividindo os dados em treino e teste
set.seed(150)
indices <- createDataPartition(dados$mean...Energy.consumption..kWh.100.km., p = 0.80, list = FALSE)
dados_treino <- dados[indices, ]
dados_teste <- dados[-indices, ]
rm(indices)

# Criando o modelo preditivo (RandomForest)
modelo <- randomForest(mean...Energy.consumption..kWh.100.km. ~ ., 
                       data = dados_treino, 
                       ntree = 100, nodesize = 10, importance = TRUE, set.seed(100))

# Realizando previsões no conjunto de teste
previsoes <- predict(modelo, newdata = dados_teste)

# Avaliando o desempenho do modelo
rmse <- sqrt(mean((previsoes - dados_teste$mean...Energy.consumption..kWh.100.km.)^2))
cat("RMSE (Root Mean Squared Error):", rmse, "\n")

mae <- mean(abs(previsoes - dados_teste$mean...Energy.consumption..kWh.100.km.))
cat("MAE (Mean Absolute Error):", mae, "\n")

rsquared <- 1 - sum((previsoes - dados_teste$mean...Energy.consumption..kWh.100.km.)^2) / sum((dados_teste$mean...Energy.consumption..kWh.100.km. - mean(dados_teste$mean...Energy.consumption..kWh.100.km.))^2)
cat("R-squared:", rsquared, "\n")

# RMSE (Root Mean Squared Error): 0.9157646
# MAE (Mean Absolute Error)     : 0.7412
# R-squared                     : 0.9424634 

rm(modelo)
rm(previsoes)
rm(rmse)
rm(mae)
rm(rsquared)
rm(dados_treino)
rm(dados_teste)




#### Versão 2

## Carregando dados
dados <- data.frame(read_xlsx("dataset/FEV-data-Excel.xlsx"))
dados <- dados[complete.cases(dados), ]


# - Modifica todas as variáveis chr para factor
# - Modifica as variáveis numéricas Number.of.seats e Number.of.doors em factor
# - Cria duas novas variáveis de relação
# - Utiliza Feature Selection
# - Cria 1 Tipo de Modelo (RandomForest)


## Engenharia de Atributos

# Convertendo a variável variáveis chr para fator
dados <- dados %>%  
  mutate_if(is.character, factor) %>%
  mutate(across(c(Number.of.seats, Number.of.doors), as.factor))

# Criando novas variáveis de relação
dados$Weight.Power.Ratio <- dados$Minimal.empty.weight..kg. / dados$Engine.power..KM.  # Relação entre Peso e Potência do Motor
dados$Battery.Range.Ratio <- dados$Battery.capacity..kWh. / dados$Range..WLTP...km.    # Relação entre Capacidade da Bateria e Alcance

str(dados)
summary(dados)

## Seleção de Variáveis (Feature Selection)
modelo <- randomForest(mean...Energy.consumption..kWh.100.km. ~ ., 
                       data = dados, 
                       ntree = 100, nodesize = 10, importance = T)
  
# Visualizando por números
print(modelo$importance)

# Visualizando por Gráficos
varImpPlot(modelo)

importancia_ordenada <- modelo$importance[order(-modelo$importance[, 1]), , drop = FALSE] 
df_importancia <- data.frame(
  Variavel = rownames(importancia_ordenada),
  Importancia = importancia_ordenada[, 1]
)
ggplot(df_importancia, aes(x = reorder(Variavel, -Importancia), y = Importancia)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  labs(title = "Importância das Variáveis", x = "Variável", y = "Importância") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1, size = 10))

rm(modelo)
rm(importancia_ordenada)
rm(df_importancia)

names(dados)


## Criando Modelo

# Dividindo os dados em treino e teste
set.seed(150)
indices <- createDataPartition(dados$mean...Energy.consumption..kWh.100.km., p = 0.80, list = FALSE)
dados_treino <- dados[indices, ]
dados_teste <- dados[-indices, ]
rm(indices)

# Criando o modelo preditivo (RandomForest)
modelo <- randomForest(mean...Energy.consumption..kWh.100.km. ~ 
                         Make + Battery.Range.Ratio + Wheelbase..cm. + Length..cm. + Engine.power..KM. +
                         Permissable.gross.weight..kg. + Maximum.speed..kph. + Minimal.empty.weight..kg. + Maximum.torque..Nm. + Drive.type, 
                       data = dados_treino, 
                       ntree = 100, nodesize = 10, importance = TRUE, set.seed(100))

# Realizando previsões no conjunto de teste
previsoes <- predict(modelo, newdata = dados_teste)

# Avaliando o desempenho do modelo
rmse <- sqrt(mean((previsoes - dados_teste$mean...Energy.consumption..kWh.100.km.)^2))
cat("RMSE (Root Mean Squared Error):", rmse, "\n")

mae <- mean(abs(previsoes - dados_teste$mean...Energy.consumption..kWh.100.km.))
cat("MAE (Mean Absolute Error):", mae, "\n")

rsquared <- 1 - sum((previsoes - dados_teste$mean...Energy.consumption..kWh.100.km.)^2) / sum((dados_teste$mean...Energy.consumption..kWh.100.km. - mean(dados_teste$mean...Energy.consumption..kWh.100.km.))^2)
cat("R-squared:", rsquared, "\n")

# Modelo 1
# RMSE (Root Mean Squared Error): 0.6272197 
# MAE (Mean Absolute Error)     : 0.5990649 
# R-squared                     : 0.9730092

rm(modelo)
rm(previsoes)
rm(rmse)
rm(mae)
rm(rsquared)




#### Versão 3

## Carregando dados
dados <- data.frame(read_xlsx("dataset/FEV-data-Excel.xlsx"))
dados <- dados[complete.cases(dados), ]


# - Modifica todas as variáveis chr para factor
# - Modifica as variáveis numéricas Number.of.seats e Number.of.doors em factor
# - Cria duas novas variáveis de relação
# - Aplica Normalização nas variáveis do tipo int
# - Utiliza Feature Selection
# - Cria 1 Tipo de Modelo (RandomForest)

## Engenharia de Atributos

# Convertendo a variável variáveis chr para fator
dados <- dados %>%  
  mutate_if(is.character, factor) %>%
  mutate(across(c(Number.of.seats, Number.of.doors), as.factor))

# Normalização dos Dados (variáveis numéricas) (Exemplo 1 coluna ao final)
numeric_columns <- sapply(dados, is.numeric)
dados_nor <- dados %>%
  mutate(across(where(is.numeric), ~ scale(., center = min(.), scale = max(.) - min(.))))
rm(numeric_columns)

# Reverter Normalização
# dados_revertidos <- dados_nor %>%
#   mutate(across(where(is.numeric), ~ (. * (max(dados[, cur_column()]) - min(dados[, cur_column()])) + min(dados[, cur_column()]))))

# Criando novas variáveis de relação
dados_nor$Weight.Power.Ratio <- dados$Minimal.empty.weight..kg. / dados$Engine.power..KM.  # Relação entre Peso e Potência do Motor
dados_nor$Battery.Range.Ratio <- dados$Battery.capacity..kWh. / dados$Range..WLTP...km.    # Relação entre Capacidade da Bateria e Alcance


## Seleção de Variáveis (Feature Selection)
modelo <- randomForest(mean...Energy.consumption..kWh.100.km. ~ ., 
                       data = dados_nor, 
                       ntree = 100, nodesize = 10, importance = T, set.seed(100))

# Visualizando por números
print(modelo$importance)

# Visualizando por Gráficos
varImpPlot(modelo)

importancia_ordenada <- modelo$importance[order(-modelo$importance[, 1]), , drop = FALSE] 
df_importancia <- data.frame(
  Variavel = rownames(importancia_ordenada),
  Importancia = importancia_ordenada[, 1]
)
ggplot(df_importancia, aes(x = reorder(Variavel, -Importancia), y = Importancia)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  labs(title = "Importância das Variáveis", x = "Variável", y = "Importância") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1, size = 10))

rm(modelo)
rm(importancia_ordenada)
rm(df_importancia)


## Criando Modelo

# Dividindo os dados em treino e teste
set.seed(150)
indices <- createDataPartition(dados_nor$mean...Energy.consumption..kWh.100.km., p = 0.80, list = FALSE)
dados_treino <- dados_nor[indices, ]
dados_teste <- dados_nor[-indices, ]
rm(indices)

# Criando o modelo preditivo (RandomForest)
modelo <- randomForest(mean...Energy.consumption..kWh.100.km. ~ Make + Wheelbase..cm.
                       + Permissable.gross.weight..kg. + Minimal.price..gross...PLN. + Length..cm. + Width..cm.,
                       data = dados_treino, 
                       ntree = 100, nodesize = 10, importance = TRUE, set.seed(100))

# Realizando previsões no conjunto de teste
previsoes <- predict(modelo, newdata = dados_teste)

# Avaliando o desempenho do modelo
rmse <- sqrt(mean((previsoes - dados_teste$mean...Energy.consumption..kWh.100.km.)^2))
cat("RMSE (Root Mean Squared Error):", rmse, "\n")

mae <- mean(abs(previsoes - dados_teste$mean...Energy.consumption..kWh.100.km.))
cat("MAE (Mean Absolute Error):", mae, "\n")

rsquared <- 1 - sum((previsoes - dados_teste$mean...Energy.consumption..kWh.100.km.)^2) / sum((dados_teste$mean...Energy.consumption..kWh.100.km. - mean(dados_teste$mean...Energy.consumption..kWh.100.km.))^2)
cat("R-squared:", rsquared, "\n")

# Modelo 1
# RMSE (Root Mean Squared Error): 0.03809704
# MAE (Mean Absolute Error)     : 0.03421335
# R-squared                     : 0.9792081

rm(modelo)
rm(previsoes)
rm(rmse)
rm(mae)
rm(rsquared)




#### Versão 4

## Carregando dados
dados <- data.frame(read_xlsx("dataset/FEV-data-Excel.xlsx"))
dados <- dados[complete.cases(dados), ]


# - Modifica todas as variáveis chr para factor
# - Modifica as variáveis numéricas Number.of.seats e Number.of.doors em factor
# - Cria novas variáveis categóricas a partir de varíaveis do tipo int
# - Cria duas novas variáveis de relação
# - Aplica Normalização nas variáveis do tipo int
# - Utiliza Feature Selection
# - Cria 1 Tipo de Modelo (RandomForest)


## Engenharia de Atributos

# Convertendo a variável variáveis chr para fator
dados <- dados %>%  
  mutate_if(is.character, factor) %>%
  mutate(across(c(Number.of.seats, Number.of.doors), as.factor))

# Cria novas variáveis categóricas a partir de varíaveis do tipo int
colunas_numericas <- sapply(dados, is.numeric)                          # Lista de colunas que são numéricas

# Função para criar variáveis fatoriais com 5 níveis
criar_variaveis_fatoriais <- function(coluna_numerica) {
  # Calcula os limites dos intervalos
  limite_inferior <- min(coluna_numerica)
  limite_superior <- max(coluna_numerica)
  
  # Calcula o tamanho do intervalo
  tamanho_intervalo <- (limite_superior - limite_inferior) / 3
  
  # Ajusta o limite inferior para evitar valores exatamente iguais
  limite_inferior <- limite_inferior - 0.001
  
  # Cria os rótulos dos níveis
  rotulos <- paste0(round(seq(limite_inferior, limite_superior, by = tamanho_intervalo), 2),
                    " a ",
                    round(seq(limite_inferior + tamanho_intervalo, limite_superior + tamanho_intervalo, by = tamanho_intervalo), 2))
  
  # Cria a variável fatorial com os rótulos, sem valores NA
  variavel_fatorial <- cut(coluna_numerica, breaks = seq(limite_inferior, limite_superior + tamanho_intervalo, by = tamanho_intervalo), labels = rotulos, na.omit = FALSE)
  
  return(variavel_fatorial)
}

# Aplica a função para criar variáveis fatoriais em cada coluna numérica
novas_variaveis_fatoriais <- lapply(dados[, colunas_numericas], criar_variaveis_fatoriais)

# Adiciona um sufixo aos nomes das novas variáveis
nomes_novas_variaveis <- paste0(names(dados[, colunas_numericas]), "_categoria")
names(novas_variaveis_fatoriais) <- nomes_novas_variaveis

# Combina as novas variáveis fatoriais ao conjunto de dados original
dados <- cbind(dados, novas_variaveis_fatoriais)
str(dados)
rm(criar_variaveis_fatoriais)
rm(novas_variaveis_fatoriais)
rm(nomes_novas_variaveis)
rm(colunas_numericas)

# Normalização dos Dados (variáveis numéricas) (Exemplo 1 coluna ao final)
numeric_columns <- sapply(dados, is.numeric)
dados_nor <- dados %>%
  mutate(across(where(is.numeric), ~ scale(., center = min(.), scale = max(.) - min(.))))
rm(numeric_columns)

# Reverter Normalização
# dados_revertidos <- dados_nor %>%
#   mutate(across(where(is.numeric), ~ (. * (max(dados[, cur_column()]) - min(dados[, cur_column()])) + min(dados[, cur_column()]))))

# Criando novas variáveis de relação
dados_nor$Weight.Power.Ratio <- dados$Minimal.empty.weight..kg. / dados$Engine.power..KM.  # Relação entre Peso e Potência do Motor
dados_nor$Battery.Range.Ratio <- dados$Battery.capacity..kWh. / dados$Range..WLTP...km.    # Relação entre Capacidade da Bateria e Alcance


## Seleção de Variáveis (Feature Selection)
modelo <- randomForest(mean...Energy.consumption..kWh.100.km. ~ ., 
                       data = dados_nor, 
                       ntree = 100, nodesize = 10, importance = T, set.seed(100))

# Visualizando por números
print(modelo$importance)

# Visualizando por Gráficos
varImpPlot(modelo)

importancia_ordenada <- modelo$importance[order(-modelo$importance[, 1]), , drop = FALSE] 
df_importancia <- data.frame(
  Variavel = rownames(importancia_ordenada),
  Importancia = importancia_ordenada[, 1]
)
ggplot(df_importancia, aes(x = reorder(Variavel, -Importancia), y = Importancia)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  labs(title = "Importância das Variáveis", x = "Variável", y = "Importância") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1, size = 10))

rm(modelo)
rm(importancia_ordenada)
rm(df_importancia)



## Criando Modelo

# Dividindo os dados em treino e teste
set.seed(150)
indices <- createDataPartition(dados_nor$mean...Energy.consumption..kWh.100.km., p = 0.80, list = FALSE)
dados_treino <- dados_nor[indices, ]
dados_teste <- dados_nor[-indices, ]
rm(indices)

# Criando o modelo preditivo (RandomForest)
modelo <- randomForest(mean...Energy.consumption..kWh.100.km. ~ Make + Wheelbase..cm.
                       + Permissable.gross.weight..kg. + Minimal.price..gross...PLN. + Length..cm. + Width..cm.
                       + Acceleration.0.100.kph..s._categoria,
                       data = dados_treino, 
                       ntree = 100, nodesize = 10, importance = TRUE, set.seed(100))

# Realizando previsões no conjunto de teste
previsoes <- predict(modelo, newdata = dados_teste)

# Avaliando o desempenho do modelo
rmse <- sqrt(mean((previsoes - dados_teste$mean...Energy.consumption..kWh.100.km.)^2))
cat("RMSE (Root Mean Squared Error):", rmse, "\n")

mae <- mean(abs(previsoes - dados_teste$mean...Energy.consumption..kWh.100.km.))
cat("MAE (Mean Absolute Error):", mae, "\n")

rsquared <- 1 - sum((previsoes - dados_teste$mean...Energy.consumption..kWh.100.km.)^2) / sum((dados_teste$mean...Energy.consumption..kWh.100.km. - mean(dados_teste$mean...Energy.consumption..kWh.100.km.))^2)
cat("R-squared:", rsquared, "\n")

# Modelo 1
# RMSE (Root Mean Squared Error): 0.04515118
# MAE (Mean Absolute Error)     : 0.0414877
# R-squared                     : 0.9707955

rm(modelo)
rm(previsoes)
rm(rmse)
rm(mae)
rm(rsquared)




#### Versão 5

## Carregando dados
dados <- data.frame(read_xlsx("dataset/FEV-data-Excel.xlsx"))
dados <- dados[complete.cases(dados), ]


# - Utiliza as mesmas configurações da Versão 3
# - Aplica Técnicas de Duplicação de Linhas


## Engenharia de Atributos

# Convertendo a variável variáveis chr para fator
dados <- dados %>%  
  mutate_if(is.character, factor) %>%
  mutate(across(c(Number.of.seats, Number.of.doors), as.factor))

# Normalização dos Dados (variáveis numéricas) (Exemplo 1 coluna ao final)
numeric_columns <- sapply(dados, is.numeric)
dados_nor <- dados %>%
  mutate(across(where(is.numeric), ~ scale(., center = min(.), scale = max(.) - min(.))))
rm(numeric_columns)

# Reverter Normalização
# dados_revertidos <- dados_nor %>%
#   mutate(across(where(is.numeric), ~ (. * (max(dados[, cur_column()]) - min(dados[, cur_column()])) + min(dados[, cur_column()]))))

# Criando novas variáveis de relação
dados_nor$Weight.Power.Ratio <- dados$Minimal.empty.weight..kg. / dados$Engine.power..KM.  # Relação entre Peso e Potência do Motor
dados_nor$Battery.Range.Ratio <- dados$Battery.capacity..kWh. / dados$Range..WLTP...km.    # Relação entre Capacidade da Bateria e Alcance


## Seleção de Variáveis (Feature Selection)
modelo <- randomForest(mean...Energy.consumption..kWh.100.km. ~ ., 
                       data = dados_nor, 
                       ntree = 100, nodesize = 10, importance = T, set.seed(100))

# Visualizando por números
print(modelo$importance)

# Visualizando por Gráficos
varImpPlot(modelo)

importancia_ordenada <- modelo$importance[order(-modelo$importance[, 1]), , drop = FALSE] 
df_importancia <- data.frame(
  Variavel = rownames(importancia_ordenada),
  Importancia = importancia_ordenada[, 1]
)
ggplot(df_importancia, aes(x = reorder(Variavel, -Importancia), y = Importancia)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  labs(title = "Importância das Variáveis", x = "Variável", y = "Importância") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1, size = 10))

rm(modelo)
rm(importancia_ordenada)
rm(df_importancia)


## Criando Modelo

# Dividindo os dados em treino e teste
set.seed(150)
indices <- createDataPartition(dados_nor$mean...Energy.consumption..kWh.100.km., p = 0.80, list = FALSE)
dados_treino <- dados_nor[indices, ]
dados_teste <- dados_nor[-indices, ]
rm(indices)
str(dados_treino)

# Duplicação de Linhas (dados_treino)
dados_treino <- dados_treino[rep(seq_len(nrow(dados_treino)), each = 2), ]

# Para uma duplicação mais equilibrada, você pode ajustar o fator 'each' conforme necessário


# Criando o modelo preditivo (RandomForest)
modelo <- randomForest(mean...Energy.consumption..kWh.100.km. ~ Make + Wheelbase..cm.
                       + Permissable.gross.weight..kg. + Minimal.price..gross...PLN. + Length..cm. + Width..cm.,
                       data = dados_treino, 
                       ntree = 100, nodesize = 10, importance = TRUE, set.seed(100))

# Realizando previsões no conjunto de teste
previsoes <- predict(modelo, newdata = dados_teste)

# Avaliando o desempenho do modelo
rmse <- sqrt(mean((previsoes - dados_teste$mean...Energy.consumption..kWh.100.km.)^2))
cat("RMSE (Root Mean Squared Error):", rmse, "\n")

mae <- mean(abs(previsoes - dados_teste$mean...Energy.consumption..kWh.100.km.))
cat("MAE (Mean Absolute Error):", mae, "\n")

rsquared <- 1 - sum((previsoes - dados_teste$mean...Energy.consumption..kWh.100.km.)^2) / sum((dados_teste$mean...Energy.consumption..kWh.100.km. - mean(dados_teste$mean...Energy.consumption..kWh.100.km.))^2)
cat("R-squared:", rsquared, "\n")

# Modelo 1
# RMSE (Root Mean Squared Error): 0.04066856
# MAE (Mean Absolute Error)     : 0.03548824
# R-squared                     : 0.9763065

rm(modelo)
rm(previsoes)
rm(rmse)
rm(mae)
rm(rsquared)




#### Versão 6 (AutoML)

## Carregando dados
dados <- data.frame(read_xlsx("dataset/FEV-data-Excel.xlsx"))
dados <- dados[complete.cases(dados), ]


# - Modifica todas as variáveis chr para factor
# - Modifica as variáveis numéricas Number.of.seats e Number.of.doors em factor
# - Cria novas variáveis categóricas a partir de varíaveis do tipo int
# - Cria duas novas variáveis de relação
# - Utilizando AutoML


## Engenharia de Atributos

# Convertendo a variável variáveis chr para fator
dados <- dados %>%  
  mutate_if(is.character, factor) %>%
  mutate(across(c(Number.of.seats, Number.of.doors), as.factor)) %>% 
  select(-Car.full.name)


# Cria novas variáveis categóricas a partir de varíaveis do tipo int
colunas_numericas <- sapply(dados, is.numeric)                          # Lista de colunas que são numéricas

# Função para criar variáveis fatoriais com 5 níveis
criar_variaveis_fatoriais <- function(coluna_numerica) {
  # Calcula os limites dos intervalos
  limite_inferior <- min(coluna_numerica)
  limite_superior <- max(coluna_numerica)
  
  # Calcula o tamanho do intervalo
  tamanho_intervalo <- (limite_superior - limite_inferior) / 3
  
  # Ajusta o limite inferior para evitar valores exatamente iguais
  limite_inferior <- limite_inferior - 0.001
  
  # Cria os rótulos dos níveis
  rotulos <- paste0(round(seq(limite_inferior, limite_superior, by = tamanho_intervalo), 2),
                    " a ",
                    round(seq(limite_inferior + tamanho_intervalo, limite_superior + tamanho_intervalo, by = tamanho_intervalo), 2))
  
  # Cria a variável fatorial com os rótulos, sem valores NA
  variavel_fatorial <- cut(coluna_numerica, breaks = seq(limite_inferior, limite_superior + tamanho_intervalo, by = tamanho_intervalo), labels = rotulos, na.omit = FALSE)
  
  return(variavel_fatorial)
}

# Aplica a função para criar variáveis fatoriais em cada coluna numérica
novas_variaveis_fatoriais <- lapply(dados[, colunas_numericas], criar_variaveis_fatoriais)

# Adiciona um sufixo aos nomes das novas variáveis
nomes_novas_variaveis <- paste0(names(dados[, colunas_numericas]), "_categoria")
names(novas_variaveis_fatoriais) <- nomes_novas_variaveis

# Combina as novas variáveis fatoriais ao conjunto de dados original
dados <- cbind(dados, novas_variaveis_fatoriais)
str(dados)
rm(criar_variaveis_fatoriais)
rm(novas_variaveis_fatoriais)
rm(nomes_novas_variaveis)
rm(colunas_numericas)


## Automl

# Inicializando o H2O (Framework de Machine Learning)
h2o.init()

# O H2O requer que os dados estejam no formato de dataframe do H2O
h2o_frame <- as.h2o(dados)
class(h2o_frame)

# Split dos dados em treino e teste (cria duas listas)
h2o_frame_split <- h2o.splitFrame(h2o_frame, ratios = 0.85)
head(h2o_frame_split)

modelo_automl <- h2o.automl(y = 'mean...Energy.consumption..kWh.100.km.',
                            training_frame = h2o_frame_split[[1]],
                            nfolds = 5,
                            leaderboard_frame = h2o_frame_split[[2]],
                            max_runtime_secs = 60 * 15,
                            sort_metric = "mae")

modelo_automl2 <- h2o.automl(y = 'mean...Energy.consumption..kWh.100.km.',
                            training_frame = h2o_frame_split[[1]],
                            nfolds = 5,
                            leaderboard_frame = h2o_frame_split[[2]],
                            max_runtime_secs = 60 * 60,
                            sort_metric = "mae")


str(dados)


# Extrai o leaderboard (dataframe com os modelos criados)
leaderboard_automl2 <- as.data.frame(modelo_automl2@leaderboard)
head(leaderboard_automl)
View(leaderboard_automl)

# Extrai o líder (modelo com melhor performance)
lider_automl2 <- modelo_automl2@leader
print(lider_automl)
View(lider_automl)

## Carregar o modelo a partir do diretório
modelo_automl_versao4 <- h2o.loadModel("modelos/versao4_modelo/GBM_grid_1_AutoML_1_20240202_151050_model_382")
modelo2_automl_versao4 <- h2o.loadModel("modelos/versao4_modelo2/DeepLearning_grid_2_AutoML_2_20240202_153518_model_10")

# Avaliação dos Modelos
ava_modelo1 <- h2o.performance(modelo_automl_versao4)
ava_modelo1
ava_modelo2 <- h2o.performance(modelo2_automl_versao4)
ava_modelo2

# Ava Modelo1
# MSE:  0.731158
# RMSE:  0.8550778
# MAE:  0.4921442
# RMSLE:  0.0397853
# Mean Residual Deviance :  0.731158

# Ava Modelo2
# MSE:  4.06569
# RMSE:  2.016356
# MAE:  1.461436
# RMSLE:  0.1026355
# Mean Residual Deviance :  4.06569



## Desliga o H2O
h2o.shutdown()


######  Escolhendo Melhor Algoritmo de Machine Learning

# - Utilizando as configurações da versão 3 (sem a criação das novas variáveis)


## Carregando dados
dados <- data.frame(read_xlsx("dataset/FEV-data-Excel.xlsx"))
dados <- dados[complete.cases(dados), ]


## Engenharia de Atributos

# Convertendo a variável variáveis chr para fator
dados <- dados %>%  
  mutate_if(is.character, factor) %>%
  mutate(across(c(Number.of.seats, Number.of.doors), as.factor))

# Normalização dos Dados (variáveis numéricas) (Exemplo 1 coluna ao final)
numeric_columns <- sapply(dados, is.numeric)
dados_nor <- dados %>%
  mutate(across(where(is.numeric), ~ scale(., center = min(.), scale = max(.) - min(.))))
rm(numeric_columns)

# Reverter Normalização
# dados_revertidos <- dados_nor %>%
#   mutate(across(where(is.numeric), ~ (. * (max(dados[, cur_column()]) - min(dados[, cur_column()])) + min(dados[, cur_column()]))))

## Selecionando variaveis
dados_nor <- dados_nor %>% 
  select(Make, Wheelbase..cm., Permissable.gross.weight..kg., 
         Minimal.price..gross...PLN., Length..cm., Width..cm.,
         mean...Energy.consumption..kWh.100.km.)
str(dados_nor)


#### Criando Modelos

## Dividindo os dados em treino e teste
set.seed(150)
indices <- createDataPartition(dados_nor$mean...Energy.consumption..kWh.100.km., p = 0.80, list = FALSE)
dados_treino <- dados_nor[indices, ]
dados_teste <- dados_nor[-indices, ]
rm(indices)

## Preparação dos dados para kNN (conversão para matriz)
dados_treino_knn <- data.matrix(dados_treino[,-7])
dados_teste_knn <- data.matrix(dados_teste[,-7])

## Preparação dos dados para Xgboost (conversão para matriz)
dados_treino_xgb <- xgb.DMatrix(data = dados_treino_knn, label = dados_treino$mean...Energy.consumption..kWh.100.km.)
dados_teste_xgb <- xgb.DMatrix(data = dados_teste_knn, label = dados_teste$mean...Energy.consumption..kWh.100.km.)

# Função para avaliar os modelos
avaliar_modelo <- function(model, model_type, dados_teste, verdadeiro = NULL) {
  if (model_type %in% c("lm", "svm", "gbm", "tree", "rm")) {
    previsoes <- predict(model, newdata = dados_teste[, -7])
  } else if (model_type == "knn") {
    previsoes <- model
    if (is.factor(previsoes)) {
      previsoes <- as.numeric(levels(previsoes))[previsoes]
    }
  } else if (model_type == "xgb") {
    previsoes <- predict(model, newdata = dados_teste)
    # Verifique se 'verdadeiro' não é nulo e use-o para as métricas
    if (!is.null(verdadeiro)) {
      rmse <- sqrt(mean((previsoes - verdadeiro)^2))
      mae <- mean(abs(previsoes - verdadeiro))
      r2 <- 1 - sum((verdadeiro - previsoes)^2) / sum((verdadeiro - mean(verdadeiro))^2)
      return(c(RMSE = rmse, MAE = mae, `R-squared` = r2))
    }
  } else {
    previsoes <- predict(model, newdata = dados_teste[, -7])
  }
  
  # Calculando as métricas de avaliação para outros modelos
  if (model_type != "xgb") {
    verdadeiro <- dados_teste[["mean...Energy.consumption..kWh.100.km."]]
    rmse <- sqrt(mean((previsoes - verdadeiro)^2))
    mae <- mean(abs(previsoes - verdadeiro))
    r2 <- cor(previsoes, verdadeiro)^2
    return(c(RMSE = rmse, MAE = mae, `R-squared` = r2))
  }
}


## Salvando avaliação dos modelos em uma List()
modelos_params <- list()

str(dados_treino)

## RandomForest
model_rf <- randomForest(mean...Energy.consumption..kWh.100.km. ~
                           Make + Wheelbase..cm. + Permissable.gross.weight..kg. + 
                           Minimal.price..gross...PLN. + Length..cm. + Width..cm.,
                        data = dados_treino, 
                        ntree = 100, nodesize = 10, importance = TRUE, set.seed(100))

avaliacao_rf <- avaliar_modelo(model_rf, "rm", dados_teste)
print(avaliacao_rf)



## SVM
model_svm <- svm(mean...Energy.consumption..kWh.100.km. ~ ., data = dados_treino)
avaliacao_svm <- avaliar_modelo(model_svm, "svm", dados_teste)
print(avaliacao_svm)



## kNN
model_knn <- knn(train = dados_treino_knn, test = dados_teste_knn, cl = dados_treino$mean...Energy.consumption..kWh.100.km., k = 1)
avaliacao_knn <- avaliar_modelo(model_knn, "knn", dados_teste)
print(avaliacao_knn)



## Árvore de Decisão
model_tree <- rpart(mean...Energy.consumption..kWh.100.km. ~ ., data = dados_treino)
avaliacao_tree <- avaliar_modelo(model_tree, "tree", dados_teste)
print(avaliacao_tree)



## Xgboost

# Definindo os parâmetros do modelo XGBoost
param <- list(
  objective = "reg:squarederror",  # Problema de regressão
  booster = "gbtree",             # Usar árvores como base
  eta = 0.3,                       # Taxa de aprendizado
  max_depth = 6,                   # Profundidade máxima da árvore
  min_child_weight = 1,            # Peso mínimo por folha
  subsample = 1,                  # Fração de amostras usadas para treinamento
  colsample_bytree = 1,           # Fração de colunas usadas por árvore
  nrounds = 100                    # Número de iterações (pode ser ajustado)
)

# Treinando o modelo XGBoost
model_xgboost <- xgboost(data = dados_treino_xgb, params = param, nrounds = 100)

# Chamando a função avaliar_modelo (precisa fornecer os rótulos verdadeiros)
verdadeiro_xgb <- getinfo(dados_teste_xgb, "label")
avaliacao_xgb <- avaliar_modelo(model_xgboost, "xgb", dados_teste_xgb, verdadeiro = verdadeiro_xgb)
print(avaliacao_xgb)




# Armazenando as avaliações em uma lista
modelos_params <- list(
  RandomForest = avaliacao_rf,
  SVM = avaliacao_svm,
  kNN = avaliacao_knn,
  DecisionTree = avaliacao_tree,
  XGBoost = avaliacao_xgb
)

# Exibindo as avaliações
print(modelos_params)















# Normalizando e revertendo uma coluna apenas
# dados_1col <- dados %>% 
#   select(Engine.power..KM.)
# max_coluna <- max(dados_1col)
# min_coluna <- min(dados_1col)
# dados_1col_nor <- as.data.frame(scale(dados_1col, center = min_coluna, scale = max_coluna - min_coluna))
# dados_1col_original <- dados_1col_nor * (max_coluna - min_coluna) + min_coluna


# dados_nor <- dados %>%
#   mutate_if(sapply(dados, is.numeric), scale)   # Utilizando dplyr