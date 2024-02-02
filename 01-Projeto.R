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
library(rpart)          # carrega algoritimo de ML (árvore de decisão)
library(e1071)          # carrega algoritimo de ML (SVM)

library(caret)          # cria confusion matrix



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
# - Cria duas novas variáveis
# - Utiliza Feature Selection
# - Cria 1 Tipo de Modelo (RandomForest)


## Engenharia de Atributos

# Convertendo a variável variáveis chr para fator
dados <- dados %>%  
  mutate_if(is.character, factor)

# Criando novas variáveis
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
# - Aplica Normalização nas variásveis do tipo int
# - Cria novas novas variáveis do tipo factor a partir de variáveis int
# - Utiliza Feature Selection
# - Cria 1 Tipo de Modelo (RandomForest)

## Engenharia de Atributos

# Convertendo a variável variáveis chr para fator
dados <- dados %>%  
  mutate_if(is.character, factor)

dados <- dados %>% 
  select(-Number.of.seats, -Number.of.doors)

# Normalização dos Dados (Menos as Variáveis Number.of.seats e Number.of.doors)
# dados_nor <- dados %>%
#   mutate_if(sapply(dados, is.numeric), scale)

# Identificar colunas numéricas e fator
numeric_columns <- sapply(dados, is.numeric)
factor_columns <- sapply(dados, is.factor)

# Se houver colunas numéricas, calcular os máximos e mínimos
if (any(numeric_columns)) {
  maxs <- apply(dados[, numeric_columns], 2, max)
  mins <- apply(dados[, numeric_columns], 2, min)
  
  # Normalizando apenas colunas numéricas
  dados_normalizados <- as.data.frame(scale(dados[, numeric_columns], center = mins, scale = maxs - mins))
  
  # Adicionar de volta as colunas do tipo factor
  dados_normalizados <- cbind(dados[, factor_columns], dados_normalizados)
} else {
  print("Não há colunas numéricas no conjunto de dados.")
}

head(dados,)
head(dados_normalizados, 2)



# Revertendo a normalização
names(dados)
dados_1col <- dados %>% 
  select(Engine.power..KM.)


# APLICAR A NORMALIZAÇÃO EM UMA ÚNICA VARIÁVEL E TENTAR REVERTER

max_coluna <- max(dados_1col)
min_coluna <- min(dados_1col)
dados_1col_nor <- as.data.frame(scale(dados_1col, center = min_coluna, scale = max_coluna - min_coluna))

# Reverter a normalização para a coluna
dados_1col_normalizado <- dados_1col_nor * (max_coluna - min_coluna) + min_coluna

head(dados_1col,2)
head(dados_1col_nor,2)
head(dados_1col_normalizado,2)

# Criando novas variáveis (Categóricas)


# TRANSFORMAR VARIAVEL QUANTIDADE DE PORTAS E QUANTIDADE DE OUTRA COISA E OUTRAS VARIAVEIS NUMERICAS EM FACTOR






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
indices <- createDataPartition(dados$mean...Energy.consumption..kWh.100.km., p = 0.80, list = FALSE)
dados_treino <- dados[indices, ]
dados_teste <- dados[-indices, ]
rm(indices)

# Criando o modelo preditivo (RandomForest)
modelo <- randomForest(mean...Energy.consumption..kWh.100.km. ~ 
                         Wheelbase..cm. + Minimal.price..gross...PLN. + Battery.Range.Ratio + Make + Permissable.gross.weight..kg. + 
                         Length..cm. + Maximum.torque..Nm. + Minimal.empty.weight..kg.,
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









#### Melhorias

# Retirar variável Car Full Name
# Transformar todas as variáveis chr em fac

# Realizar Normalização nas variáveis Numéricas
# Criar novas variáveis que serão transformadas de variáveis numéricas para variáveis categórias

# Realizar novas análises de Seleção de Variáveis
# Testar outros modelos
# Utilizar AutoML
names(dados)


## Carregando dados
dados <- data.frame(read_xlsx("dataset/FEV-data-Excel.xlsx"))


# Convertendo a variável variáveis chr para fator e removendo variável nome dos carros
dados <- dados %>%  
  mutate_if(is.character, factor) %>% 
  select(-Car.full.name)


# Realizar Normalização Dados Numéricos


# Remover Normalização


# Verificando e removendo valores ausentes
colSums(is.na(dados))
dados <- dados[complete.cases(dados), ]

# Criando novas variáveis
dados$Weight.Power.Ratio <- dados$Minimal.empty.weight..kg. / dados$Engine.power..KM.  # Relação entre Peso e Potência do Motor
dados$Battery.Range.Ratio <- dados$Battery.capacity..kWh. / dados$Range..WLTP...km.    # Relação entre Capacidade da Bateria e Alcance


## Seleção de Variáveis


## Criar Modelos