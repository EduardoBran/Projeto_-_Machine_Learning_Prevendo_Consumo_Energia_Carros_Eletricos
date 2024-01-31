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
library(caret)          # cria confusion matrix

