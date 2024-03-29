####  PROJETO COM FEEDBACK  (Contexto) ####

# Configurando o diretório de trabalho
setwd("~/Desktop/DataScience/CienciaDeDados/1.Big-Data-Analytics-com-R-e-Microsoft-Azure-Machine-Learning/20.Projetos_com_Feedback/1.Prevendo_Consumo_Energia_Carros_Eletricos")
getwd()



########################   Machine Learning em Logística Prevendo o Consumo de Energia de Carros Elétricos   ########################


## Sobre o Script

# - Este script contém a história e o contexto do projeto.



## Contexto:

# - Uma empresa da área de transporte e logística deseja migrar sua frota para carros elétricos com o objetivo de reduzir os custos.

# - Antes de tomar a decisão, a empresa gostaria de prever o consumo de nergia de carros elétricos com base em diversos fatores de
#   utilização e características dos veículos.


## Dados:

# - Este conjunto de dados lista todos os carros totalmente elétricos com seus atributos (propriedades) disponíveis atualmente no
#   mercado. A coleção não contém dados sobre carros híbridos e carros elétricos dos chamados "extensores de alcance". Os carros a
#   hidrogênio também não foram incluídos no conjunto de dados devido ao número insuficiente de modelos produzidos em masso e à
#   especificidade diferente (comparado aos veículos elétricos) do veículo, incluindo os diferentes métodos de carregamento.

# - O conjunto de dados inclui carros que, a partir de 2 de dezembro de 2020, poderiam ser adquiridos na Polônia como novos em um 
#   revendedor autorizado e aqueles disponíveis em pré-venda pública e geral, mas apenas se uma lista de preços publicamente
#   disponível com versões de equipamentos e parâmetros técnicos completos estivesse disponível. A lista não inclui carros
#   descontinuados que não podem ser adquiridos como novos de um revendedor autorizado (também quando não estão disponíveis em estoque).

# - O conjunto de dados de carros elétricos inclui todos os carros totalmente elétricos no mercado primário que foram obtidos de 
#   materiais oficiais (especificações técnicas e catálogos) fornecidos por fabricantes de automóveis com licença para vender carros
#   na Polônia. Esses materiais foram baixados de seus sites oficiais. Caso os dados fornecidos pelo fabricante estivessem incompletos,
#   as informações eram complementadas com dados do AutoCatálogo SAMAR (link disponível na seção Referências da fonte de dados).

#  <- https://data.mendeley.com/datasets/tb9yrptydn/2


# Objetivo:

# - Usando este incrível dataset com dados reais disponíveis publicamente, você deverá construir um modelo de Machine Learning capaz
#   de prever o consumo de energia de carros elétricos com base em diversos fatores, tais como o tipo e número de motores elétricos
#   do veículo, o peso do veículo, a capacidade de carga, entre outros atributos.

# - Seu trabalho é construir um modelo de Machine Learning capaz de prever o consumo de energia de veículos elétricos.
