## Sobre o Projeto

- Em uma jornada visionária rumo à sustentabilidade, uma empresa de transporte e logística decidiu explorar a viabilidade de migrar sua frota
  para veículos elétricos (EVs), um passo audacioso para reduzir custos operacionais e impacto ambiental. O coração dessa iniciativa inovadora
  era desenvolver um modelo de Machine Learning capaz de prever com precisão o consumo de energia dos EVs, baseando-se em um conjunto de dados
  detalhado que retrata os veículos elétricos disponíveis no mercado polonês até dezembro de 2020, utilizando a linguagem R para toda a análise
  de dados e modelagem.


## Preparação dos Dados

  - Antes de iniciar a construção dos modelos, o desafio inicial era lidar com dados ausentes, especialmente na variável-alvo. A decisão de remover
  linhas com essas lacunas foi crucial, pois dados incompletos poderiam introduzir viés significativo e afetar a qualidade das previsões.
  Essa abordagem assegurou a integridade do modelo desde o início, permitindo uma análise robusta e confiável do consumo de energia dos EVs.


## Explorando os Dados

  - A exploração deste conjunto de dados desencadeou uma série de inovações e experimentações, evoluindo através de várias versões do modelo,
  cada uma aprimorando a abordagem anterior:

-> Versão 1: A jornada iniciou-se mantendo os tipos originais de dados e introduzindo duas novas variáveis, optando por um modelo RandomForest.
             Este primeiro passo estabeleceu um marco inicial para comparações futuras.

-> Versão 2: Melhorias foram realizadas com a conversão de variáveis categóricas em fatores e a implementação de seleção de recursos, melhorando
             significativamente os indicadores de desempenho do modelo.

-> Versão 3: A normalização das variáveis numéricas e uma seleção de recursos ainda mais refinada marcaram um avanço na precisão do modelo,
             pavimentando o caminho para insights mais profundos sobre o consumo dos EVs.

-> Versão 4 e 5: Estas versões trouxeram inovações na engenharia de atributos e técnicas de duplicação de linhas, respectivamente, cada uma
                 visando ampliar a robustez do modelo.

-> Versão 6 (AutoML): A exploração do AutoML ofereceu uma nova perspectiva sobre a seleção de modelos, embora não tenha superado as versões
                      anteriores em termos de precisão.


## Conclusão e Impacto

- A escolha da Versão 3 e do modelo SVM emergiu como a combinação vitoriosa, demonstrando notável precisão na previsão do consumo de energia
  dos EVs. Este modelo, especificamente um SVM com kernel radial e parâmetros ajustados para epsilon-regression, se destacou pela sua habilidade
  em capturar a complexidade dos dados e fornecer previsões confiáveis.


  ## Destaque da Conclusão

  - A capacidade de prever com precisão o consumo de energia dos veículos elétricos abre portas para uma gestão de frota mais eficiente e
  sustentável. O projeto culminou na criação de uma interface gráfica, utilizando o framework Shiny no R, permitindo aos usuários visualizar e
  testar o modelo de previsão de forma interativa. Esta interface, uma inovação por si só, facilita a experimentação com diferentes configurações
  de veículos, oferecendo insights imediatos sobre o consumo de energia esperado.

- Assim, a empresa não está apenas equipada com uma ferramenta poderosa para decisões estratégicas, mas também promove uma abordagem transparente
  e acessível à gestão da frota de EVs. Este projeto exemplifica como a ciência de dados e o Machine Learning, aplicados através da linguagem R
  e explorando o potencial do algoritmo SVM, podem enfrentar desafios reais e promover a sustentabilidade no setor de transporte e logística,
  pavimentando o caminho para um futuro mais verde e eficiente.
