<div align="center">

# 🤝🏽🥇 ALLMART - FIDELIZAÇÃO DE CLIENTES 🥇🤝🏽

<img src="https://blog.acelerato.com/wp-content/uploads/2015/10/fidelidade.png" width=70% height=50% title="Health-Insurance-Ranking" alt="project_cover_image"/>

</div>

## 📖 Contexto do Negócio

> *Disclaimer: O contexto a seguir é completamente fictício. A empresa, o contexto, o CEO, as perguntas de negócio foram idealizadas para simular um problema real de negócio.*

***

A empresa All Mart é uma empresa Outlet Multimarcas, ou seja, ela comercializa produtos de segunda linha de várias marcas a um preço menor, através de um e-commerce.

Em pouco mais de 1 ano de operação, o time de marketing percebeu que alguns clientes da sua base compram produtos mais caro, com alta frequência e acabam contribuindo com uma parcela significativa do faturamento da empresa.

Baseado nessa percepção, o time de marketing vai lançar um programa de fidelidade para os melhores clientes da base, chamado _insiders_. Mas o time não tem um conhecimento avançado em análise de dados para eleger os participantes do programa.

Por esse motivo, o time de marketing requisitou aao time de dados uma seleção de clientes elegíveis ao programa, usando técnicas avançadas de manipulação de dados.

***

## 📌 Desafio do Negócio

Você faz parte do time de cientistas de dados da empresa All Mart, que precisa determinar quem são os clientes elegíveis para participar do programa _insiders_. Em posse dessta lista, o time de marketing fará uma sequência de ações personalizadas e exclusivas ao grupo, de modo a aumentar o faturamento e a frequência de compra. 

Como resultado para esse projeto, é esperado que você entregue uma lista de pesaos elegíveis a participar do programa _insiders_, junto com um relatório respondendo às seguintes perguntas:

1. Quem são as pessoas elegíveis para participar do programa _insiders_?
2. Quantos clientes farõa parte do grupo?
3. Quais as principais características desses clientes?
4. Qual a porcentagem de contribuição do faturamento, vinda do programa?
5. Qual a expectativa de faturamento (LTV) desse grupo para os próximos meses?
6. Quais as condições para uma pessoa ser elegível aos _insiders_?
7. Quais as condiçlões para um apessoa ser removida dos _insiders_?


<<<<<<< HEAD
## ⚒️ Ferramentas

- Python 3.9
- UV como gerenciador de pacotes


=======
>>>>>>> 49cc40e6c5e684a633262d1f02d26e8e2a5f25c7
## 💾 Dados

O conjunto de dados está disponível na plataforma Kaggle, através do link: https://www.kaggle.com/vik2012kvs/high-value-customers-identification

Cada linha representa uma transição de venda, que ocorreu entre o período de novembro de 2016 e dezembro de 2017.

O conjunto de dados inclui as seguintes informações:

* Invoice Number: identificador único de cada transação.
* Stock Code Product: código do item.
* Description Product: nome do item.
* Quantity: quantidade de cada item comprado por transação.
* Invoice Date: dia em que ocorreu a transação.
* Unit Price: preço do produto por unidade.
* Customer ID: identificador único do cliente.
* Country: país em que o cliente reside.


## 🔬 Proposta de Solução 

Como parte da solução do projeto, propusemos o uso da metodologia cíclica conhecida por CRISP-DS (Cross-Industry Standard Process for Data Science). Este processo baseia-se em uma separação lógica e clara dos passos para desenvolvimento da solução e em sua estrutura cíclica, de forma que um ciclo consiste em percorrer todas as fases do desenvolvimento e a entrega ágil de uma solução (Minimum Viable Product). Sua natureza cíclica permite não só o refatoramento do código como também a formulação de outras hipóteses, criação de novas features, melhora dos modelos, fine tuning, etc.

![crispds](docs/crispds_figma_corrigido.jpg)

Também desenhamos uma estratégia baseada na metodologia IOT (Input, Output & Tasks) como parte da solução, funcionando basicamente como um _Sprint Backlog_ do projeto, linkando diretamente as perguntas realizadas na elaboração do problema do negócio.

![iot](docs/IoT_method.png)

<br>

## 📉 Modelo RFM - RFM Análise

![rfm-model](docs/modelo_rfm.png)

1. Champions

    a. Compras recentes, frequentes com alto valor gasto.

    b. Prêmios para esses clientes.

2. Potential Loyalists

    a. Compras recentes, boa frequência e bom valor gasto.

<<<<<<< HEAD
    b. Programa de Fidelização e Up-sell.

3. New Customers

    a. Compra recente, baixa frequência.
=======
    b. Programa de Fidelização e Upssell

3. New Customers

    a. Compra recente, baixa frequência
>>>>>>> 49cc40e6c5e684a633262d1f02d26e8e2a5f25c7

    b. Construção de Relacionamento e ofertas especiais.

4. At Risk Customer

<<<<<<< HEAD
    a. "Faz tempo que não compra".
    
    b. Campanhas de reativação, ofertas, produtos.

5. Can't Lose Them

    a. "One Time Clients", clientes que fizeram uma única compra ou baixíssima frequência.
    
    b. Envio de e-mails perguntando se foi tudo ok.
=======
    a. "Faz tempo que não compra"
    
    b. Campnhas de reativação, ofertas, produtos

5. Can't Lose Them

TBD
>>>>>>> 49cc40e6c5e684a633262d1f02d26e8e2a5f25c7

<br>

## 🔮 Clusterização e suas Propriedades

Existem 2 principais propriedades para avaliar a performance de um agrupamento (_clusterização_):


- **Distância do centróide em relação aos pontos de seu cluster:**
        O objetivo aqui é minimizar essa distância entre todos os pontos de um mesmo cluster.

- **Distância entre clusters diferentes**:
        Aqui desejamos que a distância seja a _maior possível_, para evitar o fenômeno de *overlapping de dados*. 

![distancias](docs/distances_cores.jpg)


De uma maneira mais formal:
   1. *Compactness* (Compacidade) ou *Cohesion*

        - *WSS (Within-Cluster Sum of Squares)*: Os clusters devem ser compactos.
            Desvantagem: Não considera as distâncias entre clusters (possível _overlapping_)
            <br>
            <br>
            
   2. *Separation* (Separação)
        - *SS (Silhouette Score)* - Clusters devem ser distantes entre si.


<br>


## 📉 Resultados

TBD

<br>

## 👣 Próximos Passos

TBD

<br>

## 🔗 Referências

1. [Cohort Analysis with Python](http://www.gregreda.com/2015/08/23/cohort-analysis-with-python/)

2. 

## ✍ Autor

- [Leandro Destefani](https://github.com/leassis91)

[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/leandrodestefani) [![gmail](https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:leassis.destefani@gmail.com) [![kaggle](https://img.shields.io/badge/Kaggle-3776AB?style=for-the-badge&logo=Kaggle&logoColor=white)](https://kaggle.com/leandrodestefani)


***

## 💪 Como contribuir

1. Dê um fork no projeto.
2. Crie uma nova branch com suas mudanças: `git checkout -b my-feature`
3. Salve-as e então crie um commit com uma mensagem com o que você alterou: `git commit -m" feature: My new feature "`
4. Confirme suas alterações: `git push origin my-feature`



***