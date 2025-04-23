"""
# Scoras: Documentação de Novas Funcionalidades

Este documento fornece uma visão geral das novas funcionalidades implementadas na versão 0.3.1 da biblioteca Scoras.

## Classes Adicionadas

### Message
A classe `Message` representa uma mensagem em uma conversa entre agentes ou entre agentes e usuários.

```python
from scoras import Message

# Criar uma mensagem
mensagem = Message(
    role="user",
    content="Olá, como posso ajudar?",
    metadata={"timestamp": "2025-04-23T12:00:00Z"}
)

# Acessar propriedades da mensagem
print(mensagem.role)      # "user"
print(mensagem.content)   # "Olá, como posso ajudar?"
print(mensagem.metadata)  # {"timestamp": "2025-04-23T12:00:00Z"}
```

### Tool
A classe `Tool` representa uma ferramenta que pode ser usada por um agente para realizar ações específicas.

```python
from scoras import Tool

# Definir uma função para a ferramenta
def calcular_soma(a: int, b: int) -> int:
    return a + b

# Criar uma ferramenta
ferramenta = Tool(
    name="calculadora",
    description="Uma calculadora simples",
    function=calcular_soma
)

# Executar a ferramenta
resultado = ferramenta.execute_sync(a=5, b=3)
print(resultado)  # 8
```

### RAG
A classe `RAG` é a base para sistemas de Retrieval-Augmented Generation, combinando recuperação de documentos com geração de linguagem.

```python
from scoras import RAG, Agent, Document, SimpleRetriever

# Criar documentos
documentos = [
    Document(content="Scoras é uma biblioteca para criar agentes inteligentes.", metadata={"fonte": "docs"}),
    Document(content="Agentes podem usar ferramentas para realizar tarefas.", metadata={"fonte": "docs"})
]

# Criar um agente
agente = Agent("openai:gpt-4")

# Criar um recuperador
recuperador = SimpleRetriever(documentos)

# Criar um sistema RAG personalizado
class MeuRAG(RAG):
    async def run(self, query: str, top_k: int = 3) -> str:
        docs = await self.retriever.retrieve(query, top_k)
        contexto = "\n".join([doc.content for doc in docs])
        prompt = f"Contexto: {contexto}\n\nPergunta: {query}\n\nResposta:"
        return await self.agent.run(prompt)

# Instanciar o sistema RAG
meu_rag = MeuRAG(recuperador, agente)

# Usar o sistema RAG
resposta = meu_rag.run_sync("O que é Scoras?")
print(resposta)
```

### SimpleRAG
A classe `SimpleRAG` é uma implementação pronta para uso de um sistema RAG simples.

```python
from scoras import SimpleRAG, Agent, Document

# Criar documentos
documentos = [
    Document(content="Scoras é uma biblioteca para criar agentes inteligentes.", metadata={"fonte": "docs"}),
    Document(content="Agentes podem usar ferramentas para realizar tarefas.", metadata={"fonte": "docs"})
]

# Criar um agente
agente = Agent("openai:gpt-4")

# Criar um sistema SimpleRAG
rag = SimpleRAG(agente, documentos)

# Usar o sistema RAG
resposta = rag.run_sync("O que é Scoras?")
print(resposta)
```

### ScoreTracker
A classe `ScoreTracker` permite rastrear e analisar pontuações de complexidade em diferentes componentes e fluxos de trabalho.

```python
from scoras import ScoreTracker, Agent, Graph, Node, Edge

# Criar um rastreador de pontuação
tracker = ScoreTracker()

# Criar um agente e adicionar sua pontuação
agente = Agent("openai:gpt-4")
tracker.add_score("agente_gpt4", agente.get_complexity_score())

# Criar um grafo e adicionar sua pontuação
grafo = Graph("workflow_simples")
node1 = grafo.add_node("node1", lambda x: x * 2)
node2 = grafo.add_node("node2", lambda x: x + 1)
edge = grafo.add_edge("node1", "node2")
tracker.add_score("workflow_simples", grafo.get_complexity_score())

# Comparar pontuações
comparacao = tracker.compare_scores("agente_gpt4", "workflow_simples")
print(comparacao)
```

### ScorasConfig
A classe `ScorasConfig` fornece configurações para diversos aspectos da biblioteca Scoras.

```python
from scoras import ScorasConfig

# Criar uma configuração personalizada
config = ScorasConfig(
    enable_scoring=True,
    default_model="openai:gpt-4",
    default_temperature=0.7,
    default_max_tokens=1000,
    default_top_k=3,
    log_level="INFO"
)

# Acessar configurações
print(config.default_model)  # "openai:gpt-4"
print(config.default_temperature)  # 0.7
```

### WorkflowGraph
A classe `WorkflowGraph` estende a classe `Graph` com funcionalidades adicionais para definir fluxos de trabalho complexos.

```python
from scoras import WorkflowGraph, Node

# Criar um grafo de fluxo de trabalho
workflow = WorkflowGraph("workflow_complexo")

# Adicionar nós
node1 = workflow.add_node("node1", lambda x: x * 2)
node2 = workflow.add_node("node2", lambda x: x + 1)
node3 = workflow.add_node("node3", lambda x: x ** 2)

# Adicionar ramificação condicional
branch = workflow.add_branch("branch_condicional", lambda x: x > 10)
branch_node = branch.add_node("branch_node", lambda x: x / 2)

# Mesclar ramificação de volta ao fluxo principal
workflow.merge_branch("branch_condicional", "node3")

# Compilar e executar o fluxo de trabalho
executor = workflow.compile()
resultado = executor.run_sync(5)
print(resultado)
```

## Decoradores Adicionados

### tool
O decorador `tool` permite transformar funções em ferramentas que podem ser usadas por agentes.

```python
from scoras import tool, Agent

# Definir uma ferramenta usando o decorador
@tool(name="calculadora", description="Uma calculadora simples")
def calculadora(a: int, b: int, operacao: str = "soma") -> int:
    if operacao == "soma":
        return a + b
    elif operacao == "subtracao":
        return a - b
    elif operacao == "multiplicacao":
        return a * b
    elif operacao == "divisao":
        return a / b
    else:
        raise ValueError(f"Operação desconhecida: {operacao}")

# Usar a ferramenta diretamente
resultado = calculadora(5, 3, "soma")
print(resultado)  # 8

# Acessar a definição da ferramenta
print(calculadora.tool_name)  # "calculadora"
print(calculadora.tool_description)  # "Uma calculadora simples"
```

## Exemplos de Uso

### Exemplo Básico: Criar um Agente com Ferramentas

```python
from scoras import Agent, tool

# Definir ferramentas
@tool(name="calculadora", description="Uma calculadora simples")
def calculadora(a: int, b: int, operacao: str = "soma") -> int:
    if operacao == "soma":
        return a + b
    elif operacao == "subtracao":
        return a - b
    elif operacao == "multiplicacao":
        return a * b
    elif operacao == "divisao":
        return a / b
    else:
        raise ValueError(f"Operação desconhecida: {operacao}")

@tool(name="tradutor", description="Traduz texto para outro idioma")
def tradutor(texto: str, idioma_destino: str) -> str:
    # Simulação de tradução
    return f"[{texto}] traduzido para {idioma_destino}"

# Criar um agente
agente = Agent("openai:gpt-4")

# Adicionar ferramentas ao agente
agente.add_tool(calculadora.tool)
agente.add_tool(tradutor.tool)

# Usar o agente
resposta = agente.run_sync("Calcule 5 + 3 e traduza o resultado para francês")
print(resposta)
```

### Exemplo Avançado: Sistema RAG com Ferramentas

```python
from scoras import SimpleRAG, Agent, Document, tool

# Definir documentos
documentos = [
    Document(content="A capital do Brasil é Brasília.", metadata={"fonte": "geografia"}),
    Document(content="A capital da França é Paris.", metadata={"fonte": "geografia"}),
    Document(content="A capital do Japão é Tóquio.", metadata={"fonte": "geografia"})
]

# Definir ferramentas
@tool(name="pesquisar_populacao", description="Pesquisa a população de uma cidade")
def pesquisar_populacao(cidade: str) -> str:
    populacoes = {
        "brasília": "3,1 milhões",
        "paris": "2,1 milhões",
        "tóquio": "14 milhões"
    }
    return populacoes.get(cidade.lower(), "População desconhecida")

@tool(name="pesquisar_area", description="Pesquisa a área de uma cidade em km²")
def pesquisar_area(cidade: str) -> str:
    areas = {
        "brasília": "5.802 km²",
        "paris": "105 km²",
        "tóquio": "2.194 km²"
    }
    return areas.get(cidade.lower(), "Área desconhecida")

# Criar um agente
agente = Agent("openai:gpt-4")

# Adicionar ferramentas ao agente
agente.add_tool(pesquisar_populacao.tool)
agente.add_tool(pesquisar_area.tool)

# Criar um sistema RAG
rag = SimpleRAG(agente, documentos)

# Usar o sistema RAG
resposta = rag.run_sync("Qual é a capital da França e qual é sua população?")
print(resposta)
```

### Exemplo de Fluxo de Trabalho: Processamento de Dados

```python
from scoras import WorkflowGraph, Node, Edge

# Definir funções para os nós
def extrair_dados(entrada):
    # Simulação de extração de dados
    return {"valores": [1, 2, 3, 4, 5]}

def filtrar_dados(dados):
    # Filtrar apenas valores pares
    return {"valores": [v for v in dados["valores"] if v % 2 == 0]}

def transformar_dados(dados):
    # Multiplicar valores por 2
    return {"valores": [v * 2 for v in dados["valores"]]}

def agregar_dados(dados):
    # Calcular soma e média
    valores = dados["valores"]
    return {
        "valores": valores,
        "soma": sum(valores),
        "media": sum(valores) / len(valores) if valores else 0
    }

# Criar um grafo de fluxo de trabalho
workflow = WorkflowGraph("processamento_dados")

# Adicionar nós
workflow.add_node("extrair", extrair_dados)
workflow.add_node("filtrar", filtrar_dados)
workflow.add_node("transformar", transformar_dados)
workflow.add_node("agregar", agregar_dados)

# Adicionar arestas
workflow.add_edge("extrair", "filtrar")
workflow.add_edge("filtrar", "transformar")
workflow.add_edge("transformar", "agregar")

# Compilar e executar o fluxo de trabalho
executor = workflow.compile()
resultado = executor.run_sync({"fonte": "dados.csv"})
print(resultado)
```

## Conclusão

A versão 0.3.1 da biblioteca Scoras adiciona várias classes e funcionalidades importantes que tornam a criação de agentes, sistemas RAG e fluxos de trabalho mais flexível e poderosa. Estas novas funcionalidades mantêm a consistência com o sistema de pontuação de complexidade existente, permitindo medir e comparar a complexidade de diferentes componentes e fluxos de trabalho.
"""
