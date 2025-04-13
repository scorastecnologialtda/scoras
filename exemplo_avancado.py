"""
Exemplo avançado de uso da biblioteca Scoras.

Este script demonstra recursos avançados da biblioteca Scoras,
incluindo grafos, ferramentas HTTP e sistemas multi-agente.

Author: Anderson L. Amaral
"""

import asyncio
from typing import Dict, List, Any
from pydantic import BaseModel, Field

import scoras as sc
from scoras.agents import AssistantAgent, MultiAgentSystem
from scoras.tools import create_http_tool, create_python_tool


# Definindo modelos para resultados tipados
class ResultadoPesquisa(BaseModel):
    query: str = Field(..., description="Consulta de pesquisa")
    resultados: List[str] = Field(..., description="Lista de resultados encontrados")
    total: int = Field(..., description="Número total de resultados")


class ResultadoAnalise(BaseModel):
    texto: str = Field(..., description="Texto analisado")
    sentimento: str = Field(..., description="Sentimento do texto (positivo, negativo, neutro)")
    palavras_chave: List[str] = Field(..., description="Palavras-chave identificadas")
    resumo: str = Field(..., description="Resumo do texto")


# Funções para ferramentas
def analisar_texto(texto: str) -> Dict[str, Any]:
    """Analisa um texto e retorna informações sobre ele."""
    # Implementação simulada
    import random
    
    sentimentos = ["positivo", "negativo", "neutro"]
    palavras = texto.lower().split()
    palavras_chave = list(set([p for p in palavras if len(p) > 4]))[:5]
    
    return {
        "texto": texto,
        "sentimento": random.choice(sentimentos),
        "palavras_chave": palavras_chave,
        "resumo": f"Este é um resumo simulado de: {texto[:50]}..."
    }


async def pesquisar_web(query: str, num_results: int = 3) -> Dict[str, Any]:
    """Pesquisa informações na web."""
    # Implementação simulada
    await asyncio.sleep(1)  # Simula o tempo de resposta
    
    resultados = [
        f"Resultado {i+1} para '{query}': Informação simulada sobre {query}."
        for i in range(num_results)
    ]
    
    return {
        "query": query,
        "resultados": resultados,
        "total": num_results
    }


async def main():
    # Criando ferramentas
    ferramenta_analise = create_python_tool(
        function=analisar_texto,
        name="analisar_texto",
        description="Analisa um texto e retorna informações sobre ele"
    )
    
    ferramenta_pesquisa = create_python_tool(
        function=pesquisar_web,
        name="pesquisar_web",
        description="Pesquisa informações na web",
        is_async=True
    )
    
    # Criando agentes especializados
    agente_pesquisador = AssistantAgent(
        model="openai:gpt-4o",
        name="Pesquisador",
        tools=[ferramenta_pesquisa],
        result_type=ResultadoPesquisa
    )
    
    agente_analista = AssistantAgent(
        model="openai:gpt-4o",
        name="Analista",
        tools=[ferramenta_analise],
        result_type=ResultadoAnalise
    )
    
    # Criando um sistema multi-agente
    sistema = MultiAgentSystem({
        "pesquisador": agente_pesquisador,
        "analista": agente_analista
    })
    
    # Executando agentes individualmente
    resultado_pesquisa = await sistema.run("pesquisador", "Quais são as últimas notícias sobre IA?")
    print(f"Consulta: {resultado_pesquisa.query}")
    print(f"Resultados encontrados: {len(resultado_pesquisa.resultados)}")
    for i, resultado in enumerate(resultado_pesquisa.resultados, 1):
        print(f"  {i}. {resultado}")
    
    # Executando uma sequência de agentes
    resultados = await sistema.run_sequence(
        "Analise as notícias sobre inteligência artificial",
        ["pesquisador", "analista"]
    )
    
    # O resultado final é do último agente (analista)
    analise = resultados[-1]
    print("\nAnálise de Sentimento:")
    print(f"Texto: {analise.texto}")
    print(f"Sentimento: {analise.sentimento}")
    print(f"Palavras-chave: {', '.join(analise.palavras_chave)}")
    print(f"Resumo: {analise.resumo}")
    
    # Demonstrando o uso de grafos
    from pydantic import BaseModel
    
    class EstadoProcessamento(BaseModel):
        texto: str
        etapa: str = "inicio"
        resultado: Dict[str, Any] = {}
    
    # Criando um grafo
    grafo = sc.Graph(state_type=EstadoProcessamento)
    
    # Definindo nós do grafo
    def extrair_entidades(estado: EstadoProcessamento) -> Dict[str, Any]:
        print(f"Extraindo entidades de: {estado.texto[:30]}...")
        # Simulação de extração de entidades
        entidades = [palavra for palavra in estado.texto.split() if palavra[0].isupper()]
        return {"etapa": "entidades_extraidas", "resultado": {"entidades": entidades}}
    
    def analisar_sentimento(estado: EstadoProcessamento) -> Dict[str, Any]:
        print(f"Analisando sentimento...")
        # Simulação de análise de sentimento
        import random
        sentimento = random.choice(["positivo", "negativo", "neutro"])
        
        resultado = estado.resultado.copy()
        resultado["sentimento"] = sentimento
        
        return {"etapa": "sentimento_analisado", "resultado": resultado}
    
    def gerar_resumo(estado: EstadoProcessamento) -> Dict[str, Any]:
        print(f"Gerando resumo...")
        # Simulação de geração de resumo
        resumo = f"Resumo: {estado.texto[:50]}..."
        
        resultado = estado.resultado.copy()
        resultado["resumo"] = resumo
        
        return {"etapa": "resumo_gerado", "resultado": resultado}
    
    # Adicionando nós ao grafo
    grafo.add_node("extrair_entidades", extrair_entidades)
    grafo.add_node("analisar_sentimento", analisar_sentimento)
    grafo.add_node("gerar_resumo", gerar_resumo)
    
    # Definindo arestas
    grafo.add_edge("start", "extrair_entidades")
    grafo.add_edge("extrair_entidades", "analisar_sentimento")
    grafo.add_edge("analisar_sentimento", "gerar_resumo")
    grafo.add_edge("gerar_resumo", "end")
    
    # Compilando e executando o grafo
    grafo_compilado = grafo.compile()
    
    estado_final = await grafo_compilado.run({
        "texto": "A Inteligência Artificial está revolucionando diversos setores da economia. "
                "Empresas como Google e Microsoft estão investindo bilhões em pesquisa e desenvolvimento."
    })
    
    print("\nResultado do processamento do grafo:")
    print(f"Entidades: {estado_final['resultado'].get('entidades')}")
    print(f"Sentimento: {estado_final['resultado'].get('sentimento')}")
    print(f"Resumo: {estado_final['resultado'].get('resumo')}")


if __name__ == "__main__":
    asyncio.run(main())
