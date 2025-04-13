"""
Exemplo de uso da biblioteca Scoras.

Este script demonstra como criar e usar um agente simples com a biblioteca Scoras.

Author: Anderson L. Amaral
"""

import asyncio
from pydantic import BaseModel, Field

import scoras as sc
from scoras.tools import register_tool


# Definindo um modelo de resultado
class ResultadoClima(BaseModel):
    cidade: str = Field(..., description="Nome da cidade")
    temperatura: float = Field(..., description="Temperatura em graus Celsius")
    condicao: str = Field(..., description="Condição climática atual")


# Criando uma ferramenta
@register_tool(name="calcular", description="Realiza cálculos matemáticos")
async def calcular(expressao: str) -> float:
    """Calcula o resultado de uma expressão matemática."""
    # Implementação simplificada e segura
    import ast
    import operator
    
    # Operadores permitidos
    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
    }
    
    def eval_expr(node):
        if isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.BinOp):
            return operators[type(node.op)](eval_expr(node.left), eval_expr(node.right))
        else:
            raise ValueError(f"Operação não suportada: {type(node).__name__}")
    
    try:
        return eval_expr(ast.parse(expressao, mode='eval').body)
    except Exception as e:
        raise ValueError(f"Erro ao calcular expressão: {str(e)}")


async def main():
    # Criando um agente simples
    agente = sc.Agent(
        model="openai:gpt-4o",
        system_prompt="Você é um assistente útil e conciso."
    )
    
    # Executando o agente
    resposta = await agente.run("Qual é a capital do Brasil?")
    print(f"Resposta: {resposta}")
    
    # Criando um agente com resultado tipado
    agente_clima = sc.Agent(
        model="openai:gpt-4o",
        system_prompt="Você é um assistente especializado em clima.",
        result_type=ResultadoClima
    )
    
    # Executando o agente com resultado tipado
    resultado = await agente_clima.run("Como está o clima em São Paulo hoje?")
    print(f"Cidade: {resultado.cidade}")
    print(f"Temperatura: {resultado.temperatura}°C")
    print(f"Condição: {resultado.condicao}")
    
    # Criando um agente com ferramentas
    agente_calculadora = sc.Agent(
        model="openai:gpt-4o",
        system_prompt="Você é um assistente matemático.",
        tools=[calcular]
    )
    
    # Executando o agente com ferramentas
    resposta = await agente_calculadora.run("Quanto é 15 * 7 + 22?")
    print(f"Resultado do cálculo: {resposta}")
    
    # Criando um sistema RAG simples
    from scoras.rag import Document, create_rag_system
    
    documentos = [
        Document(content="O Brasil é um país na América do Sul com população de 214 milhões."),
        Document(content="Brasília é a capital do Brasil, fundada em 1960."),
        Document(content="São Paulo é a maior cidade do Brasil, com cerca de 12 milhões de habitantes.")
    ]
    
    sistema_rag = create_rag_system(
        agent=sc.Agent(model="openai:gpt-4o"),
        documents=documentos
    )
    
    # Executando o sistema RAG
    resposta = await sistema_rag.run("Qual é a capital do Brasil e quando foi fundada?")
    print(f"Resposta do RAG: {resposta}")


if __name__ == "__main__":
    asyncio.run(main())
