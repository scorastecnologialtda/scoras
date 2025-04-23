#!/usr/bin/env python3
"""
Script para testar a biblioteca Scoras após a atualização.
Este script verifica se todas as classes e funcionalidades implementadas estão funcionando corretamente.
"""

import sys
import os
import importlib
from typing import List, Dict, Any, Optional

def print_header(text: str) -> None:
    """Imprime um cabeçalho formatado."""
    print("\n" + "=" * 80)
    print(f" {text} ".center(80, "="))
    print("=" * 80)

def print_result(name: str, success: bool, message: Optional[str] = None) -> None:
    """Imprime o resultado de um teste."""
    status = "✅ SUCESSO" if success else "❌ FALHA"
    print(f"{status} | {name}")
    if message and not success:
        print(f"  └─ {message}")

def test_imports() -> List[Dict[str, Any]]:
    """Testa a importação de todas as classes e funções."""
    results = []
    
    # Lista de importações para testar
    imports = [
        {"name": "Graph", "module": "scoras.core"},
        {"name": "Node", "module": "scoras.core"},
        {"name": "Edge", "module": "scoras.core"},
        {"name": "Message", "module": "scoras.core"},
        {"name": "Tool", "module": "scoras.core"},
        {"name": "RAG", "module": "scoras.core"},
        {"name": "ScoreTracker", "module": "scoras.core"},
        {"name": "ScorasConfig", "module": "scoras.core"},
        {"name": "WorkflowGraph", "module": "scoras.core"},
        {"name": "ScoringMixin", "module": "scoras.core"},
        {"name": "Agent", "module": "scoras.agents"},
        {"name": "Document", "module": "scoras.rag"},
        {"name": "SimpleRAG", "module": "scoras.rag"},
        {"name": "tool", "module": "scoras.tools"},
        {"name": "ToolChain", "module": "scoras.tools"},
        {"name": "ToolRouter", "module": "scoras.tools"},
        {"name": "ToolBuilder", "module": "scoras.tools"},
        {"name": "ToolResult", "module": "scoras.tools"}
    ]
    
    for item in imports:
        name = item["name"]
        module = item["module"]
        try:
            # Importa o módulo
            mod = importlib.import_module(module)
            # Verifica se a classe/função existe no módulo
            if hasattr(mod, name):
                results.append({
                    "name": f"Importação de {name} de {module}",
                    "success": True
                })
            else:
                results.append({
                    "name": f"Importação de {name} de {module}",
                    "success": False,
                    "message": f"{name} não encontrado em {module}"
                })
        except ImportError as e:
            results.append({
                "name": f"Importação de {name} de {module}",
                "success": False,
                "message": str(e)
            })
        except Exception as e:
            results.append({
                "name": f"Importação de {name} de {module}",
                "success": False,
                "message": f"Erro inesperado: {str(e)}"
            })
    
    return results

def test_functionality() -> List[Dict[str, Any]]:
    """Testa a funcionalidade básica das classes implementadas."""
    results = []
    
    # Teste 1: Criar um agente
    try:
        from scoras import Agent
        agent = Agent("openai:gpt-4")
        score = agent.get_complexity_score()
        results.append({
            "name": "Criação de Agent e obtenção do score de complexidade",
            "success": isinstance(score, dict) and "total_score" in score
        })
    except Exception as e:
        results.append({
            "name": "Criação de Agent e obtenção do score de complexidade",
            "success": False,
            "message": str(e)
        })
    
    # Teste 2: Criar um documento
    try:
        from scoras import Document
        doc = Document(content="Este é um documento de teste", metadata={"source": "teste"})
        results.append({
            "name": "Criação de Document",
            "success": doc.content == "Este é um documento de teste" and doc.metadata["source"] == "teste"
        })
    except Exception as e:
        results.append({
            "name": "Criação de Document",
            "success": False,
            "message": str(e)
        })
    
    # Teste 3: Criar uma ferramenta usando o decorador
    try:
        from scoras import tool
        
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
        
        # Verifica se o decorador adicionou os atributos esperados
        has_tool_attr = hasattr(calculadora, "tool")
        has_name_attr = hasattr(calculadora, "tool_name") and calculadora.tool_name == "calculadora"
        
        # Testa a execução da função
        result = calculadora(5, 3, "soma")
        
        results.append({
            "name": "Decorador tool e execução da função",
            "success": has_tool_attr and has_name_attr and result == 8
        })
    except Exception as e:
        results.append({
            "name": "Decorador tool e execução da função",
            "success": False,
            "message": str(e)
        })
    
    # Teste 4: Criar um grafo de workflow
    try:
        from scoras import Graph, Node, Edge
        
        def node_func(x):
            return x * 2
        
        graph = Graph("teste")
        node1 = graph.add_node("node1", node_func)
        node2 = graph.add_node("node2", node_func)
        edge = graph.add_edge("node1", "node2")
        
        results.append({
            "name": "Criação de Graph, Node e Edge",
            "success": "node1" in graph.nodes and "node2" in graph.nodes and len(graph.edges) == 1
        })
    except Exception as e:
        results.append({
            "name": "Criação de Graph, Node e Edge",
            "success": False,
            "message": str(e)
        })
    
    # Teste 5: Criar uma mensagem
    try:
        from scoras import Message
        
        msg = Message(role="user", content="Olá, mundo!")
        
        results.append({
            "name": "Criação de Message",
            "success": msg.role == "user" and msg.content == "Olá, mundo!"
        })
    except Exception as e:
        results.append({
            "name": "Criação de Message",
            "success": False,
            "message": str(e)
        })
    
    # Teste 6: Verificar a versão
    try:
        import scoras
        
        results.append({
            "name": "Verificação da versão",
            "success": scoras.__version__ == "0.3.1",
            "message": f"Versão atual: {scoras.__version__}, esperada: 0.3.1" if scoras.__version__ != "0.3.1" else None
        })
    except Exception as e:
        results.append({
            "name": "Verificação da versão",
            "success": False,
            "message": str(e)
        })
    
    return results

def main() -> None:
    """Função principal para executar todos os testes."""
    print_header("TESTE DA BIBLIOTECA SCORAS")
    
    # Verifica se o pacote scoras está instalado
    try:
        import scoras
        print(f"Versão do Scoras: {scoras.__version__}")
    except ImportError:
        print("❌ ERRO: O pacote scoras não está instalado.")
        print("Por favor, instale o pacote antes de executar os testes:")
        print("pip install scoras")
        sys.exit(1)
    
    # Testa as importações
    print_header("TESTE DE IMPORTAÇÕES")
    import_results = test_imports()
    for result in import_results:
        print_result(result["name"], result["success"], result.get("message"))
    
    # Testa a funcionalidade
    print_header("TESTE DE FUNCIONALIDADE")
    func_results = test_functionality()
    for result in func_results:
        print_result(result["name"], result["success"], result.get("message"))
    
    # Resumo dos resultados
    print_header("RESUMO DOS RESULTADOS")
    total_tests = len(import_results) + len(func_results)
    passed_tests = sum(1 for r in import_results if r["success"]) + sum(1 for r in func_results if r["success"])
    
    print(f"Total de testes: {total_tests}")
    print(f"Testes bem-sucedidos: {passed_tests}")
    print(f"Testes com falha: {total_tests - passed_tests}")
    
    if passed_tests == total_tests:
        print("\n✅ TODOS OS TESTES PASSARAM! A biblioteca Scoras está funcionando corretamente.")
    else:
        print("\n❌ ALGUNS TESTES FALHARAM. Verifique os detalhes acima.")

if __name__ == "__main__":
    main()
