#!/usr/bin/env python3
"""
Script para testar o suporte aos protocolos MCP e A2A na biblioteca Scoras atualizada.
Este script verifica se as classes e funcionalidades relacionadas aos protocolos MCP e A2A
estão funcionando corretamente.
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

def test_mcp_imports() -> List[Dict[str, Any]]:
    """Testa a importação de classes e funções relacionadas ao protocolo MCP."""
    results = []
    
    # Lista de importações para testar
    imports = [
        {"name": "MCPServer", "module": "scoras.mcp"},
        {"name": "MCPClient", "module": "scoras.mcp"},
        {"name": "MCPSkill", "module": "scoras.mcp"},
        {"name": "create_mcp_server", "module": "scoras.mcp"},
        {"name": "create_mcp_client", "module": "scoras.mcp"}
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

def test_a2a_imports() -> List[Dict[str, Any]]:
    """Testa a importação de classes e funções relacionadas ao protocolo A2A."""
    results = []
    
    # Lista de importações para testar
    imports = [
        {"name": "A2AAgent", "module": "scoras.a2a"},
        {"name": "A2ANetwork", "module": "scoras.a2a"},
        {"name": "A2AHub", "module": "scoras.a2a"},
        {"name": "create_a2a_agent", "module": "scoras.a2a"},
        {"name": "create_a2a_network", "module": "scoras.a2a"}
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

def test_mcp_functionality() -> List[Dict[str, Any]]:
    """Testa a funcionalidade básica do protocolo MCP."""
    results = []
    
    # Teste 1: Criar um servidor MCP
    try:
        from scoras.mcp import create_mcp_server
        
        server = create_mcp_server(name="test_server")
        
        results.append({
            "name": "Criação de servidor MCP",
            "success": server is not None and hasattr(server, "name") and server.name == "test_server"
        })
    except Exception as e:
        results.append({
            "name": "Criação de servidor MCP",
            "success": False,
            "message": str(e)
        })
    
    # Teste 2: Criar um cliente MCP
    try:
        from scoras.mcp import create_mcp_client
        
        client = create_mcp_client(name="test_client", server_url="http://localhost:8000")
        
        results.append({
            "name": "Criação de cliente MCP",
            "success": client is not None and hasattr(client, "name") and client.name == "test_client"
        })
    except Exception as e:
        results.append({
            "name": "Criação de cliente MCP",
            "success": False,
            "message": str(e)
        })
    
    # Teste 3: Criar uma habilidade MCP
    try:
        from scoras.mcp import MCPSkill
        
        def skill_function(input_data):
            return {"result": input_data["value"] * 2}
        
        skill = MCPSkill(
            name="double_value",
            description="Doubles the input value",
            function=skill_function
        )
        
        results.append({
            "name": "Criação de habilidade MCP",
            "success": skill is not None and hasattr(skill, "name") and skill.name == "double_value"
        })
    except Exception as e:
        results.append({
            "name": "Criação de habilidade MCP",
            "success": False,
            "message": str(e)
        })
    
    return results

def test_a2a_functionality() -> List[Dict[str, Any]]:
    """Testa a funcionalidade básica do protocolo A2A."""
    results = []
    
    # Teste 1: Criar um agente A2A
    try:
        from scoras.a2a import create_a2a_agent
        
        agent = create_a2a_agent(name="test_agent", model="openai:gpt-4")
        
        results.append({
            "name": "Criação de agente A2A",
            "success": agent is not None and hasattr(agent, "name") and agent.name == "test_agent"
        })
    except Exception as e:
        results.append({
            "name": "Criação de agente A2A",
            "success": False,
            "message": str(e)
        })
    
    # Teste 2: Criar uma rede A2A
    try:
        from scoras.a2a import create_a2a_network
        
        network = create_a2a_network(name="test_network")
        
        results.append({
            "name": "Criação de rede A2A",
            "success": network is not None and hasattr(network, "name") and network.name == "test_network"
        })
    except Exception as e:
        results.append({
            "name": "Criação de rede A2A",
            "success": False,
            "message": str(e)
        })
    
    # Teste 3: Criar um hub A2A
    try:
        from scoras.a2a import A2AHub
        
        hub = A2AHub(name="test_hub")
        
        results.append({
            "name": "Criação de hub A2A",
            "success": hub is not None and hasattr(hub, "name") and hub.name == "test_hub"
        })
    except Exception as e:
        results.append({
            "name": "Criação de hub A2A",
            "success": False,
            "message": str(e)
        })
    
    return results

def test_protocol_integration() -> List[Dict[str, Any]]:
    """Testa a integração dos protocolos MCP e A2A com o sistema de pontuação de complexidade."""
    results = []
    
    # Teste 1: Verificar se o servidor MCP tem pontuação de complexidade
    try:
        from scoras.mcp import create_mcp_server
        
        server = create_mcp_server(name="test_server")
        score = server.get_complexity_score() if hasattr(server, "get_complexity_score") else None
        
        results.append({
            "name": "Pontuação de complexidade do servidor MCP",
            "success": score is not None and isinstance(score, dict) and "total_score" in score
        })
    except Exception as e:
        results.append({
            "name": "Pontuação de complexidade do servidor MCP",
            "success": False,
            "message": str(e)
        })
    
    # Teste 2: Verificar se o agente A2A tem pontuação de complexidade
    try:
        from scoras.a2a import create_a2a_agent
        
        agent = create_a2a_agent(name="test_agent", model="openai:gpt-4")
        score = agent.get_complexity_score() if hasattr(agent, "get_complexity_score") else None
        
        results.append({
            "name": "Pontuação de complexidade do agente A2A",
            "success": score is not None and isinstance(score, dict) and "total_score" in score
        })
    except Exception as e:
        results.append({
            "name": "Pontuação de complexidade do agente A2A",
            "success": False,
            "message": str(e)
        })
    
    return results

def main() -> None:
    """Função principal para executar todos os testes."""
    print_header("TESTE DE SUPORTE AOS PROTOCOLOS MCP E A2A")
    
    # Verifica se o pacote scoras está instalado
    try:
        import scoras
        print(f"Versão do Scoras: {scoras.__version__}")
    except ImportError:
        print("❌ ERRO: O pacote scoras não está instalado.")
        print("Por favor, instale o pacote antes de executar os testes:")
        print("pip install scoras")
        sys.exit(1)
    
    # Testa as importações do protocolo MCP
    print_header("TESTE DE IMPORTAÇÕES MCP")
    mcp_import_results = test_mcp_imports()
    for result in mcp_import_results:
        print_result(result["name"], result["success"], result.get("message"))
    
    # Testa as importações do protocolo A2A
    print_header("TESTE DE IMPORTAÇÕES A2A")
    a2a_import_results = test_a2a_imports()
    for result in a2a_import_results:
        print_result(result["name"], result["success"], result.get("message"))
    
    # Testa a funcionalidade do protocolo MCP
    print_header("TESTE DE FUNCIONALIDADE MCP")
    mcp_func_results = test_mcp_functionality()
    for result in mcp_func_results:
        print_result(result["name"], result["success"], result.get("message"))
    
    # Testa a funcionalidade do protocolo A2A
    print_header("TESTE DE FUNCIONALIDADE A2A")
    a2a_func_results = test_a2a_functionality()
    for result in a2a_func_results:
        print_result(result["name"], result["success"], result.get("message"))
    
    # Testa a integração dos protocolos com o sistema de pontuação de complexidade
    print_header("TESTE DE INTEGRAÇÃO COM PONTUAÇÃO DE COMPLEXIDADE")
    integration_results = test_protocol_integration()
    for result in integration_results:
        print_result(result["name"], result["success"], result.get("message"))
    
    # Resumo dos resultados
    print_header("RESUMO DOS RESULTADOS")
    all_results = mcp_import_results + a2a_import_results + mcp_func_results + a2a_func_results + integration_results
    total_tests = len(all_results)
    passed_tests = sum(1 for r in all_results if r["success"])
    
    print(f"Total de testes: {total_tests}")
    print(f"Testes bem-sucedidos: {passed_tests}")
    print(f"Testes com falha: {total_tests - passed_tests}")
    
    if passed_tests == total_tests:
        print("\n✅ TODOS OS TESTES PASSARAM! Os protocolos MCP e A2A estão funcionando corretamente.")
    else:
        print("\n❌ ALGUNS TESTES FALHARAM. Verifique os detalhes acima.")
        print("\nRecomendações para corrigir problemas:")
        print("1. Verifique se os módulos mcp.py e a2a.py estão presentes no pacote")
        print("2. Verifique se todas as classes e funções necessárias estão implementadas")
        print("3. Certifique-se de que os protocolos estão integrados com o sistema de pontuação de complexidade")

if __name__ == "__main__":
    main()
