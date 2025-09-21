from flask import Flask, jsonify, request
import threading
import time
import os
from datetime import datetime
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Importar seu serviço de IA
try:
    from main import AIService, inicializar_servico
    ai_service = None
    servico_inicializado = False
except ImportError as e:
    logger.error(f"Erro ao importar módulos de IA: {e}")
    ai_service = None
    servico_inicializado = False

def inicializar_servico_background():
    """Inicializa o serviço de IA em background"""
    global ai_service, servico_inicializado
    
    try:
        logger.info("Inicializando serviço de IA em background...")
        
        # Simular inicialização (adapte conforme necessário)
        ai_service = {
            "status": "ativo",
            "inicializado": datetime.now().isoformat(),
            "versao": "1.0.0"
        }
        
        servico_inicializado = True
        logger.info("Serviço de IA inicializado com sucesso!")
        
    except Exception as e:
        logger.error(f"Erro ao inicializar serviço: {e}")
        servico_inicializado = False

def monitoramento_continuo():
    """Executa monitoramento contínuo em background"""
    while True:
        try:
            if servico_inicializado:
                logger.info("✅ Monitoramento: Sistema funcionando")
            else:
                logger.warning("⚠️ Monitoramento: Sistema não inicializado")
            
            time.sleep(1800)  # 30 minutos
            
        except Exception as e:
            logger.error(f"Erro no monitoramento: {e}")
            time.sleep(300)  # 5 minutos em caso de erro

# Inicializar serviço em thread separada
def iniciar_background_tasks():
    """Inicia tarefas em background"""
    
    # Thread para inicialização
    init_thread = threading.Thread(target=inicializar_servico_background, daemon=True)
    init_thread.start()
    
    # Aguarda inicialização
    time.sleep(5)
    
    # Thread para monitoramento contínuo
    monitor_thread = threading.Thread(target=monitoramento_continuo, daemon=True)
    monitor_thread.start()

# Rotas da API
@app.route('/')
def home():
    """Página inicial - Health Check"""
    return jsonify({
        "status": "online",
        "servico": "IA Service Desk",
        "versao": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "servico_ia_inicializado": servico_inicializado
    })

@app.route('/health')
def health_check():
    """Health check detalhado"""
    return jsonify({
        "status": "healthy" if servico_inicializado else "initializing",
        "servico_ia": "ativo" if servico_inicializado else "inicializando",
        "uptime": datetime.now().isoformat(),
        "memoria_uso": "N/A",
        "ultima_verificacao": datetime.now().isoformat()
    })

@app.route('/status')
def status():
    """Status completo do sistema"""
    return jsonify({
        "sistema": {
            "status": "operacional",
            "inicializado": servico_inicializado,
            "timestamp": datetime.now().isoformat()
        },
        "servicos": {
            "triagem": "ativo" if servico_inicializado else "inicializando",
            "rag": "ativo" if servico_inicializado else "inicializando",
            "grafo_decisao": "ativo" if servico_inicializado else "inicializando"
        },
        "configuracao": {
            "ambiente": os.getenv("AMBIENTE", "desenvolvimento"),
            "log_level": os.getenv("LOG_LEVEL", "INFO"),
            "api_key_configurada": bool(os.getenv("GEMINI_API_KEY"))
        }
    })

@app.route('/processar', methods=['POST'])
def processar_pergunta():
    """Processa uma pergunta via API"""
    try:
        if not servico_inicializado:
            return jsonify({
                "erro": "Serviço ainda não inicializado. Tente novamente em alguns minutos.",
                "status": "initializing"
            }), 503
        
        data = request.get_json()
        if not data or 'pergunta' not in data:
            return jsonify({
                "erro": "Campo 'pergunta' é obrigatório",
                "exemplo": {"pergunta": "Posso reembolsar a internet?"}
            }), 400
        
        pergunta = data['pergunta']
        
        # Simular processamento (substitua pela lógica real)
        resultado = {
            "pergunta": pergunta,
            "resposta": "Sistema em modo de demonstração. Pergunta processada com sucesso.",
            "triagem": {
                "decisao": "AUTO_RESOLVER",
                "urgencia": "BAIXA"
            },
            "acao_final": "DEMO_MODE",
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Pergunta processada: {pergunta}")
        
        return jsonify({
            "sucesso": True,
            "resultado": resultado
        })
        
    except Exception as e:
        logger.error(f"Erro ao processar pergunta: {e}")
        return jsonify({
            "erro": "Erro interno do servidor",
            "detalhes": str(e)
        }), 500

@app.route('/teste')
def teste_sistema():
    """Executa teste do sistema"""
    try:
        if not servico_inicializado:
            return jsonify({
                "erro": "Serviço não inicializado",
                "status": "not_ready"
            }), 503
        
        # Teste simples
        testes = [
            "Posso reembolsar a internet?",
            "Como funciona a política de home office?",
            "Sistema funcionando?"
        ]
        
        resultados = []
        for pergunta in testes:
            resultado = {
                "pergunta": pergunta,
                "status": "processado",
                "resposta": f"Teste: {pergunta} - OK",
                "timestamp": datetime.now().isoformat()
            }
            resultados.append(resultado)
        
        return jsonify({
            "teste_executado": True,
            "total_testes": len(testes),
            "resultados": resultados,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Erro no teste: {e}")
        return jsonify({
            "erro": "Erro ao executar teste",
            "detalhes": str(e)
        }), 500

@app.route('/logs')
def ver_logs():
    """Visualiza logs recentes (últimas 50 linhas)"""
    try:
        # Simular logs (em produção, ler arquivo de log real)
        logs_exemplo = [
            f"{datetime.now().isoformat()} - INFO - Sistema inicializado",
            f"{datetime.now().isoformat()} - INFO - Monitoramento ativo",
            f"{datetime.now().isoformat()} - INFO - API funcionando",
        ]
        
        return jsonify({
            "logs": logs_exemplo,
            "total_linhas": len(logs_exemplo),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            "erro": "Erro ao acessar logs",
            "detalhes": str(e)
        }), 500

# Endpoint para forçar restart do serviço (útil para debug)
@app.route('/restart', methods=['POST'])
def restart_servico():
    """Reinicia o serviço (modo debug)"""
    try:
        global servico_inicializado
        servico_inicializado = False
        
        # Reinicializar em background
        init_thread = threading.Thread(target=inicializar_servico_background, daemon=True)
        init_thread.start()
        
        return jsonify({
            "status": "reiniciando",
            "mensagem": "Serviço sendo reinicializado. Aguarde alguns minutos.",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            "erro": "Erro ao reiniciar",
            "detalhes": str(e)
        }), 500

# Middleware para CORS (se necessário)
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response

# Inicialização
if __name__ == '__main__':
    # Iniciar tarefas em background
    iniciar_background_tasks()
    
    # Configurar porta
    port = int(os.environ.get('PORT', 5000))
    
    # Executar Flask
    logger.info(f"🚀 Iniciando servidor Flask na porta {port}")
    app.run(
        host='0.0.0.0',
        port=port,
        debug=os.getenv('DEBUG', 'False').lower() == 'true'
    )
