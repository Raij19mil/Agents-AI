import os
from typing import List

# Configurações da API Google
GOOGLE_API_KEY = os.getenv('GEMINI_API_KEY', '')

# Configurações dos modelos
MODELO_PRINCIPAL = os.getenv('MODELO_PRINCIPAL', 'gemma-3-27b-it')
MODELO_TRIAGEM = os.getenv('MODELO_TRIAGEM', 'gemini-2.5-flash')
MODELO_EMBEDDING = os.getenv('MODELO_EMBEDDING', 'models/gemini-embedding-001')

# Configurações de temperatura
TEMPERATURA_PRINCIPAL = float(os.getenv('TEMPERATURA_PRINCIPAL', '0.5'))
TEMPERATURA_TRIAGEM = float(os.getenv('TEMPERATURA_TRIAGEM', '0.0'))

# Configurações do RAG
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '300'))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '30'))
SCORE_THRESHOLD = float(os.getenv('SCORE_THRESHOLD', '0.3'))
SEARCH_K = int(os.getenv('SEARCH_K', '4'))

# Caminhos
DATA_PATH = os.getenv('DATA_PATH', './data/')
PDF_PATH = os.getenv('PDF_PATH', './data/')
LOGS_PATH = os.getenv('LOGS_PATH', './logs/')

# Configurações de agendamento
INTERVALO_MONITORAMENTO = int(os.getenv('INTERVALO_MONITORAMENTO', '30'))  # minutos
INTERVALO_TESTES = int(os.getenv('INTERVALO_TESTES', '120'))  # minutos

# Configurações de logging
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'

# Keywords para abertura de ticket
KEYWORDS_TICKET: List[str] = [
    "aprovação", "exceção", "liberação", 
    "abrir ticket", "abrir chamado", "acesso especial",
    "solicito", "preciso de", "autorização"
]

# Configurações da empresa
EMPRESA_NOME = os.getenv('EMPRESA_NOME', 'Carraro Desenvolvimento')
DEPARTAMENTOS = ['RH', 'IT', 'Financeiro', 'Operações']

# Configurações de rate limiting
DELAY_ENTRE_CHAMADAS = int(os.getenv('DELAY_CHAMADAS', '2'))  # segundos
MAX_TENTATIVAS = int(os.getenv('MAX_TENTATIVAS', '3'))

# Configurações de performance
MAX_DOCS_PROCESSAR = int(os.getenv('MAX_DOCS', '100'))
TIMEOUT_REQUESTS = int(os.getenv('TIMEOUT_REQUESTS', '30'))  # segundos

# Validações
if not GOOGLE_API_KEY:
    raise ValueError("GEMINI_API_KEY é obrigatória. Configure como variável de ambiente.")

# Configurações para diferentes ambientes
AMBIENTE = os.getenv('AMBIENTE', 'desenvolvimento')  # desenvolvimento, producao, teste

if AMBIENTE == 'producao':
    LOG_LEVEL = 'WARNING'
    DELAY_ENTRE_CHAMADAS = 5
    INTERVALO_MONITORAMENTO = 60
elif AMBIENTE == 'teste':
    LOG_LEVEL = 'DEBUG'
    DELAY_ENTRE_CHAMADAS = 1
    INTERVALO_TESTES = 5

print(f"=== CONFIGURAÇÕES CARREGADAS ===")
print(f"Ambiente: {AMBIENTE}")
print(f"Empresa: {EMPRESA_NOME}")
print(f"Modelo Principal: {MODELO_PRINCIPAL}")
print(f"Modelo Triagem: {MODELO_TRIAGEM}")
print(f"Data Path: {DATA_PATH}")
print(f"Log Level: {LOG_LEVEL}")
print(f"Intervalo Monitoramento: {INTERVALO_MONITORAMENTO}min")
print(f"=== CONFIGURAÇÃO CONCLUÍDA ===")

# Configurações específicas para deploy
class DeployConfig:
    """Configurações específicas para diferentes plataformas de deploy"""
    
    @staticmethod
    def render():
        return {
            'port': int(os.getenv('PORT', '10000')),
            'host': '0.0.0.0',
            'workers': 1
        }
    
    @staticmethod
    def railway():
        return {
            'port': int(os.getenv('PORT', '8000')),
            'host': '0.0.0.0',
            'workers': 1
        }
    
    @staticmethod
    def heroku():
        return {
            'port': int(os.getenv('PORT', '5000')),
            'host': '0.0.0.0',
            'workers': 2
        }
