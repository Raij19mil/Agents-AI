import time
import schedule
import datetime
import os
from pathlib import Path
from typing import Dict, List, Optional, TypedDict, Literal
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Importações das bibliotecas de IA
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
import re

# Configurações
GOOGLE_API_KEY = os.getenv('GEMINI_API_KEY', '')
if not GOOGLE_API_KEY:
    raise ValueError("GEMINI_API_KEY não encontrada nas variáveis de ambiente")

# Classes Pydantic
class TriagemOut(BaseModel):
    decisao: Literal["AUTO_RESOLVER", "PEDIR_INFO", "ABRIR_CHAMADO"]
    urgencia: Literal["BAIXA", "MEDIA", "ALTA"]
    campos_faltantes: List[str] = Field(default_factory=list)

class AgentState(TypedDict, total=False):
    pergunta: str
    triagem: dict
    resposta: Optional[str]
    citacoes: List[dict]
    rag_sucesso: bool
    acao_final: str

# Classe principal do serviço de IA
class AIService:
    def __init__(self):
        logger.info("Inicializando serviço de IA...")
        
        # Inicializar modelos
        self.llm = ChatGoogleGenerativeAI(
            model="gemma-3-27b-it",
            temperature=0.5,
            api_key=GOOGLE_API_KEY
        )
        
        self.llm_triagem = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.0,
            api_key=GOOGLE_API_KEY
        )
        
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=GOOGLE_API_KEY
        )
        
        # Prompt de triagem
        self.triagem_prompt = (
            "Você é um triador de Service Desk para políticas internas da empresa Carraro Desenvolvimento. "
            "Dada a mensagem do usuário, retorne SOMENTE um JSON com:\n"
            "{\n"
            '  "decisao": "AUTO_RESOLVER" | "PEDIR_INFO" | "ABRIR_CHAMADO",\n'
            '  "urgencia": "BAIXA" | "MEDIA" | "ALTA",\n'
            '  "campos_faltantes": ["..."]\n'
            "}\n"
            "Regras:\n"
            '- **AUTO_RESOLVER**: Perguntas claras sobre regras ou procedimentos descritos nas políticas.\n'
            '- **PEDIR_INFO**: Mensagens vagas ou que faltam informações.\n'
            '- **ABRIR_CHAMADO**: Pedidos de exceção, liberação, aprovação ou acesso especial.'
        )
        
        # Configurar chain de triagem
        self.triagem_chain = self.llm_triagem.with_structured_output(TriagemOut)
        
        # Configurar RAG
        self.vectorstore = None
        self.retriever = None
        self.document_chain = None
        
        # Configurar grafo
        self.grafo = None
        
        # Keywords para abertura de ticket
        self.keywords_abrir_ticket = ["aprovação", "exceção", "liberação", "abrir ticket", "abrir chamado", "acesso especial"]
        
        logger.info("Serviço de IA inicializado com sucesso!")
    
    def carregar_documentos(self, pdf_path: str = "./data/"):
        """Carrega e processa documentos PDF"""
        logger.info(f"Carregando documentos de: {pdf_path}")
        
        docs = []
        pdf_dir = Path(pdf_path)
        
        if not pdf_dir.exists():
            logger.warning(f"Diretório {pdf_path} não encontrado. Criando...")
            pdf_dir.mkdir(parents=True, exist_ok=True)
            return
        
        for pdf_file in pdf_dir.glob("*.pdf"):
            try:
                loader = PyMuPDFLoader(str(pdf_file))
                docs.extend(loader.load())
                logger.info(f"Carregado: {pdf_file.name}")
            except Exception as e:
                logger.error(f"Erro ao carregar {pdf_file.name}: {e}")
        
        if not docs:
            logger.warning("Nenhum documento PDF encontrado!")
            return
        
        # Dividir em chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
        chunks = splitter.split_documents(docs)
        
        # Criar vectorstore
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.3, "k": 4}
        )
        
        # Configurar chain RAG
        prompt_rag = ChatPromptTemplate.from_messages([
            ("system",
             "Você é um Assistente de Políticas Internas (RH/IT) da empresa Carraro Desenvolvimento. "
             "Responda SOMENTE com base no contexto fornecido. "
             "Se não houver base suficiente, responda apenas 'Não sei'."),
            ("human", "Pergunta: {input}\n\nContexto:\n{context}")
        ])
        
        self.document_chain = create_stuff_documents_chain(self.llm_triagem, prompt_rag)
        
        logger.info(f"Documentos processados: {len(chunks)} chunks criados")
    
    def configurar_grafo(self):
        """Configura o grafo de decisão"""
        logger.info("Configurando grafo de decisão...")
        
        workflow = StateGraph(AgentState)
        
        # Adicionar nós
        workflow.add_node("triagem", self.node_triagem)
        workflow.add_node("auto_resolver", self.node_auto_resolver)
        workflow.add_node("pedir_info", self.node_pedir_info)
        workflow.add_node("abrir_chamado", self.node_abrir_chamado)
        
        # Adicionar edges
        workflow.add_edge(START, "triagem")
        workflow.add_conditional_edges("triagem", self.decidir_pos_triagem, {
            "auto": "auto_resolver",
            "info": "pedir_info",
            "chamado": "abrir_chamado"
        })
        
        workflow.add_conditional_edges("auto_resolver", self.decidir_pos_auto_resolver, {
            "info": "pedir_info",
            "chamado": "abrir_chamado",
            "ok": END
        })
        
        workflow.add_edge("pedir_info", END)
        workflow.add_edge("abrir_chamado", END)
        
        self.grafo = workflow.compile()
        logger.info("Grafo configurado com sucesso!")
    
    def triagem(self, mensagem: str) -> Dict:
        """Executa triagem da mensagem"""
        try:
            saida: TriagemOut = self.triagem_chain.invoke([
                SystemMessage(content=self.triagem_prompt),
                HumanMessage(content=mensagem)
            ])
            return saida.model_dump()
        except Exception as e:
            logger.error(f"Erro na triagem: {e}")
            return {"decisao": "PEDIR_INFO", "urgencia": "MEDIA", "campos_faltantes": ["informação clara"]}
    
    def perguntar_politica_rag(self, pergunta: str) -> Dict:
        """Executa consulta RAG"""
        if not self.retriever or not self.document_chain:
            return {"answer": "Sistema de consulta não inicializado.", "citacoes": [], "contexto_encontrado": False}
        
        try:
            docs_relacionados = self.retriever.invoke(pergunta)
            
            if not docs_relacionados:
                return {"answer": "Não sei.", "citacoes": [], "contexto_encontrado": False}
            
            answer = self.document_chain.invoke({"input": pergunta, "context": docs_relacionados})
            txt = (answer or "").strip()
            
            if txt.rstrip(".!?") == "Não sei":
                return {"answer": "Não sei.", "citacoes": [], "contexto_encontrado": False}
            
            return {
                "answer": txt,
                "citacoes": self.formatar_citacoes(docs_relacionados, pergunta),
                "contexto_encontrado": True
            }
        except Exception as e:
            logger.error(f"Erro no RAG: {e}")
            return {"answer": "Erro ao processar consulta.", "citacoes": [], "contexto_encontrado": False}
    
    def formatar_citacoes(self, docs_rel: List, query: str) -> List[Dict]:
        """Formata citações dos documentos"""
        cites, seen = [], set()
        for d in docs_rel:
            src = Path(d.metadata.get("source", "")).name
            page = int(d.metadata.get("page", 0)) + 1
            key = (src, page)
            if key in seen:
                continue
            seen.add(key)
            cites.append({
                "documento": src,
                "pagina": page,
                "trecho": self.extrair_trecho(d.page_content, query)
            })
        return cites[:3]
    
    def extrair_trecho(self, texto: str, query: str, janela: int = 240) -> str:
        """Extrai trecho relevante do texto"""
        txt = re.sub(r"\s+", " ", texto or "").strip()
        termos = [t.lower() for t in re.findall(r"\w+", query or "") if len(t) >= 4]
        pos = -1
        for t in termos:
            pos = txt.lower().find(t)
            if pos != -1:
                break
        if pos == -1:
            pos = 0
        ini, fim = max(0, pos - janela//2), min(len(txt), pos + janela//2)
        return txt[ini:fim]
    
    # Nós do grafo
    def node_triagem(self, state: AgentState) -> AgentState:
        logger.info("Executando triagem...")
        return {"triagem": self.triagem(state["pergunta"])}
    
    def node_auto_resolver(self, state: AgentState) -> AgentState:
        logger.info("Executando auto resolver...")
        resposta_rag = self.perguntar_politica_rag(state["pergunta"])
        
        update: AgentState = {
            "resposta": resposta_rag["answer"],
            "citacoes": resposta_rag.get("citacoes", []),
            "rag_sucesso": resposta_rag["contexto_encontrado"],
        }
        
        if resposta_rag["contexto_encontrado"]:
            update["acao_final"] = "AUTO_RESOLVER"
        
        return update
    
    def node_pedir_info(self, state: AgentState) -> AgentState:
        logger.info("Executando pedir info...")
        faltantes = state["triagem"].get("campos_faltantes", [])
        detalhe = ",".join(faltantes) if faltantes else "Tema e contexto específico"
        
        return {
            "resposta": f"Para avançar, preciso que detalhe: {detalhe}",
            "citacoes": [],
            "acao_final": "PEDIR_INFO"
        }
    
    def node_abrir_chamado(self, state: AgentState) -> AgentState:
        logger.info("Executando abrir chamado...")
        triagem = state["triagem"]
        
        return {
            "resposta": f"Abrindo chamado com urgência {triagem['urgencia']}. Descrição: {state['pergunta'][:140]}",
            "citacoes": [],
            "acao_final": "ABRIR_CHAMADO"
        }
    
    # Decisores do grafo
    def decidir_pos_triagem(self, state: AgentState) -> str:
        logger.info("Decidindo após triagem...")
        decisao = state["triagem"]["decisao"]
        
        if decisao == "AUTO_RESOLVER":
            return "auto"
        if decisao == "PEDIR_INFO":
            return "info"
        if decisao == "ABRIR_CHAMADO":
            return "chamado"
    
    def decidir_pos_auto_resolver(self, state: AgentState) -> str:
        logger.info("Decidindo após auto resolver...")
        
        if state.get("rag_sucesso"):
            return "ok"
        
        pergunta_lower = (state["pergunta"] or "").lower()
        if any(k in pergunta_lower for k in self.keywords_abrir_ticket):
            return "chamado"
        
        return "info"
    
    def processar_pergunta(self, pergunta: str) -> Dict:
        """Processa uma pergunta usando o grafo"""
        if not self.grafo:
            logger.error("Grafo não configurado!")
            return {"erro": "Sistema não inicializado"}
        
        try:
            logger.info(f"Processando pergunta: {pergunta}")
            resultado = self.grafo.invoke({"pergunta": pergunta})
            
            # Log do resultado
            triag = resultado.get("triagem", {})
            logger.info(f"Decisão: {triag.get('decisao')} | Urgência: {triag.get('urgencia')} | Ação Final: {resultado.get('acao_final')}")
            
            return resultado
        except Exception as e:
            logger.error(f"Erro ao processar pergunta: {e}")
            return {"erro": str(e)}

# Instância global do serviço
ai_service = None

def inicializar_servico():
    """Inicializa o serviço de IA"""
    global ai_service
    try:
        logger.info("=== INICIALIZANDO SERVIÇO DE IA ===")
        ai_service = AIService()
        ai_service.carregar_documentos("./data/")
        ai_service.configurar_grafo()
        logger.info("=== SERVIÇO INICIALIZADO COM SUCESSO ===")
        return True
    except Exception as e:
        logger.error(f"Erro ao inicializar serviço: {e}")
        return False

def executar_testes():
    """Executa testes do sistema"""
    if not ai_service:
        logger.error("Serviço não inicializado!")
        return
    
    testes = [
        "Posso reembolsar a internet?",
        "Quero mais 5 dias de trabalho remoto. Como faço?",
        "Posso reembolsar cursos ou treinamentos da Alura?",
        "Quantas capivaras tem no Rio Pinheiros?"
    ]
    
    logger.info("=== EXECUTANDO TESTES ===")
    
    for pergunta in testes:
        logger.info(f"Testando: {pergunta}")
        resultado = ai_service.processar_pergunta(pergunta)
        
        if "erro" in resultado:
            logger.error(f"Erro: {resultado['erro']}")
        else:
            logger.info(f"Resposta: {resultado.get('resposta', 'N/A')}")
        
        time.sleep(2)  # Evitar rate limiting
    
    logger.info("=== TESTES CONCLUÍDOS ===")

def executar_monitoramento():
    """Executa monitoramento contínuo"""
    logger.info("Executando monitoramento do sistema...")
    
    if ai_service:
        logger.info("✅ Serviço de IA ativo")
        
        # Teste simples
        resultado = ai_service.processar_pergunta("Sistema funcionando?")
        if "erro" not in resultado:
            logger.info("✅ Sistema respondendo normalmente")
        else:
            logger.error("❌ Erro no sistema")
    else:
        logger.error("❌ Serviço não inicializado")

def main():
    """Função principal"""
    logger.info("=== INICIANDO SERVIÇO DE IA CONTÍNUO ===")
    
    # Inicializar serviço
    if not inicializar_servico():
        logger.error("Falha ao inicializar. Encerrando...")
        return
    
    # Configurar agendamentos
    schedule.every(30).minutes.do(executar_monitoramento)
    schedule.every(2).hours.do(executar_testes)
    
    # Executar testes inicial
    executar_testes()
    
    # Loop principal
    logger.info("=== ENTRANDO EM MODO DE MONITORAMENTO ===")
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Verifica a cada minuto
    except KeyboardInterrupt:
        logger.info("Serviço interrompido pelo usuário")
    except Exception as e:
        logger.error(f"Erro no loop principal: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # Modo teste único
        if inicializar_servico():
            executar_testes()
    else:
        # Modo contínuo
        main()
