import os
import subprocess
import re
import time
from datetime import timedelta
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from yt_dlp import YoutubeDL
from sentence_transformers import SentenceTransformer, util
import torch
from dotenv import load_dotenv

# ================= CONFIG ===================

load_dotenv()  # Carrega as variáveis do arquivo .env

PASTA_VIDEOS = os.getenv("PASTA_VIDEOS")
PASTA_CORTES = os.getenv("PASTA_CORTES")

# ================= UTILITÁRIOS ==============
def timestamp_to_seconds(t):
    try:
        # Remove espaços e divide por ":"
        t_clean = t.strip()
        parts = t_clean.split(":")
        
        # Converte para inteiros, ignorando texto extra
        int_parts = []
        for part in parts:
            # Remove qualquer texto não numérico da parte
            clean_part = ''.join(filter(str.isdigit, part))
            if clean_part:
                int_parts.append(int(clean_part))
            else:
                int_parts.append(0)
        
        if len(int_parts) == 2:
            minutos, segundos = int_parts
            return minutos * 60 + segundos
        elif len(int_parts) == 3:
            horas, minutos, segundos = int_parts
            return horas * 3600 + minutos * 60 + segundos
        elif len(int_parts) == 1:
            # Apenas segundos
            return int_parts[0]
        else:
            raise ValueError(f"Formato de tempo inválido: {t}")
    except Exception as e:
        raise ValueError(f"Não foi possível converter '{t}' para segundos: {e}")

def segundos_para_timestamp(segundos):
    return str(timedelta(seconds=int(segundos)))

def parse_srt(caminho_arquivo):
    with open(caminho_arquivo, 'r', encoding='utf-8') as f:
        conteudo = f.read()

    blocos = re.split(r'\n\n+', conteudo.strip())
    segmentos = []

    for bloco in blocos:
        linhas = bloco.strip().split('\n')
        if len(linhas) >= 3:
            tempo = linhas[1]
            texto = " ".join(linhas[2:]).strip()
            inicio, fim = tempo.split(' --> ')
            segmentos.append((inicio.strip(), fim.strip(), texto))
    return segmentos

def str_time_to_seconds(t):
    h, m, s = t.split(':')
    if ',' in s:
        s, ms = s.split(',')
    else:
        ms = '0'
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000

def _seconds_to_srt_time(sec_float):
    """Converte segundos (float) para string SRT HH:MM:SS,mmm"""
    if sec_float < 0:
        sec_float = 0.0
    total_ms = int(round(sec_float * 1000))
    h = total_ms // 3600000
    rem = total_ms % 3600000
    m = rem // 60000
    rem = rem % 60000
    s = rem // 1000
    ms = rem % 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

# ============ NOVO: limpeza de sobreposições ============
def limpar_sobreposicoes_srt(segmentos, margem_segundos=0.05):
    """
    Ajusta tempos de legendas para evitar sobreposições:
    se o fim da legenda atual ultrapassa o início da próxima,
    o fim é ajustado para (início_da_próxima - margem).
    """
    corrigidos = []
    for i in range(len(segmentos)):
        inicio, fim, texto = segmentos[i]
        if i < len(segmentos) - 1:
            inicio_prox, _, _ = segmentos[i + 1]
            inicio_prox_sec = str_time_to_seconds(inicio_prox)
            fim_sec = str_time_to_seconds(fim)
            ini_sec = str_time_to_seconds(inicio)

            # Ajusta fim se houver sobreposição
            limite = inicio_prox_sec - margem_segundos
            if fim_sec > limite:
                fim_sec = max(limite, ini_sec)  # evita duração negativa
                fim = _seconds_to_srt_time(fim_sec)
        corrigidos.append((inicio, fim, texto))
    return corrigidos

def limpar_arquivo_srt(caminho_srt, margem_segundos=0.05):
    """Lê, limpa sobreposições e sobrescreve o SRT no mesmo arquivo."""
    try:
        segmentos = parse_srt(caminho_srt)
        if not segmentos:
            return False
        limpos = limpar_sobreposicoes_srt(segmentos, margem_segundos=margem_segundos)
        with open(caminho_srt, 'w', encoding='utf-8') as f:
            for idx, (inicio, fim, texto) in enumerate(limpos, start=1):
                f.write(f"{idx}\n{inicio} --> {fim}\n{texto}\n\n")
        return True
    except Exception:
        return False
# ========================================================

def baixar_video_youtube(url, pasta_destino):
    if not os.path.exists(pasta_destino):
        os.makedirs(pasta_destino)

    ydl_opts = {
        'format': 'bestvideo+bestaudio/best',
        'outtmpl': os.path.join(pasta_destino, '%(title)s.%(ext)s'),
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': ['pt', 'en'],
        'subtitlesformat': 'srt',
        'quiet': True,
        'merge_output_format': 'mp4',
        'skip_download': False,
        'no_warnings': True,
        # Adiciona cookies do Chrome para autenticação
        'cookiesfrombrowser': ('chrome',),
        # Adiciona configurações para evitar problemas de impersonation
        'extractor_args': {
            'youtube': {
                'skip': ['hls', 'dash'],
                'player_skip': ['configs']
            }
        }
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        titulo = info['title']
        base_path = os.path.join(pasta_destino, titulo)
        video_path = base_path + ".mp4"

        possiveis_ext = ['.pt.srt', '.pt-BR.srt', '.en.srt', '.en-US.srt']
        legenda_path = None

        for ext in possiveis_ext:
            caminho = base_path + ext
            if os.path.exists(caminho):
                legenda_path = caminho
                break

        if not legenda_path:
            raise FileNotFoundError("Nenhuma legenda .srt encontrada em pt ou en.")

        return video_path, legenda_path, titulo

def detectar_pontos_de_corte_semantico(segmentos, intervalo_min=480, intervalo_max=720, limite_similaridade=0.6):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    modelo = SentenceTransformer('all-MiniLM-L6-v2', device=device)

    blocos = []
    buffer = []
    inicio_bloco = str_time_to_seconds(segmentos[0][0])

    for i in range(len(segmentos)):
        buffer.append(segmentos[i])
        duracao_bloco = str_time_to_seconds(segmentos[i][1]) - inicio_bloco

        if duracao_bloco >= 240:
            texto_bloco = " ".join([b[2] for b in buffer])
            blocos.append((inicio_bloco, str_time_to_seconds(segmentos[i][1]), texto_bloco))
            buffer = []
            if i + 1 < len(segmentos):
                inicio_bloco = str_time_to_seconds(segmentos[i + 1][0])

    if buffer:
        texto_bloco = " ".join([b[2] for b in buffer])
        blocos.append((inicio_bloco, str_time_to_seconds(segmentos[-1][1]), texto_bloco))

    textos = [b[2] for b in blocos]
    embeddings = modelo.encode(textos, convert_to_tensor=True)

    cortes = [blocos[0][0]]
    ultimo_corte = blocos[0][0]

    for i in range(1, len(blocos)):
        tempo_atual = blocos[i][0]
        duracao = tempo_atual - ultimo_corte
        similaridade = util.cos_sim(embeddings[i - 1], embeddings[i]).item()

        if duracao >= intervalo_min and (similaridade < limite_similaridade or duracao >= intervalo_max):
            cortes.append(tempo_atual)
            ultimo_corte = tempo_atual

    cortes.append(blocos[-1][1])
    return [segundos_para_timestamp(c) for c in cortes]

def validar_e_ajustar_cortes(cortes):
    """Apenas para cortes automáticos - limita em 12 minutos"""
    cortes_ajustados = [cortes[0]]  # Sempre mantém o primeiro corte

    for i in range(1, len(cortes)):
        tempo_anterior = timestamp_to_seconds(cortes_ajustados[-1])
        tempo_atual = timestamp_to_seconds(cortes[i])
        duracao = tempo_atual - tempo_anterior

        if duracao > 720:
            novo_tempo = tempo_anterior + 720
            cortes_ajustados.append(segundos_para_timestamp(novo_tempo))

            tempo_restante = tempo_atual - (tempo_anterior + 720)
            if tempo_restante > 720:
                segmentos_extras = int(tempo_restante // 720)
                for j in range(1, segmentos_extras + 1):
                    tempo_extra = tempo_anterior + 720 + (j * 720)
                    cortes_ajustados.append(segundos_para_timestamp(tempo_extra))

            if tempo_restante > 60:
                cortes_ajustados.append(cortes[i])
        else:
            cortes_ajustados.append(cortes[i])

    return cortes_ajustados

def processar_cortes_manuais(cortes_raw):
    """Processa cortes manuais sem limitações de duração"""
    cortes = []
    descricoes = []
    
    for c in cortes_raw:
        c = c.strip()
        if c:
            if " - " in c:
                tempo, descricao = c.split(" - ", 1)
                tempo = tempo.strip()
                descricao = descricao.strip()
            else:
                tempo = c
                descricao = ""
            
            # Validar se o tempo está em formato válido
            try:
                # Testa se consegue converter para segundos
                timestamp_to_seconds(tempo)
                cortes.append(tempo)
                descricoes.append(descricao)
            except ValueError as e:
                print(f"⚠️ Ignorando timestamp inválido: '{tempo}' - {e}")
                continue
    
    if len(cortes) < 2:
        raise ValueError("É necessário pelo menos 2 timestamps válidos para fazer cortes")
    
    # Ordena os cortes e descrições por tempo
    cortes_com_desc = list(zip(cortes, descricoes))
    cortes_com_desc.sort(key=lambda x: timestamp_to_seconds(x[0]))
    
    cortes_ordenados = [item[0] for item in cortes_com_desc]
    desc_ordenadas = [item[1] for item in cortes_com_desc]
    
    return cortes_ordenados, desc_ordenadas

def sanitizar_nome_arquivo(nome):
    """Remove caracteres inválidos para nomes de arquivo"""
    # Remove ou substitui caracteres problemáticos
    nome = re.sub(r'[<>:"/\\|?*]', '', nome)  # Remove caracteres inválidos
    nome = re.sub(r'\s+', '_', nome)  # Substitui espaços por underscore
    nome = nome.strip('._')  # Remove pontos e underscores do início/fim
    
    # Limita o comprimento do nome
    if len(nome) > 50:
        nome = nome[:47] + "..."
    
    return nome if nome else "sem_nome"

def cortar_video(video_path, cortes, pasta_saida, descricoes=None):
    """
    Corta o vídeo baseado nos timestamps fornecidos.
    Agora SEM anexar legendas diretamente no vídeo.
    """
    for i in range(len(cortes) - 1):
        inicio = cortes[i]
        fim = cortes[i + 1]
        
        # Define o nome do arquivo baseado na descrição ou genérico
        if descricoes and i < len(descricoes) - 1 and descricoes[i + 1]:
            nome_base = sanitizar_nome_arquivo(descricoes[i + 1])
        else:
            nome_base = f"corte_{i+1:02d}"
            
        video_saida = os.path.join(pasta_saida, nome_base + ".mp4")

        inicio_sec = timestamp_to_seconds(inicio)
        fim_sec = timestamp_to_seconds(fim)
        duracao = fim_sec - inicio_sec

        # Aplica o corte com fade in/out, MAS SEM legendas hardcoded
        subprocess.run([
            'ffmpeg', '-y',
            '-ss', inicio.replace(',', '.'),
            '-to', fim.replace(',', '.'),
            '-i', video_path,
            '-vf', (
                f"fade=in:0:60,"
                f"fade=out:st={duracao-3}:d=90"
            ),
            '-c:v', 'h264_nvenc',
            '-preset', 'slow',
            '-crf', '18',
            '-c:a', 'aac',
            '-b:a', '256k',
            video_saida
        ])

# ================= GUI (Tkinter) ================
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("YouTube Cutter IA - Sem Legendas Anexadas")
        self.root.geometry("700x600")
        
        # Variáveis
        self.url = tk.StringVar()
        self.modo = tk.StringVar(value="manual")
        self.fonte_video = tk.StringVar(value="youtube")  # "youtube" ou "arquivo"
        self.video_selecionado = tk.StringVar()
        self.srt_selecionado = tk.StringVar()
        self.cortes = []
        
        self.criar_interface()

    def criar_interface(self):
        # Frame principal com scroll
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # ========= SEÇÃO: FONTE DO VÍDEO =========
        fonte_frame = tk.LabelFrame(main_frame, text="Fonte do Vídeo", font=("Arial", 10, "bold"))
        fonte_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Radiobutton(
            fonte_frame, 
            text="Download do YouTube", 
            variable=self.fonte_video, 
            value="youtube",
            command=self.alternar_fonte
        ).pack(anchor=tk.W, padx=10, pady=5)
        
        tk.Radiobutton(
            fonte_frame, 
            text="Usar arquivos já baixados", 
            variable=self.fonte_video, 
            value="arquivo",
            command=self.alternar_fonte
        ).pack(anchor=tk.W, padx=10, pady=(0, 5))
        
        # ========= SEÇÃO: YOUTUBE =========
        self.youtube_frame = tk.LabelFrame(main_frame, text="YouTube", font=("Arial", 10, "bold"))
        self.youtube_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(self.youtube_frame, text="Link do vídeo YouTube:").pack(anchor=tk.W, padx=10, pady=(10, 5))
        tk.Entry(self.youtube_frame, textvariable=self.url, width=60).pack(fill=tk.X, padx=10, pady=(0, 10))
        
        # ========= SEÇÃO: ARQUIVOS LOCAIS =========
        self.arquivo_frame = tk.LabelFrame(main_frame, text="Arquivos Locais", font=("Arial", 10, "bold"))
        self.arquivo_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Seleção de vídeo
        video_frame = tk.Frame(self.arquivo_frame)
        video_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(video_frame, text="Vídeo (MP4):").pack(anchor=tk.W)
        
        video_input_frame = tk.Frame(video_frame)
        video_input_frame.pack(fill=tk.X, pady=(2, 0))
        
        self.video_entry = tk.Entry(video_input_frame, textvariable=self.video_selecionado, state="readonly")
        self.video_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        tk.Button(video_input_frame, text="Procurar", command=self.selecionar_video).pack(side=tk.RIGHT, padx=(5, 0))
        
        # Seleção de transcrição
        srt_frame = tk.Frame(self.arquivo_frame)
        srt_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(srt_frame, text="Transcrição (SRT) - apenas para análise dos cortes:").pack(anchor=tk.W)
        
        srt_input_frame = tk.Frame(srt_frame)
        srt_input_frame.pack(fill=tk.X, pady=(2, 0))
        
        self.srt_entry = tk.Entry(srt_input_frame, textvariable=self.srt_selecionado, state="readonly")
        self.srt_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        tk.Button(srt_input_frame, text="Procurar", command=self.selecionar_srt).pack(side=tk.RIGHT, padx=(5, 0))
        
        # ========= SEÇÃO: MODO DE CORTE =========
        modo_frame = tk.LabelFrame(main_frame, text="Modo de Corte", font=("Arial", 10, "bold"))
        modo_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Radiobutton(modo_frame, text="Cortes Manuais", variable=self.modo, value="manual").pack(anchor=tk.W, padx=10, pady=5)
        tk.Radiobutton(modo_frame, text="Corte Automático por IA", variable=self.modo, value="auto").pack(anchor=tk.W, padx=10, pady=(0, 5))
        
        # ========= SEÇÃO: PONTOS DE CORTE =========
        cortes_frame = tk.LabelFrame(main_frame, text="Pontos de Corte (apenas para modo manual)", font=("Arial", 10, "bold"))
        cortes_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        tk.Label(cortes_frame, text="Digite os timestamps no formato: HH:MM:SS - Descrição").pack(anchor=tk.W, padx=10, pady=(10, 5))
        
        self.txt_cortes = tk.Text(cortes_frame, height=8, width=60)
        self.txt_cortes.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        self.txt_cortes.insert("1.0", "00:00 - Introdução\n08:45 - Primeiro Tópico\n20:00 - Segundo Tópico\n30:00")
        
        # ========= SEÇÃO: CONTROLES =========
        tk.Button(main_frame, text="Iniciar Processamento", command=self.iniciar, bg="#4CAF50", fg="white", font=("Arial", 12, "bold")).pack(pady=10)
        
        # ========= SEÇÃO: LOG =========
        log_frame = tk.LabelFrame(main_frame, text="Log de Execução", font=("Arial", 10, "bold"))
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbar para o log
        scrollbar = tk.Scrollbar(log_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.log = tk.Text(log_frame, height=15, width=80, yscrollcommand=scrollbar.set)
        self.log.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        scrollbar.config(command=self.log.yview)
        
        # Configuração inicial da interface
        self.alternar_fonte()

    def alternar_fonte(self):
        """Alterna entre as seções YouTube e Arquivos Locais"""
        if self.fonte_video.get() == "youtube":
            # Mostra seção do YouTube e esconde seção de arquivos
            for widget in self.youtube_frame.winfo_children():
                widget.config(state="normal")
            for widget in self.arquivo_frame.winfo_children():
                self.desabilitar_recursivo(widget)
        else:
            # Mostra seção de arquivos e esconde seção do YouTube  
            for widget in self.youtube_frame.winfo_children():
                if isinstance(widget, tk.Entry):
                    widget.config(state="disabled")
            for widget in self.arquivo_frame.winfo_children():
                self.habilitar_recursivo(widget)

    def desabilitar_recursivo(self, widget):
        """Desabilita um widget e todos seus filhos recursivamente"""
        try:
            if hasattr(widget, 'config'):
                if isinstance(widget, (tk.Entry, tk.Button)):
                    widget.config(state="disabled")
                elif isinstance(widget, tk.Label):
                    widget.config(fg="gray")
        except:
            pass
        
        for child in widget.winfo_children():
            self.desabilitar_recursivo(child)

    def habilitar_recursivo(self, widget):
        """Habilita um widget e todos seus filhos recursivamente"""
        try:
            if hasattr(widget, 'config'):
                if isinstance(widget, tk.Button):
                    widget.config(state="normal")
                elif isinstance(widget, tk.Label):
                    widget.config(fg="black")
                elif isinstance(widget, tk.Entry):
                    if widget == self.video_entry or widget == self.srt_entry:
                        widget.config(state="readonly")
                    else:
                        widget.config(state="normal")
        except:
            pass
        
        for child in widget.winfo_children():
            self.habilitar_recursivo(child)

    def selecionar_video(self):
        """Abre dialog para selecionar arquivo de vídeo"""
        filename = filedialog.askopenfilename(
            title="Selecionar arquivo de vídeo",
            filetypes=[
                ("Arquivos de vídeo", "*.mp4 *.avi *.mkv *.mov *.wmv"),
                ("Todos os arquivos", "*.*")
            ]
        )
        if filename:
            self.video_selecionado.set(filename)

    def selecionar_srt(self):
        """Abre dialog para selecionar arquivo de transcrição"""
        filename = filedialog.askopenfilename(
            title="Selecionar arquivo de transcrição",
            filetypes=[
                ("Arquivos SRT", "*.srt"),
                ("Todos os arquivos", "*.*")
            ]
        )
        if filename:
            self.srt_selecionado.set(filename)

    def logar(self, msg):
        """Adiciona mensagem ao log"""
        self.log.insert(tk.END, msg + "\n")
        self.log.see(tk.END)
        self.root.update()

    def validar_entradas(self):
        """Valida as entradas do usuário"""
        if self.fonte_video.get() == "youtube":
            if not self.url.get().strip():
                self.logar("⚠️ Por favor, insira um URL do YouTube.")
                return False
        else:
            if not self.video_selecionado.get().strip():
                self.logar("⚠️ Por favor, selecione um arquivo de vídeo.")
                return False
            if not self.srt_selecionado.get().strip():
                self.logar("⚠️ Por favor, selecione um arquivo de transcrição SRT.")
                return False
            
            # Verifica se os arquivos existem
            if not os.path.exists(self.video_selecionado.get()):
                self.logar("⚠️ Arquivo de vídeo não encontrado.")
                return False
            if not os.path.exists(self.srt_selecionado.get()):
                self.logar("⚠️ Arquivo de transcrição não encontrado.")
                return False
        
        return True

    def iniciar(self):
        """Inicia o processamento"""
        if not self.validar_entradas():
            return
            
        modo = self.modo.get()
        
        try:
            # Obter vídeo e legenda
            if self.fonte_video.get() == "youtube":
                # Fluxo original: download do YouTube
                url = self.url.get().strip()
                self.logar("📥 Baixando vídeo e legendas do YouTube...")
                video_path, legenda_path, titulo = baixar_video_youtube(url, PASTA_VIDEOS)
                self.logar(f"✅ Vídeo baixado: {titulo}")

                # Limpeza do SRT após o download (apenas para análise)
                self.logar("🧹 Limpando sobreposições na legenda para análise...")
                if limpar_arquivo_srt(legenda_path, margem_segundos=0.05):
                    self.logar("✅ Legenda limpa com sucesso.")
                else:
                    self.logar("⚠️ Não foi possível limpar a legenda (arquivo vazio ou erro). Seguindo assim mesmo.")
            else:
                # Novo fluxo: usar arquivos selecionados
                video_path = self.video_selecionado.get()
                legenda_path = self.srt_selecionado.get()
                titulo = os.path.splitext(os.path.basename(video_path))[0]
                self.logar(f"📁 Usando arquivos selecionados:")
                self.logar(f"   Vídeo: {os.path.basename(video_path)}")
                self.logar(f"   Transcrição: {os.path.basename(legenda_path)}")

                # Limpeza do SRT para arquivos locais também (apenas para análise)
                self.logar("🧹 Limpando sobreposições na legenda para análise...")
                if limpar_arquivo_srt(legenda_path, margem_segundos=0.05):
                    self.logar("✅ Legenda limpa com sucesso.")
                else:
                    self.logar("⚠️ Não foi possível limpar a legenda (arquivo vazio ou erro). Seguindo assim mesmo.")

            # Processar cortes (lógica igual para ambos os fluxos)
            if modo == "manual":
                # CORTES MANUAIS - sem limitação de duração
                self.logar("✂️ Processando cortes manuais...")
                cortes_raw = self.txt_cortes.get("1.0", tk.END).strip().split("\n")
                cortes, descricoes = processar_cortes_manuais(cortes_raw)
                
                if len(cortes) < 2:
                    self.logar("⚠️ É necessário pelo menos 2 pontos de corte (início e fim).")
                    return
                    
            else:
                # CORTES AUTOMÁTICOS - com todas as validações e limitações
                segmentos = parse_srt(legenda_path)
                self.logar("🤖 Detectando mudanças de assunto com IA...")
                cortes = detectar_pontos_de_corte_semantico(segmentos)
                descricoes = None  # Cortes automáticos não têm descrições
                
                self.logar("⏱️ Validando duração dos cortes (máx 12 min)...")
                cortes = validar_e_ajustar_cortes(cortes)
                
                if len(cortes) < 2:
                    self.logar("⚠️ Poucos cortes detectados.")
                    return

            self.logar(f"✂️ Cortando vídeo em {len(cortes)-1} partes (sem legendas anexadas)...")
            
            # Mostrar os cortes que serão aplicados
            for i in range(len(cortes)-1):
                inicio = cortes[i]
                fim = cortes[i+1]
                duracao_seg = timestamp_to_seconds(fim) - timestamp_to_seconds(inicio)
                duracao_min = duracao_seg / 60
                
                # Determina o nome que será usado para o arquivo
                if descricoes and i < len(descricoes) - 1 and descricoes[i + 1]:
                    nome_arquivo = sanitizar_nome_arquivo(descricoes[i + 1])
                    self.logar(f"  📝 {nome_arquivo}: {inicio} → {fim} (duração: {duracao_min:.1f} min)")
                else:
                    self.logar(f"  📹 Corte {i+1}: {inicio} → {fim} (duração: {duracao_min:.1f} min)")
            
            # Criar pasta de saída se não existir
            if not os.path.exists(PASTA_CORTES):
                os.makedirs(PASTA_CORTES)
            
            # Executar os cortes (SEM legendas anexadas)
            cortar_video(video_path, cortes, PASTA_CORTES, descricoes)
            self.logar("✅ Processamento finalizado com sucesso!")
            self.logar(f"📁 Os cortes foram salvos em: {PASTA_CORTES}")
            self.logar("ℹ️ Os vídeos foram cortados baseados na análise das transcrições, mas sem legendas anexadas.")
            
        except Exception as e:
            self.logar(f"❌ Erro durante o processamento: {e}")
            import traceback
            self.logar(f"🔍 Detalhes do erro: {traceback.format_exc()}")
            return

if __name__ == '__main__':
    root = tk.Tk()
    app = App(root)
    root.mainloop()