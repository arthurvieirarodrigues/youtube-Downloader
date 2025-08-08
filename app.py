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

# Carrega as variáveis do arquivo .env

# Lê as variáveis como strings
PASTA_VIDEOS = os.getenv("PASTA_VIDEOS")
PASTA_CORTES = os.getenv("PASTA_CORTES")

# PASTA_VIDEOS = "/run/media/arthurarch/HD LINUX/youtube/videos"
# PASTA_CORTES = "/run/media/arthurarch/HD LINUX/youtube/cortes"

# ================= UTILITÁRIOS ==============
def timestamp_to_seconds(t):
    parts = list(map(int, t.strip().split(":")))
    if len(parts) == 2:
        minutos, segundos = parts
        return minutos * 60 + segundos
    elif len(parts) == 3:
        horas, minutos, segundos = parts
        return horas * 3600 + minutos * 60 + segundos
    else:
        raise ValueError(f"Formato de tempo inválido: {t}")

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

def salvar_legenda(caminho_saida, legendas):
    with open(caminho_saida, 'w', encoding='utf-8') as f:
        for idx, (inicio, fim, texto) in enumerate(legendas, start=1):
            f.write(f"{idx}\n{inicio} --> {fim}\n{texto}\n\n")

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
        'skip_download': False
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

def detectar_pontos_de_corte_semantico(segmentos, intervalo_min=480, intervalo_max=1200, limite_similaridade=0.6):

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

def filtrar_legendas_por_tempo(segmentos, tempo_inicio, tempo_fim):
    inicio_sec = timestamp_to_seconds(tempo_inicio)
    fim_sec = timestamp_to_seconds(tempo_fim)
    return [ (i, f, txt) for (i, f, txt) in segmentos if inicio_sec <= str_time_to_seconds(i) < fim_sec ]

def hardcode_legenda(video, srt, saida):
    subprocess.run([
        'ffmpeg', '-y',
        '-i', video,
        '-vf', f"subtitles='{srt}':force_style='FontName=Arial,FontSize=24,PrimaryColour=&HFFFFFF&,OutlineColour=&H000000&,BorderStyle=1,Outline=1'",
        '-c:a', 'copy',
        '-c:v', 'h264_nvenc',
        saida
    ])

def hardcode_legenda_no_video_inteiro(video_path, legenda_path):
    saida_path = os.path.splitext(video_path)[0] + "_legendado.mp4"
    hardcode_legenda(video_path, legenda_path, saida_path)
    return saida_path

def cortar_video(video_path, cortes, legenda_path, pasta_saida):
    segmentos = parse_srt(legenda_path)
    for i in range(len(cortes) - 1):
        inicio = cortes[i]
        fim = cortes[i + 1]
        nome_base = f"corte_{i+1:02d}"
        video_saida = os.path.join(pasta_saida, nome_base + ".mp4")
        srt_saida = os.path.join(pasta_saida, nome_base + ".srt")
        video_temp = os.path.join(pasta_saida, nome_base + "_temp.mp4")

        subprocess.run([
            'ffmpeg', '-y',
            '-ss', inicio.replace(',', '.'),
            '-to', fim.replace(',', '.'),
            '-i', video_path,
            '-c', 'copy',
            video_temp
        ])

        legendas_corte = filtrar_legendas_por_tempo(segmentos, inicio, fim)
        salvar_legenda(srt_saida, legendas_corte)
        hardcode_legenda(video_temp, srt_saida, video_saida)
        os.remove(video_temp)

# ================= GUI (Tkinter) ================
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("YouTube Cutter IA")
        self.url = tk.StringVar()
        self.modo = tk.StringVar(value="manual")
        self.cortes = []

        tk.Label(root, text="Link do vídeo YouTube:").pack()
        tk.Entry(root, textvariable=self.url, width=60).pack()

        tk.Radiobutton(root, text="Cortes Manuais", variable=self.modo, value="manual").pack()
        tk.Radiobutton(root, text="Corte Automático por IA", variable=self.modo, value="auto").pack()

        self.txt_cortes = tk.Text(root, height=8, width=60)
        self.txt_cortes.pack()
        self.txt_cortes.insert("1.0", "00:00\n08:45\n20:00")

        tk.Button(root, text="Iniciar", command=self.iniciar).pack(pady=10)
        self.log = tk.Text(root, height=15, width=80)
        self.log.pack()

    def logar(self, msg):
        self.log.insert(tk.END, msg + "\n")
        self.log.see(tk.END)
        self.root.update()

    def iniciar(self):
        url = self.url.get().strip()
        modo = self.modo.get()
        self.logar("📅 Baixando vídeo e legendas...")
        try:
            video_path, legenda_path, _ = baixar_video_youtube(url, PASTA_VIDEOS)
        except Exception as e:
            self.logar("❌ Erro ao baixar: " + str(e))
            return

        self.logar("🎧 Embutindo legendas no vídeo inteiro...")
        video_path = hardcode_legenda_no_video_inteiro(video_path, legenda_path)

        segmentos = parse_srt(legenda_path)

        if modo == "manual":
            cortes_raw = self.txt_cortes.get("1.0", tk.END).strip().split("\n")
            cortes = [c.strip().split(" - ")[0] for c in cortes_raw if c.strip()]
        else:
            self.logar("🧐 Detectando mudanças de assunto...")
            cortes = detectar_pontos_de_corte_semantico(segmentos)

        if len(cortes) < 2:
            self.logar("⚠️ Poucos cortes detectados.")
            return

        self.logar("✂️ Cortando vídeo em " + str(len(cortes)-1) + " partes...")
        cortar_video(video_path, cortes, legenda_path, PASTA_CORTES)
        self.logar("✅ Finalizado!")

if __name__ == '__main__':
    root = tk.Tk()
    app = App(root)
    root.mainloop()
