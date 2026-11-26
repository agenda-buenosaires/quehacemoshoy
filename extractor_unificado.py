import json
import os
import re
import time
import unicodedata
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import List, Literal, Optional
import requests
from apify_client import ApifyClient
from bs4 import BeautifulSoup
from google import genai
from google.genai import types
from PIL import Image
from pydantic import BaseModel, Field


# ==============================================================================
# CARGA DE FUENTES DESDE fuentes.json
# ==============================================================================
def cargar_fuentes(ruta_fuentes: str = "fuentes.json") -> tuple[List[str], List[str]]:
    archivo = Path(ruta_fuentes)
    if not archivo.exists():
        raise FileNotFoundError(f"No se encontró {ruta_fuentes} en el repositorio.")
    data = json.loads(archivo.read_text(encoding="utf-8"))
    paginas = data.get("paginas_web", [])
    instagrams = data.get("perfiles_instagram", [])
    return paginas, instagrams


# ==============================================================================
# 1. ESQUEMA ESTRICTO COMPATIBLE CON INDEX.HTML
# ==============================================================================
class EventoCultural(BaseModel):
    id: int = Field(description="Número identificador único y correlativo")
    titulo: str = Field(description="Nombre oficial del evento o feria")
    categoria: Literal[
        "Gourmet",
        "Entretenimiento",
        "Música y Recitales",
        "Ferias y Artesanías",
        "Arte y Cultura",
    ] = Field(description="Categoría más adecuada según la temática del evento")
    zona: Literal["Zona Oeste", "Zona Sur", "Zona Norte", "CABA"] = Field(
        description="Región de Buenos Aires a la que pertenece la localidad"
    )
    localidad: str = Field(description="Partido o barrio (ej. Hurlingham, Morón, Palermo, La Plata)")
    lugar: str = Field(
        description="Nombre del espacio y dirección exacta con calle y altura si está disponible (ej. 'Predio Municipal de Hurlingham (Av. Pedro Díaz 1710)' o 'Plaza Dorrego'). No incluir eventos con múltiples sedes sin dirección fija."
    )
    organizador: str = Field(description="Institución, municipio, feria o productora que organiza")
    fecha: str = Field(description="Fecha principal de inicio en formato estricto YYYY-MM-DD")
    fechaTexto: Optional[str] = Field(
        default=None,
        description="Texto legible si abarca varios días (ej. '27, 28 y 29 de noviembre')",
    )
    hora: str = Field(description="Horario del evento (ej. '18:00 hs' o '12:00 a 23:00 hs')")
    precio: str = Field(
        description="Costo de entrada (ej. 'Gratuito', 'Entrada Libre', 'Pago (desde $15.000)')"
    )
    urlCompra: Optional[str] = Field(
        default=None, description="Enlace de compra si es un evento pago y aparece en la fuente"
    )
    reservaPrevia: Optional[bool] = Field(
        default=False, description="True si es gratuito pero requiere reserva previa"
    )
    urlReserva: Optional[str] = Field(
        default=None, description="Enlace de reserva previa si corresponde"
    )
    descripcion: str = Field(
        description="Resumen claro y atractivo de 2 o 3 oraciones sobre el evento"
    )


class ListaEventos(BaseModel):
    eventos: List[EventoCultural]


# ==============================================================================
# 2. MÓDULO A: RECOLECCIÓN DESDE PÁGINAS WEB
# ==============================================================================
def recolectar_paginas_web(urls: List[str]) -> str:
    texto_total = ""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    }

    for url in urls:
        try:
            print(f"   [WEB] Leyendo: {url}")
            resp = requests.get(url, headers=headers, timeout=20)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()

            texto_limpio = soup.get_text(separator="\n", strip=True)[:18000]
            texto_total += f"\n--- FUENTE WEB ({url}) ---\n{texto_limpio}\n"
        except Exception as e:
            print(f"   [AVISO WEB] No se pudo leer {url}: {e}")

    return texto_total


# ==============================================================================
# 3. MÓDULO B: RECOLECCIÓN DESDE MÚLTIPLES INSTAGRAMS (VÍA APIFY)
# ==============================================================================
def recolectar_instagrams(usuarios: List[str], posts_por_perfil: int = 3) -> List[dict]:
    token = os.environ.get("APIFY_API_TOKEN")
    if not token or not usuarios:
        if not token:
            print("   [AVISO IG] No hay APIFY_API_TOKEN configurado, se omite Instagram.")
        return []

    usuarios_limpios = [
        u.replace("https://www.instagram.com/", "").replace("@", "").strip("/")
        for u in usuarios
    ]

    print(f"   [IG] Consultando perfiles de Instagram: {usuarios_limpios}...")
    client = ApifyClient(token)
    run_input = {
        "username": usuarios_limpios,
        "resultsLimit": posts_por_perfil,
    }

    run = client.actor("apify/instagram-post-scraper").call(run_input=run_input)
    posteos = []
    for item in client.dataset(run.default_dataset_id).iterate_items():
        posteos.append({
            "usuario": item.get("ownerUsername", ""),
            "texto": item.get("caption", ""),
            "imagen_url": item.get("displayUrl", ""),
        })
    return posteos


# ==============================================================================
# 4. MÓDULO C: INTELIGENCIA ARTIFICIAL UNIFICADA CON REINTENTOS Y OPTIMIZACIÓN
# ==============================================================================
def procesar_todo_con_gemini(texto_webs: str, posteos_ig: List[dict]) -> List[dict]:
    if not os.environ.get("GEMINI_API_KEY"):
        raise ValueError("Falta configurar la variable de entorno GEMINI_API_KEY.")

    client = genai.Client()
    fecha_actual = datetime.now().strftime("%Y-%m-%d")

    contenido = [
        f"""
        Sos un sistema de extracción y estructuración de datos culturales de Buenos Aires y CABA.
        La fecha actual de referencia es {fecha_actual}.
        Vas a recibir información de DOS tipos de fuentes:
        1) Texto extraído de páginas web culturales y turísticas.
        2) Posteos de Instagram (texto + imagen del flyer) de ferias y agendas locales.

        Reglas obligatorias:
        - Extraé únicamente eventos vigentes o próximos (desde {fecha_actual} en adelante) que pertenezcan exclusivamente a las fuentes entregadas. Ignorá sorteos, saludos o eventos ya pasados.
        - Descartá eventos que no tengan una sede única y específica (por ejemplo, si dicen "distintos espacios físicos de la ciudad" o "más de 15 sedes").
        - Descartá eventos realizados en el Estadio Monumental / River Plate.
        - Si un evento es pago, incluí siempre el enlace externo de compra en 'urlCompra' (si es en Movistar Arena, usá https://www.movistararena.com.ar/shows; si es en Ciudad Cultural Konex, usá https://www.cckonex.org/agenda/). Si es pago y no tiene enlace de compra disponible, no lo incluyas.
        - En los posteos de Instagram, leé con mucha atención lo que dice DENTRO DE LA IMAGEN del flyer (allí suelen estar la plaza, dirección exacta, localidad y fechas).
        - Incluí siempre la dirección con calle y altura entre paréntesis en el campo 'lugar' cuando esté disponible, para que funcione perfecto en Google Maps.
        - Inferí correctamente la 'zona' (Zona Oeste, Zona Sur, Zona Norte o CABA) según la localidad o barrio (ej. Mercedes y General Las Heras pertenecen a Zona Oeste).
        - Asigná una de las 5 categorías permitidas.
        - Formateá 'fecha' como YYYY-MM-DD.
        - No dupliques eventos si aparecen repetidos entre fuentes.
        """
    ]

    if texto_webs.strip():
        contenido.append(f"\n=== CONTENIDO DE PÁGINAS WEB ===\n{texto_webs}")

    for i, post in enumerate(posteos_ig, start=1):
        contenido.append(
            f"\n=== POST INSTAGRAM #{i} (@{post['usuario']}) ===\nTexto: {post['texto']}\n"
        )
        if post["imagen_url"]:
            try:
                img_resp = requests.get(post["imagen_url"], timeout=10)
                if img_resp.status_code == 200:
                    img = Image.open(BytesIO(img_resp.content))
                    img.thumbnail((800, 800))
                    contenido.append(img)
            except Exception as e:
                print(f"   [AVISO] No se pudo descargar imagen de @{post['usuario']}: {e}")

    modelos_a_probar = [
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.5-pro",
    ]
    response = None

    for intento in range(1, 4):
        for modelo in modelos_a_probar:
            try:
                print(f"   [IA] Intento {intento} - Analizando con {modelo}...")
                response = client.models.generate_content(
                    model=modelo,
                    contents=contenido,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=ListaEventos,
                        temperature=0.1,
                    ),
                )
                break
            except Exception as e:
                print(f"   [AVISO IA] {modelo} no disponible ({e}), probando siguiente...")
        if response is not None:
            break
        print("   [ESPERA] Reintentando en 15 segundos...")
        time.sleep(15)

    if response is None:
        raise RuntimeError("Todos los modelos de Gemini fallaron tras 3 rondas de reintentos.")

    resultado = json.loads(response.text)
    return resultado.get("eventos", [])


# ==============================================================================
# 5. VALIDACIÓN DE CALIDAD Y SINCRONIZACIÓN DE eventos.json
# ==============================================================================
def normalizar_texto(texto: str) -> str:
    texto = unicodedata.normalize("NFD", (texto or "").lower().strip())
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", texto)


def es_evento_valido(ev: dict) -> bool:
    lugar = normalizar_texto(ev.get("lugar", ""))
    titulo = normalizar_texto(ev.get("titulo", ""))
    precio = normalizar_texto(ev.get("precio", ""))
    url_compra = (ev.get("urlCompra") or "").strip()

    if not lugar:
        return False

    # Descartar sedes múltiples o genéricas
    patrones_invalidos = [
        "distintos espacios",
        "distintos puntos",
        "varios espacios",
        "varias sedes",
        "mas de 15 sedes",
        "playas y pista",
        "playon municipal, teatro",
    ]
    if any(p in lugar for p in patrones_invalidos):
        return False

    # Descartar Estadio Monumental / River Plate
    if "monumental" in lugar or "river plate" in lugar or "cancha de river" in lugar:
        return False

    # Normalizar URLs conocidas
    if "movistar arena" in lugar and not url_compra:
        ev["urlCompra"] = "https://www.movistararena.com.ar/shows"
        url_compra = ev["urlCompra"]
    elif "konex" in lugar and not url_compra and "gratuito" not in precio:
        ev["urlCompra"] = "https://www.cckonex.org/agenda/"
        url_compra = ev["urlCompra"]

    # Descartar eventos pagos sin enlace externo de compra
    es_gratis = any(k in precio for k in ["gratuito", "entrada libre", "consumicion"])
    if not es_gratis and not url_compra.startswith("http"):
        return False

    return bool(titulo)


def clave_deduplicacion(ev: dict) -> tuple:
    fecha = ev.get("fecha", "")
    localidad = normalizar_texto(ev.get("localidad", ""))
    lugar_base = normalizar_texto(ev.get("lugar", "").split("(")[0].split(",")[0])
    return (fecha, localidad, lugar_base)


def actualizar_json_historico(nuevos_eventos: List[dict], ruta_json: str = "eventos.json") -> List[dict]:
    archivo = Path(ruta_json)
    existentes = []
    if archivo.exists():
        existentes = json.loads(archivo.read_text(encoding="utf-8"))

    fecha_hoy = datetime.now().strftime("%Y-%m-%d")

    vigentes = [
        ev for ev in existentes
        if ev.get("fecha", "") >= fecha_hoy and es_evento_valido(ev)
    ]

    vistos_clave = {clave_deduplicacion(ev) for ev in vigentes}
    vistos_titulo = {(normalizar_texto(ev.get("titulo", "")), ev.get("fecha", "")) for ev in vigentes}
    combinados = list(vigentes)

    agregados = 0
    for ev in nuevos_eventos:
        if ev.get("fecha", "") < fecha_hoy or not es_evento_valido(ev):
            continue
        c_lugar = clave_deduplicacion(ev)
        c_titulo = (normalizar_texto(ev.get("titulo", "")), ev.get("fecha", ""))
        if c_lugar not in vistos_clave and c_titulo not in vistos_titulo:
            vistos_clave.add(c_lugar)
            vistos_titulo.add(c_titulo)
            combinados.append(ev)
            agregados += 1

    combinados.sort(key=lambda e: e["fecha"])
    for idx, ev in enumerate(combinados, start=1):
        ev["id"] = idx

    archivo.write_text(json.dumps(combinados, ensure_ascii=False, indent=4), encoding="utf-8")
    print(f"[OK] Se agregaron {agregados} eventos nuevos. Total vigentes: {len(combinados)}.")
    return combinados


if __name__ == "__main__":
    print("0. Cargando fuentes desde fuentes.json...")
    paginas_web, perfiles_ig = cargar_fuentes("fuentes.json")

    print("1. Recolectando datos de páginas web...")
    datos_web = recolectar_paginas_web(paginas_web)

    print("2. Recolectando datos e imágenes de perfiles de Instagram...")
    datos_ig = recolectar_instagrams(perfiles_ig, posts_por_perfil=3)

    print("3. Estructurando toda la información con Gemini AI...")
    eventos_detectados = procesar_todo_con_gemini(datos_web, datos_ig)

    print("4. Actualizando eventos.json automáticamente...")
    actualizar_json_historico(eventos_detectados, "eventos.json")
