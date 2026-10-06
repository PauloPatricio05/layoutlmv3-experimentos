import os
import pytesseract
from PIL import Image, ImageDraw
import torch
from transformers import (
    LayoutLMv3ImageProcessor,
    LayoutLMv3TokenizerFast,
    LayoutLMv3ForTokenClassification
)

# Especifica o caminho do Tesseract no Windows
if os.name == 'nt':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extrair_ocr_manual(image):
    """Extrai palavras e coordenadas (normalizadas de 0 a 1000) usando o Tesseract diretamente."""
    width, height = image.size
    ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    
    words = []
    boxes = []
    
    for i in range(len(ocr_data['text'])):
        word = ocr_data['text'][i].strip()
        if word:  # Ignora palavras vazias
            words.append(word)
            x, y, w, h = ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i]
            
            # Normaliza e limita rigidamente entre 0 e 1000
            x0 = max(0, min(1000, int(1000 * (x / width))))
            y0 = max(0, min(1000, int(1000 * (y / height))))
            x1 = max(0, min(1000, int(1000 * ((x + w) / width))))
            y1 = max(0, min(1000, int(1000 * ((y + h) / height))))
            
            boxes.append([x0, y0, x1, y1])
            
    return words, boxes

def testar_layoutlmv3(caminho_imagem):
    print(f"🔄 Carregando imagem: {caminho_imagem}")
    image = Image.open(caminho_imagem).convert("RGB")
    
    print("🔄 Executando Tesseract OCR...")
    words, boxes = extrair_ocr_manual(image)
    
    if not words:
        print("❌ Nenhum texto foi encontrado na imagem pelo Tesseract!")
        return

    print(f"✅ OCR concluído! {len(words)} palavras extraídas.")
    
    print("🔄 Carregando modelo, tokenizador e processador LayoutLMv3...")
    image_processor = LayoutLMv3ImageProcessor(apply_ocr=False)
    tokenizer = LayoutLMv3TokenizerFast.from_pretrained("nielsr/layoutlmv3-finetuned-funsd")
    model = LayoutLMv3ForTokenClassification.from_pretrained("nielsr/layoutlmv3-finetuned-funsd")

    print("🔄 Executando a inferência no LayoutLMv3...")
    
    # 1. Processa a imagem para gerar o tensor de pixels
    image_features = image_processor(images=image, return_tensors="pt")
    
    # 2. Processa o texto e as coordenadas com o LayoutLMv3TokenizerFast
    encoding = tokenizer(
        text=words,
        boxes=boxes,
        truncation=True,
        padding=True,
        return_tensors="pt"
    )
    
    # 3. Anexa os pixels ao dicionário de entrada
    encoding["pixel_values"] = image_features["pixel_values"]

    # 4. Executa a inferência
    with torch.no_grad():
        outputs = model(**encoding)

    # Extrai as predições descartando a dimensão de lote
    predictions = outputs.logits.argmax(-1).squeeze(0).tolist()
    
    if isinstance(predictions, int):
        predictions = [predictions]
        
    labels = [model.config.id2label[pred] for pred in predictions]
    input_ids = encoding.input_ids.squeeze(0).tolist()

    # -------------------------------------------------------------
    # 📊 IMPRESSÃO ORGANIZADA COM RECONSTRUÇÃO DE SUB-TOKENS
    # -------------------------------------------------------------
    print("\n" + "="*60)
    print("📌 ENTIDADES EXTRAÍDAS E RECONSTRUÍDAS:")
    print("="*60)
    
    entidades_tokens = {"HEADER": [], "QUESTION": [], "ANSWER": []}
    
    for token_id, label in zip(input_ids, labels):
        clean_label = label.replace("B-", "").replace("I-", "")
        # Agrupa os IDs de token ignorando tokens especiais
        if clean_label in entidades_tokens and token_id not in [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]:
            entidades_tokens[clean_label].append(token_id)
            
    for classe, t_ids in entidades_tokens.items():
        # O decode() junta os sub-tokens (BPE) de volta em palavras normais
        texto_limpo = tokenizer.decode(t_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        print(f"\n🔹 {classe}:")
        print(texto_limpo)
        
    print("="*60 + "\n")

    # -------------------------------------------------------------
    # 🎨 DESENHO DAS CAIXAS NA IMAGEM
    # -------------------------------------------------------------
    print("✅ Desenhando caixas na imagem...")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    
    color_map = {
        "HEADER": "blue",
        "QUESTION": "green",
        "ANSWER": "orange",
        "OTHER": "gray"
    }

    bbox_inputs = encoding.bbox.squeeze(0).tolist()
    
    for box, label in zip(bbox_inputs, labels):
        if label == "O" or box == [0, 0, 0, 0]:
            continue
            
        unnormalized_box = [
            (box[0] / 1000) * width,
            (box[1] / 1000) * height,
            (box[2] / 1000) * width,
            (box[3] / 1000) * height
        ]
        
        clean_label = label.replace("B-", "").replace("I-", "")
        color = color_map.get(clean_label, "red")
        draw.rectangle(unnormalized_box, outline=color, width=2)

    caminho_saida = os.path.join(os.path.dirname(caminho_imagem), "resultado_layoutlmv3.png")
    image.save(caminho_saida)
    print(f"🎉 Imagem salva com sucesso em: {caminho_saida}")

if __name__ == "__main__":
    pasta_atual = os.path.dirname(os.path.abspath(__file__))
    caminho_da_imagem = os.path.join(pasta_atual, "recibo_teste2.png")

    if os.path.exists(caminho_da_imagem):
        testar_layoutlmv3(caminho_da_imagem)
    else:
        print(f"⚠️ Imagem não encontrada em: {caminho_da_imagem}")
