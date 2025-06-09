import json
import pandas as pd
from pathlib import Path
import requests
import os
from urllib.parse import urlparse
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import time
import random

def create_dummy_image(place_name, place_category, save_path, width=400, height=300):
    """Membuat gambar dummy yang cantik jika download gagal"""
    try:
        # Pilih gradient warna berdasarkan kategori
        color_schemes = {
            'pantai': ['#4A90E2', '#87CEEB', '#FFFFFF'],  # Blue gradient
            'museum': ['#8B4513', '#D2691E', '#F4A460'],  # Brown gradient
            'air_terjun': ['#228B22', '#32CD32', '#98FB98'],  # Green gradient
            'lapangan': ['#228B22', '#90EE90', '#F0FFF0'],  # Light green gradient
            'situs_sejarah': ['#8B008B', '#DDA0DD', '#E6E6FA'],  # Purple gradient
            'lainnya': ['#FF6B35', '#F7931E', '#FFD700']  # Orange gradient
        }
        
        # Tentukan warna berdasarkan kategori
        if isinstance(place_category, list) and len(place_category) > 0:
            category = place_category[0]
        else:
            category = 'lainnya'
            
        colors = color_schemes.get(category, color_schemes['lainnya'])
        
        # Buat gambar dengan gradient background
        img = Image.new('RGB', (width, height), colors[0])
        draw = ImageDraw.Draw(img)
        
        # Buat efek gradient
        for y in range(height):
            ratio = y / height
            if ratio < 0.5:
                # Gradient dari warna pertama ke kedua (atas ke tengah)
                r1, g1, b1 = Image.new('RGB', (1, 1), colors[0]).getpixel((0, 0))
                r2, g2, b2 = Image.new('RGB', (1, 1), colors[1]).getpixel((0, 0))
                local_ratio = ratio * 2
            else:
                # Gradient dari warna kedua ke ketiga (tengah ke bawah)
                r1, g1, b1 = Image.new('RGB', (1, 1), colors[1]).getpixel((0, 0))
                r2, g2, b2 = Image.new('RGB', (1, 1), colors[2]).getpixel((0, 0))
                local_ratio = (ratio - 0.5) * 2
            
            r = int(r1 + (r2 - r1) * local_ratio)
            g = int(g1 + (g2 - g1) * local_ratio)
            b = int(b1 + (b2 - b1) * local_ratio)
            
            draw.line([(0, y), (width, y)], fill=(r, g, b))
        
        # Tambahkan pattern dots untuk dekorasi
        dot_color = colors[0]
        for i in range(0, width, 40):
            for j in range(0, height, 40):
                if (i + j) % 80 == 0:  # Pattern checkerboard
                    draw.ellipse([i-2, j-2, i+2, j+2], fill=dot_color, outline=None)
        
        # Tambahkan border dengan rounded corners effect
        border_color = colors[0]
        # Outer border
        draw.rectangle([5, 5, width-5, height-5], outline=border_color, width=3)
        # Inner border
        draw.rectangle([15, 15, width-15, height-15], outline=border_color, width=1)
        
        # Setup fonts
        try:
            # Coba berbagai font yang mungkin tersedia
            font_paths = [
                "C:/Windows/Fonts/arial.ttf",  # Windows
                "C:/Windows/Fonts/calibri.ttf",  # Windows
                "/System/Library/Fonts/Arial.ttf",  # macOS
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
            ]
            
            font_title = None
            for font_path in font_paths:
                try:
                    font_title = ImageFont.truetype(font_path, 28)
                    break
                except:
                    continue
            
            if not font_title:
                font_title = ImageFont.load_default()
                
        except:
            font_title = ImageFont.load_default()
        
        # Bungkus teks nama tempat dengan lebih baik
        words = place_name.split()
        lines = []
        current_line = []
        max_width = width - 60  # Margin 30px di setiap sisi
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            bbox = draw.textbbox((0, 0), test_line, font=font_title)
            text_width = bbox[2] - bbox[0]
            
            if text_width <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
        
        if current_line:
            lines.append(' '.join(current_line))
        
        # Hitung posisi teks untuk center alignment
        line_height = 35
        total_text_height = len(lines) * line_height
        start_y = (height - total_text_height) // 2
        
        # Gambar shadow teks terlebih dahulu
        shadow_offset = 2
        for i, line in enumerate(lines):
            bbox = draw.textbbox((0, 0), line, font=font_title)
            text_width = bbox[2] - bbox[0]
            x = (width - text_width) // 2
            y = start_y + i * line_height
            
            # Shadow
            draw.text((x + shadow_offset, y + shadow_offset), line, 
                     fill=(0, 0, 0, 100), font=font_title)
        
        # Gambar teks utama
        text_color = '#FFFFFF' if category in ['pantai', 'museum', 'situs_sejarah'] else '#2C3E50'
        for i, line in enumerate(lines):
            bbox = draw.textbbox((0, 0), line, font=font_title)
            text_width = bbox[2] - bbox[0]
            x = (width - text_width) // 2
            y = start_y + i * line_height
            
            # Teks utama
            draw.text((x, y), line, fill=text_color, font=font_title)
        
        # Tambahkan subtle watermark di pojok
        watermark_color = (*Image.new('RGB', (1, 1), colors[0]).getpixel((0, 0)), 50)
        try:
            font_small = ImageFont.truetype(font_paths[0], 12) if font_paths else ImageFont.load_default()
        except:
            font_small = ImageFont.load_default()
        
        draw.text((width-80, height-20), "Wisata ID", fill=watermark_color, font=font_small)
        
        # Simpan gambar dengan kualitas tinggi
        img.save(save_path, 'JPEG', quality=95, optimize=True)
        return True
        
    except Exception as e:
        print(f"Error creating dummy image: {str(e)}")
        return False

def download_image_with_fallback(url, save_path):
    """Download gambar dengan berbagai fallback method"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        # Method 1: Download langsung
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            img.save(save_path, 'JPEG', quality=85)
            return True
    except Exception as e:
        print(f"Method 1 failed for {url}: {str(e)}")
    
    try:
        # Method 2: Coba modifikasi URL Google Maps
        if 'googleusercontent.com' in url:
            # Hapus parameter yang mungkin menyebabkan masalah
            modified_url = url.split('=')[0] + '=w400-h300'
            response = requests.get(modified_url, headers=headers, timeout=10)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                img.save(save_path, 'JPEG', quality=85)
                return True
    except Exception as e:
        print(f"Method 2 failed for {url}: {str(e)}")
    
    return False

def get_unsplash_image(query, save_path):
    """Ambil gambar dari Unsplash (tanpa API key, menggunakan source URL)"""
    try:
        # Buat URL Unsplash dengan query
        unsplash_url = f"https://source.unsplash.com/400x300/?{query}"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(unsplash_url, headers=headers, timeout=15)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            img.save(save_path, 'JPEG', quality=85)
            return True
    except Exception as e:
        print(f"Unsplash download failed for {query}: {str(e)}")
    
    return False

def main():
    # Setup paths
    base_dir = Path(__file__).parent.parent
    json_dir = base_dir / 'Scrape_data' / 'json'
    image_dir = base_dir / 'data' / 'images'
    output_dir = base_dir / 'data'
    
    # Buat direktori jika belum ada
    image_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Dapatkan semua file JSON
    json_files = list(json_dir.glob('tempat_wisata_*.json'))
    print(f"Jumlah file JSON yang ditemukan: {len(json_files)}")
    
    # Proses dan gabungkan data
    all_data = []
    current_id = 1
    
    for json_file in json_files:
        print(f"Memproses file: {json_file.name}")
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        for item in data:
            item['id'] = current_id
            
            # Ambil hanya foto pertama jika ada
            if item.get('foto') and len(item['foto']) > 0:
                image_url = item['foto'][0]
                image_filename = f"wisata_{current_id}.jpg"
                image_path = image_dir / image_filename
                
                success = False
                
                # Coba download gambar asli
                print(f"Downloading image for: {item.get('nama', 'Unknown')}")
                if download_image_with_fallback(image_url, image_path):
                    success = True
                    print("✓ Original image downloaded successfully")
                else:
                    # Coba ambil dari Unsplash berdasarkan nama dan kategori
                    place_name = item.get('nama', '')
                    categories = item.get('kategori', ['tourism'])
                    
                    # Buat query untuk Unsplash
                    if 'pantai' in categories:
                        query = 'beach,ocean,indonesia'
                    elif 'museum' in categories:
                        query = 'museum,indonesia,culture'
                    elif 'air_terjun' in categories:
                        query = 'waterfall,nature,indonesia'
                    elif 'lapangan' in categories:
                        query = 'park,field,indonesia'
                    elif 'situs_sejarah' in categories:
                        query = 'historical,indonesia,heritage'
                    else:
                        query = 'indonesia,tourism,travel'
                    
                    print(f"Trying Unsplash with query: {query}")
                    if get_unsplash_image(query, image_path):
                        success = True
                        print("✓ Unsplash image downloaded successfully")
                    else:
                        # Buat gambar dummy sebagai pilihan terakhir
                        print("Creating dummy image...")
                        if create_dummy_image(
                            item.get('nama', 'Unknown Place'), 
                            item.get('kategori', ['lainnya']), 
                            image_path
                        ):
                            success = True
                            print("✓ Dummy image created successfully")
                
                if success:
                    item['foto'] = f"data/images/{image_filename}"
                else:
                    item['foto'] = None
                    print("✗ All methods failed")
                
                # Tunggu sebentar untuk menghindari rate limiting
                time.sleep(random.uniform(1, 2))
            else:
                # Buat gambar dummy jika tidak ada foto sama sekali
                image_filename = f"wisata_{current_id}.jpg"
                image_path = image_dir / image_filename
                
                if create_dummy_image(
                    item.get('nama', 'Unknown Place'), 
                    item.get('kategori', ['lainnya']), 
                    image_path
                ):
                    item['foto'] = f"data/images/{image_filename}"
                else:
                    item['foto'] = None
            
            current_id += 1
            
        all_data.extend(data)
    
    print(f"Total tempat wisata yang digabungkan: {len(all_data)}")
    
    # Buat DataFrame
    df = pd.DataFrame(all_data)
    columns = ['id'] + [col for col in df.columns if col != 'id']
    df = df[columns]
    
    # Simpan ke JSON
    output_json = output_dir / 'tempat_wisata_indonesia.json'
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)
    print(f"Data berhasil disimpan ke {output_json}")
    
    # Simpan ke CSV
    output_csv = output_dir / 'tempat_wisata_indonesia.csv'
    df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"Data berhasil disimpan ke {output_csv}")
    
    # Statistik
    successful_downloads = sum(1 for item in all_data if item.get('foto') is not None)
    print(f"\nStatistik:")
    print(f"Total tempat wisata: {len(all_data)}")
    print(f"Gambar berhasil diproses: {successful_downloads}")
    print(f"Tingkat keberhasilan: {(successful_downloads/len(all_data)*100):.1f}%")

if __name__ == "__main__":
    main()