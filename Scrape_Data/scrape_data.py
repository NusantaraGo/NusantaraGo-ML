import os
import time
import pandas as pd
import random
import re
import json
import undetected_chromedriver as uc
import requests
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from selenium.common.exceptions import TimeoutException
from urllib.parse import urlparse, parse_qs
import wikipedia
import wikipediaapi

class GoogleMapsScraper:
    def __init__(self):
        # Inisialisasi Wikipedia API
        self.wiki = wikipediaapi.Wikipedia(
            language='id',
            user_agent='NusantaraGo/1.0 (https://github.com/NusantaraGo/NusantaraGo-ML;nusantarago245@gmail.com) Python/3.10'
        )
        
        options = uc.ChromeOptions()
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--window-size=1920,1080")
        options.add_argument('--headless')
        
        # Menambahkan user-agent dan opsi keamanan
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36"
        ]
        options.add_argument(f'user-agent={random.choice(user_agents)}')
        
        # Menambahkan opsi keamanan tambahan
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_argument('--disable-extensions')
        options.add_argument('--disable-popup-blocking')
        options.add_argument('--disable-notifications')
        options.add_argument('--disable-infobars')
        options.add_argument('--disable-gpu')
        options.add_argument('--disable-software-rasterizer')
        options.add_argument('--ignore-certificate-errors')
        options.add_argument('--allow-running-insecure-content')
        options.add_argument('--disable-web-security')
        options.add_argument('--disable-features=IsolateOrigins,site-per-process')

        try:
            self.driver = uc.Chrome(options=options)
            self.wait = WebDriverWait(self.driver, 20)
            self.scraped_urls = set()

            # Handle cookie consent
            self.driver.get("https://www.google.com/maps")
            try:
                consent_buttons = [
                    '//button[contains(., "Reject all") or contains(., "Reject") or contains(., "Decline")]',
                    '//button[contains(., "Tolak semua") or contains(., "Tolak")]',
                    '//button[contains(@jsname, "tWT92d")]',
                    '//div[contains(@role, "dialog")]//button',
                ]

                for xpath in consent_buttons:
                    try:
                        reject_button = self.wait.until(EC.element_to_be_clickable((By.XPATH, xpath)))
                        reject_button.click()
                        print("Successfully clicked consent button")
                        time.sleep(2)
                        break
                    except Exception:
                        continue
            except Exception as e:
                print(f"Info: No cookie consent popup found: {str(e)}")
        except Exception as e:
            print(f"Error initializing Chrome: {str(e)}")
            raise

    def clean_filename(self, name):
        return re.sub(r'[\\/*?:"<>|]', "", name).replace(" ", "_")

    def search_places(self, province):
        search_query = f"tempat wisata terkenal di {province}"
        print(f"Mencari: {search_query}")
        self.driver.get(f"https://www.google.com/maps/search/{search_query.replace(' ', '+')}")
        time.sleep(5)

        try:
            self.wait.until(EC.presence_of_element_located(
                (By.CSS_SELECTOR, 'div[role="feed"], div.section-result-content, div[role="region"]')
            ))
        except TimeoutException:
            print("Tidak ada hasil pencarian")
            return False

        scroll_attempt = 0
        last_count = 0
        max_scroll_attempts = 10  # Meningkatkan jumlah scroll attempts

        while scroll_attempt < max_scroll_attempts:
            scroll_targets = [
                'div[role="feed"]',
                'div.section-scrollbox',
                'div[aria-label*="Results"]',
                'div.m6QErb[role="region"]',
                'div.m6QErb',
                'div.section-layout',
                'div[role="region"]',
            ]

            scrolled = False
            for selector in scroll_targets:
                try:
                    scrollable_div = self.driver.find_element(By.CSS_SELECTOR, selector)
                    self.driver.execute_script(
                        'arguments[0].scrollTop = arguments[0].scrollHeight',
                        scrollable_div
                    )
                    scrolled = True
                    break
                except Exception:
                    continue

            if not scrolled:
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")

            time.sleep(random.uniform(2.0, 3.0))

            current_results = []
            result_selectors = [
                'a[href*="/maps/place/"]',
                'a[data-item-id*="place"]',
                'div[jsaction*="placeCard"] a',
                'div.section-result a',
                'div[role="article"] a',
            ]

            for selector in result_selectors:
                elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                if elements:
                    current_results.extend(elements)

            print(f"Ditemukan {len(current_results)} hasil setelah scroll {scroll_attempt + 1}")

            if len(current_results) >= 100:  # Meningkatkan target jumlah hasil
                break

            if len(current_results) == last_count:
                scroll_attempt += 1
            last_count = len(current_results)

        return last_count > 0

    def get_place_urls(self, max_places=100):
        urls = []
        selectors = [
            'a[href*="/maps/place/"]',
            'a[data-item-id*="place"]',
            'div[jsaction*="placeCard"] a',
            'div.section-result a',
            'div[role="article"] a',
        ]

        for selector in selectors:
            elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
            for element in elements:
                url = element.get_attribute('href')
                if url and '/maps/place/' in url and url not in urls:
                    urls.append(url)
                    if len(urls) >= max_places:
                        break
            if len(urls) >= max_places:
                break

        return urls

    def get_wikipedia_description(self, place_name):
        """Mengambil deskripsi dari Wikipedia berdasarkan nama tempat"""
        try:
            # Tambahkan kata kunci untuk mencari lebih spesifik
            search_queries = [
                place_name,
                f"{place_name} (tempat wisata)",
                f"{place_name} (pantai)",  # Jika mengandung kata pantai
                f"{place_name} (gunung)",  # Jika mengandung kata gunung
                f"{place_name} (danau)",   # Jika mengandung kata danau
                f"{place_name} (wisata)",  # Tambahan kata wisata
            ]
            
            # Coba setiap query
            for query in search_queries:
                # Coba cari artikel Wikipedia
                page = self.wiki.page(query)
                if page.exists():
                    # Periksa apakah judul artikel mengandung nama tempat
                    if place_name.lower() in page.title.lower():
                        summary = page.summary
                        if summary:
                            # Batasi panjang deskripsi
                            return summary[:500] + "..." if len(summary) > 500 else summary
                
                # Jika tidak ditemukan, coba cari dengan kata kunci
                search_results = wikipedia.search(query, results=1)
                if search_results:
                    page = self.wiki.page(search_results[0])
                    if page.exists():
                        # Periksa apakah judul artikel mengandung nama tempat
                        if place_name.lower() in page.title.lower():
                            summary = page.summary
                            if summary:
                                return summary[:500] + "..." if len(summary) > 500 else summary
            
            return 'N/A'
        except Exception as e:
            print(f"Error getting Wikipedia description for {place_name}: {str(e)}")
            return 'N/A'

    def get_place_photos(self, url, place_name, province):
        """Mengambil link foto-foto dari tempat wisata"""
        try:
            # Buka halaman foto
            photo_url = url.replace('/place/', '/place/photo/')
            self.driver.get(photo_url)
            time.sleep(3)

            # Tunggu sampai foto muncul
            try:
                self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'img[src*="googleusercontent"]')))
            except TimeoutException:
                print(f"No photos found for {place_name}")
                return []

            # Ambil semua link foto
            photo_urls = []
            photo_elements = self.driver.find_elements(By.CSS_SELECTOR, 'img[src*="googleusercontent"]')
            
            for idx, element in enumerate(photo_elements[:5]):  # Ambil maksimal 5 foto
                try:
                    photo_url = element.get_attribute('src')
                    if photo_url and 'googleusercontent' in photo_url:
                        photo_urls.append(photo_url)
                        print(f"Found photo {idx+1} for {place_name}")
                        time.sleep(random.uniform(1, 2))
                except Exception as e:
                    print(f"Error getting photo {idx+1}: {str(e)}")
                    continue

            return photo_urls
        except Exception as e:
            print(f"Error getting photos for {place_name}: {str(e)}")
            return []

    def get_category(self, name, description):
        """Menentukan kategori tempat wisata berdasarkan nama dan deskripsi"""
        categories = {
            'pantai': ['pantai', 'beach', 'laut', 'pesisir', 'teluk'],
            'gunung': ['gunung', 'mountain', 'bukit', 'hill', 'pegunungan', 'puncak'],
            'danau': ['danau', 'lake', 'telaga', 'waduk', 'bendungan'],
            'air_terjun': ['air terjun', 'waterfall', 'curug'],
            'taman': ['taman', 'park', 'garden', 'kebun', 'taman sari'],
            'museum': ['museum', 'galeri', 'gallery'],
            'candi': ['candi', 'temple', 'pura', 'vihara'],
            'taman_nasional': ['taman nasional', 'national park'],
            'pulau': ['pulau', 'island', 'kepulauan'],
            'goa': ['goa', 'cave', 'gua'],
            'situs_sejarah': ['situs', 'sejarah', 'historical', 'heritage', 'monumen', 'tugu'],
            'taman_rekreasi': ['rekreasi', 'recreation', 'hiburan', 'entertainment'],
            'benteng': ['benteng', 'fort', 'fortress', 'castle', 'keraton'],
            'lapangan': ['lapangan', 'field', 'stadium', 'blang'],
            'rumah_adat': ['rumah adat', 'rumoh', 'rumah tradisional', 'traditional house'],
            'masjid': ['masjid', 'mosque', 'masigit'],
            'makam': ['makam', 'grave', 'kuburan', 'cemetery'],
            'pasar': ['pasar', 'market', 'bazar'],
            'taman_hutan': ['taman hutan', 'hutan raya', 'forest park'],
            'wisata_alam': ['wisata alam', 'nature', 'alam']
        }

        text = (name + ' ' + description).lower()
        matched_categories = []

        for category, keywords in categories.items():
            if any(keyword in text for keyword in keywords):
                matched_categories.append(category)

        return matched_categories if matched_categories else ['lainnya']

    def clean_address(self, address):
        """Membersihkan alamat dari nomor dan kata-kata yang tidak perlu"""
        try:
            # Hapus karakter khusus di awal kalimat
            address = re.sub(r'^[^\w\s]+', '', address)
            
            # Pisahkan alamat berdasarkan koma
            parts = address.split(',')
            
            # Hapus bagian yang mengandung nomor atau kata-kata yang tidak perlu
            cleaned_parts = []
            for part in parts:
                part = part.strip()
                # Skip jika bagian mengandung nomor
                if any(char.isdigit() for char in part):
                    continue
                # Skip jika bagian mengandung kata-kata yang tidak perlu
                skip_words = ['regency', 'kabupaten', 'kota', 'city', 'district']
                if any(word.lower() in part.lower() for word in skip_words):
                    continue
                cleaned_parts.append(part)
            
            # Gabungkan kembali dengan koma
            return ', '.join(cleaned_parts)
        except:
            return address

    def parse_place_details(self, url, province):
        for retry in range(3):
            try:
                print(f"Processing URL: {url}")
                self.driver.get(url)
                time.sleep(random.uniform(3, 5))

                try:
                    self.wait.until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, 'h1, div[role="heading"]'))
                    )
                except TimeoutException:
                    print(f"Timeout waiting for page to load, retrying... (attempt {retry+1}/3)")
                    continue

                soup = BeautifulSoup(self.driver.page_source, 'html.parser')

                # Extract name first
                name = 'N/A'
                name_selectors = [
                    'h1.fontHeadlineLarge',
                    'h1',
                    'div[role="heading"]',
                ]
                for selector in name_selectors:
                    element = soup.select_one(selector)
                    if element:
                        name = element.text.strip()
                        break

                # Get photos if name is found
                photo_urls = []
                description = 'N/A'
                if name != 'N/A':
                    photo_urls = self.get_place_photos(url, name, province)
                    description = self.get_wikipedia_description(name)

                # Extract coordinates
                coords = None
                coords_match = re.search(r'@(-?\d+\.\d+),(-?\d+\.\d+)', url)
                if coords_match:
                    coords = {
                        "latitude": float(coords_match.group(1)),
                        "longitude": float(coords_match.group(2))
                    }

                if not coords:
                    lat_match = re.search(r'!3d(-?\d+\.\d+)', url)
                    lng_match = re.search(r'!4d(-?\d+\.\d+)', url)
                    if lat_match and lng_match:
                        coords = {
                            "latitude": float(lat_match.group(1)),
                            "longitude": float(lng_match.group(1))
                        }

                coordinates = coords if coords else 'N/A'

                # Extract reviews count
                reviews_count = 'N/A'
                reviews_selectors = [
                    'div.fontBodyMedium > span > span:first-child',
                    'span.section-rating-term',
                    'span[aria-label*="ulasan"]',
                    'span[aria-label*="review"]',
                ]
                for selector in reviews_selectors:
                    element = soup.select_one(selector)
                    if element:
                        text = element.text.strip()
                        num_match = re.search(r'(\d[\d.,]*)', text)
                        if num_match:
                            reviews_count = num_match.group(1).replace(',', '')
                            break

                # Extract rating
                rating = 'N/A'
                rating_selectors = [
                    'div.fontDisplayLarge',
                    'span[aria-hidden="true"]',
                    'span.section-star-display',
                ]
                for selector in rating_selectors:
                    element = soup.select_one(selector)
                    if element:
                        text = element.text.strip()
                        if text and re.match(r'^\d+(\.\d+)?$', text) and float(text) <= 5.0:
                            rating = text
                            break

                # Extract address
                address = 'N/A'
                address_selectors = [
                    'button[data-item-id="address"]',
                    'div[data-item-id="address"]',
                    'button[aria-label*="alamat"]',
                ]
                for selector in address_selectors:
                    element = soup.select_one(selector)
                    if element:
                        text = element.text.strip()
                        if text:
                            address = self.clean_address(text)
                            break

                # Determine category
                categories = self.get_category(name, description)

                return {
                    'nama': name,
                    'alamat': address,
                    'rating': rating,
                    'jumlah_review': reviews_count,
                    'deskripsi': description,
                    'koordinat': json.dumps(coordinates, ensure_ascii=False),
                    'url': url,
                    'provinsi': province,
                    'foto': json.dumps(photo_urls, ensure_ascii=False),
                    'kategori': json.dumps(categories, ensure_ascii=False)
                }
            except Exception as e:
                print(f"Error on attempt {retry+1}: {str(e)}")
                if retry == 2:
                    print(f"Failed to scrape: {url} after 3 attempts")
                    return None
                time.sleep(5)

    def scrape_province(self, province, max_places=15):
        print(f"\nStarting scraping for province: {province}")
        all_data = []
        temp_data = []  # Menyimpan data sementara untuk diurutkan

        try:
            if self.search_places(province):
                place_urls = self.get_place_urls(max_places)

                for idx, url in enumerate(place_urls):
                    if url in self.scraped_urls:
                        continue

                    print(f"  Place {idx+1}/{len(place_urls)}")
                    place_data = self.parse_place_details(url, province)
                    if place_data:
                        # Konversi rating dan jumlah review ke float untuk perhitungan
                        try:
                            rating = float(place_data['rating']) if place_data['rating'] != 'N/A' else 0
                            reviews = float(place_data['jumlah_review'].replace(',', '')) if place_data['jumlah_review'] != 'N/A' else 0
                            # Hitung skor berdasarkan rating dan jumlah review
                            # Rating memiliki bobot 0.6 dan jumlah review memiliki bobot 0.4
                            score = (rating * 0.6) + (min(reviews/1000, 5) * 0.4)  # Normalisasi jumlah review
                            place_data['score'] = score
                            temp_data.append(place_data)
                        except:
                            # Jika ada error dalam konversi, tetap simpan data dengan skor 0
                            place_data['score'] = 0
                            temp_data.append(place_data)
                        
                        self.scraped_urls.add(url)
                        time.sleep(random.uniform(2.0, 4.0))

                # Urutkan data berdasarkan skor tertinggi
                temp_data.sort(key=lambda x: x['score'], reverse=True)
                
                # Ambil max_places data teratas
                all_data = temp_data[:max_places]
                
                # Hapus kolom score sebelum menyimpan
                for data in all_data:
                    data.pop('score', None)

            else:
                print(f"No results found for {province}")

        except Exception as e:
            print(f"Error processing {province}: {str(e)}")

        return all_data

    def save_data(self, data, province):
        if not data:
            print(f"No data to save for {province}")
            return

        safe_name = self.clean_filename(province)
        os.makedirs('json', exist_ok=True)

        # Save to JSON dengan format yang lebih rapi
        json_path = os.path.join('json', f'tempat_wisata_{safe_name}.json')
        # Konversi DataFrame ke dict dengan format yang diinginkan
        records = data
        for record in records:
            # Format foto menjadi list yang lebih rapi
            if record['foto'] != 'N/A':
                try:
                    photos = json.loads(record['foto'])
                    record['foto'] = photos  # Simpan sebagai list Python, bukan string JSON
                except:
                    record['foto'] = []
            
            # Format koordinat menjadi objek yang berisi latitude dan longitude
            if record['koordinat'] != 'N/A':
                try:
                    coords = json.loads(record['koordinat'])
                    record['koordinat'] = coords
                except:
                    record['koordinat'] = 'N/A'
            
            # Format kategori menjadi list yang lebih rapi
            if record['kategori'] != 'N/A':
                try:
                    categories = json.loads(record['kategori'])
                    record['kategori'] = categories  # Simpan sebagai list Python, bukan string JSON
                except:
                    record['kategori'] = []
        
        # Simpan dengan indentasi yang lebih baik
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"JSON data saved to: {json_path}")

    def close(self):
        try:
            self.driver.quit()
            print("Browser closed successfully")
        except Exception as e:
            print(f"Error closing browser: {str(e)}")

def main():
    MAX_PLACES = 15
    COOLDOWN = 30

    os.makedirs('json', exist_ok=True)

    provinces = input("Masukkan nama provinsi (pisahkan dengan koma jika lebih dari satu): ").split(',')
    provinces = [p.strip() for p in provinces if p.strip()]

    if not provinces:
        print("Tidak ada provinsi yang dimasukkan!")
        return

    for province in provinces:
        try:
            print(f"\n{'='*50}")
            print(f"MULAI SCRAPING UNTUK: {province.upper()}")
            print(f"{'='*50}")

            scraper = GoogleMapsScraper()
            data = scraper.scrape_province(province, MAX_PLACES)
            if data:
                scraper.save_data(data, province)
                print(f"\nRingkasan untuk {province}:")
                print(f"- Total tempat: {len(data)}")

                # Calculate average rating
                ratings = []
                for d in data:
                    try:
                        if d['rating'] != 'N/A':
                            ratings.append(float(d['rating'].replace(',', '.')))
                    except:
                        pass

                if ratings:
                    avg_rating = sum(ratings) / len(ratings)
                    print(f"- Rata-rata rating: {avg_rating:.2f}")
                else:
                    print("- Rata-rata rating: N/A")

                # Show data completion rates
                fields = ['nama', 'alamat', 'rating', 'koordinat', 'deskripsi', 'kategori']
                completion_rates = {}

                for field in fields:
                    valid_count = sum(1 for d in data if d[field] != 'N/A')
                    completion_rates[field] = (valid_count / len(data)) * 100

                print("\nTingkat kelengkapan data:")
                for field, rate in completion_rates.items():
                    print(f"- {field}: {rate:.1f}%")

            else:
                print(f"Tidak ada data yang berhasil di-scrape untuk {province}")
        except Exception as e:
            print(f"Error saat memproses {province}: {str(e)}")
        finally:
            try:
                scraper.close()
            except:
                pass

            if province != provinces[-1]:
                print(f"\nMenunggu {COOLDOWN} detik sebelum provinsi berikutnya...")
                time.sleep(COOLDOWN)

def install_dependencies():
    try:
        import google.colab
        IN_COLAB = True
    except ImportError:
        IN_COLAB = False

    if IN_COLAB:
        print("Installing dependencies for Google Colab...")
        import subprocess
        subprocess.run(['pip', 'install', 'selenium', 'webdriver_manager', 'beautifulsoup4'])
        subprocess.run(['apt-get', 'update'])
        subprocess.run(['apt', 'install', '-y', 'chromium-chromedriver'])
        subprocess.run(['cp', '/usr/lib/chromium-browser/chromedriver', '/usr/bin'])

if __name__ == "__main__":
    try:
        install_dependencies()
        main()
    except Exception as e:
        print(f"Main error: {str(e)}")
        import traceback
        traceback.print_exc()