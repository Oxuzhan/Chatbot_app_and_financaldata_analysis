import streamlit as st
import json
import re
import os
import google.generativeai as genai
from datetime import datetime
from typing import Dict, Any, Optional
import time

# Page configuration
st.set_page_config(
    page_title="Araç Finansmanı Chatbot",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: #2c3e50;
        padding: 2rem;
        border-radius: 10px;
        color: #ffffff;
        text-align: center;
        margin-bottom: 2rem;
        font-family: 'Segoe UI', 'Roboto', Arial, sans-serif;
        font-weight: 600;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .main-header h1 {
        color: #ffffff;
        margin-bottom: 0.5rem;
        font-size: 2.5rem;
    }

    .main-header p {
        color: #ecf0f1;
        font-size: 1.1rem;
        margin: 0;
    }

    .chat-message {
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #3498db;
        font-family: 'Segoe UI', 'Roboto', Arial, sans-serif;
        font-size: 1rem;
        line-height: 1.5;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }

    .user-message {
        background-color: #ffffff;
        border-left-color: #3498db;
        color: #2c3e50;
    }

    .user-message strong {
        color: #2980b9;
    }

    .bot-message {
        background-color: #f8f9fa;
        border-left-color: #27ae60;
        color: #2c3e50;
    }

    .bot-message strong {
        color: #27ae60;
    }

    .info-card {
        background: #ffffff;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border: 1px solid #e9ecef;
    }

    .info-card h3 {
        color: #2c3e50;
        margin-bottom: 1rem;
        font-family: 'Segoe UI', 'Roboto', Arial, sans-serif;
    }

    .info-card ul {
        color: #495057;
        line-height: 1.6;
    }

    .info-card li {
        margin-bottom: 0.5rem;
    }

    .status-success {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-weight: 500;
    }

    .status-error {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-weight: 500;
    }

    .metric-card {
        background: #3498db;
        color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        font-family: 'Segoe UI', 'Roboto', Arial, sans-serif;
        font-weight: 600;
    }

    /* Streamlit component styling */
    .stButton > button {
        background-color: #3498db;
        color: #ffffff;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-family: 'Segoe UI', 'Roboto', Arial, sans-serif;
        font-weight: 500;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        background-color: #2980b9;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }

    .stTextInput > div > div > input {
        border-radius: 6px;
        border: 2px solid #e9ecef;
        font-family: 'Segoe UI', 'Roboto', Arial, sans-serif;
    }

    .stTextInput > div > div > input:focus {
        border-color: #3498db;
        box-shadow: 0 0 0 0.2rem rgba(52, 152, 219, 0.25);
    }

    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }

    /* Main content area */
    .css-18e3th9 {
        padding-top: 2rem;
    }

    /* Chat input styling */
    .stChatInput > div {
        background-color: #ffffff;
        border: 2px solid #e9ecef;
        border-radius: 25px;
    }

    .stChatInput > div:focus-within {
        border-color: #3498db;
        box-shadow: 0 0 0 0.2rem rgba(52, 152, 219, 0.25);
    }

    .stChatInput input {
        color: #2c3e50 !important;
        font-family: 'Segoe UI', 'Roboto', Arial, sans-serif !important;
        background-color: #ffffff !important;
        font-size: 16px !important;
    }

    .stChatInput input::placeholder {
        color: #6c757d !important;
    }

    .stChatInput textarea {
        color: #2c3e50 !important;
        font-family: 'Segoe UI', 'Roboto', Arial, sans-serif !important;
        background-color: #ffffff !important;
        font-size: 16px !important;
    }

    .stChatInput textarea::placeholder {
        color: #6c757d !important;
    }
</style>
""", unsafe_allow_html=True)


class VehicleFinanceChatbot:
    def __init__(self, api_key: str, config_file: str = "chatbot_config.json"):
        """Initialize the chatbot"""
        # Configure Gemini API
        genai.configure(api_key=api_key)
        # Changed to free model: gemini-2.5-flash-preview-05-20
        self.model = genai.GenerativeModel('gemini-2.5-flash-preview-05-20')

        # Load configuration file
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            # Create default config if file doesn't exist
            self.config = self._create_default_config()
            self._save_config(config_file)

        # Store user data
        self.user_data = {}
        self.current_step = "greeting"
        self.application_type = None  # "new" or "used"

    def _create_default_config(self):
        """Create default configuration"""
        return {
            "finance_rules": {
                "new": {
                    "max_vehicle_value": 7000000,
                    "max_financing_ratio": 0.6,
                    "guarantor_threshold": 5000000
                },
                "used": {
                    "max_vehicle_age": 5,
                    "max_financing_ratio": 0.4,
                    "max_loan_amount": 3000000
                }
            },
            "faq": {
                "supported_brands": "Tüm marka ve modeller (ticari araçlar hariç)",
                "interest_rates": "Güncel piyasa koşullarına göre belirlenir",
                "loan_terms": "12-60 ay vade seçenekleri"
            }
        }

    def _save_config(self, config_file: str):
        """Save configuration to file"""
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, ensure_ascii=False, indent=2)

    def get_system_prompt(self) -> str:
        """
        Generate system prompt for the AI model

        🔧 SYSTEM PROMPT CUSTOMIZATION POINT:
        Modify this method to change how the AI responds to users.
        You can adjust the personality, rules, or response style here.
        """
        return f"""
        # ARAÇ FİNANSMANI UZMANI SİSTEM PROMPT

        ## ROL VE KİMLİK
        Sen profesyonel bir banka araç finansmanı uzmanısın. İsmin "Araç Finansman Asistanı" ve müşterilere araç kredisi konusunda kapsamlı yardım sağlıyorsun. Deneyimli, güvenilir ve çözüm odaklı bir yaklaşımın var.

        ## TEMEL İLKELER
        ### İletişim Tarzı:
        - Her zaman Türkçe konuş ve saygılı bir dil kullan
        - Samimi ama profesyonel bir ton benimse
        - Müşterinin seviyesine uygun açıklamalar yap
        - Emojileri uygun yerlerde kullan (💡, ✅, ❌, 🚗, 💰)

        ### Yanıt Kalitesi:
        - Kısa, net ve anlaşılır yanıtlar ver
        - Teknik terimleri basit Türkçe ile açıkla
        - Her yanıtta bir sonraki adımı belirt
        - Belirsizlik durumunda soru sor

        ## GÖREV KAPSAMI
        ### ANA SORUMLULUKLARIN:
        1. **Araç finansmanı başvuru süreci** - Adım adım rehberlik
        2. **Kredi koşulları bilgilendirme** - Faiz, vade, teminat açıklamaları
        3. **Uygunluk değerlendirmesi** - Gerçek zamanlı kontroller
        4. **Dokümantasyon rehberliği** - Gerekli evrak listesi
        5. **Çapraz satış fırsatları** - HGS, sigorta vb. ürün önerileri

        ### KONU DIŞI DURUMLAR:
        Araç finansmanı dışındaki konularda:
        - Kibar bir şekilde konu dışı olduğunu belirt
        - Mümkünse araç finansmanı ile bağlantı kur
        - Genel sohbette samimi ama odakta kal
        - Örnek: "Bu konu uzmanlık alanım dışında ama araç finansmanı için size nasıl yardımcı olabilirim? 🚗"

        ## FİNANSMAN KURALLARI VE LİMİTLER
        {json.dumps(self.config['finance_rules'], ensure_ascii=False, indent=2)}

        ## SIK SORULAN SORULAR VE YANITLAR
        {json.dumps(self.config['faq'], ensure_ascii=False, indent=2)}

        ## DOĞRULAMA VE GÜVENLİK
        ### Kesinlikle YAPMA:
        - Kişisel verileri (TCKN, telefon, adres) paylaşma veya kaydetme
        - Gerçek faiz oranları ve kesin meblağlar vermek (güncel değişken bilgiler)
        - Müşteri adına karar vermek
        - Yanıltıcı veya yanlış bilgi vermek

        ### Her Zaman YAP:
        - Girilen verileri anında doğrula
        - Hata durumunda net açıklama yap
        - Güvenlik uyarılarını belirt
        - Şüpheli durumlarda şubeye yönlendir

        ## KONUŞMA AKIŞI YÖNETİMİ
        ### Başlangıç:
        - Sıcak karşılama ve kendini tanıt
        - Hizmet seçeneklerini sun (yeni/ikinci el)
        - Süreci kısaca açıkla

        ### Bilgi Toplama:
        - Her seferinde tek bilgi iste
        - Girilen bilgiyi onaylayarak tekrarla
        - Doğrulama hatalarını anında bildir
        - İlerleme durumunu göster

        ### Sonlandırma:
        - Başvuru özetini detaylı sun
        - Çapraz satış teklifi yap
        - Teşekkür et ve yeni başvuru için davet et

        ## HATA YÖNETİMİ
        Teknik sorun durumunda:
        "Üzgünüm, sistemde geçici bir sorun yaşanıyor. Lütfen şu bilgiyi tekrar girebilir misiniz? Sorun devam ederse şubelerimizden destek alabilirsiniz. 🔧"

        Kural ihlali durumunda:
        "Bu işlem kurallarımıza uymuyor. [Sebep açıklaması]. Alternatif çözüm: [Öneri] 💡"

        ## PERFORMANS HEDEFLERİ
        - Müşteri memnuniyeti odaklı yaklaşım
        - Hızlı ve doğru bilgi sağlama
        - Başvuru tamamlama oranını artırma
        - Çapraz satış fırsatlarını değerlendirme

        Şimdi müşteriyle doğal, yardımsever ve profesyonel bir sohbet başlat!
        """

    def validate_data(self, field: str, value: Any, app_type: str) -> tuple[bool, str]:
        """Validate input data"""
        rules = self.config['finance_rules'][app_type]

        if field == "vehicle_value":
            if app_type == "new":
                if value > 7000000:
                    return False, "7M TL üzeri araçlar için başvuru yapılamaz"
            return True, ""

        elif field == "vehicle_age":
            if app_type == "used" and value > 5:
                return False, "5 yaş üstü araçlar için başvuru oluşturulamaz"
            return True, ""

        elif field == "loan_amount":
            vehicle_value = self.user_data.get('vehicle_value', 0)
            if app_type == "new":
                max_amount = vehicle_value * 0.6
                if value > max_amount:
                    return False, f"Araç fiyatının en fazla %60'ı ({max_amount:,.0f} TL) talep edilebilir"
            else:  # used
                max_amount = min(vehicle_value * 0.4, 3000000)
                if value > max_amount:
                    return False, f"Araç kasko değerinin en fazla %40'ı veya 3M TL ({max_amount:,.0f} TL) talep edilebilir"
            return True, ""

        elif field == "tckn":
            if not re.match(r'^\d{11}$', str(value)):
                return False, "TCKN 11 haneli olmalıdır"
            return True, ""

        return True, ""

    def extract_info_from_text(self, text: str, expected_type: str) -> Optional[Any]:
        """Extract information from text"""
        if expected_type == "number":
            # Find numbers
            numbers = re.findall(r'\b\d+(?:\.\d+)?\b', text.replace(',', '').replace('.', ''))
            return int(numbers[0]) if numbers else None
        elif expected_type == "tckn":
            # Find TCKN
            tckn = re.findall(r'\b\d{11}\b', text)
            return tckn[0] if tckn else None
        elif expected_type == "text":
            return text.strip()
        return None

    def generate_response(self, user_message: str) -> str:
        """Generate response using AI - only when needed"""
        try:
            # Simple FAQ responses without LLM
            if any(word in user_message.lower() for word in ["hangi model", "model", "marka", "araç türü"]):
                return "Bankamızda tüm marka ve modeller için finansman sağlıyoruz. Sadece ticari araçlar (kamyon, minibüs, otobüs) hariçtir. Hangi araç modelini tercih ediyorsunuz?"

            if any(word in user_message.lower() for word in ["faiz", "oran", "vade"]):
                return "Faiz oranları güncel piyasa koşullarına göre belirlenir. 12-60 ay vade seçenekleri mevcuttur. Detaylı bilgi için şubelerimize başvurabilirsiniz."

            # Use AI for complex responses
            prompt = self.get_system_prompt() + f"\n\nKullanıcı mesajı: {user_message}"
            response = self.model.generate_content(prompt)
            return response.text

        except Exception as e:
            return "Üzgünüm, teknik bir sorun yaşandı. Lütfen tekrar deneyin."

    def _get_update_options(self) -> str:
        """Güncellenebilir alanları listeler"""
        options = []
        if self.application_type == "new":
            options = [
                "1. Araç Değeri",
                "2. Araç Modeli",
                "3. Finansman Tutarı"
            ]
            if 'guarantor_tckn' in self.user_data:
                options.append("4. Kefil TCKN")
        else:  # used
            options = [
                "1. Kasko Değeri",
                "2. Araç Yaşı",
                "3. Finansman Tutarı"
            ]
            if 'seller_tckn' in self.user_data:
                options.append("4. Satıcı TCKN")
        
        return "\n".join(options)

    def _handle_update_selection(self, user_message: str) -> Dict[str, Any]:
        """Kullanıcının güncelleme seçimini işler"""
        try:
            # Eğer kullanıcı evet/hayır gibi bir cevap verdiyse burada kontrol et
            if user_message.strip().lower() in ["hayır", "hayir", "yok", "istemiyorum"]:
                self.current_step = "confirmation"
                return {
                    "response": self._generate_confirmation_message(),
                    "step": self.current_step,
                    "data": self.user_data
                }
            elif user_message.strip().lower() in ["evet", "yes", "istiyorum"]:
                return {
                    "response": "Hangi bilgiyi güncellemek istersiniz?\n" + self._get_update_options(),
                    "step": "update_selection",
                    "data": self.user_data
                }
            # Seçim numarası girildiyse
            choice = int(user_message.strip())
            if self.application_type == "new":
                if choice == 1:
                    self.user_data.pop('vehicle_value', None)
                    self.current_step = "update_field_input"
                    self._last_update_field = 'vehicle_value'
                    return {
                        "response": "Yeni araç değerini giriniz:",
                        "step": self.current_step,
                        "data": self.user_data
                    }
                elif choice == 2:
                    self.user_data.pop('vehicle_model', None)
                    self.current_step = "update_field_input"
                    self._last_update_field = 'vehicle_model'
                    return {
                        "response": "Yeni araç modelini giriniz:",
                        "step": self.current_step,
                        "data": self.user_data
                    }
                elif choice == 3:
                    self.user_data.pop('loan_amount', None)
                    self.current_step = "update_field_input"
                    self._last_update_field = 'loan_amount'
                    return {
                        "response": "Yeni finansman tutarını giriniz:",
                        "step": self.current_step,
                        "data": self.user_data
                    }
                elif choice == 4 and 'guarantor_tckn' in self.user_data:
                    self.user_data.pop('guarantor_tckn', None)
                    self.current_step = "update_field_input"
                    self._last_update_field = 'guarantor_tckn'
                    return {
                        "response": "Yeni kefil TCKN giriniz:",
                        "step": self.current_step,
                        "data": self.user_data
                    }
            else:  # used
                if choice == 1:
                    self.user_data.pop('vehicle_value', None)
                    self.current_step = "update_field_input"
                    self._last_update_field = 'vehicle_value'
                    return {
                        "response": "Yeni kasko değerini giriniz:",
                        "step": self.current_step,
                        "data": self.user_data
                    }
                elif choice == 2:
                    self.user_data.pop('vehicle_age', None)
                    self.current_step = "update_field_input"
                    self._last_update_field = 'vehicle_age'
                    return {
                        "response": "Yeni araç yaşını giriniz:",
                        "step": self.current_step,
                        "data": self.user_data
                    }
                elif choice == 3:
                    self.user_data.pop('loan_amount', None)
                    self.current_step = "update_field_input"
                    self._last_update_field = 'loan_amount'
                    return {
                        "response": "Yeni finansman tutarını giriniz:",
                        "step": self.current_step,
                        "data": self.user_data
                    }
                elif choice == 4 and 'seller_tckn' in self.user_data:
                    self.user_data.pop('seller_tckn', None)
                    self.current_step = "update_field_input"
                    self._last_update_field = 'seller_tckn'
                    return {
                        "response": "Yeni satıcı TCKN giriniz:",
                        "step": self.current_step,
                        "data": self.user_data
                    }
            
            return {
                "response": "Geçersiz seçim. Lütfen tekrar deneyiniz:\n" + self._get_update_options(),
                "step": "update_selection",
                "data": self.user_data
            }
        except ValueError:
            return {
                "response": "Lütfen bir sayı giriniz:\n" + self._get_update_options(),
                "step": "update_selection",
                "data": self.user_data
            }

    def _handle_update_field_input(self, user_message: str) -> Dict[str, Any]:
        """Güncellenecek alan için yeni değeri alır ve tekrar güncelleme isteyip istemediğini sorar"""
        field = getattr(self, '_last_update_field', None)
        if not field:
            return {"response": "Bir hata oluştu. Lütfen tekrar deneyin.", "step": "confirmation", "data": self.user_data}
        # Alan tipine göre veri çek
        if field in ["vehicle_value", "loan_amount", "vehicle_age"]:
            value = self.extract_info_from_text(user_message, "number")
            if value is None:
                return {"response": "Lütfen geçerli bir değer giriniz:", "step": "update_field_input", "data": self.user_data}
            # Doğrulama
            app_type = self.application_type
            is_valid, error = self.validate_data(field, value, app_type)
            if not is_valid:
                return {"response": error, "step": "update_field_input", "data": self.user_data}
            self.user_data[field] = value
        elif field in ["vehicle_model"]:
            self.user_data[field] = user_message.strip()
        elif field in ["guarantor_tckn", "seller_tckn"]:
            tckn = self.extract_info_from_text(user_message, "tckn")
            if not tckn:
                return {"response": "Lütfen geçerli bir TCKN giriniz:", "step": "update_field_input", "data": self.user_data}
            is_valid, error = self.validate_data('tckn', tckn, self.application_type)
            if not is_valid:
                return {"response": error, "step": "update_field_input", "data": self.user_data}
            self.user_data[field] = tckn
        else:
            return {"response": "Bir hata oluştu. Lütfen tekrar deneyin.", "step": "confirmation", "data": self.user_data}
        # Güncelleme sonrası tekrar sor
        self.current_step = "update_selection"
        return {
            "response": "Başka bir değişiklik yapmak ister misiniz? (Evet/Hayır)",
            "step": self.current_step,
            "data": self.user_data
        }

    def process_message(self, user_message: str) -> Dict[str, Any]:
        """Process message and update state"""
        user_message_lower = user_message.strip().lower()

        # Exit controls - valid at every step
        if any(word in user_message_lower for word in
               ['çık', 'çıkış', 'quit', 'exit', 'bye', 'görüşürüz', 'hoşça kal', 'bitir', 'kapat']):
            return {
                "response": "Başvuru işleminiz yarıda kesildi. Teşekkürler, iyi günler! 👋",
                "step": "exit",
                "data": self.user_data,
                "should_exit": True
            }

        # Cancel/restart controls
        if any(word in user_message_lower for word in
               ['iptal', 'sıfırla', 'yeniden başla', 'restart', 'baştan', 'temizle']):
            self.user_data = {}
            self.current_step = "greeting"
            self.application_type = None
            return {
                "response": "Başvuru sıfırlandı. Yeniden başlamak için 'merhaba' yazabilirsiniz.",
                "step": self.current_step,
                "data": self.user_data
            }

        # Initial greeting
        if self.current_step == "greeting":
            if any(word in user_message_lower for word in ["merhaba", "selam", "iyi", "başla"]):
                self.current_step = "determine_type"
                return {
                    "response": "Merhaba! Araç finansmanı konusunda size yardımcı olmaktan mutluluk duyarım. Yeni araç mı yoksa ikinci el araç finansmanı mı istiyorsunuz?",
                    "step": self.current_step,
                    "data": self.user_data
                }

        # Determine application type
        elif self.current_step == "determine_type":
            if "yeni" in user_message_lower:
                self.application_type = "new"
                self.current_step = "collect_new_vehicle_info"
                return {
                    "response": "Yeni araç finansmanı için başvurunuzu alıyorum. Öncelikle aracın proforma fatura değerini öğrenebilir miyim? \n💡 İpucu: İstediğiniz zaman 'çık' yazarak çıkabilirsiniz.",
                    "step": self.current_step,
                    "data": self.user_data
                }
            elif any(word in user_message_lower for word in ["ikinci", "2.", "eski", "kullanılmış"]):
                self.application_type = "used"
                self.current_step = "collect_used_vehicle_info"
                return {
                    "response": "İkinci el araç finansmanı için başvurunuzu alıyorum. Öncelikle aracın kasko değerini öğrenebilir miyim? \n💡 İpucu: İstediğiniz zaman 'çık' yazarak çıkabilirsiniz.",
                    "step": self.current_step,
                    "data": self.user_data
                }

        # Collect new vehicle info
        elif self.current_step == "collect_new_vehicle_info":
            return self._handle_new_vehicle_collection(user_message)

        # Collect used vehicle info
        elif self.current_step == "collect_used_vehicle_info":
            return self._handle_used_vehicle_collection(user_message)

        # Handle confirmation step
        elif self.current_step == "confirmation":
            if any(word in user_message_lower for word in ["hayır", "hayir", "güncelle", "guncelle", "değiştir", "degistir"]):
                self.current_step = "update_selection"
                return {
                    "response": "Hangi bilgiyi güncellemek istersiniz?\n" + self._get_update_options(),
                    "step": self.current_step,
                    "data": self.user_data
                }
            elif any(word in user_message_lower for word in ["evet", "onayla", "tamam"]):
                # Save application
                save_result = self.save_application()
                if save_result["success"]:
                    self.current_step = "hgs_offer"
                    return {
                        "response": "🎉 Başvurunuz kaydedildi! Başvuru No: {}\n\nHGS ürünümüzü de almak ister misiniz? (Evet/Hayır)".format(save_result['application_id']),
                        "step": self.current_step,
                        "data": self.user_data
                    }
                else:
                    return {"response": f"❌ Başvuru kaydedilemedi: {save_result['error']}", "step": self.current_step, "data": self.user_data}
        # Handle HGS offer step
        elif self.current_step == "hgs_offer":
            if any(word in user_message_lower for word in ["evet", "isterim", "almak", "olsun"]):
                self.current_step = "end"
                return {
                    "response": "✅ HGS başvurunuz da alınmıştır. Tüm başvurularınız başarıyla tamamlandı! Yeni bir başvuru için 'merhaba' yazabilirsiniz. 👋",
                    "step": self.current_step,
                    "data": self.user_data
                }
            elif any(word in user_message_lower for word in ["hayır", "hayir", "istemem", "olmasın", "yok"]):
                self.current_step = "end"
                return {
                    "response": "Anlaşıldı, HGS başvurusu alınmadı. Tüm başvurularınız başarıyla tamamlandı! Yeni bir başvuru için 'merhaba' yazabilirsiniz. 👋",
                    "step": self.current_step,
                    "data": self.user_data
                }
            else:
                return {
                    "response": "Lütfen 'Evet' veya 'Hayır' şeklinde yanıt veriniz. HGS ürünümüzü almak ister misiniz?",
                    "step": self.current_step,
                    "data": self.user_data
                }

        # Handle update selection
        elif self.current_step == "update_selection":
            return self._handle_update_selection(user_message)

        # Handle update field input
        elif self.current_step == "update_field_input":
            return self._handle_update_field_input(user_message)

        # General AI response
        response = self.generate_response(user_message)
        return {
            "response": response,
            "step": self.current_step,
            "data": self.user_data
        }

    def _handle_new_vehicle_collection(self, user_message: str) -> Dict[str, Any]:
        """Handle new vehicle information collection"""
        # Vehicle value
        if 'vehicle_value' not in self.user_data:
            value = self.extract_info_from_text(user_message, "number")
            if value:
                is_valid, error = self.validate_data('vehicle_value', value, 'new')
                if is_valid:
                    self.user_data['vehicle_value'] = value
                    return {
                        "response": f"Araç değeri {value:,} TL olarak kaydedildi. Şimdi araç modelini belirtir misiniz?",
                        "step": self.current_step,
                        "data": self.user_data
                    }
                else:
                    return {"response": error, "step": self.current_step, "data": self.user_data}

        # Vehicle model
        elif 'vehicle_model' not in self.user_data:
            if any(word in user_message for word in ["hangi", "model", "marka", "tür", "neler"]):
                return {
                    "response": "Bankamızda tüm marka ve modeller için finansman sağlıyoruz (Toyota, Volkswagen, BMW, Mercedes, Renault, Ford vb.). Sadece ticari araçlar (kamyon, minibüs, otobüs) hariçtir. Hangi araç modelini seçtiniz?",
                    "step": self.current_step,
                    "data": self.user_data
                }

            if any(word in user_message for word in ["ticari", "kamyon", "minibüs", "otobüs"]):
                return {
                    "response": "Üzgünüm, ticari modeller için başvuru yapılamaz. Farklı bir araç modeli var mı?",
                    "step": self.current_step,
                    "data": self.user_data
                }

            self.user_data['vehicle_model'] = user_message

            if self.user_data['vehicle_value'] >= 5000000:
                return {
                    "response": f"Araç modeli kaydedildi. Araç fiyatı 5M ve üzeri olduğu için kefil TCKN'i gereklidir. Kefil TCKN'ini giriniz:",
                    "step": self.current_step,
                    "data": self.user_data
                }
            else:
                return {
                    "response": "Araç modeli kaydedildi. Son olarak istediğiniz finansman tutarını belirtiniz:",
                    "step": self.current_step,
                    "data": self.user_data
                }

        # Guarantor TCKN
        elif self.user_data['vehicle_value'] >= 5000000 and 'guarantor_tckn' not in self.user_data:
            tckn = self.extract_info_from_text(user_message, "tckn")
            if tckn:
                is_valid, error = self.validate_data('tckn', tckn, 'new')
                if is_valid:
                    self.user_data['guarantor_tckn'] = tckn
                    return {
                        "response": "Kefil TCKN kaydedildi. Son olarak istediğiniz finansman tutarını belirtiniz:",
                        "step": self.current_step,
                        "data": self.user_data
                    }
                else:
                    return {"response": error, "step": self.current_step, "data": self.user_data}

        # Loan amount
        elif 'loan_amount' not in self.user_data:
            amount = self.extract_info_from_text(user_message, "number")
            if amount:
                is_valid, error = self.validate_data('loan_amount', amount, 'new')
                if is_valid:
                    self.user_data['loan_amount'] = amount
                    self.current_step = "confirmation"
                    return {
                        "response": self._generate_confirmation_message(),
                        "step": self.current_step,
                        "data": self.user_data
                    }
                else:
                    return {"response": error, "step": self.current_step, "data": self.user_data}

        return {"response": "Lütfen geçerli bir değer giriniz.", "step": self.current_step, "data": self.user_data}

    def _handle_used_vehicle_collection(self, user_message: str) -> Dict[str, Any]:
        """Handle used vehicle information collection"""
        # Kasko value
        if 'vehicle_value' not in self.user_data:
            value = self.extract_info_from_text(user_message, "number")
            if value:
                self.user_data['vehicle_value'] = value
                return {
                    "response": f"Araç kasko değeri {value:,} TL olarak kaydedildi. Aracın yaşını belirtir misiniz?",
                    "step": self.current_step,
                    "data": self.user_data
                }

        # Vehicle age
        elif 'vehicle_age' not in self.user_data:
            age = self.extract_info_from_text(user_message, "number")
            if age:
                is_valid, error = self.validate_data('vehicle_age', age, 'used')
                if is_valid:
                    self.user_data['vehicle_age'] = age
                    return {
                        "response": "Araç yaşı kaydedildi. İstediğiniz finansman tutarını belirtiniz:",
                        "step": self.current_step,
                        "data": self.user_data
                    }
                else:
                    return {"response": error, "step": self.current_step, "data": self.user_data}

        # Loan amount
        elif 'loan_amount' not in self.user_data:
            amount = self.extract_info_from_text(user_message, "number")
            if amount:
                is_valid, error = self.validate_data('loan_amount', amount, 'used')
                if is_valid:
                    self.user_data['loan_amount'] = amount
                    return {
                        "response": "Finansman tutarı kaydedildi. Satıcı T.C. kimlik numarası var mı? (İsteğe bağlı - 'hayır' veya 'yok' diyebilirsiniz)",
                        "step": self.current_step,
                        "data": self.user_data
                    }
                else:
                    return {"response": error, "step": self.current_step, "data": self.user_data}

        # Seller TCKN (optional)
        elif 'seller_tckn' not in self.user_data:
            if any(word in user_message for word in ["hayır", "yok", "istemiyorum", "gerek yok"]):
                self.user_data['seller_tckn'] = None
            else:
                tckn = self.extract_info_from_text(user_message, "tckn")
                if tckn:
                    is_valid, error = self.validate_data('tckn', tckn, 'used')
                    if is_valid:
                        self.user_data['seller_tckn'] = tckn
                    else:
                        return {"response": error, "step": self.current_step, "data": self.user_data}
                else:
                    self.user_data['seller_tckn'] = None

            self.current_step = "confirmation"
            return {
                "response": self._generate_confirmation_message(),
                "step": self.current_step,
                "data": self.user_data
            }

        return {"response": "Lütfen geçerli bir değer giriniz.", "step": self.current_step, "data": self.user_data}

    def _generate_confirmation_message(self) -> str:
        """Generate confirmation message"""
        msg = "Başvuru bilgilerinizi kontrol ediniz:\n\n"

        if self.application_type == "new":
            msg += f"• Başvuru Türü: Yeni Araç\n"
            msg += f"• Araç Değeri: {self.user_data['vehicle_value']:,} TL\n"
            msg += f"• Araç Modeli: {self.user_data['vehicle_model']}\n"
            if 'guarantor_tckn' in self.user_data:
                msg += f"• Kefil TCKN: {self.user_data['guarantor_tckn']}\n"
            msg += f"• Finansman Tutarı: {self.user_data['loan_amount']:,} TL\n"
        else:
            msg += f"• Başvuru Türü: İkinci El Araç\n"
            msg += f"• Kasko Değeri: {self.user_data['vehicle_value']:,} TL\n"
            msg += f"• Araç Yaşı: {self.user_data['vehicle_age']} yıl\n"
            msg += f"• Finansman Tutarı: {self.user_data['loan_amount']:,} TL\n"
            if self.user_data.get('seller_tckn'):
                msg += f"• Satıcı TCKN: {self.user_data['seller_tckn']}\n"

        msg += "\nBilgiler doğru mu? 'Evet' derseniz başvurunuzu tamamlarım, 'Hayır' derseniz güncelleyebilirsiniz."
        return msg

    def save_application(self) -> Dict[str, Any]:
        """Save application to file"""
        application = {
            "id": f"APP_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "type": self.application_type,
            "data": self.user_data.copy(),
            "timestamp": datetime.now().isoformat(),
            "status": "pending"
        }

        try:
            applications = []
            if os.path.exists("applications.json"):
                with open("applications.json", "r", encoding="utf-8") as f:
                    applications = json.load(f)

            applications.append(application)

            with open("applications.json", "w", encoding="utf-8") as f:
                json.dump(applications, f, ensure_ascii=False, indent=2)

            return {"success": True, "application_id": application["id"]}
        except Exception as e:
            return {"success": False, "error": str(e)}


def init_session_state():
    """Initialize session state variables"""
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'api_key_validated' not in st.session_state:
        st.session_state.api_key_validated = False
    if 'current_step' not in st.session_state:
        st.session_state.current_step = "greeting"


def validate_api_key(api_key: str) -> bool:
    """Validate Google Gemini API key"""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        # Test with a simple prompt
        response = model.generate_content("Test")
        return True
    except Exception:
        return False


def display_chat_message(role: str, content: str):
    """Display a chat message with styling"""
    if role == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>👤 Siz:</strong><br>
            {content}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message bot-message">
            <strong>🤖 Bot:</strong><br>
            {content}
        </div>
        """, unsafe_allow_html=True)


def main():
    """Main Streamlit application"""
    init_session_state()

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🚗 Araç Finansmanı Chatbot</h1>
        <p>Yeni ve ikinci el araç finansmanı başvuruları için akıllı asistan</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar for API key and controls
    with st.sidebar:
        st.markdown("### ⚙️ Ayarlar")

        # API Key Input
        if not st.session_state.api_key_validated:
            st.markdown("#### 🔑 Google Gemini API Key")
            api_key = st.text_input(
                "API Key giriniz:",
                type="password",
                help="Google AI Studio'dan alacağınız ücretsiz API key"
            )

            if st.button("🔓 API Key Doğrula", type="primary"):
                if api_key:
                    with st.spinner("API Key doğrulanıyor..."):
                        if validate_api_key(api_key):
                            st.session_state.api_key_validated = True
                            st.session_state.chatbot = VehicleFinanceChatbot(api_key)
                            st.success("✅ API Key başarıyla doğrulandı!")
                            st.rerun()
                        else:
                            st.error("❌ Geçersiz API Key! Lütfen kontrol edin.")
                else:
                    st.warning("⚠️ Lütfen API Key giriniz.")

            # Instructions
            st.markdown("""
            #### 📖 API Key Nasıl Alınır?
            1. [Google AI Studio](https://aistudio.google.com/app/apikey) adresine gidin
            2. "Create API Key" butonuna tıklayın
            3. API Key'i kopyalayın ve yukarıya yapıştırın

            **Not**: Gemini 1.5 Flash modeli ücretsizdir!
            """)

        else:
            st.success("✅ API Key aktif")

            # Chat controls
            st.markdown("#### 🎮 Kontroller")
            if st.button("🔄 Sohbeti Yeniden Başlat"):
                st.session_state.messages = []
                st.session_state.chatbot = VehicleFinanceChatbot(st.session_state.chatbot.model.model_name)
                st.rerun()

            if st.button("🚪 Çıkış Yap"):
                st.session_state.api_key_validated = False
                st.session_state.chatbot = None
                st.session_state.messages = []
                st.rerun()

            # Statistics
            st.markdown("#### 📊 İstatistikler")
            try:
                if os.path.exists("applications.json"):
                    with open("applications.json", "r", encoding="utf-8") as f:
                        apps = json.load(f)

                    total_apps = len(apps)
                    new_apps = len([a for a in apps if a['type'] == 'new'])
                    used_apps = len([a for a in apps if a['type'] == 'used'])

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Toplam", total_apps)
                    with col2:
                        st.metric("Yeni", new_apps)
                    with col3:
                        st.metric("İkinci El", used_apps)
                else:
                    st.info("Henüz başvuru yok")
            except:
                st.info("İstatistik yüklenemedi")

    # Main chat interface
    if not st.session_state.api_key_validated:
        st.markdown("""
        <div class="info-card">
            <h3>🎯 Chatbot Özellikleri</h3>
            <ul>
                <li>✅ Yeni ve ikinci el araç finansmanı</li>
                <li>✅ Otomatik veri doğrulama</li>
                <li>✅ Türkçe doğal dil işleme</li>
                <li>✅ Adım adım başvuru süreci</li>
                <li>✅ Çapraz satış fırsatları</li>
                <li>✅ Güvenli veri saklama</li>
            </ul>
            <p><strong>👈 Başlamak için soldaki panelden API Key giriniz!</strong></p>
        </div>
        """, unsafe_allow_html=True)
        return

    # Chat interface
    st.markdown("### 💬 Sohbet")

    # Display chat messages
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            display_chat_message(message["role"], message["content"])

    # Chat input
    user_input = st.chat_input("Mesajınızı yazın... (Başlamak için 'merhaba' yazın)")

    if user_input:
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Process message with chatbot
        try:
            with st.spinner("Bot yanıt hazırlıyor..."):
                result = st.session_state.chatbot.process_message(user_input)
                bot_response = result["response"]

                # Add bot response to chat
                st.session_state.messages.append({"role": "bot", "content": bot_response})

                # Handle special cases
                if result.get("should_exit", False):
                    st.balloons()
                    time.sleep(2)
                    st.session_state.messages = []
                    st.session_state.chatbot = VehicleFinanceChatbot(st.session_state.chatbot.model._model_name)

                # Handle confirmation step
                elif st.session_state.chatbot.current_step == "confirmation":
                    # Show current application data
                    if st.session_state.chatbot.user_data:
                        with st.expander("📋 Başvuru Detayları", expanded=True):
                            data = st.session_state.chatbot.user_data
                            app_type = st.session_state.chatbot.application_type

                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("**Başvuru Türü:**")
                                st.write("Yeni Araç" if app_type == "new" else "İkinci El Araç")

                                if app_type == "new":
                                    st.markdown("**Araç Değeri:**")
                                    st.write(f"{data.get('vehicle_value', 0):,} TL")
                                    st.markdown("**Araç Modeli:**")
                                    st.write(data.get('vehicle_model', 'Belirtilmedi'))
                                else:
                                    st.markdown("**Kasko Değeri:**")
                                    st.write(f"{data.get('vehicle_value', 0):,} TL")
                                    st.markdown("**Araç Yaşı:**")
                                    st.write(f"{data.get('vehicle_age', 0)} yıl")

                            with col2:
                                st.markdown("**Finansman Tutarı:**")
                                st.write(f"{data.get('loan_amount', 0):,} TL")

                                if data.get('guarantor_tckn'):
                                    st.markdown("**Kefil TCKN:**")
                                    st.write(data['guarantor_tckn'])

                                if data.get('seller_tckn'):
                                    st.markdown("**Satıcı TCKN:**")
                                    st.write(data['seller_tckn'])

                    # Confirmation buttons
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("✅ Evet, Onayla", type="primary", use_container_width=True):
                            # Save application
                            save_result = st.session_state.chatbot.save_application()
                            if save_result["success"]:
                                st.success(f"🎉 Başvurunuz kaydedildi! ID: {save_result['application_id']}")

                                # Cross-selling opportunity
                                st.info("💡 HGS ürünümüzü de almak ister misiniz?")

                                # Reset for new application
                                time.sleep(3)
                                st.session_state.messages.append({
                                    "role": "bot",
                                    "content": f"Başvurunuz başarıyla kaydedildi! Başvuru No: {save_result['application_id']}\n\nYeni bir başvuru için 'merhaba' yazabilirsiniz."
                                })

                                # Reset chatbot
                                api_key = st.session_state.chatbot.model._api_key if hasattr(
                                    st.session_state.chatbot.model, '_api_key') else None
                                if api_key:
                                    st.session_state.chatbot = VehicleFinanceChatbot(api_key)

                                st.balloons()
                            else:
                                st.error(f"❌ Başvuru kaydedilemedi: {save_result['error']}")

                    with col2:
                        if st.button("❌ Hayır, Güncelle", use_container_width=True):
                            st.session_state.messages.append({
                                "role": "bot",
                                "content": "Hangi bilgiyi güncellemek istiyorsunuz? Lütfen metin olarak belirtin."
                            })
                            st.rerun()

        except Exception as e:
            st.error(f"❌ Hata oluştu: {str(e)}")
            st.session_state.messages.append({
                "role": "bot",
                "content": "Üzgünüm, bir hata oluştu. Lütfen tekrar deneyin."
            })

        st.rerun()

    # Quick start buttons
    if len(st.session_state.messages) == 0:
        st.markdown("### 🚀 Hızlı Başlangıç")
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("👋 Merhaba", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": "Merhaba"})
                st.rerun()

        with col2:
            if st.button("🆕 Yeni Araç", use_container_width=True):
                st.session_state.messages.extend([
                    {"role": "user", "content": "Merhaba"},
                    {"role": "user", "content": "Yeni araç"}
                ])
                st.rerun()

        with col3:
            if st.button("🔄 İkinci El Araç", use_container_width=True):
                st.session_state.messages.extend([
                    {"role": "user", "content": "Merhaba"},
                    {"role": "user", "content": "İkinci el araç"}
                ])
                st.rerun()

    # Footer information
    with st.expander("ℹ️ Önemli Bilgiler"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **Yeni Araç Finansmanı:**
            - Maksimum araç değeri: 7M TL
            - Finansman oranı: %60'a kadar
            - 5M TL üzeri için kefil gerekli
            - Ticari araçlar hariç
            """)

        with col2:
            st.markdown("""
            **İkinci El Araç Finansmanı:**
            - Maksimum araç yaşı: 5 yıl
            - Finansman oranı: %40'a kadar
            - Maksimum tutar: 3M TL
            - Kasko değeri baz alınır
            """)

        st.markdown("""
        **Komutlar:**
        - `çık` - Uygulamadan çıkış
        - `iptal` - Başvuruyu sıfırla
        - `yeniden başla` - Baştan başla
        """)


# Create default config file if it doesn't exist
def create_default_config():
    """Create default configuration file"""
    if not os.path.exists("chatbot_config.json"):
        config = {
            "finance_rules": {
                "new": {
                    "max_vehicle_value": 7000000,
                    "max_financing_ratio": 0.6,
                    "guarantor_threshold": 5000000
                },
                "used": {
                    "max_vehicle_age": 5,
                    "max_financing_ratio": 0.4,
                    "max_loan_amount": 3000000
                }
            },
            "faq": {
                "supported_brands": "Tüm marka ve modeller (ticari araçlar hariç)",
                "interest_rates": "Güncel piyasa koşullarına göre belirlenir",
                "loan_terms": "12-60 ay vade seçenekleri"
            }
        }

        with open("chatbot_config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    create_default_config()
    main()