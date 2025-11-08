from flask import Flask, render_template, Response, jsonify
import joblib
from scapy.all import sniff, IP, TCP, UDP, Raw
from threading import Thread, Lock
import json
import re
import nltk
from nltk.corpus import stopwords
import logging
import os
import time
import random
from datetime import datetime
import psutil
import ctypes
import ssl

def check_and_display_admin_status():
    if is_admin():
        logger.info("✅ Running with Administrator privileges - Real capture AVAILABLE")
        logger.info("📡 You can now use 'Real Capture' mode")
    else:
        logger.warning("⚠️  Running WITHOUT Administrator privileges")
        logger.warning("💡 Real capture will not work - use Simulation mode")
        logger.info("💡 To enable real capture: Run VS Code as Administrator")

# Handle SSL for NLTK
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_unverified_https_context = _create_unverified_https_context

# Configure Flask app and logging
app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download NLTK data if needed
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
    stop_words = set(stopwords.words('english'))
except LookupError:
    print("📥 Downloading NLTK data...")
    nltk.download('stopwords')
    nltk.download('punkt')
    stop_words = set(stopwords.words('english'))

# Global variables
model = None
vectorizer = None
live_results = []
live_results_lock = Lock()
sniffing_active = False
simulation_active = True
packet_count = 0
current_interface = None

def is_admin():
    """Check if running as administrator"""
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

def get_active_interface():
    """Find the best active network interface"""
    try:
        interfaces = psutil.net_if_stats()
        net_io = psutil.net_io_counters(pernic=True)
        
        # Prefer WiFi and Ethernet interfaces
        preferred_interfaces = ['wi-fi', 'wifi', 'wireless', 'ethernet', 'local area connection']
        
        for interface_name, stats in interfaces.items():
            if (stats.isup and 
                interface_name != 'lo' and 
                'virtual' not in interface_name.lower() and
                'npcap' not in interface_name.lower() and
                'bluetooth' not in interface_name.lower()):
                
                # Check if interface has traffic
                has_traffic = False
                if interface_name in net_io:
                    if net_io[interface_name].bytes_sent > 0 or net_io[interface_name].bytes_recv > 0:
                        has_traffic = True
                
                # Priority to preferred interfaces with traffic
                if any(pref in interface_name.lower() for pref in preferred_interfaces) and has_traffic:
                    logger.info(f"✅ Preferred interface found: {interface_name}")
                    return interface_name
        
        # Fallback to any interface with traffic
        for interface_name, stats in interfaces.items():
            if stats.isup and interface_name != 'lo':
                has_traffic = False
                if interface_name in net_io:
                    if net_io[interface_name].bytes_sent > 0 or net_io[interface_name].bytes_recv > 0:
                        has_traffic = True
                
                if has_traffic:
                    logger.info(f"✅ Fallback interface: {interface_name}")
                    return interface_name
        
        logger.warning("❌ No active network interface found")
        return None
        
    except Exception as e:
        logger.error(f"Error finding interface: {e}")
        return None

def test_interface_connectivity(interface_name):
    """Test if interface can capture packets"""
    logger.info(f"🧪 Testing interface: {interface_name}")
    
    test_packets = []
    
    def test_callback(packet):
        if packet.haslayer(IP):
            test_packets.append(packet)
            return len(test_packets) >= 2
    
    try:
        sniff(iface=interface_name, prn=test_callback, timeout=5, store=False)
        
        if test_packets:
            logger.info(f"✅ Successfully captured {len(test_packets)} real packets")
            return True
        else:
            logger.warning("⚠️ No packets captured (interface may be idle)")
            return True
    except Exception as e:
        logger.error(f"❌ Interface test failed: {e}")
        return False

def load_model():
    """Load trained model and vectorizer"""
    global model, vectorizer
    try:
        if os.path.exists('model.pkl') and os.path.exists('vectorizer.pkl'):
            model = joblib.load('model.pkl')
            vectorizer = joblib.load('vectorizer.pkl')
            logger.info("✅ Model and vectorizer loaded successfully")
            return True
        else:
            logger.error("❌ Model files not found. Run train_model.py first.")
            return False
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        return False

def preprocess_text(text):
    """Clean and preprocess text data"""
    if not isinstance(text, str):
        text = str(text)
    
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = text.lower()
    words = text.split()
    filtered = [w for w in words if w not in stop_words and len(w) > 1]
    return ' '.join(filtered)

def packet_handler(packet):
    """Handle incoming real packets and classify them"""
    global model, vectorizer, live_results, sniffing_active, packet_count

    if not sniffing_active or model is None:
        return

    try:
        if not packet.haslayer(IP):
            return
            
        ip_layer = packet[IP]
        ip_src = ip_layer.src
        ip_dst = ip_layer.dst
        
        protocol = "TCP" if packet.haslayer(TCP) else "UDP" if packet.haslayer(UDP) else "OTHER"
        
        sport, dport = '', ''
        if packet.haslayer(TCP):
            sport = packet[TCP].sport
            dport = packet[TCP].dport
        elif packet.haslayer(UDP):
            sport = packet[UDP].sport
            dport = packet[UDP].dport
        
        payload = ''
        if packet.haslayer(Raw):
            try:
                raw_load = bytes(packet[Raw].load)
                payload = raw_load[:50].decode('utf-8', errors='ignore')
            except:
                payload = str(raw_load[:30])
        
        raw_text = f"{protocol} {ip_src}:{sport} -> {ip_dst}:{dport}"
        if payload:
            raw_text += f" {payload}"
        
        clean_text = preprocess_text(raw_text)
        
        if len(clean_text.strip()) > 5:
            vec = vectorizer.transform([clean_text])
            prediction = model.predict(vec)[0]
            probabilities = model.predict_proba(vec)[0]
            confidence = max(probabilities) * 100
            
            packet_count += 1
            result = {
                "id": packet_count,
                "text": clean_text[:80] + "..." if len(clean_text) > 80 else clean_text,
                "prediction": prediction,
                "confidence": round(confidence, 2),
                "src": ip_src,
                "dst": ip_dst,
                "sport": str(sport),
                "dport": str(dport),
                "protocol": protocol,
                "timestamp": datetime.now().isoformat(),
                "real_packet": True,
            }
            
            with live_results_lock:
                live_results.append(result)
            
            if prediction == 'malicious' and confidence > 70:
                logger.warning(f"🚨 MALICIOUS ({confidence:.1f}%): {ip_src} -> {ip_dst}")
            else:
                logger.info(f"📡 REAL {prediction.upper()} ({confidence:.1f}%): {ip_src} -> {ip_dst}")
            
    except Exception as e:
        logger.debug(f"Packet processing error: {e}")

def simulate_packets():
    """Simulate network packets for testing"""
    global live_results, simulation_active, packet_count

    sample_packets = [
        {
            "text": "GET /index.html HTTP/1.1 Host: example.com User-Agent: Mozilla/5.0",
            "pred": "benign", 
            "src": "192.168.1.100", "dst": "93.184.216.34",
            "sport": "54321", "dport": "80", "protocol": "TCP"
        },
        {
            "text": "POST /api/login HTTP/1.1 Content-Type: application/json",
            "pred": "benign",
            "src": "192.168.1.101", "dst": "203.0.113.1", 
            "sport": "54322", "dport": "443", "protocol": "TCP"
        },
        {
            "text": "GET /admin/exec.php?cmd=whoami HTTP/1.1",
            "pred": "malicious",
            "src": "10.0.0.50", "dst": "192.168.1.1",
            "sport": "4444", "dport": "80", "protocol": "TCP"
        },
        {
            "text": "DNS Query for google.com",
            "pred": "benign",
            "src": "192.168.1.105", "dst": "8.8.8.8",
            "sport": "54323", "dport": "53", "protocol": "UDP"
        }
    ]

    while simulation_active:
        packet = random.choice(sample_packets)
        packet_count += 1
        
        base_conf = 85 if packet["pred"] == "benign" else 75
        confidence = random.randint(base_conf - 10, base_conf + 5)
        
        result = {
            "id": packet_count,
            "text": packet["text"],
            "prediction": packet["pred"],
            "confidence": confidence,
            "src": packet["src"],
            "dst": packet["dst"],
            "sport": packet["sport"],
            "dport": packet["dport"],
            "protocol": packet["protocol"],
            "timestamp": datetime.now().isoformat(),
            "real_packet": False
        }
        
        with live_results_lock:
            live_results.append(result)
        
        logger.info(f"🎮 SIM {packet['pred'].upper()} ({confidence}%): {packet['src']} -> {packet['dst']}")
        time.sleep(random.uniform(1.5, 3.0))

def start_real_capture():
    """Start real packet capture on active interface"""
    global sniffing_active, current_interface, simulation_active
    
    if not is_admin():
        logger.error("❌ ADMIN RIGHTS REQUIRED!")
        return False
    
    simulation_active = False
    time.sleep(1)
    sniffing_active = True
    
    try:
        active_interface = get_active_interface()
        if not active_interface:
            return False
        
        current_interface = active_interface
        
        logger.info(f"🎯 Selected interface: {active_interface}")
        
        logger.info("🧪 Testing interface connectivity...")
        test_success = test_interface_connectivity(active_interface)
        
        if not test_success:
            return False
        
        logger.info("✅ Interface test passed! Starting real capture...")
        
        logger.info(f"🔍 Starting REAL packet capture on: {active_interface}")
        sniff(iface=active_interface, prn=packet_handler, store=0, filter="ip")
        return True
        
    except Exception as e:
        logger.error(f"❌ Real capture failed: {e}")
        return False

def start_simulation():
    """Start packet simulation"""
    global simulation_active, sniffing_active
    sniffing_active = False
    simulation_active = True
    sim_thread = Thread(target=simulate_packets, daemon=True)
    sim_thread.start()
    logger.info("🎮 Started packet simulation")
    return True

# Flask Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/stream')
def stream():
    def event_stream():
        while True:
            with live_results_lock:
                if live_results:
                    result = live_results.pop(0)
                    yield f"data: {json.dumps(result)}\n\n"
            time.sleep(0.1)
    return Response(event_stream(), mimetype="text/event-stream")

@app.route('/status')
def status():
    return jsonify({
        "model_loaded": model is not None,
        "sniffing_active": sniffing_active,
        "simulation_active": simulation_active,
        "total_packets": packet_count,
        "current_interface": current_interface,
        "is_admin": is_admin()
    })

@app.route('/start_real')
def start_real():
    if not is_admin():
        return jsonify({
            "status": "error", 
            "message": "🔐 ADMIN RIGHTS REQUIRED! Run as Administrator."
        })
    
    simulation_active = False
    time.sleep(1)
    
    try:
        sniff_thread = Thread(target=start_real_capture, daemon=True)
        sniff_thread.start()
        return jsonify({
            "status": "success", 
            "message": "Real packet capture started"
        })
    except Exception as e:
        return jsonify({
            "status": "error", 
            "message": f"Failed to start real capture: {str(e)}"
        })

@app.route('/start_simulation')
def start_sim():
    global simulation_active, sniffing_active
    sniffing_active = False
    time.sleep(1)
    
    success = start_simulation()
    if success:
        return jsonify({
            "status": "success", 
            "message": "Simulation mode activated"
        })
    else:
        return jsonify({
            "status": "error", 
            "message": "Failed to start simulation"
        })

@app.route('/stop_capture')
def stop_capture():
    global sniffing_active, simulation_active
    sniffing_active = False
    simulation_active = False
    return jsonify({
        "status": "success", 
        "message": "Capture stopped"
    })

@app.route('/check_admin')
def check_admin():
    return jsonify({"is_admin": is_admin()})

if __name__ == '__main__':
    check_and_display_admin_status()
    if load_model():
        start_simulation()
        logger.info("🚀 Starting web server on http://localhost:5000")
        logger.info("💡 Access: http://localhost:5000")
        app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
    else:
        logger.error("❌ Failed to start - train model first: python train_model.py")
