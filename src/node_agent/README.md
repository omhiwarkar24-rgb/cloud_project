# Live Node Monitoring Agent

This folder contains the standalone Live Node Monitoring Agent for the **Adaptive Cloud Security & Predictive Maintenance System**.

---

## 🚀 Quick Start on Node Laptop

### Option 1: Double-Click Launcher (Windows)
Double-click **`run_node_agent.bat`**.
* It will ask for the Central Server IP address (default: `10.175.163.77`).
* Press **Enter** to connect!

---

### Option 2: Run via Command Prompt / Terminal
Open Command Prompt in this folder and specify your Central Server's IP address:

```bash
# Point directly to the Central Server laptop:
python collector.py --server-url http://10.175.163.77:8000

# Or using the shorthand IP flag:
python collector.py --server-ip 10.175.163.77
```

---

### Option 3: Using a text file
Create a file named `server_url.txt` in this folder containing:
```text
http://10.175.163.77:8000
```
Then simply run:
```bash
python collector.py
```

---

## 📋 Requirements on the Node Laptop
The Node Agent only requires:
```bash
pip install psutil
```
*(Standard Python libraries `urllib`, `socket`, `platform`, `json`, and `threading` are built into Python).*

---

## 🔍 How to Find Central Server Laptop IP
On your Central Server laptop (where the dashboard is running):
1. Open Command Prompt.
2. Run: `ipconfig`
3. Look for your Wi-Fi or Ethernet **IPv4 Address** (e.g. `10.175.163.77` or `192.168.1.X`).
4. Both laptops must be connected to the same Wi-Fi or local network.
