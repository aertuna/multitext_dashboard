# 🧠 MultiText Dashboard

MultiText Dashboard is a lightweight Flask-based web application that allows users to upload Excel files and compare input sentences against structured data. It is designed for **text analysis**, **keyword-based filtering**, and **real-time matching** using Python and Pandas.

---

## 🚀 Features

- 📁 Upload Excel (`.xlsx`) files via the web interface
- ✍️ Enter multiple sentences or keywords for analysis
- ✅ Case sensitivity toggle for refined control
- 🔍 Logical text matching (e.g., AND/OR keyword filters)
- 📊 Display matching rows from the uploaded Excel file
- 🖥️ Clean, minimal dashboard UI (Jinja2 + HTML/CSS)

---

## 🛠 Technologies Used

- Python 3
- Flask
- Pandas
- OpenPyXL
- HTML5 / CSS3 (Jinja2 templating)
- JavaScript (basic usage for UI control)

---

## 📦 Installation

1. **Clone the repository**

```bash
git clone https://github.com/aertuna/multitext_dashboard.git
cd multitext_dashboard
```

2. **Create a virtual environment (optional but recommended)**

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Run the application**

```bash
python app.py
```

5. **Open in browser**

Go to [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 📂 Project Structure

```
multitext_dashboard/
│
├── app.py                # Main Flask app
├── requirements.txt      # Python dependencies
├── templates/            # HTML templates (index.html)
├── static/               # CSS, JS, and assets
│   └── style.css
├── uploads/              # Uploaded Excel files (temporary)
└── README.md
```

---

## 🧪 How It Works

1. The user uploads an Excel file (with text data in columns).
2. The user inputs one or more sentences or phrases.
3. The backend reads the Excel using Pandas.
4. A comparison is made between the user input and the Excel data.
5. Matching results are displayed in a formatted table.

---

## 📸 Screenshot

<img width="1680" height="1050" alt="Ekran Resmi 2025-05-06 22 48 41 (1)" src="https://github.com/user-attachments/assets/a2c5359e-fb30-46e5-875f-29cbffc6cdd4" />
<img width="1680" height="1050" alt="Ekran Resmi 2025-05-06 22 48 31 (1)" src="https://github.com/user-attachments/assets/184471d5-e5ce-48d9-a884-022138f65fc4" />
<img width="1680" height="1050" alt="Ekran Resmi 2025-05-06 22 48 23 (1)" src="https://github.com/user-attachments/assets/24797105-a9b8-49c9-b6d3-2f68f898051c" />
<img width="1680" height="1050" alt="Ekran Resmi 2025-05-06 22 47 30 (1)" src="https://github.com/user-attachments/assets/55efb959-47de-4d55-8f7a-2909c05be74d" />


---

## ❓ Use Cases

- Keyword-based document or dataset scanning
- Simple NLP preprocessing tasks
- Internal tools for data filtering from Excel
- Legal or academic text review dashboards

---

## 📬 Contact

**Author**: Alperen ERTUNA  
🔗 GitHub: [github.com/aertuna](https://github.com/aertuna)

Feel free to open issues or submit pull requests for improvements.

---
