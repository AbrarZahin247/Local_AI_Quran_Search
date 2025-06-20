# Quranic Semantic Search Engine (Bangla)

This project is a fully local, AI-powered search engine for exploring the Holy Quran in Bengali. It uses a hybrid search approach (combining semantic and keyword search) and a local Large Language Model (LLM) via Ollama to provide context-aware answers and summaries based on the provided text.

The entire system runs on your local machine, ensuring 100% privacy.

![Application Screenshot](app_screenshot.PNG)
*(This screenshot shows the main user interface of the application.)*

## Features

-   **Interactive Web Interface:** A user-friendly search page built with Flask.
-   **Hybrid Search:** Combines the power of semantic (vector) search to understand meaning and lexical (keyword) search for precision.
-   **AI-Powered Summaries:** Uses a local LLM (e.g., Gemma, Llama 3) to generate coherent summaries from the retrieved text chunks.
-   **Source Attribution:** Clearly shows the Surah and Ayah range for every piece of text shown.
-   **100% Local & Private:** Your data and search queries never leave your computer.
-   **Smart Chunking:** Groups verses together to provide better context for search and summarization.

## Technology Stack

-   **Backend:** Python, Flask
-   **AI / ML:** LangChain, Sentence-Transformers, PyTorch
-   **Vector Search:** FAISS (from Meta AI)
-   **Keyword Search:** rank-bm25
-   **Local LLM:** Ollama

## Prerequisites

Before you begin, ensure you have the following installed on your system:
1.  **Miniconda or Anaconda:** For managing Python environments. [Download Miniconda](https://docs.conda.io/en/latest/miniconda.html)
2.  **Git:** For cloning the repository (optional, if you are sharing via Git).

## Installation & Setup Guide

Follow these steps carefully to set up and run the project on your local machine.

### Step 1: Clone the Repository

First, get the project files. If the project is on GitHub, clone it. Otherwise, simply copy the project folder.

```bash
git clone <your-repository-url>
cd <repository-folder>
```

### Step 2: Install and Set Up Ollama

This application requires the Ollama service to be running to provide LLM capabilities.

1.  **Download Ollama:** Go to the official website [https://ollama.com](https://ollama.com) and download the installer for your operating system (Windows, macOS, or Linux).
2.  **Install Ollama:** Run the installer. On Windows, ensure you allow it to add `ollama` to your system PATH if prompted.
3.  **Start the Ollama Application:**
    -   **On Windows/macOS:** Run the Ollama desktop application from your Start Menu or Applications folder. You should see its icon appear in your system tray/menu bar. This means the server is running in the background.
    -   **On Linux:** Open a terminal and run `ollama serve`. You must leave this terminal running.
4.  **Download an LLM Model:** Now, you need to download a model for Ollama to use. We recommend Google's `gemma:2b` for its balance of performance and size. Open a **new terminal** and run:
    ```bash
    ollama pull gemma:2b
    ```
    This will download the model, which may take some time.

### Step 3: Create the Conda Environment

We will create a dedicated Python environment to keep all dependencies organized.

1.  Open your terminal (Anaconda Prompt for Windows).
2.  Navigate to the project folder.
3.  Create and activate the new environment:
    ```bash
    # Create an environment named 'quran_search' with Python 3.11
    conda create -n quran_search python=3.11 -y

    # Activate the environment
    conda activate quran_search
    ```

### Step 4: Install Dependencies

1.  **Install Core Libraries with Conda:** For best performance and compatibility, install PyTorch and FAISS using Conda from the `pytorch` channel.
    ```bash
    conda install pytorch faiss-cpu -c pytorch -y
    ```
2.  **Install Remaining Libraries with Pip:** Install the rest of the application's dependencies.
    ```bash
    pip install flask langchain langchain-community sentence-transformers rank-bm25 ollama
    ```

### Step 5: Prepare the Search Index

The application will create a search index from your `quran_text.txt` file.

-   If a folder named `faiss_index_bangla` already exists in your project directory, **delete it**. This ensures a fresh, correct index is built the first time you run the app.

### Step 6: Run the Application

You are now ready to launch the web server!

```bash
python app.py
```

The terminal will show that the models are loading, and finally, it will say the server is running on `http://127.0.0.1:5000`.

## Usage

1.  Make sure the **Ollama application is running** in the background.
2.  Run `python app.py` from your activated `quran_search` conda environment.
3.  Open your web browser and navigate to **http://127.0.0.1:5000**.
4.  Type your query in Bengali into the search box and click "অনুসন্ধান করুন".

## Project Structure

```
.
├── faiss_index_bangla/  # Auto-generated folder for the vector search index.
├── static/              # Contains the CSS file for styling the web page.
│   └── style.css
├── templates/           # Contains the HTML file for the web page.
│   └── index.html
├── app.py               # The main Flask application with all backend logic.
├── quran_text.txt       # The source data file containing the Quranic text.
├── app_screenshot.png   # An image showing the application's user interface.
└── README.md            # This file.
```

## Troubleshooting (Common Errors)

-   **Error: `Connection Refused` or `Failed to establish a new connection`**
    -   **Cause:** The Ollama server is not running.
    -   **Solution:** Start the Ollama desktop application or run `ollama serve` in a separate terminal.

-   **Error: `404 Model Not Found`**
    -   **Cause:** The specific model (e.g., `gemma:2b`) has not been downloaded yet.
    -   **Solution:** Run `ollama pull <model_name>` in your terminal (e.g., `ollama pull gemma:2b`).

-   **Error: `Only one usage of each socket address... is normally permitted.`**
    -   **Cause:** You are trying to run `ollama serve` when the Ollama server (likely from the desktop app) is already running.
    -   **Solution:** You don't need to do anything! The server is already running. Just proceed to run `python app.py`.

-   **Error: `'ollama' is not recognized as an internal or external command`**
    -   **Cause:** The Ollama installation directory was not added to your system's PATH.
    -   **Solution:** The easiest fix is to reinstall Ollama, ensuring any "Add to PATH" options are checked. Alternatively, just use the desktop app to start the server instead of the command line."# Local_AI_Quran_Search" 
