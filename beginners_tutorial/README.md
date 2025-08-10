## 🤖 AI-Powered Structured Data Extraction

A simple project using the **Google LangExtract** library to demonstrate extracting entities and relationships from unstructured text. This project processes a sample text and outputs a structured JSON file and an interactive HTML visualization.

-----

### ✨ Features

  * **LLM Extraction**: Uses the Google Gemini model to identify and extract key entities and relationships.
  * **Structured Output**: Generates a JSON Lines (`.jsonl`) file with a consistent schema.
  * **Interactive Visualization**: Creates an HTML file to visually inspect and verify the extracted entities, with highlighting on the source text.
  * **Customizable**: Easily modify the extraction task by editing the prompt and few-shot examples.

-----

### 🛠️ Prerequisites

  * Python 3.7+
  * Google AI Studio API Key
  * A **`requirements.txt`** file in the project directory, which includes:
      * `langextract`
      * `libmagic`

-----

### 🔑 How to Create and Configure Your API Key

1.  Go to your [Google AI Studio API Key page](https://aistudio.google.com/app/apikey).
2.  Click **"Create API key"** and choose a new project if necessary.
3.  Copy the generated key.
4.  Set the API key as an environment variable (recommended for security):
      * **Windows**:
        ```bash
        setx LANGEXTRACT_API_KEY "YOUR_API_KEY_HERE"
        ```
      * **macOS/Linux**:
        ```bash
        export LANGEXTRACT_API_KEY="YOUR_API_KEY_HERE"
        ```
      * Alternatively, you can pass the key directly in the script using the `api_key` parameter.

-----

### 📥 Installation

1.  Clone the repository and navigate to the project directory:
    ```bash
    git clone https://github.com/jenilsoni-ai/google-langextract.git
    cd beginners_tutorial/
    ```
2.  Install dependencies from the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

-----

### 🚀 Usage

1.  Run the main script from your terminal:
    ```bash
    python langextract_tutorial.py
    ```
2.  The script will:
      * Process the input text using the Gemini model.
      * Print the extracted entities and relationships in a formatted JSON output to the console.
      * Create a file named `extraction_results.jsonl` in your directory.
      * Create a file named `visualization.html`. Open this file in your web browser to view the interactive visualization.

-----

### ⚙️ Customizing the Extraction

You can easily adapt this project to different domains by editing the extraction task in `langextract_tutorial.py`:

  * **Change the Prompt**: Modify the `prompt` variable to describe a different extraction task.
  * **Update the Examples**: Provide new `lx.data.ExampleData` examples that match your new prompt and the data you want to extract.
  * **Change the Model**: You can experiment with different Gemini models by changing the `model_id` parameter in the `lx.extract()` function.

-----

### 🛠️ Troubleshooting

  * **`API key must be provided...`**: Ensure you have set the `LANGEXTRACT_API_KEY` environment variable correctly, or pass the key directly in the code as shown in the **"How to Create and Configure"** section above.
  * **`'charmap' codec can't encode characters`**: This is a Windows-specific encoding error. When saving the HTML file, ensure you specify `encoding='utf-8'`:
    ```python
    with open("visualization.html", "w", encoding='utf-8') as f:
        f.write(html_content)
    ```

-----
