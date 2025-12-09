# Gen-AI Powered UI/UX Design Platform

This project is a Gen-AI powered platform for UI/UX design, featuring voice input, text input, language selection, and AI-generated web content.

## Features

- **Dual Input Methods**: Voice-to-text input and text input
- **Multilingual Support**: Support for 10+ Indian languages
- **AI-Powered HTML Generation**: Creates beautiful HTML/Tailwind CSS websites
- **Smart Fallbacks**: Automatic fallback to lighter models and template generation
- **Error Handling**: Robust error handling with retry logic
- **Instant Preview**: Live HTML preview in the browser
- **Database Integration**: Vector similarity search for similar designs

## How to Run Locally

1. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```

2. Set up environment variables (optional):
    - Create a `.env` file with your API keys
    - Or modify the API keys directly in `Works.py` (not recommended for production)

3. Start the app:
    ```sh
    streamlit run Works.py
    ```

4. Open your browser and go to:  
   [http://localhost:8501](http://localhost:8501)

## Project Structure

- `Works.py` - Main Streamlit application
- `local.py` - Local embedding generation script
- `uploader.py` - Database upload utility
- `style.css` - Custom styling
- `requirements.txt` - Python dependencies

## Technologies Used

- **Streamlit** - Web framework
- **Gemini API** - LLM for HTML generation
- **Sentence Transformers** - Embeddings fallback
- **SingleStoreDB** - Vector database
- **Tailwind CSS** - Styling framework

## Features in Detail

### Input Methods
- **Text Input**: Type your website description directly
- **Voice Input**: Record audio in multiple languages

### Smart Model Selection
- Primary: Gemini 1.5 Flash (lighter, faster)
- Fallback: Template-based generation if APIs fail
- Automatic retry with exponential backoff

### Error Handling
- Rate limit detection and automatic retry
- Graceful fallback to template generation
- Clear error messages and status indicators

## Repository

[GitHub Repo](https://github.com/Chaithraaradhya/genaiabl)

## License

This project is open source and available for use. 