
from pathlib import Path
from src.scanner_effect import DocumentScanner

if __name__ == "__main__":
    base_dir = Path(__file__).parent.resolve()
    input_dir = base_dir / 'input'
    output_dir = base_dir / 'output'
    
    # Create input directory if it doesn't exist, just in case
    input_dir.mkdir(parents=True, exist_ok=True)
    
    app = DocumentScanner(input_dir, output_dir)
    app.run()
