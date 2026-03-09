#!/usr/bin/env python3
"""
Convert Python source files to PDF with proper formatting
"""
from fpdf import FPDF
import sys
import os

class CodePDF(FPDF):
    def __init__(self):
        super().__init__()
        self.add_page()
        self.set_auto_page_break(auto=True, margin=15)
    
    def header(self):
        self.set_font('Courier', 'B', 10)
        self.set_text_color(0, 0, 0)
    
    def footer(self):
        self.set_y(-15)
        self.set_font('Courier', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def convert_py_to_pdf(input_file, output_file):
    """Convert a Python file to PDF with monospace font"""
    print(f"Converting {input_file} to {output_file}...")
    
    # Create PDF object
    pdf = CodePDF()
    
    # Set title based on filename
    pdf.set_title(os.path.basename(input_file))
    
    # Add title page
    pdf.set_font('Courier', 'B', 16)
    pdf.cell(0, 10, os.path.basename(input_file), 0, 1, 'C')
    pdf.ln(5)
    
    # Read source file
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {input_file}: {e}")
        return False
    
    # Set font for code
    pdf.set_font('Courier', '', 8)
    pdf.set_text_color(0, 0, 0)
    
    # Process line by line
    lines = content.split('\n')
    for i, line in enumerate(lines, 1):
        # Replace tabs with spaces
        line = line.replace('\t', '    ')
        
        # Handle empty lines
        if not line.strip():
            pdf.ln(4)
            continue
        
        # Handle very long lines by splitting
        if len(line) > 95:
            # Split long lines
            chunks = [line[i:i+95] for i in range(0, len(line), 95)]
            for chunk in chunks:
                try:
                    pdf.cell(0, 4, chunk, 0, 1)
                except Exception as e:
                    # Handle encoding issues
                    try:
                        clean_chunk = chunk.encode('latin-1', 'replace').decode('latin-1')
                        pdf.cell(0, 4, clean_chunk, 0, 1)
                    except:
                        pdf.cell(0, 4, '[encoding error]', 0, 1)
        else:
            try:
                pdf.cell(0, 4, line, 0, 1)
            except Exception as e:
                # Handle encoding issues
                try:
                    clean_line = line.encode('latin-1', 'replace').decode('latin-1')
                    pdf.cell(0, 4, clean_line, 0, 1)
                except:
                    pdf.cell(0, 4, '[encoding error]', 0, 1)
    
    # Save PDF
    try:
        pdf.output(output_file)
        print(f"✓ Successfully created {output_file}")
        return True
    except Exception as e:
        print(f"✗ Error creating PDF: {e}")
        return False

def main():
    # Files to convert
    files = [
        'algorithms_detail.py',
        'algorithms_example.py',
        'chess.py'
    ]
    
    base_dir = '/Users/kusk/Desktop/Algorithm Design'
    success_count = 0
    
    for filename in files:
        input_path = os.path.join(base_dir, filename)
        output_path = os.path.join(base_dir, filename.replace('.py', '.pdf'))
        
        if not os.path.exists(input_path):
            print(f"✗ File not found: {input_path}")
            continue
        
        if convert_py_to_pdf(input_path, output_path):
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"Conversion complete: {success_count}/{len(files)} files converted successfully")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
