#!/usr/bin/env python3
from fpdf import FPDF
import re

class MarkdownPDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Algorithm Comparison Guide', 0, 0, 'C')
        self.ln(15)
    
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def clean_line(line):
    """Remove emojis and problematic unicode"""
    # Remove common emojis
    line = re.sub(r'[\U0001F300-\U0001F9FF]', '', line)
    # Remove other problematic symbols
    line = re.sub(r'[✅❌⚠️★✓]', '', line)
    return line

print("Converting markdown to PDF...")

pdf = MarkdownPDF()
pdf.add_page()
pdf.set_auto_page_break(auto=True, margin=15)

with open('algorithm_comparison_guide.md', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for line in lines:
    line = line.rstrip()
    
    if not line.strip():
        pdf.ln(3)
        continue
    
    # Clean the line
    line = clean_line(line)
    
    # Skip table separator lines
    if re.match(r'^[\|\-\s]+$', line):
        continue
    
    # Handle headers
    if line.startswith('# '):
        pdf.set_font('Arial', 'B', 16)
        text = line[2:].strip()
    elif line.startswith('## '):
        pdf.set_font('Arial', 'B', 13)
        text = line[3:].strip()
    elif line.startswith('### '):
        pdf.set_font('Arial', 'B', 11)
        text = line[4:].strip()
    elif line.startswith('#### '):
        pdf.set_font('Arial', 'B', 10)
        text = line[5:].strip()
    elif line.startswith('- ') or line.startswith('* '):
        pdf.set_font('Arial', '', 9)
        text = '  • ' + line[2:].strip()
    elif line.startswith('|'):
        pdf.set_font('Courier', '', 8)
        text = line
    else:
        pdf.set_font('Arial', '', 9)
        text = line
    
    # Write the line
    try:
        pdf.multi_cell(0, 5, text)
    except Exception as e:
        # If encoding fails, try latin-1
        try:
            clean_text = text.encode('latin-1', 'replace').decode('latin-1')
            pdf.multi_cell(0, 5, clean_text)
        except:
            # Last resort - skip the line
            pass

pdf.output('algorithm_comparison_guide.pdf')
print('✓ Successfully created algorithm_comparison_guide.pdf')
