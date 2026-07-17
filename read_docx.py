import zipfile
import xml.etree.ElementTree as ET
import os

def read_docx(file_path):
    if not os.path.exists(file_path):
        return f"File not found: {file_path}"
    try:
        with zipfile.ZipFile(file_path) as docx:
            xml_content = docx.read('word/document.xml')
            root = ET.fromstring(xml_content)
            
            # Namespace map
            ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
            
            # Extract text
            text_parts = []
            for para in root.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p'):
                para_text = "".join(node.text for node in para.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t') if node.text)
                if para_text:
                    text_parts.append(para_text)
            return "\n".join(text_parts)
    except Exception as e:
        return f"Error reading {file_path}: {e}"

if __name__ == "__main__":
    for filename in ["MathMatch_Day1_Notes.docx", "MathMatch_Day2_Notes.docx"]:
        txt_name = filename.replace(".docx", ".txt")
        text = read_docx(filename)
        with open(txt_name, "w") as f:
            f.write(text)
        print(f"Wrote {txt_name} with {len(text)} characters")
