import pytest
from app import allowed_file

def test_allowed_file():
    # Test valid extensions
    assert allowed_file("document.pdf") == True
    assert allowed_file("image.jpg") == True
    assert allowed_file("notes.txt") == True
    assert allowed_file("data.docx") == True
    
    # Test invalid extensions
    assert allowed_file("malicious.exe") == False
    assert allowed_file("script.sh") == False
    assert allowed_file("no_extension_file") == False