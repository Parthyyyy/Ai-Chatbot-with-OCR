import pytest
from app import allowed_file
from email_validator import is_valid_email_format, clean_email

def test_allowed_file():
    assert allowed_file("document.pdf") == True
    assert allowed_file("notes.txt") == True
    assert allowed_file("malicious.exe") == False

def test_email_validator():
    assert is_valid_email_format("test@example.com") == True
    assert is_valid_email_format("invalid-email") == False
    
def test_clean_email():
    assert clean_email("@@test@example.com..") == "test@example.com"