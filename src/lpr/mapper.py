import re

def is_arabic_letter(char):
    # Basic check if character is Arabic letter (including potential transliterations if model uses them)
    # This is a simplification; we assume OCR returns proper names or arabic chars.
    # Typical arabic unicode range: '\u0600' to '\u06FF'
    if not char:
        return False
    if '\u0600' <= char <= '\u06ff':
        return True
    
    # If the model returns english names for arabic letters
    arabic_letter_names = ["alif", "baa", "taa", "thaa", "jeem", "haa", "khaa", "dal", "thal", "raa", "zaay", "seen", "sheen", "saad", "daad", "taa", "zaa", "ayn", "ghayn", "faa", "qaaf", "kaaf", "laam", "meem", "noon", "haa", "waw", "yaa"]
    if char.lower() in arabic_letter_names:
        return True
        
    return False

def is_digit(char):
    return char.isdigit() or (char >= '\u0660' and char <= '\u0669') # Arabic numerals check

def classify_governorate(letters, digits):
    """
    Classify the governate based on Egyptian License Plate rules.
    letters: list of characters from OCR (from right to left in physical space)
    digits: list of digits from OCR (from right to left in physical space)
    """
    letter_count = len(letters)
    digit_count = len(digits)
    
    if letter_count == 0 and digit_count == 0:
        return "Unknown"

    plate_letters_str = "".join(letters)

    # Basic structural rules for Major ones
    if letter_count == 3 and digit_count == 3:
        return "Cairo"
    elif letter_count == 2 and digit_count == 4:
        return "Giza"
    elif letter_count == 3 and digit_count == 4:
        # Other governorates depend on the leading letter(s) 
        # (right-most physically, which should be letters[0] if sorted correctly Right-To-Left)
        first_letter = letters[0] if letters else ""
        
        # We try to match with known prefixes (assumes arabic characters are handled)
        if first_letter == "س":
            return "Alexandria"
        elif first_letter == "ر":
            return "Sharqia"
        elif first_letter == "د":
            return "Dakahlia"
        elif first_letter == "م":
            return "Monufia"
        elif first_letter == "ب":
            return "Beheira"
        elif first_letter == "ل":
            return "Kafr El-Sheikh"
        elif first_letter == "ع":
            return "Gharbia"
        elif first_letter == "ق":
            return "Qalyubia"
        elif first_letter == "ف":
            return "Fayoum"
        elif first_letter == "و":
            return "Beni Suef"
        elif first_letter == "ن":
            return "Minya"
        elif first_letter == "ي":
            return "Assiut"
        elif first_letter == "ه" or first_letter == "هـ":
            return "Sohag"
            
        # 2-letter prefix for some governorates
        prefix_2 = "".join(letters[0:2]) if len(letters) > 1 else ""
        
        if prefix_2 == "طس":
            return "Suez"
        elif prefix_2 == "طص":
            return "Ismailia"
        elif prefix_2 == "طع":
            return "Port Said"
        elif prefix_2 == "طد":
            return "Damietta"
        elif prefix_2 == "طا":
            return "North Sinai"
        elif prefix_2 == "طج":
            return "South Sinai"
        elif prefix_2 == "طر":
            return "Red Sea"
        elif prefix_2 == "جه" or prefix_2 == "جهـ":
            return "Matrouh"
        elif prefix_2 == "جو":
            return "New Valley"
        elif prefix_2 == "صا":
            return "Qena"
        elif prefix_2 == "صق":
            return "Luxor"
        elif prefix_2 == "صو":
            return "Aswan"
            
        return "Other (3L 4D)"

    return "Unknown"
