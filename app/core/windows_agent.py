import google.generativeai as genai
from typing import Dict, Any
import os
import re
from dotenv import load_dotenv
from app.config.settings import GEMINI_MODEL

class WindowsCommandAgent:
    def __init__(self):
        """Initialize the Windows Command Agent."""
        load_dotenv()
        
        # Configure Gemini AI
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
            
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(GEMINI_MODEL)
        
    def is_windows_query(self, query: str) -> bool:
        """
        Determine if the query is related to Windows commands.
        
        Args:
            query (str): The user's query
            
        Returns:
            bool: True if the query is Windows-related, False otherwise
        """
        windows_indicators = [
            r'\b(cmd|command prompt|powershell|batch|\.bat|\.ps1)\b',
            r'\bwindows\b',
            r'\b(dir|copy|xcopy|robocopy|del|rd|md|mkdir|type|echo|set)\b'
        ]
        
        query = query.lower()
        return any(re.search(pattern, query, re.IGNORECASE) for pattern in windows_indicators)
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a Windows-related query using Gemini AI.
        
        Args:
            query (str): The user's query
            
        Returns:
            Dict[str, Any]: Response containing the generated answer
        """
        prompt = f'''
You are a Windows Command Line Assistant. Format your response using strict markdown structure.
Your response MUST follow this exact format:

### Command Overview
[Brief description of the command's purpose and functionality]

### Syntax
```cmd
[command] [options] [arguments]
```

### Key Options
- `/option`: [Description]
- `/long-option`: [Description]

### Examples
```cmd
# Example 1: Basic usage
[command] [/option] [value]

# Example 2: Advanced usage
[command] [/long-option] [value]
```

### Notes
[Important considerations or warnings about Windows-specific behavior]

Based on the query: "{query}"

Remember:
1. ALWAYS include all five sections (Command Overview, Syntax, Key Options, Examples, Notes)
2. Use ```cmd for code blocks (not bash)
3. Use /switches for Windows commands (not -options)
4. Include at least 2 examples with comments
5. Mention CMD vs PowerShell compatibility if relevant
'''
        
        try:
            print("\n=== DEBUG: Sending prompt to Gemini ===")
            print(prompt)
            
            response = self.model.generate_content(prompt)
            
            print("\n=== DEBUG: Raw Response from Gemini ===")
            print(response.text)
            
            # Ensure the response has the correct markdown structure
            response_text = response.text.strip()
            if not response_text.startswith('### Command Overview'):
                print("\n=== DEBUG: Adding missing header ===")
                response_text = f"### Command Overview\n{response_text}"
            
            print("\n=== DEBUG: Final Processed Response ===")
            print(response_text)
            print("\n=== DEBUG: End of Response ===\n")
            
            return {
                'response': response_text,
                'source': 'gemini',
                'is_windows_command': True,
                'platform': 'Windows'
            }
            
        except Exception as e:
            print(f"\n=== DEBUG: Error occurred ===")
            print(f"Error: {str(e)}")
            return {
                'error': f"Error processing Windows command query: {str(e)}",
                'is_windows_command': True
            } 